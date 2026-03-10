#!/usr/bin/env python3
"""
Extended Video Score Calculator
Supports two modes:
1. Single directory mode: Compare videos in ORIG_DIR vs ADAPT_DIR
2. Parent directory mode: Compare videos in subdirectories of ORIG_PARENT_DIR vs ADAPT_PARENT_DIR

Calculates three metrics:
- SSF (Single-Step Fidelity): Cosine similarity between video embeddings
  * Higher is better, 1.0 is perfect
- SS-FD (Single-Step Fréchet Distance): Fréchet distance between distributions
  * Lower is better, 0.0 is perfect
- DVS (Distribution Variance Score): Trace(Sigma_Adapted) / Trace(Sigma_Baseline)
  * Measures relative variance/spread of the distribution
  * DVS = 1.0: Same variance as baseline
  * DVS > 1.0: More variance (more spread out)
  * DVS < 1.0: Less variance (more concentrated)
"""

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
import numpy as np
from scipy import linalg
from tqdm import tqdm
from PIL import Image
import cv2
import json
import argparse
from collections import defaultdict

# --- 1. Fréchet Distance Helper ---
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Fréchet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might have complex components
    try:
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
    except linalg.LinAlgError:
        # Handle cases where the matrix might be singular
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# --- 2. Core Logic: Video Embedding ---
@torch.no_grad()
def get_video_embedding(video_path, clip_model, clip_processor, device):
    """
    Reads a video file frame by frame, gets CLIP embeddings
    for each frame, and returns the mean-pooled embedding
    for the entire video.
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    # 1. Read all frames from video
    batch_of_images = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame from OpenCV's BGR format to PIL's RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch_of_images.append(Image.fromarray(frame_rgb))

    cap.release()

    if not batch_of_images:
        print(f"Warning: No frames read from {video_path}")
        return None

    # 2. Process all decoded images for CLIP
    inputs = clip_processor(
        images=batch_of_images,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # 3. Get embeddings for all frames in batches
    all_frame_embeddings = []

    # Process in CLIP batches to avoid OOM
    CLIP_BATCH_SIZE = 64
    pixel_values = inputs['pixel_values']

    for i in range(0, len(pixel_values), CLIP_BATCH_SIZE):
        i_end = min(i + CLIP_BATCH_SIZE, len(pixel_values))
        batch = pixel_values[i:i_end]
        frame_embeddings = clip_model.get_image_features(pixel_values=batch)
        all_frame_embeddings.append(frame_embeddings)

    # Concat all frame embeddings: [num_frames, embedding_dim]
    all_frame_embeddings = torch.cat(all_frame_embeddings, dim=0)

    # 4. Average to get the single semantic embedding for the video
    video_embedding = torch.mean(all_frame_embeddings, dim=0)

    return video_embedding


def calculate_scores_for_directory(orig_dir, adapt_dir, clip_model, clip_processor, device, file_extension="*.mp4"):
    """
    Calculate SSF and SS-FD scores for videos in a single directory pair.

    Returns:
        dict with keys: 'ssf', 'ss_fd', 'pair_count'
    """
    total_ssf_score = 0.0
    pair_count = 0
    all_orig_embeddings = []
    all_adapt_embeddings = []

    orig_files = sorted(list(orig_dir.glob(file_extension)))
    if not orig_files:
        print(f"Warning: No files found in {orig_dir} with extension {file_extension}")
        return {'ssf': 0.0, 'ss_fd': 0.0, 'pair_count': 0}

    for orig_path in tqdm(orig_files, desc=f"Processing {orig_dir.name}", leave=False):
        adapt_path = adapt_dir / orig_path.name

        if not adapt_path.exists():
            print(f"Warning: Skipping {orig_path.name}, no matching file in {adapt_dir}")
            continue

        # Get the single semantic vector for each video
        emb_orig = get_video_embedding(orig_path, clip_model, clip_processor, device)
        emb_adapt = get_video_embedding(adapt_path, clip_model, clip_processor, device)

        if emb_orig is None or emb_adapt is None:
            print(f"Skipping pair {orig_path.name} due to read error.")
            continue

        # Store for SS-FD calculation later
        all_orig_embeddings.append(emb_orig)
        all_adapt_embeddings.append(emb_adapt)

        # Calculate per-prompt score
        score = F.cosine_similarity(emb_orig.unsqueeze(0), emb_adapt.unsqueeze(0))

        total_ssf_score += score.item()
        pair_count += 1

    if pair_count == 0:
        print(f"Warning: No matching pairs found in {orig_dir.name}")
        return {'ssf': 0.0, 'ss_fd': 0.0, 'pair_count': 0}

    avg_ssf_score = total_ssf_score / pair_count

    # Calculate SS-FD and DVS
    feat_orig = torch.stack(all_orig_embeddings).detach().cpu().numpy()
    feat_adapt = torch.stack(all_adapt_embeddings).detach().cpu().numpy()

    mu_orig = np.mean(feat_orig, axis=0)
    sigma_orig = np.cov(feat_orig, rowvar=False)

    mu_adapt = np.mean(feat_adapt, axis=0)
    sigma_adapt = np.cov(feat_adapt, rowvar=False)

    ss_fd_score = calculate_frechet_distance(mu_orig, sigma_orig, mu_adapt, sigma_adapt)

    # Calculate DVS (Distribution Variance Score)
    # DVS = Trace(Sigma_Adapted) / Trace(Sigma_Baseline)
    # Measures relative variance/spread of the distribution
    # DVS > 1.0: Adapted has more variance (more spread out)
    # DVS = 1.0: Same variance
    # DVS < 1.0: Adapted has less variance (more concentrated)
    trace_orig = np.trace(sigma_orig)
    trace_adapt = np.trace(sigma_adapt)
    dvs_score = trace_adapt / trace_orig if trace_orig > 0 else 0.0

    return {
        'ssf': float(avg_ssf_score),
        'ss_fd': float(ss_fd_score),
        'dvs': float(dvs_score),
        'pair_count': pair_count,
        'orig_embeddings': all_orig_embeddings,  # For total calculation
        'adapt_embeddings': all_adapt_embeddings
    }


def calculate_total_scores(all_embeddings_dict):
    """
    Calculate total SSF and SS-FD scores across all categories.

    Args:
        all_embeddings_dict: Dict mapping category -> {'orig_embeddings': [], 'adapt_embeddings': []}

    Returns:
        dict with keys: 'ssf', 'ss_fd', 'pair_count'
    """
    # Collect all embeddings across categories
    all_orig_embeddings = []
    all_adapt_embeddings = []

    for category, emb_data in all_embeddings_dict.items():
        all_orig_embeddings.extend(emb_data['orig_embeddings'])
        all_adapt_embeddings.extend(emb_data['adapt_embeddings'])

    if not all_orig_embeddings:
        return {'ssf': 0.0, 'ss_fd': 0.0, 'pair_count': 0}

    pair_count = len(all_orig_embeddings)

    # Calculate total SSF
    total_ssf_score = 0.0
    for emb_orig, emb_adapt in zip(all_orig_embeddings, all_adapt_embeddings):
        score = F.cosine_similarity(emb_orig.unsqueeze(0), emb_adapt.unsqueeze(0))
        total_ssf_score += score.item()

    avg_ssf_score = total_ssf_score / pair_count

    # Calculate total SS-FD and DVS
    feat_orig = torch.stack(all_orig_embeddings).detach().cpu().numpy()
    feat_adapt = torch.stack(all_adapt_embeddings).detach().cpu().numpy()

    mu_orig = np.mean(feat_orig, axis=0)
    sigma_orig = np.cov(feat_orig, rowvar=False)

    mu_adapt = np.mean(feat_adapt, axis=0)
    sigma_adapt = np.cov(feat_adapt, rowvar=False)

    ss_fd_score = calculate_frechet_distance(mu_orig, sigma_orig, mu_adapt, sigma_adapt)

    # Calculate DVS (Distribution Variance Score)
    trace_orig = np.trace(sigma_orig)
    trace_adapt = np.trace(sigma_adapt)
    dvs_score = trace_adapt / trace_orig if trace_orig > 0 else 0.0

    return {
        'ssf': float(avg_ssf_score),
        'ss_fd': float(ss_fd_score),
        'dvs': float(dvs_score),
        'pair_count': pair_count
    }


def main():
    parser = argparse.ArgumentParser(
        description='Calculate video similarity scores (SSF and SS-FD)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single directory mode
  python video_score_calculator_extended.py \\
    --orig_dir output/clean_42 \\
    --adapt_dir output/epoch1000 \\
    --output_file scores.json

  # Parent directory mode (with subdirectories)
  python video_score_calculator_extended.py \\
    --orig_parent_dir output/onestep/clean_category_42 \\
    --adapt_parent_dir output/onestep/20251008_20-21-45/epoch1000 \\
    --output_file scores_epoch1000.json
        """
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--orig_dir', type=str, help='Original video directory (single dir mode)')
    mode_group.add_argument('--orig_parent_dir', type=str, help='Original video parent directory (parent dir mode)')

    parser.add_argument('--adapt_dir', type=str, help='Adapted video directory (required for single dir mode)')
    parser.add_argument('--adapt_parent_dir', type=str, help='Adapted video parent directory (required for parent dir mode)')

    parser.add_argument('--output_file', type=str, required=True, help='Output JSON file path for scores')
    parser.add_argument('--file_extension', type=str, default='*.mp4', help='Video file extension pattern (default: *.mp4)')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-large-patch14', help='CLIP model name')

    args = parser.parse_args()

    # Validate arguments
    if args.orig_dir and not args.adapt_dir:
        parser.error("--adapt_dir is required when using --orig_dir")
    if args.orig_parent_dir and not args.adapt_parent_dir:
        parser.error("--adapt_parent_dir is required when using --orig_parent_dir")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load CLIP model
    print(f"Loading CLIP model: {args.clip_model}...")
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)

    results = {}

    # --- Single Directory Mode ---
    if args.orig_dir:
        print("\n" + "="*80)
        print("MODE: Single Directory")
        print("="*80)

        orig_dir = Path(args.orig_dir)
        adapt_dir = Path(args.adapt_dir)

        if not orig_dir.exists():
            print(f"Error: Original directory does not exist: {orig_dir}")
            return
        if not adapt_dir.exists():
            print(f"Error: Adapted directory does not exist: {adapt_dir}")
            return

        print(f"Original: {orig_dir}")
        print(f"Adapted:  {adapt_dir}")

        scores = calculate_scores_for_directory(
            orig_dir, adapt_dir, clip_model, clip_processor, device, args.file_extension
        )

        results = {
            'ssf': scores['ssf'],
            'ss_fd': scores['ss_fd'],
            'dvs': scores['dvs'],
            'pair_count': scores['pair_count']
        }

        print("\n" + "="*80)
        print("RESULTS:")
        print("="*80)
        print(f"SSF Score:   {results['ssf']:.6f} (higher is better, 1.0 is perfect)")
        print(f"SS-FD Score: {results['ss_fd']:.6f} (lower is better, 0.0 is perfect)")
        print(f"DVS Score:   {results['dvs']:.6f} (1.0 is same variance, >1.0 more spread, <1.0 more concentrated)")
        print(f"Pair Count:  {results['pair_count']}")
        print("="*80)

    # --- Parent Directory Mode ---
    else:  # args.orig_parent_dir
        print("\n" + "="*80)
        print("MODE: Parent Directory (with subdirectories)")
        print("="*80)

        orig_parent = Path(args.orig_parent_dir)
        adapt_parent = Path(args.adapt_parent_dir)

        if not orig_parent.exists():
            print(f"Error: Original parent directory does not exist: {orig_parent}")
            return
        if not adapt_parent.exists():
            print(f"Error: Adapted parent directory does not exist: {adapt_parent}")
            return

        print(f"Original Parent: {orig_parent}")
        print(f"Adapted Parent:  {adapt_parent}")

        # Discover subdirectories
        orig_subdirs = sorted([d for d in orig_parent.iterdir() if d.is_dir()])

        if not orig_subdirs:
            print(f"Error: No subdirectories found in {orig_parent}")
            return

        print(f"\nFound {len(orig_subdirs)} subdirectories in original parent")

        # Calculate scores for each subdirectory
        category_results = {}
        all_embeddings = {}

        for orig_subdir in tqdm(orig_subdirs, desc="Processing categories"):
            category = orig_subdir.name
            adapt_subdir = adapt_parent / category

            if not adapt_subdir.exists():
                print(f"Warning: Skipping category '{category}', no matching subdirectory in adapted parent")
                continue

            print(f"\n📂 Processing category: {category}")

            scores = calculate_scores_for_directory(
                orig_subdir, adapt_subdir, clip_model, clip_processor, device, args.file_extension
            )

            category_results[category] = {
                'ssf': scores['ssf'],
                'ss_fd': scores['ss_fd'],
                'dvs': scores['dvs'],
                'pair_count': scores['pair_count']
            }

            # Store embeddings for total calculation
            if scores['pair_count'] > 0:
                all_embeddings[category] = {
                    'orig_embeddings': scores['orig_embeddings'],
                    'adapt_embeddings': scores['adapt_embeddings']
                }

            print(f"  SSF: {scores['ssf']:.6f}, SS-FD: {scores['ss_fd']:.6f}, DVS: {scores['dvs']:.6f}, Pairs: {scores['pair_count']}")

        # Calculate total scores across all categories
        print("\n" + "="*80)
        print("Calculating TOTAL scores across all categories...")
        print("="*80)

        total_scores = calculate_total_scores(all_embeddings)

        results = {
            'categories': category_results,
            'total': {
                'ssf': total_scores['ssf'],
                'ss_fd': total_scores['ss_fd'],
                'dvs': total_scores['dvs'],
                'pair_count': total_scores['pair_count']
            }
        }

        # Print summary
        print("\n" + "="*80)
        print("RESULTS BY CATEGORY:")
        print("="*80)
        for category in sorted(category_results.keys()):
            cat_scores = category_results[category]
            print(f"{category:15s} | SSF: {cat_scores['ssf']:.6f} | SS-FD: {cat_scores['ss_fd']:.6f} | DVS: {cat_scores['dvs']:.6f} | Pairs: {cat_scores['pair_count']}")

        print("\n" + "="*80)
        print("TOTAL RESULTS:")
        print("="*80)
        print(f"SSF Score:   {total_scores['ssf']:.6f} (higher is better, 1.0 is perfect)")
        print(f"SS-FD Score: {total_scores['ss_fd']:.6f} (lower is better, 0.0 is perfect)")
        print(f"DVS Score:   {total_scores['dvs']:.6f} (1.0 is same variance, >1.0 more spread, <1.0 more concentrated)")
        print(f"Total Pairs: {total_scores['pair_count']}")
        print("="*80)

    # Save results to JSON file
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
