import argparse
import torch
import torch.nn.functional as F
# Original CLIP imports
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
# --- NEW: X-CLIP imports ---
from transformers import XCLIPModel, XCLIPProcessor as XCLIPProcessorClass
import cv2
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- 1. Re-usable Helper Functions ---

def get_all_frames_as_pil(video_path):
    """
    Decodes a video file frame-by-frame and returns a list of PIL Images.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None

        batch_of_images = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            batch_of_images.append(pil_image)
        
        cap.release()

        if not batch_of_images:
            print(f"Error: No frames read from {video_path}")
            return None
        
        return batch_of_images

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

# --- We no longer need sample_frames_from_list ---

@torch.no_grad()
def get_text_embedding(text_prompt, clip_text_model, clip_tokenizer, device):
    """
    Gets the CLIP text embedding for a single prompt string. (For Metric 1)
    """
    inputs = clip_tokenizer(
        [text_prompt],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    text_embedding = clip_text_model.get_text_features(**inputs)
    return text_embedding.squeeze(0) # Remove batch dim

# --- 3. Main Script ---
def main(videos_dir, prompt_file):
    # --- Configuration ---
    VIDEOS_DIR = Path(videos_dir)
    PROMPT_FILE = Path(prompt_file)
    VIDEO_EXTENSION = "*.mp4"
    WINDOW_SIZE = 32 # X-CLIP expects 32 frames
    # ---------------------

    # --- 1. Load All Models ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Metric 1 Models (Frame-Averaged CLIP) ---
    print("Loading CLIP (ViT-L/14) models...")
    clip_model_name = "openai/clip-vit-large-patch14"
    clip_image_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_text_model = clip_image_model
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    
    # --- Metric 2 Models (X-CLIP) ---
    print("Loading X-CLIP (microsoft/xclip-base-patch16-zero-shot) models...")
    xclip_model_name = "microsoft/xclip-base-patch16-zero-shot"
    xclip_model = XCLIPModel.from_pretrained(xclip_model_name).to(device)
    xclip_processor = XCLIPProcessorClass.from_pretrained(xclip_model_name)
    
    print("All models loaded.")

    # --- 2. Load Prompts from File ---
    # (This section is unchanged)
    if not PROMPT_FILE.exists():
        print(f"Error: Prompt file not found at {PROMPT_FILE}")
        return
    print(f"Loading prompts from {PROMPT_FILE}...")
    try:
        prompts = PROMPT_FILE.read_text().splitlines()
        prompts = [p.strip() for p in prompts if p.strip()]
        if not prompts:
            print("Error: Prompt file is empty.")
            return
        print(f"Loaded {len(prompts)} prompts.")
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return

    # --- 3. Find video paths ---
    # (This section is unchanged)
    video_files = list(VIDEOS_DIR.glob(f"**/{VIDEO_EXTENSION}"))
    if not video_files:
        print(f"Error: No video files with extension {VIDEO_EXTENSION} found in {VIDEOS_DIR}")
        return
    print(f"Found {len(video_files)} videos to evaluate...")

    # --- Initialize lists for both scores ---
    all_frame_avg_scores = []
    all_xclip_scores = []
    
    text_embedding_cache = {}

    for video_path in tqdm(video_files, desc="Calculating All Scores"):
        
        # --- 4. Get Text Prompt from Filename Index ---
        # (This section is unchanged)
        video_id_str = video_path.stem
        try:
            video_index = int(video_id_str)
            prompt_index_in_list = video_index - 1
            if not (0 <= prompt_index_in_list < len(prompts)):
                print(f"Warning: Skipping {video_path.name}. ID '{video_index}' out of range.")
                continue
            text_prompt = prompts[prompt_index_in_list]
        except ValueError:
            print(f"Warning: Skipping {video_path.name}. Filename stem not a valid integer.")
            continue
        
        # --- 5. Get All Video Frames ---
        all_frames = get_all_frames_as_pil(video_path)
        if all_frames is None:
            print(f"Warning: Skipping {video_path.name}, could not read frames.")
            continue
        
        total_frames = len(all_frames)

        # --- 6. CALCULATE METRIC 1 (Frame-Averaged CLIP Score) ---
        # (This section is unchanged, it correctly uses *all* frames)
        try:
            if text_prompt not in text_embedding_cache:
                text_embedding_cache[text_prompt] = get_text_embedding(
                    text_prompt, clip_text_model, clip_tokenizer, device
                )
            text_emb = text_embedding_cache[text_prompt]

            inputs = clip_processor(
                images=all_frames,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            with torch.no_grad():
                all_frame_embeddings = clip_image_model.get_image_features(**inputs)
                video_emb = torch.mean(all_frame_embeddings, dim=0)
            
            video_emb_norm = F.normalize(video_emb, p=2, dim=0)
            text_emb_norm = F.normalize(text_emb, p=2, dim=0)
            
            frame_avg_score = F.cosine_similarity(video_emb_norm, text_emb_norm, dim=0)
            all_frame_avg_scores.append(frame_avg_score.item())
        except Exception as e:
            print(f"Warning: Failed to calculate Metric 1 for {video_path.name}: {e}")

        # --- 7. CALCULATE METRIC 2 (X-CLIP Score with Sliding Window) ---
        try:
            # Check if video is long enough for at least one window
            if total_frames < WINDOW_SIZE:
                print(f"Warning: Skipping X-CLIP for {video_path.name}, "
                      f"has only {total_frames} frames (need {WINDOW_SIZE}).")
                continue

            # --- This is the new sliding window logic ---
            # 1. Define the frame-slices for the 3 windows
            start_mid = (total_frames - WINDOW_SIZE) // 2
            
            clip_windows = [
                all_frames[0:WINDOW_SIZE],                             # First 32
                all_frames[start_mid : start_mid + WINDOW_SIZE],       # Middle 32
                all_frames[total_frames - WINDOW_SIZE : total_frames]  # Last 32
            ]
            
            window_scores = []
            
            # 2. Calculate score for each window
            for window in clip_windows:
                inputs = xclip_processor(
                    text=[text_prompt],
                    videos=[window], # Use the 32-frame window
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                with torch.no_grad():
                    outputs = xclip_model(**inputs)
                
                score = outputs.logits_per_video[0][0].item()
                window_scores.append(score)
            
            # 3. Choose the MAX score from the 3 windows
            max_score = max(window_scores)
            all_xclip_scores.append(max_score)

        except Exception as e:
            print(f"Warning: Failed to calculate Metric 2 for {video_path.name}: {e}")


    # --- 8. Calculate Final Average Scores ---
    print("\n--- Evaluation Complete ---")
    print(f"Checkpoint Directory: {VIDEOS_DIR.name}")
    print(f"Total Videos Evaluated: {len(all_frame_avg_scores)}") # Assumes both ran

    if all_frame_avg_scores:
        final_avg_frame_clip_score = np.mean(all_frame_avg_scores)
        print(f"Average Frame-Avg CLIP Score (Thematic): {final_avg_frame_clip_score:.6f}")
    
    if all_xclip_scores:
        final_avg_xclip_score = np.mean(all_xclip_scores)
        print(f"Average X-CLIP Score (Temporal):    {final_avg_xclip_score:.6f} (Max-Window)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate CLIP and X-CLIP scores for generated videos.")
    parser.add_argument("--videos_dir", required=True, help="Directory containing generated .mp4 files.")
    parser.add_argument("--prompt_file", default="metric/high_quality_prompts_96.txt", help="Text file with one prompt per line.")
    args = parser.parse_args()
    main(args.videos_dir, args.prompt_file)