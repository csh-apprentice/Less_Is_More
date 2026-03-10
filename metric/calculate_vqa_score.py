import argparse
import t2v_metrics
import numpy as np
from pathlib import Path
from tqdm import tqdm

def main(videos_dir, prompt_file):
    # --- Configuration ---
    VIDEOS_DIR = Path(videos_dir)
    PROMPT_FILE = Path(prompt_file)

    # Optional: Change to "*.gif" or "*.avi" if needed
    VIDEO_EXTENSION = "*.mp4"
    # ---------------------

    # --- 1. Load VQA Model ---
    print("Loading VQAScore (qwen2.5-vl-7b)...")
    # This will load the model, which may take some time and VRAM.
    # The library handles device placement (e.g., to CUDA if available).
    try:
        vqa_scorer = t2v_metrics.VQAScore(model='qwen2.5-vl-7b')
    except Exception as e:
        print(f"Error: Failed to load VQAScore model: {e}")
        print("Please ensure 't2v_metrics' and its dependencies (like transformers, torch) are installed.")
        return
    print("VQAScore model loaded.")

    # --- 2. Load Prompts from File ---
    if not PROMPT_FILE.exists():
        print(f"Error: Prompt file not found at {PROMPT_FILE}")
        return
        
    print(f"Loading prompts from {PROMPT_FILE}...")
    try:
        prompts = PROMPT_FILE.read_text().splitlines()
        prompts = [p.strip() for p in prompts if p.strip()] # Clean up lines
        if not prompts:
            print("Error: Prompt file is empty.")
            return
        print(f"Loaded {len(prompts)} prompts.")
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return

    # --- 3. Find video paths ---
    video_files = list(VIDEOS_DIR.glob(f"**/{VIDEO_EXTENSION}"))
    if not video_files:
        print(f"Error: No video files with extension {VIDEO_EXTENSION} found in {VIDEOS_DIR}")
        return
        
    print(f"Found {len(video_files)} videos to evaluate...")

    all_vqa_scores = []
    
    # --- 4. Loop Through Videos and Calculate VQA Score ---
    for video_path in tqdm(video_files, desc="Calculating VQA Scores"):
        
        # --- Get Text Prompt from Filename Index ---
        video_id_str = video_path.stem
        
        try:
            # Convert "00001" -> 1
            video_index = int(video_id_str)
            # Get the prompt from the list (1-indexed)
            prompt_index_in_list = video_index - 1
            
            if not (0 <= prompt_index_in_list < len(prompts)):
                print(f"Warning: Skipping {video_path.name}. ID '{video_index}' is out of range for prompts file (1 to {len(prompts)}).")
                continue
                
            text_prompt = prompts[prompt_index_in_list]
            
        except ValueError:
            print(f"Warning: Skipping {video_path.name}. Filename stem '{video_id_str}' is not a valid integer.")
            continue
        
        # --- 5. Calculate VQA Score ---
        try:
            # VQAScore is simple: it takes the video path (as a string) and text.
            # It returns a 1x1 tensor, so we use .item() to get the float.
            vqa_score = vqa_scorer(
                images=[str(video_path)], # Pass the file path directly
                texts=[text_prompt],
                fps=8.0 # Use recommended 8.0 fps for Qwen
            ).item()
            
            all_vqa_scores.append(vqa_score)
        
        except Exception as e:
            print(f"Warning: Failed to calculate VQA Score for {video_path.name}: {e}")

    # --- 6. Calculate Final Average Score ---
    if not all_vqa_scores:
        print("Error: No VQA scores were calculated.")
        return
        
    final_avg_vqa_score = np.mean(all_vqa_scores)
    
    print("\n--- Evaluation Complete ---")
    print(f"Checkpoint Directory: {VIDEOS_DIR.name}")
    print(f"Total Videos Evaluated: {len(all_vqa_scores)}")
    print(f"Average VQA Score (Qwen-VL): {final_avg_vqa_score:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate VQA score for generated videos.")
    parser.add_argument("--videos_dir", required=True, help="Directory containing generated .mp4 files.")
    parser.add_argument("--prompt_file", default="metric/high_quality_prompts_96.txt", help="Text file with one prompt per line.")
    args = parser.parse_args()
    main(args.videos_dir, args.prompt_file)