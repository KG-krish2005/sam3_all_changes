import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import numpy as np
import argparse
import torch
import pandas as pd
import sys
import gc
from math import floor
import supervision as sv

try:
    from sam3.model_builder import build_sam3_video_predictor
except ImportError:
    print("CRITICAL ERROR: 'sam3' library not found.")
    sys.exit(1)

# --- CONFIGURATION ---
DEFAULT_SAM_CONFIG = "sam3_hiera_large.yaml"
DEFAULT_SAM_CHECKPOINT = "checkpoints/sam3_hiera_large.pt"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_PROMPT = "person"

def get_id_suffix(n):
    n = int(n) - 1
    result = ""
    while n >= 0:
        result = chr(ord('a') + (n % 26)) + result
        n = floor(n / 26) - 1
    return result

# --- CORE LOGIC (UNCHANGED) ---
def process_single_video(input_path, output_path, args):
    """
    Process a single video. Logic preserved exactly as requested.
    """
    print(f"[INFO] Processing: {os.path.basename(input_path)}")

    predictor = build_sam3_video_predictor(
        async_loading_frames=True
    )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): return {}, 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    palette = sv.ColorPalette.from_hex(["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFD700", "#FFA07A"])
    mask_annotator = sv.MaskAnnotator(color=palette, color_lookup=sv.ColorLookup.TRACK)
    box_annotator = sv.BoxAnnotator(color=palette, color_lookup=sv.ColorLookup.TRACK, thickness=2)
    label_annotator = sv.LabelAnnotator(color=palette, text_position=sv.Position.TOP_CENTER, color_lookup=sv.ColorLookup.TRACK)

    state_output = predictor.start_session(input_path)#1
    # Handle API variations
    if isinstance(state_output, dict):
        session_id = state_output.get("session_id")
        inference_state = state_output.get("inference_state")
    else:
        inference_state = state_output
        session_id = None

    predictor.add_prompt(session_id=session_id, frame_idx=0, text=TEXT_PROMPT)#2

    generator = predictor.propagate_in_video(
        session_id if session_id else inference_state, 
        propagation_direction="forward", 
        start_frame_idx=0, 
        max_frame_num_to_track=None
    )#3
    
    tracking_history = {} 

    for frame_out in generator:
        sam_frame_idx = frame_out.get("frame_index")
        outputs = frame_out.get("outputs")
        
        if outputs is None: continue
        obj_ids = outputs.get("out_obj_ids")
        video_res_masks = outputs.get("out_binary_masks")
        if obj_ids is None or video_res_masks is None: continue

        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_pos != sam_frame_idx:
            if 0 < (sam_frame_idx - current_pos) < 5:
                 for _ in range(sam_frame_idx - current_pos): cap.grab()
            else:
                 cap.set(cv2.CAP_PROP_POS_FRAMES, sam_frame_idx)
            
        ret, frame_bgr = cap.read()
        if not ret: break

        if len(obj_ids) > 0:
            masks_tensor = np.asarray(video_res_masks)
            if masks_tensor.ndim == 4 and masks_tensor.shape[1] == 1:
                masks_tensor = masks_tensor.squeeze(1)
            
            has_pixels = masks_tensor.reshape(masks_tensor.shape[0], -1).any(axis=1) 
            valid_indices = np.where(has_pixels)[0]
            
            if len(valid_indices) > 0:
                current_masks = masks_tensor[valid_indices]
                current_ids = [int(obj_ids[i]) for i in valid_indices]
                
                for oid in current_ids:
                    if oid not in tracking_history:
                        tracking_history[oid] = {'start_frame': sam_frame_idx, 'end_frame': sam_frame_idx}
                    else:
                        tracking_history[oid]['end_frame'] = sam_frame_idx

                detections = sv.Detections(
                    mask=current_masks, xyxy=sv.mask_to_xyxy(current_masks), tracker_id=np.array(current_ids)
                )
                labels = [f"ID: {tid}" for tid in detections.tracker_id]
                annotated_frame = mask_annotator.annotate(frame_bgr.copy(), detections)
                annotated_frame = box_annotator.annotate(annotated_frame, detections)
                annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=labels)
                out.write(annotated_frame)
            else:
                out.write(frame_bgr)
        else:
            out.write(frame_bgr)

        # if sam_frame_idx % 50 == 0:
        #     print(f"   Frame {sam_frame_idx}/{total_frames}", end='\r')

    cap.release()
    out.release()
    del predictor 
    del inference_state
    del session_id
    del generator
    gc.collect()
    torch.cuda.empty_cache()
    
    return tracking_history, total_frames, fps

# --- MAIN (ADAPTED FOR SINGLE RUN) ---
def main():
    parser = argparse.ArgumentParser()
    # Arguments simplified for single file processing
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_video", type=str, required=True)
    parser.add_argument("--output_csv_fragment", type=str, required=True)
    parser.add_argument("--clip_name", type=str, required=True, help="Used for ID generation")
    
    parser.add_argument("--sam_config", type=str, default=DEFAULT_SAM_CONFIG)
    parser.add_argument("--sam_checkpoint", type=str, default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    
    args = parser.parse_args()
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    with torch.inference_mode(), torch.autocast(device_type=args.device, dtype=dtype):
        try:
            video_stats, total_frames, fps = process_single_video(
                args.input_video, 
                args.output_video,
                args
            )
            
            # Format Data
            all_tracking_data = []
            objects_list = [{'original_id': k, 'start': v['start_frame'], 'end': v['end_frame']} 
                          for k, v in video_stats.items()]
            objects_list.sort(key=lambda x: (x['start'], x['end']))

            for i, obj in enumerate(objects_list):
                all_tracking_data.append({
                    "clip_name": args.clip_name,
                    "unique_people_id": f"{args.clip_name}{get_id_suffix(i+1)}",
                    "start_frame": obj['start'],
                    "end_frame": obj['end'],
                    "total_frames": total_frames,
                    "fps": fps
                })
            
            # Save Fragment CSV (Headers included, Bash script will handle merging)
            if all_tracking_data:
                pd.DataFrame(all_tracking_data).to_csv(args.output_csv_fragment, index=False)
                print(f"[SUCCESS] Saved CSV fragment to {args.output_csv_fragment}")
            else:
                # Create empty file to signal completion but no data
                open(args.output_csv_fragment, 'a').close()

        except Exception as e:
            print(f"[ERROR] Failed to process {args.input_video}: {e}")
            sys.exit(1) # Exit with error for Bash to catch

if __name__ == "__main__":
    main()