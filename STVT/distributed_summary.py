import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
import os
import argparse
from train import parse_args
from STVT.build_model import build_model
from summarize_video import (
    summarize_video, 
    create_summary_video, 
    extract_features, 
    create_segment_tensor, 
    simple_shot_segmentation, 
    select_shots_knapsack
)
# -------------------------
# Helper: extract frames from video with chunking support
# -------------------------
def extract_frames(video_path, start_frame=None, end_frame=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set default values if not specified
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = total_frames
    
    # Seek to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    frame_idx = start_frame
    
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize to 224x224 and convert BGR to RGB
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_idx += 1
        
    cap.release()
    return frames, total_frames

# Rest of the existing helper functions remain the same...
# extract_features, create_segment_tensor, simple_shot_segmentation, select_shots_knapsack

# -------------------------
# Worker function for distributed processing
# -------------------------
def process_video_chunk(rank, world_size, args):
    # Initialize the process group
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        init_method=f'tcp://localhost:{args.port}',
        rank=rank, 
        world_size=world_size
    )
    
    print(f"Worker {rank} initialized")
    
    # Set device for this process
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    else:
        device = torch.device('cpu')
    torch.cuda.set_device(device)
    
    # Load model
    stvt_model = build_model(args=args)
    state_dict = torch.load(args.model_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    stvt_model.load_state_dict(new_state_dict)
    stvt_model = DDP(stvt_model.to(device), device_ids=[device.index] if device.type == 'cuda' else None)
    
    # Load feature extractor
    resnet = models.resnet18(pretrained=True)
    feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    feature_extractor.to(device)
    feature_extractor.eval()
    
    # Get video metadata
    cap = cv2.VideoCapture(args.input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Calculate chunk for this rank
    frames_per_rank = total_frames // world_size
    start_frame = rank * frames_per_rank
    end_frame = total_frames if rank == world_size - 1 else (rank + 1) * frames_per_rank
    
    print(f"Rank {rank} processing frames {start_frame} to {end_frame}")
    
    # Extract frames for this chunk
    frames, _ = extract_frames(args.input_video_path, start_frame, end_frame)
    
    # Process frames
    features = extract_features(frames, feature_extractor, device)
    segments_tensor, num_segments = create_segment_tensor(features, args.sequence_length)
    segments_tensor = segments_tensor.to(device)
    
    # Run model
    stvt_model.eval()
    with torch.no_grad():
        outputs = stvt_model(segments_tensor)
    outputs = outputs.cpu()
    
    # Calculate frame scores
    frame_scores = []
    for seg in range(num_segments):
        seg_out = outputs[:, seg, :]
        seg_prob = F.softmax(seg_out, dim=1)
        seg_scores = seg_prob[:, 1]
        frame_scores.extend(seg_scores.tolist())
    
    # Trim extra scores (if padded)
    frame_scores = frame_scores[:len(frames)]
    
    # Save results to temporary file with rank in filename
    chunk_data = {
        'start_frame': start_frame,
        'end_frame': end_frame,
        'frame_scores': frame_scores,
        'features': [f.cpu().numpy() for f in features]
    }
    
    # Create temp dir if it doesn't exist
    os.makedirs(args.temp_dir, exist_ok=True)
    
    # Save chunk data
    chunk_file = os.path.join(args.temp_dir, f'chunk_{rank}.pt')
    torch.save(chunk_data, chunk_file)
    
    # Wait for all processes to finish
    dist.barrier(timeout=60)
    dist.destroy_process_group()

# -------------------------
# Function to aggregate chunk results and create summary
# -------------------------
def aggregate_results(args, world_size):
    print("Aggregating results from all workers...")
    
    # Load all chunk data
    all_frame_scores = []
    all_features = []
    max_frame_idx = 0
    
    # First pass: determine total frames count
    for rank in range(world_size):
        chunk_file = os.path.join(args.temp_dir, f'chunk_{rank}.pt')
        if not os.path.exists(chunk_file):
            print(f"Warning: Chunk file for rank {rank} not found!")
            continue
            
        try:
            chunk_data = torch.load(chunk_file)
            max_frame_idx = max(max_frame_idx, chunk_data['end_frame'])
        except Exception as e:
            print(f"Error loading chunk {rank}: {e}")
    
    # Initialize arrays with placeholders
    all_frame_scores = [0.0] * max_frame_idx
    all_features = [None] * max_frame_idx
    
    # Second pass: fill in the data
    for rank in range(world_size):
        chunk_file = os.path.join(args.temp_dir, f'chunk_{rank}.pt')
        if not os.path.exists(chunk_file):
            continue
            
        try:
            print(f"Processing chunk {rank}/{world_size}")
            chunk_data = torch.load(chunk_file)
            
            start_frame = chunk_data['start_frame']
            end_frame = chunk_data['end_frame']
            chunk_scores = chunk_data['frame_scores']
            chunk_features = chunk_data['features']
            
            # Copy data into pre-allocated arrays
            for i, (score, feature) in enumerate(zip(chunk_scores, chunk_features)):
                frame_idx = start_frame + i
                if frame_idx < max_frame_idx:  # Safety check
                    all_frame_scores[frame_idx] = score
                    all_features[frame_idx] = torch.tensor(feature)
        except Exception as e:
            print(f"Error processing chunk {rank}: {e}")
    
    # Filter out None values before shot segmentation
    valid_features = [f for f in all_features if f is not None]
    
    # Summarization post-processing
    shots = simple_shot_segmentation(valid_features, threshold=args.shot_threshold)
    print(f"Identified {len(shots)} shots.")
    
    # Compute average importance for each shot
    shot_scores = []
    for (start, end) in shots:
        if start < len(all_frame_scores) and end < len(all_frame_scores):
            valid_scores = [s for s in all_frame_scores[start:end+1] if s is not None and isinstance(s, (int, float))]
            shot_score = np.mean(valid_scores) if valid_scores else 0.0
            shot_scores.append(shot_score)
        else:
            shot_scores.append(0.0)
    
    # Select shots using knapsack
    selected_shot_indices = select_shots_knapsack(shots, shot_scores, len(valid_features), args.summary_ratio)
    print(f"Selected {len(selected_shot_indices)} shots for the summary.")
    
    # Create final summary
    summary_frame_indices = []
    for idx in selected_shot_indices:
        if idx < len(shots):
            s, e = shots[idx]
            summary_frame_indices.extend(list(range(s, e+1)))
    
    return summary_frame_indices, all_frame_scores

# -------------------------
# Function to create summary video from selected frames
# -------------------------
def create_distributed_summary_video(summary_frame_indices, output_path, input_video_path):
    """Create summary video by extracting only the needed frames"""
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Sort frame indices to ensure chronological order
    summary_frame_indices = sorted(summary_frame_indices)
    
    # Process frames
    current_pos = -1
    for frame_idx in summary_frame_indices:
        # Skip to the next frame position
        if frame_idx > current_pos + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        else:
            # If sequential, just read next frame
            pass
        
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            current_pos = frame_idx
    
    cap.release()
    out.release()
    print(f"Summary video saved to {output_path}")

# -------------------------
# Main function to coordinate distributed processing
# -------------------------
def main_distributed():
    
    args=parse_args()
    
    # Create temp dir if needed
    os.makedirs(args.temp_dir, exist_ok=True)
    
    # Determine world size (number of processes)
    world_size = min(args.num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    if world_size > 1:
        print(f"Using distributed processing with {world_size} GPUs")
    else:
        print("Using single process mode (no distribution)")
    
    # Launch processes
    if world_size > 1:
        mp.spawn(
            process_video_chunk,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
        
        # Aggregate results
        summary_frame_indices, frame_scores = aggregate_results(args, world_size)
        
        # Create summary video
        create_distributed_summary_video(summary_frame_indices, args.output_video_path, args.input_video_path)
    else:
        # Single GPU/CPU mode, use the original pipeline
        print("Running in single process mode")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load model
        stvt_model = build_model(args=args)
        state_dict = torch.load(args.model_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        stvt_model.load_state_dict(new_state_dict)
        stvt_model.to(device)
        
        # Load feature extractor
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        feature_extractor.to(device)
        
        # Run original summarization
        summary_frames, frame_importance, frames = summarize_video(
            args.input_video_path, stvt_model, feature_extractor, device,
            sequence_length=args.sequence_length, summary_ratio=args.summary_ratio
        )
        
        # Create summary video
        create_summary_video(frames, summary_frames, args.output_video_path, args.input_video_path)
        
    print("Summarization complete!")
    
    # Cleanup temp files
    if world_size > 1:
        for rank in range(world_size):
            chunk_file = os.path.join(args.temp_dir, f'chunk_{rank}.pt')
            if os.path.exists(chunk_file):
                os.remove(chunk_file)

if __name__ == "__main__":
    main_distributed()