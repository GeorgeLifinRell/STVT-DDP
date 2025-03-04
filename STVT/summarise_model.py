import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from train import parse_args
from STVT.build_model import build_model

# -------------------------
# Helper: extract frames from video
# -------------------------
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize to 224x224 and convert BGR to RGB
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

# -------------------------
# Helper: extract deep features from each frame using a CNN
# -------------------------
def extract_features(frames, feature_extractor, device):
    # Define the transform (normalize as used for ResNet)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    
    features = []
    feature_extractor.eval()
    with torch.no_grad():
        for frame in frames:
            inp = preprocess(frame).unsqueeze(0).to(device)  # shape: (1,3,224,224)
            feat = feature_extractor(inp)  # expect output shape (1, 512, 1, 1) or (1, 512)
            # Squeeze extra dimensions so we get (512,)
            feat = feat.squeeze()
            features.append(feat)
    return features

# -------------------------
# Helper: group frame features into segments
#
# For each segment, we assume a sequence_length (e.g. 16 frames) and 
# arrange them into a 4x4 grid. Each frame’s feature is reshaped to (512,1,1)
# and concatenated along spatial dimensions to produce a tensor of shape (512,4,4).
# -------------------------
def create_segment_tensor(features, sequence_length=16):
    num_frames = len(features)
    num_segments = math.ceil(num_frames / sequence_length)
    # Pad the last segment if necessary (repeat last feature)
    if num_frames % sequence_length != 0:
        last_feat = features[-1]
        for _ in range(sequence_length - (num_frames % sequence_length)):
            features.append(last_feat)
    
    segments = []
    for seg in range(num_segments):
        seg_feats = features[seg*sequence_length : (seg+1)*sequence_length]
        # Each feature is (512,), reshape to (512,1,1)
        patches = [f.view(512, 1, 1) for f in seg_feats]
        # Arrange patches into a 4x4 grid (assuming sequence_length == 16)
        row_list = []
        for r in range(4):
            # Concatenate 4 patches along width (axis=2)
            row = torch.cat(patches[r*4:(r+1)*4], dim=2)
            row_list.append(row)
        # Concatenate the rows along height (axis=1)
        grid = torch.cat(row_list, dim=1)  # shape: (512,4,4)
        segments.append(grid)
    # Stack segments into a batch tensor
    segments_tensor = torch.stack(segments, dim=0)  # shape: (num_segments, 512,4,4)
    return segments_tensor, num_segments

# -------------------------
# A simple (placeholder) shot segmentation function.
# In the paper, KTS (Kernel Temporal Segmentation) is used.
# Here we implement a very basic segmentation based on frame feature differences.
# -------------------------
def simple_shot_segmentation(frame_features, threshold=0.3):
    num_frames = len(frame_features)
    boundaries = [0]
    prev = frame_features[0].cpu().numpy()
    for i in range(1, num_frames):
        curr = frame_features[i].cpu().numpy()
        diff = np.linalg.norm(curr - prev)
        if diff > threshold:
            boundaries.append(i)
        prev = curr
    boundaries.append(num_frames - 1)
    # Group boundaries into shot segments (each segment is a tuple: (start, end))
    shots = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        shots.append((start, end))
    return shots

# -------------------------
# Helper: knapsack solver for shot selection.
#
# Given a list of shots with lengths and average importance scores,
# select a subset such that the total length is below max_length.
# The value for each shot is computed as (shot_length * avg_importance).
# -------------------------
def select_shots_knapsack(shots, shot_scores, total_frames, summary_ratio=0.15):
    max_length = int(total_frames * summary_ratio)
    num_shots = len(shots)
    dp = [[0]*(max_length+1) for _ in range(num_shots+1)]
    keep = [[False]*(max_length+1) for _ in range(num_shots+1)]
    
    for i in range(1, num_shots+1):
        shot_len = shots[i-1][1] - shots[i-1][0] + 1
        shot_value = shot_scores[i-1] * shot_len
        for w in range(max_length+1):
            if shot_len <= w:
                if dp[i-1][w] < dp[i-1][w-shot_len] + shot_value:
                    dp[i][w] = dp[i-1][w-shot_len] + shot_value
                    keep[i][w] = True
                else:
                    dp[i][w] = dp[i-1][w]
            else:
                dp[i][w] = dp[i-1][w]
    # Backtracking to determine selected shots
    w = max_length
    selected_indices = []
    for i in range(num_shots, 0, -1):
        if keep[i][w]:
            selected_indices.append(i-1)
            shot_len = shots[i-1][1] - shots[i-1][0] + 1
            w -= shot_len
    selected_indices = sorted(selected_indices)
    return selected_indices

# -------------------------
# Main summarization function
# -------------------------
def summarize_video(video_path, stvt_model, feature_extractor, device,
                    sequence_length=16, summary_ratio=0.15):
    """
    video_path: path to the input video file.
    stvt_model: the trained STVT model (in eval mode).
    feature_extractor: the CNN model to extract deep features.
    device: torch device (e.g., 'cuda' or 'cpu').
    sequence_length: number of frames per segment (e.g., 16).
    summary_ratio: maximum summary length as a fraction of the total frames.
    """
    # 1. Extract frames from video
    frames = extract_frames(video_path)
    total_frames = len(frames)
    print(f"Extracted {total_frames} frames from video.")

    # 2. Extract deep features for each frame (each feature is a 512-dim tensor)
    features = extract_features(frames, feature_extractor, device)

    # 3. Group features into segments (each segment is arranged into a 4x4 grid)
    segments_tensor, num_segments = create_segment_tensor(features, sequence_length)
    segments_tensor = segments_tensor.to(device)
    
    # 4. Pass segments through the STVT model to obtain frame importance predictions
    stvt_model.eval()
    with torch.no_grad():
        # The model expects input shape (batch, channels, height, width)
        # For each segment, the model outputs a tensor of shape (num_patches, batch, out_dim)
        outputs = stvt_model(segments_tensor)  # shape: (16, num_segments, 2)
    outputs = outputs.cpu()
    
    # 5. For each segment, compute importance scores per frame.
    # We take the softmax probability of the “important” class (assumed to be index 1).
    frame_scores = []
    for seg in range(num_segments):
        seg_out = outputs[:, seg, :]  # shape: (16, 2)
        seg_prob = F.softmax(seg_out, dim=1)  # shape: (16, 2)
        seg_scores = seg_prob[:, 1]  # probability for class 1 (importance)
        frame_scores.extend(seg_scores.tolist())
    # Trim extra scores (if padded)
    frame_scores = frame_scores[:total_frames]
    
    # 6. Shot segmentation.
    shots = simple_shot_segmentation(features, threshold=0.3)
    print(f"Identified {len(shots)} shots.")
    
    # 7. Compute average importance for each shot.
    shot_scores = []
    for (start, end) in shots:
        shot_score = np.mean(frame_scores[start:end+1])
        shot_scores.append(shot_score)
    
    # 8. Use knapsack selection to choose shots so that total frames in summary <= summary_ratio * total_frames.
    selected_shot_indices = select_shots_knapsack(shots, shot_scores, total_frames, summary_ratio)
    print(f"Selected {len(selected_shot_indices)} shots for the summary.")
    
    # 9. Gather the frame indices for the selected shots.
    summary_frame_indices = []
    for idx in selected_shot_indices:
        s, e = shots[idx]
        summary_frame_indices.extend(list(range(s, e+1)))
    
    # Return both the summary frame indices and the full list of frames for reconstruction.
    return summary_frame_indices, frame_scores, frames

# -------------------------
# Function to reconstruct and save summary video from selected frames.
# -------------------------
def create_summary_video(frames, summary_frame_indices, output_path, original_video_path):
    """
    frames: list of frames (RGB numpy arrays) extracted from the original video.
    summary_frame_indices: list of frame indices to include in the summary.
    output_path: path to save the summary video.
    original_video_path: path of the original video (to extract fps and frame size).
    """
    # Open original video to get fps and frame dimensions.
    cap = cv2.VideoCapture(original_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Assuming frames are all the same size.
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for idx in summary_frame_indices:
        # Convert RGB back to BGR for OpenCV VideoWriter
        frame_bgr = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Summary video saved to {output_path}")

if __name__ == "__main__":
    args = parse_args()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    stvt_model = build_model(args=args)
    
    # Load the state dict
    state_dict = torch.load("/home/user0123/STVT/STVT/STVT/model/SumMe/Record_1.pth", map_location=device)

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    # Load the modified state dict
    stvt_model.load_state_dict(new_state_dict)
    
    stvt_model.to(device)
    
    # Set up the CNN feature extractor.
    # We use a pre-trained ResNet-18 and remove the final fully connected layer.
    resnet = models.resnet18(pretrained=True)
    feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    feature_extractor.to(device)
    feature_extractor.eval()
    
    # Path to the user video file
    video_path = args.input_video_path
    
    # Run the summarization function
    summary_frames, frame_importance, frames = summarize_video(
        video_path, stvt_model, feature_extractor, device,
        sequence_length=16, summary_ratio=0.15
    )
    
    print("Summary frame indices:", summary_frames)
    
    # Reconstruct and save the summary video.
    output_video_path = args.output_video_path
    create_summary_video(frames, summary_frames, output_video_path, video_path)
