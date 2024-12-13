
import decord
from decord import VideoReader
decord.bridge.set_bridge("torch")
import numpy as np 
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Lambda, Resize, InterpolationMode, Normalize

def get_frame_indices(num_frames, vlen):
    intervals = np.linspace(start=0, stop=vlen, num=num_frames+1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    
    frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
    return frame_indices

def read_frames_decord(
        video_path, num_frames):
    num_threads = 1 if video_path.endswith('.webm') else 0 # make ssv2 happy
    video_reader = VideoReader(video_path, num_threads=num_threads)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    frame_indices = get_frame_indices(num_frames, vlen)
    frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames

def preprocess_video_frames(frames):
    # Resize frames to the expected input size
    transforms = Compose(
        [   
            Lambda(lambda x: x.float().div(255.0)),
            Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    processed_frames = transforms(frames)
    return processed_frames.unsqueeze(0)

def process_video_frames_batch(video_paths, num_frames=8):
    video_inputs = []
    for video_path in video_paths:
        frames = read_frames_decord(video_path, num_frames=num_frames)
        video_inputs.append(preprocess_video_frames(frames))
    return torch.cat(video_inputs, dim=0)

def compute_metrics(ranks):
    ranks = np.array(ranks)
    r1 = 100.0 * np.mean(ranks < 1)
    r5 = 100.0 * np.mean(ranks < 5)
    r10 = 100.0 * np.mean(ranks < 10)
    medr = np.median(ranks) + 1
    meanr = np.mean(ranks) + 1
    return {'R@1': r1, 'R@5': r5, 'R@10': r10, 'Median Rank': medr, 'Mean Rank': meanr}