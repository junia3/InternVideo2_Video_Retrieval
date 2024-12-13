import os
import json
from torch.utils.data import Dataset

class MSRVTTDataset(Dataset):
    def __init__(self, data_dir : str, split : str):
        super(MSRVTTDataset, self).__init__()
        self.data_dir = data_dir
        self.split = split
        self.video_ids = []
        self.video_paths = {}
        self.captions = {}
        self.load_data()

    def load_data(self):
        # Load data from data_path
        video_path = os.path.join(self.data_dir, "videos", "all")
        caption_path = os.path.join(self.data_dir, "annotation", "MSR_VTT.json")
        split_path = os.path.join(self.data_dir,
                                  "high-quality",
                                  "structured-symlinks",
                                  self.split+".txt")
        
        with open(split_path, "r") as f:
            video_id_list = f.read().splitlines()

        with open(caption_path, "r") as f:
            caption_data = json.load(f)

        for video_id in video_id_list:
            self.video_ids.append(video_id)
            self.video_paths[video_id] = os.path.join(video_path, video_id+".mp4")
            self.captions[video_id] = []

        for anno_data in caption_data["annotations"]:
            if anno_data["image_id"] in self.captions.keys():
                video_id = anno_data["image_id"]
                caption = anno_data["caption"]
                self.captions[video_id].append(caption)
                
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        return video_id, self.video_paths[video_id], self.captions[video_id]
    

def collate_func(batch):
    video_ids = []
    video_paths = []
    captions = []

    for (video_id, video_path, video_captions) in batch:
        video_ids.extend([video_id]*len(video_captions))
        video_paths.append(video_path)
        captions.extend(video_captions)

    return {
        "video_ids": video_ids,
        "video_paths": video_paths,
        "captions": captions,
    }