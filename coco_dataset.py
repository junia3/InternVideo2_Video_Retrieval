import os
from torch.utils.data import Dataset
import pandas as pd

class COCODataset(Dataset):
    def __init__(self, data_dir : str):
        super(COCODataset, self).__init__()
        self.data_dir = data_dir
        self.data = [] # (filepath, raw captions)
        self.load_data()

    def load_data(self):
        df = pd.read_csv(os.path.join(self.data_dir, "test_5k_mscoco_2014.csv"))
        for filename, fileid, captions in zip(df["filename"], df["imgid"], df["raw"]):
            captions = eval(captions)
            fileid = eval(fileid)
            self.data.append((os.path.join(self.data_dir, "images_mscoco_2014_5k_test", filename), captions, fileid))
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_func(batch):
    image_paths = []
    captions = []
    file_ids = []
    for image_path, caption, fileid in batch:
        image_paths.append(image_path)
        captions.extend(caption)
        file_ids.extend(fileid)
    return image_paths, captions, file_ids