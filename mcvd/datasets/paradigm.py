import numpy as np
import os
import pickle
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ParadigmDataset(Dataset):

    def __init__(self, data_dir, frames_per_sample=5, train=True, random_time=True, random_horizontal_flip=True,
                 total_videos=-1, with_target=True, start_at=0):

        self.data_dir = data_dir
        self.train = train
        self.frames_per_sample = frames_per_sample
        self.random_time = random_time
        self.random_horizontal_flip = random_horizontal_flip
        self.total_videos = total_videos
        self.with_target = with_target
        self.start_at = start_at

        self.train_dir = data_dir
        if self.train:
            self.train_video_dirs = [os.path.join(self.train_dir, f'video_{str(i).zfill(5)}') for i in range(2000, 2100)]
        else:
            self.train_video_dirs = [os.path.join(self.train_dir, f'video_{str(i).zfill(5)}') for i in range(1000, 1100)]

        print(f"Dataset length: {self.__len__()}")

    def len_of_vid(self, index):
        video_index = index % self.__len__()
        video_dir = self.train_video_dirs[video_index]
        return len(os.listdir(video_dir))

    def __len__(self):
        return self.total_videos if self.total_videos > 0 else len(self.train_video_dirs)

    def max_index(self):
        return len(self.train_video_dirs)

    def __getitem__(self, index, time_idx=0):

        # Use `index` to select the video directory
        video_dir = self.train_video_dirs[index]
        frames = []

        # Collect frames from the video directory
        for filename in sorted(os.listdir(video_dir)):
            if filename.endswith('.png'):
                img_path = os.path.join(video_dir, filename)
                img = Image.open(img_path)
                frames.append(transforms.ToTensor()(img))

        # Randomly choose a window of frames if random_time is enabled
        video_len = len(frames)
        if self.random_time and video_len > self.frames_per_sample:
            time_idx = np.random.randint(video_len - self.frames_per_sample)
        time_idx += self.start_at
        prefinals = frames[time_idx:time_idx+self.frames_per_sample]

        # Apply random horizontal flip if enabled
        flip_p = np.random.randint(2) == 0 if self.random_horizontal_flip else 0
        prefinals = [transforms.RandomHorizontalFlip(flip_p)(img) for img in prefinals]

        # Load target
        target = int(os.path.basename(video_dir).split('_')[1]) - 1

        if self.with_target:
            return torch.stack(prefinals), torch.tensor(target)
        else:
            return torch.stack(prefinals)
