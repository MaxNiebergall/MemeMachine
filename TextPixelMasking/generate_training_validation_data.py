#see https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files

import os
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from image_with_text_functions import generate_text_on_image_and_pixel_mask_from_path
import numpy as np

class CustomImageDataset(Dataset):
    #Transform options: 'random_crop', TBA 'detect-resize'
    def __init__(self, img_dir, x_size, y_size, n_channels=3, transform=None, RNGseed=None, target_transform=None):
        self.x_size = x_size
        self.y_size = y_size
        self.n_channels = n_channels
        self.img_dir = img_dir
        self.transform = transform 
        self.target_transform = target_transform
        self.img_paths = []
        self.RNGseed = RNGseed
        for file_ in  os.listdir(img_dir+"/"):
            self.img_paths.append(str(img_dir+"/"+file_))

    def __len__(self):
        return len(self.img_paths)
 
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        text_img, mask = generate_text_on_image_and_pixel_mask_from_path(img_path, self.x_size, self.y_size, self.n_channels, RNGseed=self.RNGseed)

        image = ToTensor()(text_img)
        label_image = ToTensor()(mask)
        label_image = label_image.squeeze().flatten()
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label_image = self.target_transform(label_image)
        return image, label_image