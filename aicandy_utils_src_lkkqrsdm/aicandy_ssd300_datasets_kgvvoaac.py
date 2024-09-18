"""

@author:  AIcandy 
@website: aicandy.vn

"""

import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from aicandy_utils_src_lkkqrsdm.aicandy_ssd300_utils_xslstyan import transform


class VOC_Dataset(Dataset):
    def __init__(self, folder_path, dataset_split, include_difficult=False):
        self.dataset_split = dataset_split.lower()
        assert self.dataset_split in {'train', 'test'}
        self.folder_path = folder_path
        self.include_difficult = include_difficult
        with open(os.path.join(folder_path, self.dataset_split + '_images.json'), 'r') as img_file:
            self.image_list = json.load(img_file)
        with open(os.path.join(folder_path, self.dataset_split + '_objects.json'), 'r') as obj_file:
            self.object_list = json.load(obj_file)

        assert len(self.image_list) == len(self.object_list)

    def __getitem__(self, index):
        img = Image.open(self.image_list[index], mode='r')
        img = img.convert('RGB')
        obj_details = self.object_list[index]
        bbox = torch.FloatTensor(obj_details['boxes'])
        obj_labels = torch.LongTensor(obj_details['labels']) 
        obj_difficulties = torch.ByteTensor(obj_details['difficulties'])

        if not self.include_difficult:
            bbox = bbox[1 - obj_difficulties]
            obj_labels = obj_labels[1 - obj_difficulties]
            obj_difficulties = obj_difficulties[1 - obj_difficulties]

        img, bbox, obj_labels, obj_difficulties = transform(img, bbox, obj_labels, obj_difficulties, split=self.dataset_split)

        return img, bbox, obj_labels, obj_difficulties

    def __len__(self):
        return len(self.image_list)

    def collate_fn(self, batch):
        img_batch = list()
        bbox_batch = list()
        label_batch = list()
        difficulty_batch = list()

        for item in batch:
            img_batch.append(item[0])
            bbox_batch.append(item[1])
            label_batch.append(item[2])
            difficulty_batch.append(item[3])

        img_batch = torch.stack(img_batch, dim=0)

        return img_batch, bbox_batch, label_batch, difficulty_batch
