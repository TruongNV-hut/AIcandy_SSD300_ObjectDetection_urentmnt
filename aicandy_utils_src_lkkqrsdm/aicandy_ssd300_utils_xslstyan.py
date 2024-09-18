"""

@author:  AIcandy 
@website: aicandy.vn

"""

import json
import os
import torch
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}


def decimate(tensor, m):
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor


def xy_to_cxcy(xy):
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2, 
                      xy[:, 2:] - xy[:, :2]], 1)  


def cxcy_to_xy(cxcy):
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10), 
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2], 
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1) 


def find_intersection(set_1, set_2):
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0)) 
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0) 
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1] 


def find_jaccard_overlap(set_1, set_2):
    intersection = find_intersection(set_1, set_2) 
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1]) 
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection 

    return intersection / union 

def expand(image, boxes, filler):
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)
    filler = torch.FloatTensor(filler)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1) 
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0) 

    return new_image, new_boxes

def random_crop_image(img, bboxes, lbls, diffs):
    img_height = img.size(1)
    img_width = img.size(2)
    while True:
        overlap_threshold = random.choice([0., .1, .3, .5, .7, .9, None])
        if overlap_threshold is None:
            return img, bboxes, lbls, diffs
        num_attempts = 50
        for _ in range(num_attempts):
            min_scale = 0.3
            scale_height = random.uniform(min_scale, 1)
            scale_width = random.uniform(min_scale, 1)
            crop_height = int(scale_height * img_height)
            crop_width = int(scale_width * img_width)
            aspect_ratio = crop_height / crop_width
            if not 0.5 < aspect_ratio < 2:
                continue

            crop_left = random.randint(0, img_width - crop_width)
            crop_right = crop_left + crop_width
            crop_top = random.randint(0, img_height - crop_height)
            crop_bottom = crop_top + crop_height
            crop_rect = torch.FloatTensor([crop_left, crop_top, crop_right, crop_bottom])
            overlap = find_jaccard_overlap(crop_rect.unsqueeze(0), bboxes)
            overlap = overlap.squeeze(0)
            if overlap.max().item() < overlap_threshold:
                continue
            cropped_img = img[:, crop_top:crop_bottom, crop_left:crop_right]
            box_centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2.
            centers_in_crop = (box_centers[:, 0] > crop_left) * (box_centers[:, 0] < crop_right) * (box_centers[:, 1] > crop_top) * (box_centers[:, 1] < crop_bottom)
            if not centers_in_crop.any():
                continue

            cropped_boxes = bboxes[centers_in_crop, :]
            cropped_labels = lbls[centers_in_crop]
            cropped_diffs = diffs[centers_in_crop]
            cropped_boxes[:, :2] = torch.max(cropped_boxes[:, :2], crop_rect[:2])
            cropped_boxes[:, :2] -= crop_rect[:2]
            cropped_boxes[:, 2:] = torch.min(cropped_boxes[:, 2:], crop_rect[2:])
            cropped_boxes[:, 2:] -= crop_rect[:2]

            return cropped_img, cropped_boxes, cropped_labels, cropped_diffs



def flip(image, boxes):
    new_image = FT.hflip(image)
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    new_image = FT.resize(image, dims)
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims 

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def photometric_distort(image):
    new_image = image
    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ == 'adjust_hue':
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                adjust_factor = random.uniform(0.5, 1.5)
            new_image = d(new_image, adjust_factor)

    return new_image


def transform(image, boxes, labels, difficulties, split):
    assert split in {'train', 'test'}
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    if split == 'train':
        new_image = photometric_distort(new_image)
        new_image = FT.to_tensor(new_image)
        if random.random() < 0.5:
            new_image, new_boxes = expand(new_image, boxes, filler=mean)
        new_image, new_boxes, new_labels, new_difficulties = random_crop_image(new_image, new_boxes, new_labels, new_difficulties)
        new_image = FT.to_pil_image(new_image)
        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)
    new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))
    new_image = FT.to_tensor(new_image)
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels, new_difficulties


def save_checkpoint(epoch, model, optimizer):
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filepath = 'aicandy_model_out_gnloibxd/aicandy_checkpoint_ssd300.pth'
    torch.save(state, filepath)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
