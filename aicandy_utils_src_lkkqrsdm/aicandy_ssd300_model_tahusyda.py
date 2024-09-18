"""

@author:  AIcandy 
@website: aicandy.vn

"""

from torch import nn
from aicandy_utils_src_lkkqrsdm.aicandy_ssd300_utils_xslstyan import *
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGGBase(nn.Module):
    def __init__(self):
        super(VGGBase, self).__init__()

        # Standard convolutional layers in VGG16
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1) 
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6) 
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.load_pretrained_layers()

    def forward(self, image):
        out = F.relu(self.conv1_1(image))
        out = F.relu(self.conv1_2(out))
        out = self.pool1(out)

        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = self.pool2(out) 

        out = F.relu(self.conv3_1(out))
        out = F.relu(self.conv3_2(out)) 
        out = F.relu(self.conv3_3(out)) 
        out = self.pool3(out) 

        out = F.relu(self.conv4_1(out))
        out = F.relu(self.conv4_2(out)) 
        out = F.relu(self.conv4_3(out)) 
        conv4_3_feats = out 
        out = self.pool4(out)

        out = F.relu(self.conv5_1(out)) 
        out = F.relu(self.conv5_2(out)) 
        out = F.relu(self.conv5_3(out)) 
        out = self.pool5(out) 

        out = F.relu(self.conv6(out)) 
        conv7_feats = F.relu(self.conv7(out)) 

        return conv4_3_feats, conv7_feats

    def load_pretrained_layers(self):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        for i, param in enumerate(param_names[:-4]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias'] 
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3]) 
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4]) 

        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  

        self.load_state_dict(state_dict)


class AuxiliaryConvolutions(nn.Module):
    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0) 
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  
        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0) 
        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  

        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):
        out = F.relu(self.conv8_1(conv7_feats)) 
        out = F.relu(self.conv8_2(out))  
        conv8_2_feats = out  
        out = F.relu(self.conv9_1(out))  
        out = F.relu(self.conv9_2(out))  
        conv9_2_feats = out  
        out = F.relu(self.conv10_1(out)) 
        out = F.relu(self.conv10_2(out))  
        conv10_2_feats = out  
        out = F.relu(self.conv11_1(out)) 
        conv11_2_feats = F.relu(self.conv11_2(out))  

        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats


class PredictionConvolutions(nn.Module):
    def __init__(self, n_classes):
        super(PredictionConvolutions, self).__init__()
        self.n_classes = n_classes
        n_boxes = {'conv4_3': 4,
                   'conv7': 6,
                   'conv8_2': 6,
                   'conv9_2': 6,
                   'conv10_2': 4,
                   'conv11_2': 4}

        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * 4, kernel_size=3, padding=1)

        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * n_classes, kernel_size=3, padding=1)

        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        batch_size = conv4_3_feats.size(0)
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3,  1).contiguous() 
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4) 
        l_conv7 = self.loc_conv7(conv7_feats)  
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous() 
        l_conv7 = l_conv7.view(batch_size, -1, 4) 
        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  
        l_conv9_2 = self.loc_conv9_2(conv9_2_feats) 
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous() 
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)
        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()  
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4) 
        l_conv11_2 = self.loc_conv11_2(conv11_2_feats) 
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()  
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)  
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats) 
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous() 
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.n_classes)  
        c_conv7 = self.cl_conv7(conv7_feats)  
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous() 
        c_conv7 = c_conv7.view(batch_size, -1, self.n_classes)  
        c_conv8_2 = self.cl_conv8_2(conv8_2_feats) 
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()  
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)  
        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous() 
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)  
        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)  
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()  
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes) 
        c_conv11_2 = self.cl_conv11_2(conv11_2_feats) 
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous() 
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)  

        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1) 
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2], dim=1)

        return locs, classes_scores


class SSD300(nn.Module):
    def __init__(self, n_classes):
        super(SSD300, self).__init__()
        self.n_classes = n_classes
        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes)
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescale_factors, 20)
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        conv4_3_feats, conv7_feats = self.base(image)
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt() 
        conv4_3_feats = conv4_3_feats / norm 
        conv4_3_feats = conv4_3_feats * self.rescale_factors  
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = \
            self.aux_convs(conv7_feats)  
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats)  

        return locs, classes_scores

    def create_prior_boxes(self):
        fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())
        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1) 

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap):
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2) 
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy)) 
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  
            for c in range(1, self.n_classes):
                class_scores = predicted_scores[i][:, c] 
                score_above_min_score = class_scores > min_score
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score] 
                class_decoded_locs = decoded_locs[score_above_min_score]
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True) 
                class_decoded_locs = class_decoded_locs[sort_ind]
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)

                # Thay đổi từ torch.uint8 sang torch.bool
                suppress = torch.zeros((n_above_min_score), dtype=torch.bool).to(device) 
                for box in range(class_decoded_locs.size(0)):
                    if suppress[box] == 1:
                        continue

                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    suppress[box] = 0

                image_boxes.append(class_decoded_locs[~suppress])  # Sử dụng ~ thay vì 1 -
                image_labels.append(torch.LongTensor((~suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[~suppress])
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            image_boxes = torch.cat(image_boxes, dim=0) 
            image_labels = torch.cat(image_labels, dim=0)
            image_scores = torch.cat(image_scores, dim=0) 
            n_objects = image_scores.size(0)

            if n_objects > 200:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:200] 
                image_boxes = image_boxes[sort_ind][:200] 
                image_labels = image_labels[sort_ind][:200] 

            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores 


class MultiBoxLoss(nn.Module):
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()  
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)

        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)  
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)
            _, prior_for_each_object = overlap.max(dim=1) 
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)
            overlap_for_each_prior[prior_for_each_object] = 1.
            label_for_each_prior = labels[i][object_for_each_prior] 
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  
            true_classes[i] = label_for_each_prior
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)
        positive_priors = true_classes != 0 
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])

        n_positives = positive_priors.sum(dim=1) 
        n_hard_negatives = self.neg_pos_ratio * n_positives 
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1)) 
        conf_loss_all = conf_loss_all.view(batch_size, n_priors) 
        conf_loss_pos = conf_loss_all[positive_priors] 
        conf_loss_neg = conf_loss_all.clone() 
        conf_loss_neg[positive_priors] = 0. 
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True) 
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1) 
        conf_loss_hard_neg = conf_loss_neg[hard_negatives] 
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  

        return conf_loss + self.alpha * loc_loss
