"""

@author:  AIcandy 
@website: aicandy.vn

"""

import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from aicandy_utils_src_lkkqrsdm.aicandy_ssd300_model_tahusyda import SSD300, MultiBoxLoss
from aicandy_utils_src_lkkqrsdm.aicandy_ssd300_datasets_kgvvoaac import VOC_Dataset
from aicandy_utils_src_lkkqrsdm.aicandy_ssd300_utils_xslstyan import *


# python aicandy_ssd300_train_haykkxnu.py --train_dir /aicandy/datasets/aicandy_voc_nskpbsgv --num_epochs 500 --batch_size 8 --last_checkpoint 'aicandy_model_out_gnloibxd/aicandy_checkpoint_ssd300.pth' 



retain_difficult = True 
num_classes = len(label_map) 
num_workers = 4 
learning_rate = 1e-3  
optim_momentum = 0.9  
weight_decay_factor = 5e-4  
gradient_clip = None 
cudnn.benchmark = True


def train(train_dir, num_epochs, batch_size, last_checkpoint):
    global start_epoch, label_map, epoch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model_checkpoint = last_checkpoint 
    if model_checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=num_classes)
        bias_params = list()
        non_bias_params = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    bias_params.append(param)
                else:
                    non_bias_params.append(param)
        optimizer = torch.optim.SGD(params=[{'params': bias_params, 'lr': 2 * learning_rate}, {'params': non_bias_params}],
                                    lr=learning_rate, momentum=optim_momentum, weight_decay=weight_decay_factor)

    else:
        model_checkpoint = torch.load(model_checkpoint)
        start_epoch = model_checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = model_checkpoint['model']
        optimizer = model_checkpoint['optimizer']

    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)
    train_data = VOC_Dataset(train_dir,
                                  dataset_split='train',
                                  include_difficult=retain_difficult)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_data.collate_fn, num_workers=num_workers,
                                               pin_memory=True) 

    for epoch in range(start_epoch, num_epochs):
        loss = train_one_epoch(train_loader=train_loader,
                               model=model,
                               criterion=criterion,
                               optimizer=optimizer)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')
        if epoch == start_epoch:
            min_loss = loss
            save_checkpoint(epoch, model, optimizer)
            print(f'Saved best model with loss: {min_loss:.4f}')

        if loss < min_loss:
            min_loss = loss
            save_checkpoint(epoch, model, optimizer)
            print(f'Saved best model with loss: {min_loss:.4f}')


def train_one_epoch(train_loader, model, criterion, optimizer):
    model.train()
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        predicted_locs, predicted_scores = model(images)
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)
        optimizer.zero_grad()
        loss.backward()
        if gradient_clip is not None:
            clip_gradient(optimizer, gradient_clip)
        optimizer.step()

    del predicted_locs, predicted_scores, images, boxes, labels

    return loss


if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='AIcandy.vn')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--last_checkpoint', type=str, default='None', help='Path to the last checkpoint')

    args = parser.parse_args()
    train(args.train_dir, args.num_epochs, args.batch_size, args.last_checkpoint)
