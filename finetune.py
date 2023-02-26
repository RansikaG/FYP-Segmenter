from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import segm.utils.torch as ptu
from dataloader import Image_and_Masks
from segm.data.utils import IGNORE_LABEL
from segm.model.factory import load_model
from torchsummary import summary


def finetune(model_path='E:/GitHub Repos/segmenter_model_data/checkpoint.pth', gpu=True):
    ptu.set_gpu_mode(gpu)

    model_dir = Path(model_path).parent
    loaded_model, variant = load_model(model_path)
    model = loaded_model
    model.to(ptu.device)

    # print(loaded_model)
    # summary(model, (3,224, 224),2)
    dataset = Image_and_Masks(root_dir='E:/GitHub Repos/V7_masks')
    dataloader_train = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)

    amp_autocast = torch.cuda.amp.autocast

    ##Training loop

    if torch.cuda.is_available():
        loaded_model.cuda()
    optimizer = optim.Adam(loaded_model.parameters(), lr=0.00001)
    epoch = 1
    for ep in range(epoch):
        model.train()
        print('\nStarting epoch %d / %d :' % (ep + 1, epoch))
        pbar = tqdm(total=len(dataloader_train))
        for batch_idx, data in enumerate(dataloader_train):
            image, mask, viewpoint = data
            image = image.to(ptu.device)
            mask = mask.to(ptu.device)
            with amp_autocast():
                seg_pred = model.forward(image)
                pred_maps = create_attention_maps(seg_pred)
            loss_mask, loss_area, loss_div, foreground_loss = Loss(pred_maps, seg_pred[:, 0:1, :, :], mask, viewpoint)
            loss = loss_mask + loss_area + loss_div + foreground_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(1)
        pbar.close()


def create_attention_maps(seg_pred):
    softmax = nn.Softmax2d()
    seg_soft_masks = softmax(seg_pred)
    seg_soft_masks = seg_soft_masks *255
    return seg_soft_masks[:, :3, :, :]


def create_segementer(model_path):
    loaded_model, variant = load_model(model_path)
    # encoder = nn.Sequential(*list(loaded_model.children())[0:1])
    print(loaded_model)
    state_dict = loaded_model.state_dict()
    cls = loaded_model.decoder.cls_emb
    print(state_dict['decoder.cls_emb'].shape)


def Loss(pred, seg_pred, target, view):
    bs, c, h, w = pred.size()
    device = pred.device

    # Foreground loss

    criterion_foreground = torch.nn.CrossEntropyLoss()

    loss_foreground = criterion_foreground(seg_pred, target)
    ''' 1st loss: Mask Reconstruction loss '''
    pred_mask = torch.zeros_like(pred)
    for i in range(bs):  # F/R/S iterating over the batch dimension
        if view[i] == 0:
            pred_mask[i] = torch.LongTensor([1, 0, 0]).view(-1, 1, 1).repeat(1, h, w)
        elif view[i] == 1:
            pred_mask[i] = torch.LongTensor([0, 1, 0]).view(-1, 1, 1).repeat(1, h, w)
        elif view[i] == 2:
            pred_mask[i] = torch.LongTensor([0, 0, 1]).view(-1, 1, 1).repeat(1, h, w)
        elif view[i] == 3:
            pred_mask[i] = torch.LongTensor([1, 0, 1]).view(-1, 1, 1).repeat(1, h, w)
        elif view[i] == 4:
            pred_mask[i] = torch.LongTensor([0, 1, 1]).view(-1, 1, 1).repeat(1, h, w)
    pred_mask = pred_mask.to(device)

    criterion_mask = nn.MSELoss()
    pred_mask = torch.sum(pred * pred_mask, dim=1, keepdim=True)
    loss_mask = criterion_mask(pred_mask, target)

    ''' 2nd loss: Area Constraint loss '''
    mask_area = pred.view(bs, c, -1).sum(2)
    area = target.view(bs, -1).sum(1, keepdim=True).expand_as(mask_area)
    mask_area_max = torch.zeros_like(mask_area)
    for i in range(bs):
        if view[i] == 0:
            mask_area_max[i] = torch.FloatTensor([1, 0, 0])
        elif view[i] == 1:
            mask_area_max[i] = torch.FloatTensor([0, 1, 0])
        elif view[i] == 2:
            mask_area_max[i] = torch.FloatTensor([0, 0, 1])
        elif view[i] == 3:
            mask_area_max[i] = torch.FloatTensor([0.7, 0, 0.4])
        elif view[i] == 4:
            mask_area_max[i] = torch.FloatTensor([0, 0.7, 0.4])
    mask_area_max = mask_area_max.to(device)

    criterion_area = nn.ReLU()
    loss_area = criterion_area(mask_area / area - mask_area_max)

    ''' 3rd loss: Spatial Diversity loss '''
    criterion_div = nn.ReLU()
    loss_divFR = criterion_div((pred[:, 0] * pred[:, 1]).mean())
    loss_divFS = criterion_div((pred[:, 0] * pred[:, 2]).mean() - 0.04)
    loss_divRS = criterion_div((pred[:, 1] * pred[:, 2]).mean() - 0.04)
    loss_div = loss_divFR + loss_divFS + loss_divRS
    return loss_mask, 0.5 * loss_area.mean(), loss_div, loss_foreground


if __name__ == "__main__":
    # create_segementer(model_path='E:/GitHub Repos/segmenter_model_data/checkpoint.pth')
    finetune(model_path='E:/GitHub Repos/segmenter_model_data/checkpoint.pth')
