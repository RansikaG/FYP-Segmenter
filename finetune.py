import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T
import segm.utils.torch as ptu
from dataloader import Image_and_Masks
from segm.model.factory import load_model, create_decoder
from torchsummary import summary


def finetune(model_path='E:/GitHub Repos/segmenter_model_data/checkpoint.pth', gpu=True):
    ptu.set_gpu_mode(gpu)

    loaded_model, variant = load_new_model(model_path)
    model = loaded_model
    model.to(ptu.device)

    # print(loaded_model)
    # summary(model, (3,224, 224),2)
    # dataset = Image_and_Masks(root_dir='E:/GitHub Repos/V7_masks')
    train_dataset = Image_and_Masks(root_dir='dataset', mode='train')
    valid_dataset = Image_and_Masks(root_dir='dataset', mode='validate')
    dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    dataloader_valid = DataLoader(valid_dataset, batch_size=21)
    amp_autocast = torch.cuda.amp.autocast

    ##Training loop

    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    epoch = 100

    for ep in range(epoch):
        train_mask_loss = 0.
        train_area_loss = 0.
        train_div_loss = 0.
        model.train()

        print('\nStarting epoch %d / %d :' % (ep + 1, epoch))
        pbar = tqdm(total=len(dataloader_train))
        for batch_idx, data in enumerate(dataloader_train):
            image, mask, viewpoint = data
            image = image.to(ptu.device)
            mask = mask.to(ptu.device)  # between 0-1
            with amp_autocast():
                seg_pred = model.forward(image)
                pred_maps = create_attention_maps(seg_pred)

            loss_mask, loss_area, loss_div = Loss(pred_maps, mask, viewpoint)
            loss = loss_mask + loss_area + loss_div

            train_mask_loss += loss_mask.item()
            train_area_loss += loss_area.item()
            train_div_loss += loss_div.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'mask_loss': ' {0:1.3f}'.format(train_mask_loss / (batch_idx + 1))})
            pbar.update(1)
        if (ep + 1) % 10 == 0:
            save_model(model, model_path, ep + 1)
            print('Evaluating in ' + str(ep+1))
            evaluate_images(model=model, path='dataset', validloader=dataloader_valid, ep=ep)
        pbar.close()




def save_model(model, model_path, epoch):
    snapshot = dict(
        model=model.state_dict(),
    )
    torch.save(snapshot, model_path.replace('.pth', '_' + str(epoch) + '_epoch.pth'))


def evaluate_images(model, path, validloader, ep):
    inv = T.Compose([T.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                     T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])
    data, mask, view = iter(validloader).next()
    pred = model(data.to(ptu.device))

    data = [inv(x).permute(1, 2, 0).cpu().detach().numpy() for x in data]
    view = view.detach().numpy()
    pred = pred.detach().cpu().numpy()

    orien_dict = {0: 'Front', 1: 'Rear', 2: 'Side', 3: 'Front-Side', 4: 'Rear-Side'}
    plt.figure()
    for i in range(21):
        plt.subplot(7, 12, (4 * i + 1));
        plt.axis('off');
        plt.title(orien_dict[view[i]], fontsize=6, fontweight="bold")
        plt.imshow(data[i])
        plt.subplot(7, 12, (4 * i + 2));
        plt.axis('off');
        plt.title('Front', fontsize=6)
        plt.imshow(data[i] * np.tile(pred[i, 0, :, :, np.newaxis], (1, 1, 3)))
        plt.subplot(7, 12, (4 * i + 3));
        plt.axis('off');
        plt.title('Rear', fontsize=6)
        plt.imshow(data[i] * np.tile(pred[i, 1, :, :, np.newaxis], (1, 1, 3)))
        plt.subplot(7, 12, (4 * i + 4));
        plt.axis('off');
        plt.title('Side', fontsize=6)
        plt.imshow(data[i] * np.tile(pred[i, 2, :, :, np.newaxis], (1, 1, 3)))
    image_name = os.path.join(path, '%i.png' % (ep + 1))
    plt.savefig(image_name);
    plt.close()


def create_attention_maps(seg_pred):
    softmax = nn.Softmax2d()
    seg_soft_masks = softmax(seg_pred)
    seg_soft_masks = seg_soft_masks
    return seg_soft_masks


def load_new_model(model_path):
    loaded_model, variant = load_model(model_path)
    net_kwargs = variant["net_kwargs"]
    # embed_dim = int(net_kwargs["d_model"])

    decoder_cfg = net_kwargs.pop("decoder")
    decoder_cfg["n_cls"] = 3
    decoder = create_decoder(loaded_model.encoder, decoder_cfg)

    loaded_model.decoder = decoder
    state_dict = loaded_model.state_dict()
    # cls = loaded_model.decoder.cls_emb
    # classes = torch.tensor(cls[:, 57, :])
    # cls_embeddings = torch.rand(1, 3, embed_dim)
    # state_dict['decoder.cls_emb'] = cls_embeddings
    # state_dict['decoder.mask_norm.bias'] = torch.rand(3)
    # state_dict['decoder.mask_norm.weight'] = torch.rand(3)
    del loaded_model, variant
    modified_model_path = model_path.replace('.pth', '_new.pth')

    snapshot = dict(
        model=state_dict,
    )
    torch.save(snapshot, modified_model_path)

    new_model, variant = load_model(modified_model_path, modify=True)
    # new_model.encoder.reqiresgrad(False)
    children = new_model.children()
    for i, child in enumerate(children):
        if i > 0:
            for param in child.parameters():
                param.requires_grad = False
    return new_model, variant


def Loss(pred, target, view):
    bs, c, h, w = pred.size()
    device = pred.device

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
    return loss_mask, 0.5 * loss_area.mean(), loss_div


if __name__ == "__main__":
    # load_new_model(model_path='E:/GitHub Repos/segmenter_model_data/checkpoint.pth')
    # finetune(model_path='E:/GitHub Repos/segmenter_model_data/checkpoint.pth')
    finetune(model_path='/home/fyp3-2/Desktop/BATCH18/FYP-Segmenter/PretrainedModels/checkpoint.pth')
