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

    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    amp_autocast = torch.cuda.amp.autocast

    ##Training loop
    criterion = torch.nn.CrossEntropyLoss()

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
                print(seg_pred.shape, mask.shape)
                loss = criterion(seg_pred[:, 0:1, :, :], mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(1)
        pbar.close()


def create_segementer(model_path):
    loaded_model, variant = load_model(model_path)
    # encoder = nn.Sequential(*list(loaded_model.children())[0:1])
    print(loaded_model)
    state_dict = loaded_model.state_dict()
    cls = loaded_model.decoder.cls_emb
    print(state_dict['decoder.cls_emb'])

if __name__ == "__main__":
    create_segementer(model_path='E:/GitHub Repos/segmenter_model_data/checkpoint.pth')
    # finetune(model_path='E:/GitHub Repos/segmenter_model_data/checkpoint.pth')
