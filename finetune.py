from pathlib import Path

import torch
from torch import nn

import segm.utils.torch as ptu
from segm.data.utils import IGNORE_LABEL
from segm.model.factory import load_model
from torchsummary import summary


def finetune(model_path='E:/GitHub Repos/segmenter_model_data/checkpoint.pth', gpu=True):
    ptu.set_gpu_mode(gpu)

    model_dir = Path(model_path).parent
    loaded_model, variant = load_model(model_path)
    loaded_model
    loaded_model.to(ptu.device)

    print(loaded_model)
    # summary(model, (3,224, 224),2)

    ##Training loop
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)


def create_segementer(model_path):
    loaded_model, variant = load_model(model_path)
    encoder = nn.Sequential(*list(loaded_model.children())[0:1])
    print(loaded_model)
    print('###############')
    print(encoder)

if __name__ == "__main__":
    create_segementer(model_path='E:/GitHub Repos/segmenter_model_data/checkpoint.pth')
