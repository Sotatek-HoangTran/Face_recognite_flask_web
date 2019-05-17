import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

# import torchvision.datasets as datasets
from torch.utils.data.dataset import Dataset

from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from head.metrics import ArcFace
from loss.focal import FocalLoss
from util.utils import make_weights_for_balanced_classes, get_val_data, separate_irse_bn_paras, separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val, AverageMeter, accuracy

# from tensorboardX import SummaryWriter
import os
import csv
import pandas as pd
import numpy as np
import io

from PIL import Image
from tqdm import tqdm
from torchvision import datasets
from matplotlib import cm

RGB_MEAN = [0.5, 0.5, 0.5]
INPUT_SIZE = [112, 112]
RGB_STD = [0.5, 0.5, 0.5]
EMBEDDING_SIZE = 512

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GPU_ID = [0]

val_transform = transforms.Compose([
    transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]), # smaller side resized
    transforms.TenCrop((INPUT_SIZE[0], INPUT_SIZE[1])),
    
    transforms.Lambda(lambda crops: 
            [transforms.ToTensor()(crop) for crop in crops]
    ),
    transforms.Lambda(lambda norms: torch.stack(
            [transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)(norm) for norm in norms]
        )
    ),
])

def image_loader(image):
	image = Image.open(image)
	image = val_transform(image).float()
	image = image.clone().detach().requires_grad_(True)
	image = image.unsqueeze(0)
	return image

def predict(image):

	image = image_loader(image=image)

	BACKBONE = IR_50(INPUT_SIZE)
	HEAD = ArcFace(in_features = EMBEDDING_SIZE, out_features = 1000, device_id = GPU_ID)

	BACKBONE = BACKBONE.to(DEVICE)
	HEAD = HEAD.to(DEVICE)

	BACKBONE.load_state_dict(torch.load('./trained_model/Backbone_IR_50_ArcFace_30.pth'))
	HEAD.load_state_dict(torch.load('./trained_model/Head_IR_50_ArcFace_30.pth'))

	BACKBONE.eval()    
	HEAD.eval()

	image = image.to(DEVICE)
	bs, ncrops, c, h, w = image.size()
	inputs = image.view(-1, c, h, w)
	features = BACKBONE(inputs)
	outputs = HEAD(features, None)
	outputs = outputs.view(bs, ncrops, -1).mean(1)
	top_probs, top_labs = outputs.data.topk(1)
	top_labs = top_labs.cpu().numpy()
	top_probs = top_probs.cpu().numpy()
	return int(top_labs), float(top_probs)

if __name__ == '__main__':
	print(predict('./img/00aa05306ba14492aecfad21aac28b76.png'))