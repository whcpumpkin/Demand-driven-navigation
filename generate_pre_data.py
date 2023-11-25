import torchvision.transforms as transforms
import os
import datetime
import random
import ctypes
import time
import sys
from utils.args import parse_arguments
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from PIL import Image
from model.detr import DETRModel
# from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import json
import h5py
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
import torchvision
from utils.util import make_env_fn
from vector_env import VectorEnv
import copy
import clip
import math
import orjson
BICUBIC = InterpolationMode.BICUBIC


def get_valid_transforms(h, w):
    return A.Compose(
        [A.Resize(height=h, width=w, p=1.0), ToTensorV2(p=1.0)],
        p=1.0,
        # bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0, label_fields=['labels'])
    )


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def pre_traj_crop():
    with open(args.path2dataset + "bc_{}_check.json".format(args.dataset_mode), 'r', encoding='utf8') as fp:
        data = json.load(fp)
    detr = DETRModel(2, 100)
    detr.load_state_dict(torch.load("./pretrained_model/detr_new_18.pth"))
    detr = detr.to(args.device).eval()
    transform_detr = get_valid_transforms(h=300, w=300)
    local_image_encoder, local_image_encoder_preprocess = clip.load("ViT-B/32")
    local_image_encoder = local_image_encoder.to(args.device).eval()

    def crop_from_bounding_box(image, bbox, top_k_idx):
        crop_start_time = time.time()
        crop_cur = []
        img = torch.tensor(image.copy()).permute(2, 0, 1).to(torch.float32).to(args.device)
        for idx_top in range(args.top_k):

            bbox_idx = top_k_idx[idx_top]
            x = math.floor(bbox[bbox_idx, 0] * 300)
            y = math.floor(bbox[bbox_idx, 1] * 300)
            w = math.ceil(bbox[bbox_idx, 2] * 300 + 0.5)
            h = math.ceil(bbox[bbox_idx, 3] * 300 + 0.5)
            crop_1 = torchvision.transforms.functional.crop(img, y, x, h, w)
            crop_1_PIL = torchvision.transforms.ToPILImage()(crop_1)
            crop_cur.append(local_image_encoder_preprocess(crop_1_PIL))

        crop_cur = torch.stack(crop_cur, dim=0).to(args.device)
        crop_feature = local_image_encoder.encode_image(crop_cur)
        crop_feature = crop_feature / crop_feature.norm(dim=-1, keepdim=True)
        # print("crop time: ", time.time()-crop_start_time)
        return crop_feature

    pre_data = []

    for i in tqdm(range(args.start, len(data))):
        # print(i)
        pre_data_item = {}
        path = data[i]["path"]
        path = args.dataset_mode + path.split(args.dataset_mode)[-1]
        pre_data_item['path'] = path

        seq_len = data[i]["seq_len"]
        pre_data_item['image_pre'] = {}
        if os.path.exists(os.path.join(args.path2dataset, "bc/" + path)) is False:
            continue
        listfiles = os.listdir(os.path.join(args.path2dataset, "bc/" + path))
        depth=0
        for idx in range(len(listfiles)):
            if "depth" in listfiles[idx]:
                depth+=1
        if seq_len != len(listfiles)-depth:
            print("seq_len error")
            print(path)
            continue

        for j in range(seq_len):
            image_path = os.path.join(args.path2dataset, "bc/" + path, str(j) + ".jpg")
            image = np.array(Image.open(image_path))
            detr_image = transform_detr(image=image.copy() / 255.0)["image"].to(torch.float32).to(args.device)
            detr_output = detr([detr_image])

            prob = detr_output['pred_logits'].softmax(dim=-1).detach().cpu().numpy()[:, :, 0]
            top_k_idx = torch.tensor(np.argsort(prob, axis=-1)[:, -args.top_k:]).to(args.device)
            bbox_top_k = torch.index_select(detr_output['pred_boxes'][0], dim=0, index=top_k_idx[0])
            logits_top_k = torch.index_select(detr_output['pred_logits'][0], dim=0, index=top_k_idx[0])
            local_feature = crop_from_bounding_box(image, detr_output['pred_boxes'][0], top_k_idx[0])

            pre_data_item['image_pre'][str(j)] = torch.cat([local_feature, bbox_top_k, logits_top_k], dim=-1).cpu().detach().numpy()
            # pre_data_item['image_pre'][str(j)]["feature"] = local_feature.cpu().detach().numpy()
            # pre_data_item['image_pre'][str(j)]["bbox"] = bbox_top_k.cpu().detach().numpy()
            # pre_data_item['image_pre'][str(j)]["logits"] = logits_top_k.cpu().detach().numpy()
        pre_data.append(pre_data_item)
        # json.dump(pre_data_item, f_json)
        # break
        if (i + 1) % 6000 == 0:
            serialize = orjson.dumps(pre_data, option=orjson.OPT_SERIALIZE_NUMPY)
            f_json = open("./dataset/bc_{}_{}_pre.json".format(args.dataset_mode, i), 'wb')
            f_json.write(serialize)
            pre_data = []
    serialize = orjson.dumps(pre_data, option=orjson.OPT_SERIALIZE_NUMPY)
    f_json = open("./dataset/bc_{}_{}_pre.json".format(args.dataset_mode, i), 'wb')
    f_json.write(serialize)
    
def merge_pre_crop_json():
    all_data = []
    for idx in [args.start]:
        hf = h5py.File("./dataset/bc_val_0_pre.h5", 'w')
        num = 0
        pa_set = set()
        with open("./dataset/bc_val_{}_pre.json".format(idx), "rb") as f:
            data = orjson.loads(f.read())
            for item in tqdm(data):
                if item["path"].replace("/", "_") in pa_set:
                    continue
                else:
                    pa_set.add(item["path"].replace("/", "_"))
                for l in range(len(item["image_pre"])):
                    p = item["path"].replace("/", "_") + "_" + str(l)
                    hf[p] = item["image_pre"][str(l)]
        hf.close()

if __name__ == "__main__":
    args = parse_arguments()

    pre_traj_crop()
