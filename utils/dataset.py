from torch.utils.data import Dataset

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from omegaconf import OmegaConf
import os.path as op
import random
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
import argparse
import json
import torch
import numpy as np
if __name__ != "__main__":
    from utils.args import parse_arguments
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from model.detr import DETRModel
import h5py
import torchvision.transforms as transforms
from transformers import AutoFeatureExtractor
import pandas as pd
import orjson


class instruction_LGO_dataset(Dataset):

    def __init__(self, args, mode="train"):
        super(instruction_LGO_dataset, self).__init__()
        self.args = args
        with open(args.path2instruction_bert_features, 'r', encoding='utf8') as fp:
            self.instruction_features = json.load(fp)
        with open(args.path2LGO_features, 'r', encoding='utf8') as fp:
            self.LGO_features = json.load(fp)
        with open(args.path2dataset + "/" + "instruction_{}_check.json".format(mode), 'r', encoding='utf8') as fp:
            self.instruction = json.load(fp)
        num = 0
        del_key = []
        for key in self.LGO_features.keys():
            if len(list(self.LGO_features[key].keys())) < 2:
                del_key.append(key)

        for key in del_key:
            del self.LGO_features[key]
            del self.instruction_features[key]
            if key in self.instruction.keys():
                del self.instruction[key]
        self.check()
        self.instruction = set(self.instruction.keys()).intersection(set(self.instruction_features.keys())).intersection(set(self.LGO_features.keys()))

    def __len__(self):
        return len(list(self.instruction))

    def check(self):
        del_key = []
        for key in self.instruction_features.keys():
            if key not in self.LGO_features.keys():
                del_key.append(key)
        for key in del_key:
            del self.instruction_features[key]
            if key in self.instruction.keys():
                del self.instruction[key]

        del_key = []
        for key in self.instruction.keys():
            if key not in self.LGO_features.keys():
                del_key.append(key)
        for key in del_key:
            del self.instruction[key]

    def __getitem__(self, idx):
        instruction = list(self.instruction)[idx]
        list_obj = list(self.LGO_features[instruction].keys())
        obj_idx = random.sample(range(0, len(list_obj)), 2)
        positive_obj_feature_1 = torch.tensor(self.LGO_features[instruction][list_obj[obj_idx[0]]])
        positive_obj_feature_1 = positive_obj_feature_1 / positive_obj_feature_1.norm(dim=-1, keepdim=True)

        positive_obj_feature_2 = torch.tensor(self.LGO_features[instruction][list_obj[obj_idx[1]]])
        positive_obj_feature_2 = positive_obj_feature_2 / positive_obj_feature_2.norm(dim=-1, keepdim=True)

        instruction_feature = torch.tensor(self.instruction_features[instruction][1][0][0])
        positive_demand_feature_1 = torch.cat((instruction_feature, positive_obj_feature_1))
        positive_demand_feature_2 = torch.cat((instruction_feature, positive_obj_feature_2))

        negative_demand_feature_diff_instruction = []
        negative_demand_feature_diff_object = []
        for i in range((self.args.mini_batch_size - 2) // 2):
            negative_instruction = random.choice(list(self.instruction_features.keys()))
            negative_instruction_feature = torch.tensor(self.instruction_features[negative_instruction][1][0][0])
            negative_demand_feature = torch.cat((negative_instruction_feature, positive_obj_feature_1))
            negative_demand_feature_diff_instruction.append(negative_demand_feature)

            random_instruction = random.choice(list(self.instruction_features.keys()))
            negative_obj_feature = torch.tensor(random.choice(list(self.LGO_features[random_instruction].values())))
            negative_obj_feature = negative_obj_feature / negative_obj_feature.norm(dim=-1, keepdim=True)

            negative_demand_feature = torch.cat((instruction_feature, negative_obj_feature))
            negative_demand_feature_diff_object.append(negative_demand_feature)
        feature_all = []
        feature_all.append(positive_demand_feature_1)
        feature_all.append(positive_demand_feature_2)
        feature_all.extend(negative_demand_feature_diff_instruction)
        feature_all.extend(negative_demand_feature_diff_object)
        feature_all = torch.stack(feature_all, dim=0)
        # return (positive_demand_feature_1, positive_demand_feature_2, negative_demand_feature_diff_instruction, negative_demand_feature_diff_object)
        return feature_all


def get_valid_transforms(h, w):
    return A.Compose(
        [A.Resize(height=h, width=w, p=1.0), ToTensorV2(p=1.0)],
        p=1.0,
        # bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0, label_fields=['labels'])
    )


class pretrain_dataset(Dataset):

    def __init__(self, args, mode="train"):
        super(pretrain_dataset, self).__init__()
        self.args = args
        self.mode = mode
        # datanames = os.listdir(path)
        # self.data = []
        # for name in datanames:
        #     if "json" in name:
        #         with open(path+name, 'r', encoding='utf8') as fp:
        #             json_data = fp.readlines()[0].split("}")[:-1]
        #         for line in json_data:
        #             self.data.append(eval(line+"}"))
        with open(args.path2dataset + "instruction_bert_features.json", 'r', encoding='utf8') as fp:
            self.instruction_feature = json.load(fp)
        with open(args.path2dataset + "traj_{}_metadata.json".format(mode), 'r', encoding='utf8') as fp:
            self.data = json.load(fp)
        del_key = []
        self.new_data = []
        for item in self.data:
            if item["instruction"] in self.instruction_feature.keys():
                self.new_data.append(item)
        self.data = self.new_data
        self.detr_data = h5py.File(args.path2dataset + "/crop/crop_image_{}.hdf5".format(mode), "r")
        self.transform_global = transforms.Compose([transforms.ToPILImage(), transforms.Resize(args.input_size, interpolation=3), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform_global_depth = transforms.Compose([transforms.ToPILImage(), transforms.Resize(args.input_size, interpolation=3), transforms.CenterCrop(224), transforms.ToTensor()])
        self.resnet_feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
        t = 1

    def __len__(self):
        return len(list(self.data))

    def __getitem__(self, idx):
        if self.mode == "val":
            image_path = os.path.join(self.args.path2dataset, self.data[idx]["locate"][2:15] + self.mode + self.data[idx]["locate"][18:])
        else:
            image_path = os.path.join(self.args.path2dataset, self.data[idx]["locate"][2:15] + self.mode + self.data[idx]["locate"][20:])
        image = self.transform_global(np.array(Image.open(image_path + ".jpg")))
        depth = torch.tensor(np.array(Image.open(image_path + "_depth.jpg").convert('L').resize((300, 300)))) / 255.0
        # detr_image = self.transform_detr(image=image.copy() / 255.0)["image"].to(torch.float32).to(self.args.device)
        # detr_output = self.detr([detr_image])
        crop = torch.tensor(np.array(self.detr_data[image_path.replace("/", "_")[1:] + "_crop"]))
        bbox = torch.tensor(np.array(self.detr_data[image_path.replace("/", "_")[1:] + "_pred_boxes"]))
        logits = torch.tensor(np.array(self.detr_data[image_path.replace("/", "_")[1:] + "_pred_logits"]))
        instruction = self.data[idx]["instruction"]
        instruction_feature = torch.tensor(self.instruction_feature[instruction][1][0][0])
        target_action = self.data[idx]["action"]
        # return tuple(image, crop, bbox, logits, depth, instruction_feature, target_action)
        return {"crop": crop, "bbox": bbox, "logits": logits, "instruction": instruction, "instruction_feature": instruction_feature, "target_action": target_action, "depth": depth, "image": image}


class Pre_VG_dataset(Dataset):

    def __init__(self, args, mode="train"):
        super(Pre_VG_dataset, self).__init__()
        self.args = args
        self.mode = mode
        data = pd.read_csv(args.path2dataset + "data_grounding_multiple/{}_check.csv".format(mode))
        self.image_id = sorted(list(set(data["image_id"].tolist())))
        t = 1
        # self.bbox = data[" bbox"].tolist()
        # self.object_name = data[" object name"].tolist()
        # with open(args.path2dataset + "answer_small.json", 'r', encoding='utf8') as fp:
        #     self.answer_small = json.load(fp)
        # # load bert feature
        # with open(args.path2dataset + "instruction_bert_features.json", 'r', encoding='utf8') as fp:
        #     self.instruction_feature = json.load(fp)
        # idx = 0
        # while idx < len(self.image_id):
        #     if self.object_name[idx] not in self.answer_small.keys():
        #         del self.image_id[idx]
        #         del self.bbox[idx]
        #         del self.object_name[idx]
        #     else:
        #         idx += 1

    def __len__(self):
        return len(self.image_id)

    def __getitem__(self, idx):
        image_path = os.path.join(self.args.path2dataset, "data_grounding_multiple/{}".format(self.mode), self.image_id[idx] + ".jpg")
        path = os.path.join(self.args.path2dataset, "data_grounding_multiple/{}".format(self.mode), self.image_id[idx])
        path = path.replace("/", "_")[1:]
        image = np.array(Image.open(image_path))
        # bbox = eval(self.bbox[idx])
        # object_name = self.object_name[idx]
        # instruction = random.choice(self.answer_small[object_name])
        # instruction_feature = self.instruction_feature[instruction][1][0][0]
        return (image, path)


class VG_dataset(Dataset):

    def __init__(self, args, mode="train"):
        super(VG_dataset, self).__init__()
        self.args = args
        self.mode = mode
        data = pd.read_csv(args.path2dataset + "data_grounding_multiple/{}_check.csv".format(mode))
        self.image_id = data["image_id"].tolist()
        self.bbox = data[" bbox"].tolist()
        self.object_name = data[" object name"].tolist()
        with open(args.path2dataset + "answer_small.json", 'r', encoding='utf8') as fp:
            self.answer_small = json.load(fp)
        # load bert feature
        with open(args.path2dataset + "instruction_bert_features.json", 'r', encoding='utf8') as fp:
            self.instruction_feature = json.load(fp)
        # with open(args.path2dataset + "data_grounding/pre_data_{}_or.json".format(args.dataset_mode), 'rb') as fp:
        #     self.pre_data = orjson.loads(fp)
        if mode == "train":
            self.f = []
            self.value_f = {}
            self.ref_f = {}
            for i in range(5):
                self.f.append(h5py.File("./dataset/data_grounding_multiple/pre_data_{}_{}.hdf5".format(args.dataset_mode, i), "r"))
            for i in range(5):
                # for key in tqdm(list(self.f[i].keys()), desc="Train_f:"+str(i)):
                #     self.value_f[key] = np.array(self.f[i][key])
                for key in tqdm(list(self.f[i].keys()), desc="Train_f:" + str(i)):
                    self.ref_f[key] = i
        else:
            self.value_f = {}
            self.f = h5py.File("./dataset/data_grounding_multiple/pre_data_{}.hdf5".format(mode), "r")
            # for key in tqdm(list(self.f.keys()), desc="Val_f"):
            #     self.value_f[key] = np.array(self.f[key])
        # with open(args.path2dataset + "data_grounding/pre_data_{}.json".format(mode), 'r', encoding='utf8') as fp:
        #     self.pre_data = json.load(fp)
        # idx = 0
        # while idx < len(self.image_id):
        #     if self.object_name[idx] not in self.answer_small.keys():
        #         del self.image_id[idx]
        #         del self.bbox[idx]
        #         del self.object_name[idx]
        #     else:
        #         idx += 1

    def __len__(self):
        if self.mode == "train":
            return len(self.image_id)
        elif self.mode == "val":
            return len(self.image_id)

    def __getitem__(self, idx):
        if self.mode == "val":
            image_path = os.path.join(self.args.path2dataset, "data_grounding_multiple/{}".format(self.mode), self.image_id[idx] + ".jpg")
            path = os.path.join(self.args.path2dataset, "data_grounding_multiple/{}".format(self.mode), self.image_id[idx])
            path = path.replace("/", "_")[1:]
            image = np.array(Image.open(image_path))
            bbox = eval(self.bbox[idx])
            object_name = self.object_name[idx]
            instruction = random.choice(self.answer_small[object_name])
            instruction_feature = self.instruction_feature[instruction][1][0][0]
            crop_feature = np.array(self.f[path])
            # crop_feature = self.value_f[path]
        else:
            image_path = os.path.join(self.args.path2dataset, "data_grounding_multiple/{}".format(self.mode), self.image_id[idx] + ".jpg")
            path = os.path.join(self.args.path2dataset, "data_grounding_multiple/{}".format(self.mode), self.image_id[idx])
            path = path.replace("/", "_")[1:]
            image = np.array(Image.open(image_path))
            bbox = eval(self.bbox[idx])
            object_name = self.object_name[idx]
            instruction = random.choice(self.answer_small[object_name])
            instruction_feature = self.instruction_feature[instruction][1][0][0]
            crop_feature = np.array(self.f[self.ref_f[path]][path])
            # crop_feature = self.value_f[path]
        return (image, bbox, object_name, instruction_feature, path, crop_feature)


class Traj_dataset(Dataset):

    def __init__(self, args, mode="train"):
        super(Traj_dataset, self).__init__()
        self.args = args
        self.mode = mode

        with open(args.path2dataset + "answer_small.json", 'r', encoding='utf8') as fp:
            self.answer_small = json.load(fp)
        # load bert feature
        if "wo_BERT" not in self.args.mode or 'wo_MAE' in self.args.mode or "resnet" in self.args.mode:
            with open(args.path2dataset + "instruction_bert_features.json", 'r', encoding='utf8') as fp:
                self.instruction_feature = json.load(fp)
        with open(args.path2dataset + "bc_{}_check.json".format(args.dataset_mode), 'r', encoding='utf8') as fp:
            self.data = json.load(fp)
        self.hf_list = []
        self.hf_dict = {}
        lens = 5 if args.dataset_mode == "train" else 1
        for i in range(lens):
            self.hf_list.append(h5py.File("./dataset/bc_{}_{}_pre.h5".format(args.dataset_mode, i), "r"))
            key_list = list(self.hf_list[-1].keys())
            for key in tqdm(key_list):
                self.hf_dict[key] = i
        self.transform_global = transforms.Compose([transforms.Resize(args.input_size, interpolation=3), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]["path"]
        instruction = self.data[idx]["instruction"]
        start_position = self.data[idx]["start_position"]
        start_rotation = self.data[idx]["start_rotation"]
        start_horizon = self.data[idx]["start_horizon"]
        bbox = self.data[idx]["bbox"]
        object_name = self.data[idx]["object"]
        seq_len = self.data[idx]["seq_len"]
        action = self.data[idx]["action"]
        image = []
        image_PIL=[]
        for i in range(seq_len):
            image_path = os.path.join(self.args.path2dataset, "bc/" + path, str(i) + ".jpg")
            ima_PIL=Image.open(image_path)
            image_PIL.append(np.array(ima_PIL))
            image.append(self.transform_global(ima_PIL))
        p = path.replace("/", "_")
        crop_feature = []
        for i in range(seq_len):
            crop_feature.append(np.array(self.hf_list[self.hf_dict[p + "_" + str(i)]][p + "_" + str(i)]))
        if "wo_BERT" in self.args.mode or 'wo_MAE' in self.args.mode or "resnet" in self.args.mode:
            return (image,image_PIL, bbox, object_name, [instruction], start_position, start_rotation, start_horizon, action, seq_len, crop_feature)
        else:
            return (image,image_PIL, bbox, object_name, self.instruction_feature[instruction][1][0][0], start_position, start_rotation, start_horizon, action, seq_len, crop_feature)

if __name__ == "__main__":
    from args import parse_arguments
    args = parse_arguments()
    dataset = pretrain_dataset(args)
    dataset.__getitem__(19)
    dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=32, num_workers=8)
    for step, batch in tqdm(enumerate(dataloader)):
        t = 1

    t = 1
