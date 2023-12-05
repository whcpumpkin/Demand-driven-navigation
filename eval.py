from utils.args import parse_arguments
from vector_env import VectorEnv
from env import Human_Demand_Env, Human_Demand_Env_GPT, Random_Human_Demand_Env, Object_Env
import argparse
import time
import os
import sys
import copy
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.util import make_env_fn
import torch.optim as optim
import numpy as np
from agent import Agent
from utils.dataset import pretrain_dataset, VG_dataset, instruction_LGO_dataset, Traj_dataset
import utils
import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from copy import deepcopy
import clip
from model.detr import DETRModel
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from PIL import Image
from utils.util import get_valid_transforms, _convert_image_to_rgb
from torch.utils.tensorboard import SummaryWriter
from model.VG_model import VG_model
import pandas as pd
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision import transforms
import gc
import math
import json
from transformers import BertTokenizer, BertModel



def eval_DDN(args):
    start_time_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    writer = SummaryWriter(args.eval_path + "/runs")
    f = open(args.eval_path + "/eval_{}_{}.txt".format(args.dataset_mode, start_time_str), "a")
    device = args.device
    agent = Agent(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    listFiles = sorted(list(os.listdir(args.eval_path)))[:-1]
    list_dict = {}
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    for name in listFiles:
        if "checkpoint" not in name:
            continue
        ckpt_idx = eval(name.split("_")[-1].split(".")[0])
        list_dict[ckpt_idx] = name
    list_ckpt = sorted(list(list_dict.keys()))
    dataset_val = Traj_dataset(args, args.dataset_mode)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_val = torch.utils.data.BatchSampler(sampler_val, 1, drop_last=False)
    data_loader_val = DataLoader(dataset_val, batch_sampler=batch_sampler_val, num_workers=args.workers)
    for model_ckpt in list_ckpt:
        if args.eval_ckpt>-1:
            model_path = os.path.join(args.eval_path, list_dict[args.eval_ckpt])
            ckpt_idx = args.eval_ckpt
        else:
            model_path = os.path.join(args.eval_path, list_dict[model_ckpt])
            ckpt_idx = model_ckpt

        agent = agent.to("cpu")
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))
        agent.load_state_dict(model_dict["model_state_dict"])
        agent = agent.to(args.device)
        agent.eval()
        total = 0
        correct = 0
        for i, (image,image_PIL, bbox, object_name, instruction_feature, start_position, start_rotation, start_horizon, action, seq_len, crop_feature) in tqdm(enumerate(data_loader_val), total=data_loader_val.__len__()):
            image = torch.cat(image).to(device)
            # instruction_feature = torch.stack(instruction_feature).to(device)
            inputs = {}
            inputs["crop"] = torch.stack(crop_feature).to(device).squeeze(dim=1).float()

            inputs["image"] = image.float()
            inputs["image_PIL"]=image_PIL

            instruction_feature = torch.stack(instruction_feature).to(device)
            inputs["instruction_feature"] = instruction_feature.permute(1, 0).repeat(seq_len, 1)
            other_inputs = {}
            other_inputs['prev_action'] = (torch.ones((1, 1, 6)).to(args.device)) * np.log(1 / 6)
            other_inputs["prev_hidden_h"] = torch.zeros((agent.lstm_layer_num, 1, args.rnn_hidden_state_dim)).to(args.device)
            other_inputs["prev_hidden_c"] = torch.zeros((agent.lstm_layer_num, 1, args.rnn_hidden_state_dim)).to(args.device)
            outputs = agent.forward_pre(inputs, other_inputs).squeeze(dim=1).squeeze(dim=1)
            action = torch.stack(action).squeeze(dim=1).to(device)
            losses = criterion(outputs, action)
            _, predicted = torch.max(outputs, dim=1)
            total += action.size(0)
            correct += (predicted == action).sum().item()
        print("-------------{}----------------".format(ckpt_idx))
        print('ckpt_idx: {:4d}   accuracy: {}'.format(ckpt_idx, correct / total))
        print("-------------{}----------------".format(ckpt_idx))
        f.write('ckpt_idx: {:4d}   accuracy: {} \n'.format(ckpt_idx, correct / total))
        writer.add_scalar('val/pretrain_accuracy', correct / total, ckpt_idx)
        # writer.add_scalar('val/pretrain_loss', total_loss, ckpt_idx)
        if args.eval_ckpt>-1:
            break



@torch.no_grad()
def test_DDN(args):

    device = args.device
    VG = VG_model(args)
    VG.load_state_dict(torch.load("./pretrained_model/VG.pt", map_location=torch.device('cpu'))["model_state_dict"])
    VG.to(device)
    detr = DETRModel(2, 100)
    detr.load_state_dict(torch.load("./pretrained_model/detr_new_18.pth", map_location=torch.device('cpu')))
    detr.to(device)

    

    def get_VG_model_answer(image, instruction_feature):

        instruction_feature = instruction_feature

        detr_image = [torch.tensor(image.copy()).to(args.device).permute(2, 0, 1) / 255.0]
        detr_output = detr(detr_image,return_feature=True)
        prob = detr_output[0]['pred_logits'].softmax(dim=-1).detach().cpu().numpy()[:, :, 0]
        top_k_idx = torch.tensor(np.argsort(prob, axis=-1)[:, -args.top_k:]).to(args.device)
        bbox_top_k_features = torch.index_select(detr_output[1][0][0], dim=0, index=top_k_idx[0]).detach()
        bbox_top_k = torch.index_select(detr_output[0]['pred_boxes'][0], dim=0, index=top_k_idx[0]).detach()
        logits_top_k = (torch.index_select(detr_output[0]['pred_logits'][0], dim=0, index=top_k_idx[0]).detach())
        crop_feature = torch.cat([bbox_top_k_features, bbox_top_k, logits_top_k], dim=-1)
        image = np.expand_dims(image, axis=0)
        output = VG(image, crop_feature.unsqueeze(0), instruction_feature.unsqueeze(0))
        predicted = torch.argmax(output, dim=-1)
        bbox_pred = torch.index_select(bbox_top_k, dim=0, index=predicted[0])[0].detach()
        x = math.floor(bbox_pred[0] * 300)
        y = math.floor(bbox_pred[1] * 300)
        w = math.ceil(bbox_pred[2] * 300 + 0.5)
        h = math.ceil(bbox_pred[3] * 300 + 0.5)
        return (x, y, x + w, y + h)

    start_time_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    writer = SummaryWriter(args.eval_path + "/runs")
    f = open(args.eval_path + "/eval_{}_{}_{}.txt".format(args.dataset_mode, start_time_str, args.seen_instruction), "a")
    device = args.device
    args.num_category = 109
    agent = Agent(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    listFiles = sorted(list(os.listdir(args.eval_path)))[:-1]
    list_dict = {}

    for name in listFiles:
        if "checkpoint" not in name:
            continue
        ckpt_idx = eval(name.split("_")[-1].split(".")[0])
        list_dict[ckpt_idx] = name
    list_ckpt = sorted(list(list_dict.keys()))

    envs = Human_Demand_Env(args)
    obs = envs.reset()
    local_image_encoder, _ = clip.load("ViT-B/32")
    local_image_encoder = local_image_encoder.to(args.device)
    transform_global = transforms.Compose([transforms.ToPILImage(), transforms.Resize(args.input_size, interpolation=3), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def get_inputs(obs):
        images = []

        images.append(transform_global(torch.tensor(obs.frame).permute(2, 0, 1)).clone())

        images = torch.stack(images)
        local_feature = {}

        with torch.no_grad():
            lf = torch.stack([local_image_encoder.encode_image(obs.local_feature['feature'].to(device))], dim=0).clone()
        lf = lf / lf.norm(dim=-1, keepdim=True)
        local_feature["features"] = lf.cpu().detach().clone()
        local_feature["bboxes"] = torch.stack([obs.local_feature['bbox']], dim=0).clone()
        local_feature["logits"] = torch.stack([obs.local_feature['logits']], dim=0).clone()
        inputs = {}
        inputs["rgb"] = images.clone()
        # inputs["depth"] = depth_images
        inputs["local_feature"] = deepcopy(local_feature)
        inputs["instruction"] = torch.stack([obs.instruction_features]).clone()
        return inputs

    for model_ckpt in list_ckpt:
        if args.eval_ckpt > 0:
            model_path = args.eval_path + list_dict[args.eval_ckpt]
            ckpt_idx = args.eval_ckpt
        else:
            model_path = args.eval_path + list_dict[model_ckpt]
            ckpt_idx = model_ckpt
        agent = agent.to("cpu")
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))
        agent.load_state_dict(model_dict["model_state_dict"])
        agent = agent.to(args.device)
        navi_success_num = 0
        dis_goal = 0
        spl = 0
        sele_success_num = 0
        for j in range(args.epoch):
            inputs = get_inputs(obs)
            # del obs
            other_inputs = {}
            other_inputs['prev_action'] = (torch.ones((1, 1, 6)).to(args.device)) * np.log(1 / 6)
            other_inputs["prev_hidden_h"] = torch.zeros((agent.lstm_layer_num, 1, args.rnn_hidden_state_dim)).to(args.device)
            other_inputs["prev_hidden_c"] = torch.zeros((agent.lstm_layer_num, 1, args.rnn_hidden_state_dim)).to(args.device)
            for step in tqdm(range(args.max_step), desc=args.dataset_mode + " Epoch: " + str(j)):
                value, action_out, action_log_probs, hx, cx, action_dist = agent.act(inputs, other_inputs)
                action_select = None
                if action_out.squeeze(dim=1).squeeze(dim=1).tolist()[0] == 5:
                    action_select = get_VG_model_answer(obs.frame, obs.instruction_features)
                action_navi = {"action": {"action": {"action": action_out.squeeze(dim=1).squeeze(dim=1).tolist()[0]}}}
                next_obs, reward, done, infos = envs.step(action_navi, action_select)

                obs = deepcopy(next_obs)
                del next_obs
                del inputs
                if done is True:

                    if infos["navigation_success"]:
                        navi_success_num += 1
                        spl += infos["spl"]
                    if infos["select_success"]:
                        sele_success_num += 1
                    dis_goal += obs.min_dis
                    break
                inputs = get_inputs(obs)
                # del obs
                del infos
                del other_inputs
                other_inputs = {}
                other_inputs['prev_action'] = deepcopy(action_dist.logits).to(args.device)

                other_inputs["prev_hidden_h"] = deepcopy(hx).to(args.device)
                other_inputs["prev_hidden_c"] = deepcopy(cx).to(args.device)
                gc.collect()
            obs = envs.reset()
        writer.add_scalar("sele_success_num", sele_success_num / args.epoch, ckpt_idx)
        writer.add_scalar("navi_success_num", navi_success_num / args.epoch, ckpt_idx)
        writer.add_scalar("spl", spl / args.epoch, ckpt_idx)
        writer.add_scalar("dis_goal", dis_goal / args.epoch, ckpt_idx)
        # writer.add_scalar("step", np.mean(step_all), ckpt_idx)
        print("ckpt: ", ckpt_idx, "sele_success_num: ", sele_success_num / args.epoch, "navi_success_num: ", navi_success_num / args.epoch, "spl: ", spl / args.epoch, "dis_goal: ", dis_goal / args.epoch)
        f.write("ckpt: " + str(ckpt_idx) + " sele_success_num: " + str(sele_success_num / args.epoch) + " navi_success_num: " + str(navi_success_num / args.epoch) + " spl: " + str(spl / args.epoch) + " dis_goal: " + str(dis_goal / args.epoch) + "\n")

        if args.eval_ckpt > 0:
            break


if __name__ == '__main__':
    args = parse_arguments()
    print(args.mode)
    start_time_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    args.path2saved_checkpoints = args.path2saved_checkpoints + "/" + args.mode + "/" + start_time_str   
    if args.mode=="test_DDN":
        test_DDN(args)
    if args.mode=="eval_DDN":
        eval_DDN(args)