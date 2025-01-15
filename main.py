import gc
from utils.args import parse_arguments
from vector_env import VectorEnv
from env import Human_Demand_Env, Random_Human_Demand_Env, Object_Env, Image_Env
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
from torch.utils.data.distributed import DistributedSampler
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
import queue
from statistics import mean
import pandas as pd
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from torchvision import transforms
import itertools
import json
from transformers import BertTokenizer, BertModel
import math
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from glob import glob
import logging

BICUBIC = InterpolationMode.BICUBIC


def create_logger(logging_dir, log_name="log.txt", mode='train'):
    """
    Create a logger that writes to a log file and stdout.
    """
    if mode == 'train':
        if dist.get_rank() == 0:  # real logger
            logging.basicConfig(level=logging.INFO, format='[\033[34m%(asctime)s\033[0m] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/{log_name}")])
            logger = logging.getLogger(__name__)
        else:  # dummy logger (does nothing)
            logger = logging.getLogger(__name__)
            logger.addHandler(logging.NullHandler())
    else:
        logging.basicConfig(level=logging.INFO, format='[\033[34m%(asctime)s\033[0m] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/{log_name}")])
        logger = logging.getLogger(__name__)
    return logger


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def save_checkpoint(args, epoch, model, optimizer):
    '''
    Checkpointing. Saves model and optimizer state_dict() and current epoch and global training steps.
    '''
    checkpoint_path = os.path.join(args.path2saved_checkpoints, f'checkpoint_{epoch}.pt')
    save_num = 0
    while (save_num < 10):
        try:

            # if args.n_gpu > 1:
            if False:
                torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
            else:
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)

            break
        except:
            save_num += 1

    return


def train_DDN(args):
    writer = SummaryWriter(args.path2saved_checkpoints + "/runs")
    f = open(args.path2saved_checkpoints + "/log.txt", "w")
    device = args.device
    args.num_category = 109
    agent = Agent(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # model_dict = torch.load("./saved_checkpoints/pre_Ours/2023-04-21_01-32-33/checkpoint_19.pt", map_location=torch.device('cpu'))
    # agent.load_state_dict(model_dict['model_state_dict'])
    agent.optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    # BERT_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    # BERT_model = BertModel.from_pretrained("bert-large-uncased").to(device)
    # agent.optimizer.load_state_dict(model_dict['optimizer_state_dict'])
    # for state in agent.optimizer.state.values():
    #     for k, v in state.items():
    #         if torch.is_tensor(v):
    #             state[k] = v.to(device)
    agent = agent.to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.cuda()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(agent.optimizer, args.pretrain_lr_drop)
    dataset_train = Traj_dataset(args, "train")
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, 1, drop_last=False)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.workers)
    # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train)
    for epoch in tqdm(range(args.epoch)):
        agent.train()
        criterion.train()

        print_freq = 1000
        epoch_time = time.time()
        total = 0
        correct = 0
        total_loss = 0
        feq_loss = 0
        for i, (image, image_PIL, bbox, object_name, instruction_feature, start_position, start_rotation, start_horizon, action, seq_len, crop_feature) in tqdm(enumerate(data_loader_train), total=data_loader_train.__len__()):
            image = torch.cat(image).to(device)

            inputs = {}
            inputs["crop"] = torch.stack(crop_feature).to(device).squeeze(dim=1).float()

            inputs["image"] = image.float()
            inputs["image_PIL"] = image_PIL

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
            total += image.shape[0]
            correct += (predicted == action).sum().item()
            agent.optimizer.zero_grad()
            losses.backward()
            total_loss += losses.item()
            feq_loss += losses.item()
            if args.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_norm)
            agent.optimizer.step()
            if (i + 1) % print_freq == 0:
                print('Epoch: {:4d} cost: {:.2f}  accuracy: {}, loss:{}'.format(epoch, time.time() - epoch_time, correct / total, feq_loss))
                feq_loss = 0
        print("-------------{}----------------".format(epoch))
        print('Epoch: {:4d} cost: {:.2f}  accuracy: {}, loss:{}'.format(epoch, time.time() - epoch_time, correct / total, total_loss))
        print("-------------{}----------------".format(epoch))
        f.write('Epoch: {:4d} cost: {:.2f}  accuracy: {} \n'.format(epoch, time.time() - epoch_time, correct / total, total_loss))
        writer.add_scalar('pretrain_accuracy', correct / total, epoch)
        writer.add_scalar('pretrain_loss', total_loss, epoch)
        save_checkpoint(args, epoch, agent, agent.optimizer)


def train_DDN_Split(args):
    writer = SummaryWriter(args.path2saved_checkpoints + "/runs")
    f = open(args.path2saved_checkpoints + "/log.txt", "w")
    device = args.device
    args.num_category = 109
    agent = Agent(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    agent.optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    agent = agent.to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.cuda()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(agent.optimizer, args.pretrain_lr_drop)
    dataset_train = Traj_dataset(args, "train")
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, 1, drop_last=False)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.workers)
    # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train)
    for epoch in tqdm(range(args.epoch)):
        agent.train()
        criterion.train()

        print_freq = 1000
        epoch_time = time.time()
        total = 0
        correct = 0
        total_loss = 0
        feq_loss = 0
        for i, (image, image_PIL, bbox, object_name, instruction_feature, start_position, start_rotation, start_horizon, action, seq_len, crop_feature) in tqdm(enumerate(data_loader_train), total=data_loader_train.__len__()):
            patch_size = args.patch_size
            patch_num = math.ceil(seq_len / patch_size)
            other_inputs = {}
            other_inputs['prev_action'] = (torch.ones((1, 1, 6)).to(args.device)) * np.log(1 / 6)
            other_inputs['prev_hidden_h'] = torch.zeros((2, 1, 1024)).to(args.device)
            other_inputs['prev_hidden_c'] = torch.zeros((2, 1, 1024)).to(args.device)
            for patch_idx in range(patch_num):
                image = torch.cat(image).to(device)

                inputs = {}
                inputs["crop"] = torch.stack(crop_feature).to(device).squeeze(dim=1).float()
                inputs["crop"] = inputs["crop"][patch_idx * patch_size:min(seq_len, (patch_idx + 1) * patch_size)]

                inputs["image"] = image.float()
                inputs["image"] = inputs["image"][patch_idx * patch_size:min(seq_len, (patch_idx + 1) * patch_size)]
                inputs["image_PIL"] = image_PIL
                inputs["image_PIL"] = inputs["image_PIL"][patch_idx * patch_size:min(seq_len, (patch_idx + 1) * patch_size)]

                instruction_feature = torch.stack(instruction_feature).to(device)
                inputs["instruction_feature"] = instruction_feature.permute(1, 0).repeat(seq_len, 1)
                inputs["instruction_feature"] = inputs["instruction_feature"][patch_idx * patch_size:min(seq_len, (patch_idx + 1) * patch_size)]
                outputs, hx, cx = agent.forward_pre_split(inputs, other_inputs).squeeze(dim=1).squeeze(dim=1)
                action_patch = torch.stack(action).squeeze(dim=1).to(device)
                action_patch = action_patch[patch_idx * patch_size:min(seq_len, (patch_idx + 1) * patch_size)]
                losses = criterion(outputs, action_patch)
                _, predicted = torch.max(outputs, dim=1)
                total += image.shape[0]
                correct += (predicted == action_patch).sum().item()
                agent.optimizer.zero_grad()
                losses.backward()
                total_loss += losses.item()
                feq_loss += losses.item()
                other_inputs["prev_action"] = outputs[-1]
                other_inputs["prev_hidden_h"] = hx
                other_inputs["prev_hidden_c"] = cx
                if args.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_norm)
                agent.optimizer.step()
            if (i + 1) % print_freq == 0:
                print('Epoch: {:4d} cost: {:.2f}  accuracy: {}, loss:{}'.format(epoch, time.time() - epoch_time, correct / total, feq_loss))
                feq_loss = 0
        print("-------------{}----------------".format(epoch))
        print('Epoch: {:4d} cost: {:.2f}  accuracy: {}, loss:{}'.format(epoch, time.time() - epoch_time, correct / total, total_loss))
        print("-------------{}----------------".format(epoch))
        f.write('Epoch: {:4d} cost: {:.2f}  accuracy: {} \n'.format(epoch, time.time() - epoch_time, correct / total, total_loss))
        writer.add_scalar('pretrain_accuracy', correct / total, epoch)
        writer.add_scalar('pretrain_loss', total_loss, epoch)
        save_checkpoint(args, epoch, agent, agent.optimizer)


def train_DDN_Multi_GPU(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    if rank == 0:
        os.makedirs(args.path2saved_checkpoints, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.path2saved_checkpoints}/*"))
        model_string_name = "DDN"
        experiment_dir = f"{args.path2saved_checkpoints}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        # save the args to the experiment folder as json
        with open(f"{experiment_dir}/args.json", "w") as f:
            json.dump(vars(args), f, indent=4)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
    args.num_category = 109
    agent = Agent(args)
    agent = DDP(agent.to(device), device_ids=[rank])
    np.random.seed(args.seed)
    random.seed(args.seed)

    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    agent = agent.to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.cuda()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.pretrain_lr_drop)
    dataset_train = Traj_dataset(args, "train")
    sampler = DistributedSampler(dataset_train, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed)
    data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=False, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    agent.train()
    # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train)
    for epoch in tqdm(range(args.epoch)):
        sampler.set_epoch(epoch)
        criterion.train()
        logger.info(f"Beginning epoch {epoch}...")
        print_freq = 1000
        epoch_time = time.time()
        total = 0
        correct = 0
        total_loss = 0
        feq_loss = 0
        for i, (image, image_PIL, bbox, object_name, instruction_feature, start_position, start_rotation, start_horizon, action, seq_len, crop_feature) in tqdm(enumerate(data_loader_train), total=data_loader_train.__len__()):
            image = torch.cat(image).to(device)

            inputs = {}
            inputs["crop"] = torch.stack(crop_feature).to(device).squeeze(dim=1).float()

            inputs["image"] = image.float()
            inputs["image_PIL"] = image_PIL

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
            total += image.shape[0]
            correct += (predicted == action).sum().item()
            optimizer.zero_grad()
            losses.backward()
            total_loss += losses.item()
            feq_loss += losses.item()
            if args.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_norm)
            optimizer.step()
            if (i + 1) % print_freq == 0:
                logger.info('Epoch: {:4d} cost: {:.2f}  accuracy: {}, loss:{}'.format(epoch, time.time() - epoch_time, correct / total, feq_loss))
                feq_loss = 0

        logger.info('Epoch: {:4d} cost: {:.2f}  accuracy: {}, loss:{}'.format(epoch, time.time() - epoch_time, correct / total, total_loss))
        if rank == 0:
            save_checkpoint(args, epoch, agent, agent.optimizer)
        dist.barrier()
    cleanup()


def main(args):
    start_time_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    args.path2saved_checkpoints = args.path2saved_checkpoints + "/" + args.mode + "/" + start_time_str
    args.time = start_time_str
    if not os.path.exists(args.path2saved_checkpoints):
        os.makedirs(args.path2saved_checkpoints)
    print('\n Training started from: {}'.format(time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time()))))
    if args.mode == "train_DDN":
        train_DDN(args)
    if args.mode == "train_DDN_Split":
        train_DDN_Split(args)
    if args.mode == "train_DDN_Multi_GPU":
        train_DDN_Multi_GPU(args)


if __name__ == "__main__":

    args = parse_arguments()
    main(args)
