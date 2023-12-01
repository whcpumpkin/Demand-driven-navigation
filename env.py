
import gc
from copy import deepcopy
import math
import clip
from utils.util import get_valid_transforms, _convert_image_to_rgb
from model.detr import DETRModel
# from thortils.navigation import (
#     get_shortest_path_to_object, )
from utils.util import make_env_fn
from vector_env import VectorEnv
from PIL import Image
import os
from utils.args import parse_arguments
import numpy as np
from ai2thor.util.metrics import (get_shortest_path_to_object)
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import gzip
import prior
import random
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import copy
import argparse
import torch
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
import albumentations as A
import cv2
import time

import warnings

warnings.filterwarnings('ignore')

BICUBIC = InterpolationMode.BICUBIC
try:
    from prior import LazyJsonDataset
except:
    raise ImportError("Please update the prior package (pip install --upgrade prior).")


def compute_single_spl(path, shortest_path, successful_path):
    """
    Computes SPL for a path dict(x=float, y=float, z=float)
    :param path: Sequence of dict(x=float, y=float, z=float) representing the path to evaluate
    :param shortest_path: Sequence of dict(x=float, y=float, z=float) representing the shortest oath
    :param successful_path: boolean indicating if the path was successful, 0 for a failed path or 1 for a successful one
    :return:
    """
    Si = 1 if successful_path == True or successful_path == 1 else 0
    li = path_distance(shortest_path)
    pi = path_distance(path)
    if max(pi, li) > 0:
        pl_ratio = li / max(pi, li)
    else:
        pl_ratio = 1.0
    spl = Si * pl_ratio
    return spl


def vector_distance(v0, v1):
    dx = v0["x"] - v1["x"]
    dz = v0["z"] - v1["z"]
    return math.sqrt(dx * dx + dz * dz)


def path_distance(path):
    distance = 0
    for i in range(0, len(path) - 1):
        distance += vector_distance(path[i], path[i + 1])
    return distance


class Object_Env():

    def __init__(self, args=None):
        self.args = args
        self.dataset = self.load_dataset()[args.dataset_mode]
        self.current_step = self.args.max_step
        self.reset_time = 0
        self.observation_space = (300, 300, 4)
        self.action_space = 6
        self.number_of_episodes = args.epoch
        self.controller = Controller(scene=self.dataset[0], renderDepthImage=True, gridSize=0.05, snapToGrid=True, visibilityDistance=1.5, width=300, height=300)
        with open(args.path2dataset + "obj2idx.json", 'r', encoding='utf8') as fp:
            self.obj2idx = json.load(fp)
        self.current_house = 0
        if "Ours" in self.args.mode:
            self.detr = DETRModel(2, 100)
            self.detr.load_state_dict(torch.load("./pretrained_model/detr_new_18.pth", map_location="cpu"))
            self.detr = self.detr.to(args.device).eval()
            self.transform_detr = get_valid_transforms(h=300, w=300)

    def load_dataset(self) -> prior.DatasetDict:
        """Load the houses dataset."""
        print("[AI2-THOR WARNING] There has been an update to ProcTHOR-10K that must be used with AI2-THOR version 5.0+. To use the new version of ProcTHOR-10K, please update AI2-THOR to version 5.0+ by running:\n"
              "    pip install --upgrade ai2thor\n"
              "Alternatively, to downgrade to the old version of ProcTHOR-10K, run:\n"
              '   prior.load_dataset("procthor-10k", revision="ab3cacd0fc17754d4c080a3fd50b18395fae8647")')
        data = {}
        dic = {"train": 10000, "val": 1000, "test": 1000}
        for split, size in [("train", 10_000), ("val", 1_000), ("test", 1_000)]:
            with gzip.open(f"dataset/{split}.jsonl.gz", "r") as f:
                houses = [line for line in tqdm(f, total=size, desc=f"Loading {split}")]
            data[split] = LazyJsonDataset(data=houses, dataset="procthor-dataset", split=split)
        return prior.DatasetDict(**data)

    def reset(self):

        self.done = False
        self.navigation_success = False
        self.reset_time += 1
        if self.current_step == self.args.max_step or "test" in self.args.mode:
            self.house_idx = random.randint(0, 199)
            self.house = self.dataset[int(self.house_idx)]
            self.controller.reset(scene=self.house, renderDepthImage=True, renderInstanceSegmentation=False, snapToGrid=True, visibilityDistance=1.5, gridSize=0.05, width=300, height=300)
            self.current_house = (self.current_house + 1) % 5
            self.all_position = self.get_reachable_position()
            self.current_step = 0
            self.reset_time = 0
            self.current_path = []

        event = self.controller.last_event
        obj_list = set()
        for obj in event.metadata["objects"]:
            if obj['name'].split("|")[0] in list(self.obj2idx.keys()):
                obj_list.add(obj['name'].split("|")[0])
        self.target_object_name=random.choice(list(obj_list))
        self.target_object = self.obj2idx[self.target_object_name]
        self.target_objects_metadata = []
        for obj in event.metadata["objects"]:
            if obj['name'].split("|")[0] in list(self.obj2idx.keys()):
                if self.obj2idx[obj['name'].split("|")[0]] == self.target_object:
                    self.target_objects_metadata.append(obj)
                    self.target_object_position = [obj["position"]['x'], obj["position"]['z']]
                    event.position = [obj["position"]['x'], obj["position"]['z']]

        self.target_object_onehot = ([0] * 109)
        self.target_object_onehot[self.target_object] = 1

        if len(self.all_position) > 0 and (self.reset_time % 5 == 0 or "test" in self.args.mode):
            position = random.choice(self.all_position)
            rotation = {'x': 0, 'y': random.choice([0, 90, 180, 270]), 'z': 0}
            horizon = random.choice([0, 30, -30])
            event = self.teleport(position, rotation, horizon)
        # self.target_object_path = []
        self.current_path = []
        self.init_agent_pose = {}
        self.init_agent_pose['position'] = event.metadata['agent']['position']
        self.init_agent_pose['rotation'] = event.metadata['agent']['rotation']
        if "Ours" in self.args.mode:
            with torch.no_grad():
                detr_image = self.transform_detr(image=event.frame.copy() / 255.0)["image"].to(torch.float32).to(self.args.device)
                detr_output, hs = self.detr([detr_image])
            prob = detr_output['pred_logits'].softmax(dim=-1).detach().cpu().numpy()[:, :, 0]
            top_k_idx = torch.tensor(np.argsort(prob, axis=-1)[:, -self.args.top_k:].copy()).to(self.args.device)
            bbox_top_k = torch.index_select(detr_output['pred_boxes'][0], dim=0, index=top_k_idx[0])
            logits_top_k = torch.index_select(detr_output['pred_logits'][0], dim=0, index=top_k_idx[0])
            local_feature = self.crop_from_bounding_box(event.frame, detr_output['pred_boxes'][0], top_k_idx[0])
            del detr_output
            del hs
            del detr_image
            event.local_feature = {}
            event.local_feature["feature"] = local_feature.cpu().detach().clone()
            event.local_feature["bbox"] = bbox_top_k.cpu().detach().clone()
            event.local_feature["logits"] = logits_top_k.cpu().detach().clone()
        # for obj in self.target_objects_metadata:
        #     path,_=get_shortest_path_to_object(
        #         self.controller, obj['objectId'], self.controller.last_event.metadata['agent']['position']), self.controller.last_event.metadata['agent']['rotation']
        #     self.target_object_path.append(path)
        self.current_path.append(self.controller.last_event.metadata['agent']['position'])
        self.min_distance_demand_objs = self.get_shortest_distance_to_target(event)
        event.min_dis = self.min_distance_demand_objs
        event.target_object = self.target_object
        event.target_object_onehot = self.target_object_onehot
        event.agent_position = [event.metadata["agent"]["position"]['x'], event.metadata["agent"]["position"]['z']]
        event.target_object_position = self.target_object_position
        event.target_object_name=self.target_object_name
        self.init_event = deepcopy(event)
        return event

    def step(self, action):
        # if self.done:
        #     event = self.controller.last_event
        #     reward = 0
        #     info = {}
        #     info["navigation_success"] = self.navigation_success
        #     event.min_dis = self.min_distance_demand_objs
        #     return event, reward, self.done, info
        self.current_step += 1
        # action: RotateRight RotateLeft LookUp LookDown MoveAhead Done
        s_time = time.time()
        action_navi = self.trans_action(action['action']['action']['action'])
        if action_navi == "Done":
            event = self.controller.step(action=action_navi, renderInstanceSegmentation=True)
            # event = self.controller.step(action=action_navi)
        else:
            if "Ahead" in action_navi:
                event = self.controller.step(action=action_navi, moveMagnitude=0.25, renderInstanceSegmentation=False)
                # event = self.controller.step(action=action_navi, moveMagnitude=0.25)
            else:
                event = self.controller.step(action=action_navi, degrees=30, renderInstanceSegmentation=False)
        self.current_path.append(self.controller.last_event.metadata['agent']['position'])
        # event = self.controller.step(action=action_navi, degrees=30)
        # print("controller step: ", time.time()-s_time)
        # event = self.controller.step(action=self.trans_action(action_navi["action"]["action"]["action"]))
        # print("step_time:", time.time()-s_time)
        # obj = self.controller.last_event.metadata['objects']
        reward = self.reward_function(event)
        # reward -= 0.001
        info = {}
        info["navigation_success"] = False
        if action_navi == "Done" or self.args.max_step < self.current_step:
            self.done = True
            info["navigation_success"], info['spl'] = self.success_check(event)
            if info["navigation_success"]:
                self.navigation_success = True
                reward = self.args.navigation_success_reward

                # print("navigation success!!")
            # if info["select_success"]:
            #     reward = self.args.select_success_reward
        # info["instruction"] = self.instruction
        # info["instruction_features"] = self.instruction_features[self.instruction]
        if "Ours" in self.args.mode:
            with torch.no_grad():
                detr_image = self.transform_detr(image=event.frame.copy() / 255.0)["image"].to(torch.float32).to(self.args.device)
                detr_output, hs = self.detr([detr_image])
            prob = detr_output['pred_logits'].softmax(dim=-1).detach().cpu().numpy()[:, :, 0]
            top_k_idx = torch.tensor(np.argsort(prob, axis=-1)[:, -self.args.top_k:].copy()).to(self.args.device)
            bbox_top_k = torch.index_select(detr_output['pred_boxes'][0], dim=0, index=top_k_idx[0])
            logits_top_k = torch.index_select(detr_output['pred_logits'][0], dim=0, index=top_k_idx[0])
            local_feature = self.crop_from_bounding_box(event.frame, detr_output['pred_boxes'][0], top_k_idx[0])
            del detr_output
            del hs
            del detr_image
            event.local_feature = {}
            event.local_feature["feature"] = local_feature.cpu().detach().clone()
            event.local_feature["bbox"] = bbox_top_k.cpu().detach().clone()
            event.local_feature["logits"] = logits_top_k.cpu().detach().clone()
        event.target_object = self.target_object
        event.target_object_onehot = self.target_object_onehot
        event.min_dis = self.min_distance_demand_objs
        event.target_object_position = self.target_object_position
        event.agent_position = [event.metadata["agent"]["position"]['x'], event.metadata["agent"]["position"]['z']]
        event.target_object_name=self.target_object_name
        # print("step time:{} action: {}".format(time.time()-s_time, action_navi))
        return event, reward, self.done, info

    def crop_from_bounding_box(self, image, bbox, top_k_idx):
        crop_start_time = time.time()
        local_image_encoder_preprocess = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        crop_cur = []
        img = torch.tensor(image.copy()).permute(2, 0, 1).to(torch.float32).to(self.args.device)
        for idx_top in range(self.args.top_k):

            bbox_idx = top_k_idx[idx_top]
            x = math.floor(bbox[bbox_idx, 0] * 300)
            y = math.floor(bbox[bbox_idx, 1] * 300)
            w = math.ceil(bbox[bbox_idx, 2] * 300 + 0.5)
            h = math.ceil(bbox[bbox_idx, 3] * 300 + 0.5)
            crop_1 = torchvision.transforms.functional.crop(img, y, x, h, w)
            crop_1_PIL = torchvision.transforms.ToPILImage()(crop_1)
            crop_cur.append(deepcopy(local_image_encoder_preprocess(crop_1_PIL)))

        crop_cur = torch.stack(crop_cur, dim=0)
        # print("crop time: ", time.time()-crop_start_time)
        return crop_cur
    
    def success_check(self, event):
        if self.args.max_step < self.current_step:
            return False
        navigation_success = False
        spl = 0
        for obj in self.target_objects_metadata:
            agent_position = event.metadata["agent"]["position"]
            agent_pos = event.metadata["agent"]["position"]
            obj_pos = obj["position"]
            dis = np.sqrt((agent_pos['x'] - obj_pos['x'])**2 + (agent_pos['z'] - obj_pos['z'])**2)
            bbox = self.get_bounding_box(event, obj["objectId"])
            # shortest_path = get_shortest_path_to_object(
            #         self.controller, obj['objectId'], self.init_agent_pose['position'], self.init_agent_pose['rotation'])
            if event.instance_detections2D == None:
                print("warning! detection is none")
            if dis <= 1.5 and bbox is not None:
                navigation_success = True
                if "test" in self.args.mode:

                    shortest_path = get_shortest_path_to_object(self.controller, obj['objectId'], self.init_agent_pose['position'], self.init_agent_pose['rotation'])
                    if shortest_path is not None:
                        spl = compute_single_spl(shortest_path, self.current_path, True)
                    else:
                        spl = 1
                break
        # for candidate_demand_object in candidate_demand_objects:
        #     IoU = self.IoU(self.get_bounding_box(event, candidate_demand_object["name"]), action_select)

        #     if IoU > 0.5:
        #         select_success = True
        #         break
        return navigation_success, spl

    def get_bounding_box(self, event, objectId):
        if event.instance_detections2D == None:
            return None
        return event.instance_detections2D.get(objectId)

    def trans_action(self, action):
        action_idx = {0: 'MoveAhead', 1: 'RotateLeft', 2: 'RotateRight', 3: 'LookUp', 4: 'LookDown', 5: 'Done'}
        return action_idx[int(action)]

    def reward_function(self, event=None):
        min_dis = self.get_shortest_distance_to_target(event)
        reward = max(self.min_distance_demand_objs - min_dis, 0)
        self.min_distance_demand_objs = min_dis
        return reward

    def get_shortest_distance_to_target(self, event):
        min_dis = 99999999
        for obj in self.target_objects_metadata:
            agent_pos = event.metadata["agent"]["position"]
            obj_pos = obj["position"]
            dis = np.sqrt((agent_pos['x'] - obj_pos['x'])**2 + (agent_pos['z'] - obj_pos['z'])**2)
            min_dis = min(min_dis, dis)
        return min_dis

    def get_reachable_position(self):
        event = self.controller.step(action="GetReachablePositions")
        return event.metadata["actionReturn"]

    def teleport(self, position, rotation, horizon):
        event = self.controller.step(action="Teleport", position=position, rotation=rotation, horizon=horizon)
        return event

    def seed(self, rank):
        random.seed(rank)
        np.random.seed(rank)

    def close(self):
        pass


class Image_Env():

    def __init__(self, args=None):
        self.args = args
        self.dataset = self.load_dataset()[args.dataset_mode]
        self.current_step = self.args.max_step
        self.reset_time = 0
        self.observation_space = (300, 300, 4)
        self.action_space = 6
        self.number_of_episodes = args.epoch
        self.controller = Controller(scene=self.dataset[0], renderDepthImage=True, gridSize=0.05, snapToGrid=True, visibilityDistance=1.5, width=300, height=300)
        # with open(args.path2dataset + "obj2idx.json", 'r', encoding='utf8') as fp:
        #     self.obj2idx = json.load(fp)
        self.current_house = 0

    def load_dataset(self) -> prior.DatasetDict:
        """Load the houses dataset."""
        print("[AI2-THOR WARNING] There has been an update to ProcTHOR-10K that must be used with AI2-THOR version 5.0+. To use the new version of ProcTHOR-10K, please update AI2-THOR to version 5.0+ by running:\n"
              "    pip install --upgrade ai2thor\n"
              "Alternatively, to downgrade to the old version of ProcTHOR-10K, run:\n"
              '   prior.load_dataset("procthor-10k", revision="ab3cacd0fc17754d4c080a3fd50b18395fae8647")')
        data = {}
        dic = {"train": 10000, "val": 1000, "test": 1000}
        for split, size in [("train", 10_000), ("val", 1_000), ("test", 1_000)]:
            with gzip.open(f"dataset/{split}.jsonl.gz", "r") as f:
                houses = [line for line in tqdm(f, total=size, desc=f"Loading {split}")]
            data[split] = LazyJsonDataset(data=houses, dataset="procthor-dataset", split=split)
        return prior.DatasetDict(**data)

    def reset(self):

        self.done = False
        self.navigation_success = False
        self.reset_time += 1
        if self.current_step == self.args.max_step or "test" in self.args.mode:
            self.house_idx = random.randint(0, 199)
            self.house = self.dataset[int(self.house_idx)]
            self.controller.reset(scene=self.house, renderDepthImage=True, renderInstanceSegmentation=False, snapToGrid=True, visibilityDistance=1.5, gridSize=0.05, width=300, height=300)
            self.current_house = (self.current_house + 1) % 5
            self.all_position = self.get_reachable_position()
            self.current_step = 0
            self.reset_time = 0

        event = self.controller.last_event
        self.goal_agent_pose = {}
        self.goal_agent_pose['position'] = event.metadata['agent']['position']
        self.goal_agent_pose['rotation'] = event.metadata['agent']['rotation']
        self.goal = event.frame

        position = random.choice(self.all_position)
        rotation = {'x': 0, 'y': random.choice([0, 90, 180, 270]), 'z': 0}
        horizon = random.choice([0, 30, -30])
        event = self.teleport(position, rotation, horizon)
        # self.target_object_path = []
        self.current_path = []
        self.init_agent_pose = {}
        self.init_agent_pose['position'] = event.metadata['agent']['position']
        self.init_agent_pose['rotation'] = event.metadata['agent']['rotation']
        # for obj in self.target_objects_metadata:
        #     path,_=get_shortest_path_to_object(
        #         self.controller, obj['objectId'], self.controller.last_event.metadata['agent']['position']), self.controller.last_event.metadata['agent']['rotation']
        #     self.target_object_path.append(path)
        self.current_path.append(self.controller.last_event.metadata['agent']['position'])
        self.min_distance = np.sqrt((self.goal_agent_pose['position']['x'] - self.init_agent_pose['position']['x'])**2 +
                                    (self.goal_agent_pose['position']['z'] - self.init_agent_pose['position']['z'])**2)  # euclidean distance
        event.goal = self.goal
        event.min_dis = self.min_distance
        event.agent_position = [event.metadata["agent"]["position"]['x'], event.metadata["agent"]["position"]['z']]
        event.target_position = self.goal_agent_pose
        return event

    def step(self, action):
        # if self.done:
        #     event = self.controller.last_event
        #     reward = 0
        #     info = {}
        #     info["navigation_success"] = self.navigation_success
        #     event.min_dis = self.min_distance_demand_objs
        #     return event, reward, self.done, info
        self.current_step += 1
        # action: RotateRight RotateLeft LookUp LookDown MoveAhead Done
        s_time = time.time()
        action_navi = self.trans_action(action['action']['action']['action'])
        if action_navi == "Done":
            event = self.controller.step(action=action_navi, renderInstanceSegmentation=True)
            # event = self.controller.step(action=action_navi)
        else:
            if "Ahead" in action_navi:
                event = self.controller.step(action=action_navi, moveMagnitude=0.25, renderInstanceSegmentation=False)
                # event = self.controller.step(action=action_navi, moveMagnitude=0.25)
            else:
                event = self.controller.step(action=action_navi, degrees=30, renderInstanceSegmentation=False)
        self.current_path.append(self.controller.last_event.metadata['agent']['position'])
        # event = self.controller.step(action=action_navi, degrees=30)
        # print("controller step: ", time.time()-s_time)
        # event = self.controller.step(action=self.trans_action(action_navi["action"]["action"]["action"]))
        # print("step_time:", time.time()-s_time)
        # obj = self.controller.last_event.metadata['objects']
        reward = self.reward_function(event)
        # reward -= 0.001
        info = {}
        info["navigation_success"] = False
        if action_navi == "Done" or self.args.max_step < self.current_step:
            self.done = True
            info["navigation_success"], info['spl'] = self.success_check(event)
            if info["navigation_success"]:
                self.navigation_success = True
                reward = self.args.navigation_success_reward

                # print("navigation success!!")
            # if info["select_success"]:
            #     reward = self.args.select_success_reward
        # info["instruction"] = self.instruction
        # info["instruction_features"] = self.instruction_features[self.instruction]
        event.goal = self.goal
        event.min_dis = self.min_distance
        event.target_position = self.goal_agent_pose
        event.agent_position = [event.metadata["agent"]["position"]['x'], event.metadata["agent"]["position"]['z']]
        # print("step time:{} action: {}".format(time.time()-s_time, action_navi))
        return event, reward, self.done, info

    def success_check(self, event):
        if self.args.max_step < self.current_step:
            return False
        navigation_success = False
        spl = 0

        agent_position = event.metadata["agent"]["position"]
        dis = np.sqrt((self.goal_agent_pose['position']['x'] - agent_position['x'])**2 + (self.goal_agent_pose['position']['z'] - agent_position['z'])**2)
        delta_rotation = abs(self.goal_agent_pose['rotation']['y'] - event.metadata['agent']['rotation']['y'])
        if dis <= 1.5 and delta_rotation < 31:
            navigation_success = True
            if "test" in self.args.mode:
                shortest_path = get_shortest_path_to_object(self.controller, obj['objectId'], self.init_agent_pose['position'], self.init_agent_pose['rotation'])
                spl = compute_single_spl(shortest_path, self.current_path, True)

        # for candidate_demand_object in candidate_demand_objects:
        #     IoU = self.IoU(self.get_bounding_box(event, candidate_demand_object["name"]), action_select)

        #     if IoU > 0.5:
        #         select_success = True
        #         break
        return navigation_success, spl

    def get_bounding_box(self, event, objectId):
        if event.instance_detections2D == None:
            return None
        return event.instance_detections2D.get(objectId)

    def trans_action(self, action):
        action_idx = {0: 'MoveAhead', 1: 'RotateLeft', 2: 'RotateRight', 3: 'LookUp', 4: 'LookDown', 5: 'Done'}
        return action_idx[int(action)]

    def reward_function(self, event=None):
        agent_position = event.metadata["agent"]["position"]
        min_dis = np.sqrt((self.goal_agent_pose['position']['x'] - agent_position['x'])**2 + (self.goal_agent_pose['position']['z'] - agent_position['z'])**2)
        reward = max(self.min_distance - min_dis, 0)
        self.min_distance = min_dis
        return reward

    def get_shortest_distance_to_target(self, event):
        min_dis = 99999999
        for obj in self.target_objects_metadata:
            agent_pos = event.metadata["agent"]["position"]
            obj_pos = obj["position"]
            dis = np.sqrt((agent_pos['x'] - obj_pos['x'])**2 + (agent_pos['z'] - obj_pos['z'])**2)
            min_dis = min(min_dis, dis)
        return min_dis

    def get_reachable_position(self):
        event = self.controller.step(action="GetReachablePositions")
        return event.metadata["actionReturn"]

    def teleport(self, position, rotation, horizon):
        event = self.controller.step(action="Teleport", position=position, rotation=rotation, horizon=horizon)
        return event

    def seed(self, rank):
        random.seed(rank)
        np.random.seed(rank)

    def close(self):
        pass


class Random_Human_Demand_Env():

    def __init__(self, args=None):
        self.args = args
        self.dataset = self.load_dataset()[args.dataset_mode]
        self.demand_answer = self.load_answer()
        self.demand_instruction = self.load_instruction()
        self.obj_house_idx, self.house_idx_obj = self.load_obj_house_idx()
        self.instruction_features = self.load_instuction_features()
        self.current_step = args.max_step
        self.reset_time = 5
        self.observation_space = (300, 300, 4)
        self.action_space = 6
        self.number_of_episodes = args.epoch
        self.controller = Controller(scene=self.dataset[0], renderDepthImage=False, gridSize=0.05, snapToGrid=True, visibilityDistance=1.5, width=300, height=300)

    def load_instuction_features(self):
        with open(self.args.path2dataset + "/" + "instruction_bert_features" + ".json", 'r', encoding='utf8') as fp:
            instuction_features = json.load(fp)
        return instuction_features

    def load_answer(self):
        with open(self.args.path2answer, 'r', encoding='utf8') as fp:
            answer = json.load(fp)
        return answer

    def load_obj_house_idx(self):
        with open(self.args.path2dataset + "/" + "obj_house_idx_{}".format(self.args.dataset_mode) + ".json", 'r', encoding='utf8') as fp:
            obj_house_idx = json.load(fp)
        with open(self.args.path2dataset + "/" + "house_idx_obj_{}".format(self.args.dataset_mode) + ".json", 'r', encoding='utf8') as fp:
            house_idx_obj = json.load(fp)
        return obj_house_idx, house_idx_obj

    def load_instruction(self):
        with open(self.args.path2dataset + "/" + "instruction_{}_check.json".format(self.args.dataset_mode), 'r', encoding='utf8') as fp:
            instruction = json.load(fp)
        return instruction

    def load_dataset(self) -> prior.DatasetDict:
        """Load the houses dataset."""
        print("[AI2-THOR WARNING] There has been an update to ProcTHOR-10K that must be used with AI2-THOR version 5.0+. To use the new version of ProcTHOR-10K, please update AI2-THOR to version 5.0+ by running:\n"
              "    pip install --upgrade ai2thor\n"
              "Alternatively, to downgrade to the old version of ProcTHOR-10K, run:\n"
              '   prior.load_dataset("procthor-10k", revision="ab3cacd0fc17754d4c080a3fd50b18395fae8647")')
        data = {}
        dic = {"train": 10000, "val": 1000, "test": 1000}
        for split, size in [("train", 10_000), ("val", 1_000), ("test", 1_000)]:
            with gzip.open(f"dataset/{split}.jsonl.gz", "r") as f:
                houses = [line for line in tqdm(f, total=size, desc=f"Loading {split}")]
            data[split] = LazyJsonDataset(data=houses, dataset="procthor-dataset", split=split)
        return prior.DatasetDict(**data)

    def reset(self):
        # house_idx = random.choice(list(self.house_idx_obj.keys()))
        # self.house = self.dataset[int(house_idx)]
        # obj_name = random.choice(self.house_idx_obj[house_idx])
        # self.instruction = random.choice(self.demand_answer[obj_name])
        # self.demand_objs_name = []
        # for obj in self.demand_instruction[self.instruction]:
        #     if int(house_idx) in self.obj_house_idx[obj]:
        #         self.demand_objs_name.append(obj)
        s_time = time.time()
        self.current_step = 0
        while True:
            while True:
                self.instruction = random.choice(list(self.demand_instruction.keys()))
                obj_name = random.choice(self.demand_instruction[self.instruction])
                if len(self.obj_house_idx[obj_name]) > 0:
                    self.house_idx = random.choice(self.obj_house_idx[obj_name])
                    break
            self.demand_objs_name = []
            self.house = self.dataset[int(self.house_idx)]
            for obj in self.demand_instruction[self.instruction]:
                if int(self.house_idx) in self.obj_house_idx[obj]:
                    self.demand_objs_name.append(obj)

            self.controller.reset(scene=self.house, renderDepthImage=False, renderInstanceSegmentation=False, snapToGrid=True, visibilityDistance=1.5, gridSize=0.05, width=300, height=300)
            event = self.controller.last_event
            self.demand_objs = []
            for name in self.demand_objs_name:
                for obj in event.metadata['objects']:
                    if name in obj["name"]:
                        self.demand_objs.append(obj)

            all_position = self.get_reachable_position()
            position = random.choice(all_position)

            rotation = {'x': 0, 'y': random.choice([0, 90, 180, 270]), 'z': 0}
            horizon = random.choice([0, 30, -30])
            event = self.teleport(position, rotation, horizon)

            # self.min_distance_demand_objs = self.get_min_distance_demand_objs(event)
            # if self.min_distance_demand_objs < 999999999:

            #     break
            # print("Warning: fail to reset!")
            break

        return event

    def step(self, action_navi, action_select=None):
        self.current_step += 1
        # action: RotateRight RotateLeft LookUp LookDown MoveAhead Done
        s_time = time.time()
        action_navi = self.trans_action(action_navi["action"]["action"]["action"])
        if action_navi == "Done":
            event = self.controller.step(action=action_navi, renderInstanceSegmentation=True)
        else:
            if "Ahead" in action_navi:
                event = self.controller.step(action=action_navi, moveMagnitude=0.25, renderInstanceSegmentation=False)
            else:
                event = self.controller.step(action=action_navi, degrees=30, renderInstanceSegmentation=False)

        # event = self.controller.step(action=self.trans_action(action_navi["action"]["action"]["action"]))
        # print("step_time:", time.time()-s_time)
        # obj = self.controller.last_event.metadata['objects']
        # reward = self.reward_function(event)
        reward = -0.001
        info = {}
        done = False
        if action_navi == "Done" or self.args.max_step < self.current_step:
            done = True
            info["navigation_success"], info["select_success"] = self.success_check(event, action_select)
            if info["navigation_success"]:
                reward = self.args.navigation_success_reward
                # print("navigation success!!")
            if info["select_success"]:
                reward = self.args.select_success_reward
        info["instruction"] = self.instruction
        info["instruction_features"] = self.instruction_features[self.instruction]

        return event, reward, done, info

    def success_check(self, event, action_select):
        if self.args.max_step <= self.current_step:
            return False, False
        navigation_success = False
        select_success = False
        candidate_demand_objects = []
        for demand_object in self.demand_objs:
            agent_position = event.metadata["agent"]["position"]
            agent_pos = event.metadata["agent"]["position"]
            obj_pos = demand_object["position"]
            dis = np.sqrt((agent_pos['x'] - obj_pos['x'])**2 + (agent_pos['z'] - obj_pos['z'])**2)
            bbox = self.get_bounding_box(event, demand_object["objectId"])
            if event.instance_detections2D == None:
                print("warning! detection is none")
            if dis <= 1.5 and bbox is not None:
                navigation_success = True
                candidate_demand_objects.append(demand_object)
        # for candidate_demand_object in candidate_demand_objects:
        #     IoU = self.IoU(self.get_bounding_box(event, candidate_demand_object["name"]), action_select)

        #     if IoU > 0.5:
        #         select_success = True
        #         break
        return navigation_success, select_success

    def get_bounding_box(self, event, objectId):
        if event.instance_detections2D == None:
            return None
        return event.instance_detections2D.get(objectId)

    def trans_action(self, action):
        action_idx = {0: 'MoveAhead', 1: 'RotateLeft', 2: 'RotateRight', 3: 'LookUp', 4: 'LookDown', 5: 'Done'}
        return action_idx[int(action)]

    def get_reachable_position(self):
        event = self.controller.step(action="GetReachablePositions")
        return event.metadata["actionReturn"]

    def teleport(self, position, rotation, horizon):
        event = self.controller.step(action="Teleport", position=position, rotation=rotation, horizon=horizon)
        return event

    def seed(self, rank):
        random.seed(rank)
        np.random.seed(rank)

    def close(self):
        pass


class Human_Demand_Env():

    def __init__(self, args=None):
        self.args = args
        # mode: train, val, test
        if args.dataset_mode == "debug":
            args.dataset_mode = "train"
            self.dataset = self.load_dataset()["train"]
        else:
            self.dataset = self.load_dataset()[args.dataset_mode]
        self.mode = self.args.mode
        self.image_size = (300, 300)
        self.demand_answer = self.load_answer()
        self.demand_instruction = self.load_instruction()
        self.obj_house_idx, self.house_idx_obj = self.load_obj_house_idx()
        self.instruction_features = self.load_instuction_features()
        self.current_step = args.max_step
        self.reset_time = 0
        self.observation_space = (self.image_size[0], self.image_size[1], 4)
        self.action_space = 6
        self.number_of_episodes = args.epoch

        self.controller = Controller(scene=self.dataset[0], renderDepthImage=False, gridSize=0.05, snapToGrid=True, visibilityDistance=1.5, width=300, height=300,server_timeout=500)
        # self.controller.start()
        # self.local_image_encoder, self.local_image_encoder_preprocess = clip.load(
        #     "ViT-B/32")
        self.local_image_encoder_preprocess = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        for ins in self.demand_instruction:
            if ins not in self.instruction_features.keys():
                print("{} not in instruction_features".format(ins))
        # del self.local_image_encoder
        if "DDN" in self.mode:
            self.detr = DETRModel(2, 100)
            self.detr.load_state_dict(torch.load("./pretrained_model/detr_new_18.pth", map_location="cpu"))
            self.detr = self.detr.to(args.device).eval()
            self.transform_detr = get_valid_transforms(h=300, w=300)
        # if self.args.collect_data:
        #     self.collect_data()
        # self.w2v_model = models.word2vec.Word2Vec.load('wiki.model')
        # self.object2idx = self.load_object2idx()

    def load_object2idx(self):
        with open(self.args.path2dataset + "/" + "obj2idx" + ".json", 'r', encoding='utf8') as fp:
            obj2idx = json.load(fp)
        return obj2idx

    def load_instuction_features(self):
        with open(self.args.path2dataset + "/" + "instruction_bert_features" + ".json", 'r', encoding='utf8') as fp:
            instuction_features = json.load(fp)
        return instuction_features

    def load_answer(self):
        if self.args.seen_instruction > 0:
            with open(self.args.path2dataset + "answer_small.json", 'r', encoding='utf8') as fp:
                answer = json.load(fp)
        else:
            with open(self.args.path2dataset + "answer_unseen.json", 'r', encoding='utf8') as fp:
                answer = json.load(fp)
            print("unseen~~~~~~!!!!!!!!!!!!!!")
        return answer

    def load_obj_house_idx(self):
        with open(self.args.path2dataset + "/env/" + "obj_house_idx_{}".format(self.args.dataset_mode) + ".json", 'r', encoding='utf8') as fp:
            obj_house_idx = json.load(fp)
        with open(self.args.path2dataset + "/env/" + "house_idx_obj_{}".format(self.args.dataset_mode) + ".json", 'r', encoding='utf8') as fp:
            house_idx_obj = json.load(fp)
        return obj_house_idx, house_idx_obj

    def load_instruction(self):
        if self.args.seen_instruction > 0:
            with open(self.args.path2dataset + "/" + "instruction_small.json".format(self.args.dataset_mode), 'r', encoding='utf8') as fp:
                instruction = json.load(fp)
        else:
            with open(self.args.path2dataset + "/" + "instruction_unseen.json".format(self.args.dataset_mode), 'r', encoding='utf8') as fp:
                instruction = json.load(fp)
            print("unseen~~~~~~!!!!!!!!!!!!!!")
        return instruction

    def load_dataset(self) -> prior.DatasetDict:
        """Load the houses dataset."""
        print("[AI2-THOR WARNING] There has been an update to ProcTHOR-10K that must be used with AI2-THOR version 5.0+. To use the new version of ProcTHOR-10K, please update AI2-THOR to version 5.0+ by running:\n"
              "    pip install --upgrade ai2thor\n"
              "Alternatively, to downgrade to the old version of ProcTHOR-10K, run:\n"
              '   prior.load_dataset("procthor-10k", revision="ab3cacd0fc17754d4c080a3fd50b18395fae8647")')
        data = {}
        dic = {"train": 10000, "val": 1000, "test": 1000}
        for split, size in [("train", 10_000), ("val", 1_000), ("test", 1_000)]:
            with gzip.open(f"dataset/env/{split}.jsonl.gz", "r") as f:
                houses = [line for line in tqdm(f, total=size, desc=f"Loading {split}")]
            data[split] = LazyJsonDataset(data=houses, dataset="procthor-dataset", split=split)
        return prior.DatasetDict(**data)

    def collect_data(self):
        num = 0
        for idx in tqdm(range(self.args.start, self.args.end)):
            self.house = self.dataset[idx]
            self.controller.reset(scene=self.house, renderDepthImage=self.args.is_depth, renderInstanceSegmentation=False,
                                  gridSize=0.25, snapToGrid=True, visibilityDistance=1.5, width=300, height=300)
            reachable_position = self.get_reachable_position()
            path = "./dataset/images/RGB/house_" + str(idx)
            if os.path.exists(path) is False:
                os.makedirs(path)
            else:
                continue
            ratio = 200 / len(reachable_position)
            for position in reachable_position:
                if random.random() > ratio:
                    continue
                rotation = random.choice([{'x': 0, 'y': 0, 'z': 0}, {'x': 0, 'y': 90, 'z': 0}, {'x': 0, 'y': 180, 'z': 0}, {'x': 0, 'y': 270, 'z': 0}])
                horizon = random.choice([0, 30, -30])
                event = self.teleport(position, rotation, horizon)
                name = str(position['x'])+"_"+str(position['y'])+"_" + \
                    str(position['z'])+"_"+str(rotation['y'])+"_"+str(horizon)
                im = Image.fromarray(event.frame)
                im.save(path + "/" + name + ".jpg")
                # for rotation in [{'x': 0, 'y': 0, 'z': 0}, {'x': 0, 'y': 90, 'z': 0}, {'x': 0, 'y': 180, 'z': 0}, {'x': 0, 'y': 270, 'z': 0}]:
                #     for horizon in [0, 30, -30]:
                #         event = self.teleport(position, rotation, horizon)
                #         name = str(position['x'])+"_"+str(position['y'])+"_"+str(position['z'])+"_"+str(rotation['y'])+"_"+str(horizon)
                #         im = Image.fromarray(event.frame)
                #         im.save(path+"/"+name+".jpg")
                #         num += 1
                num += 1
        print("start: {} end: {}  num: {}".format(self.args.start, self.args.end, num))

    # def get_target_objet(self):
    #     L_object = random.choice(self.demand_instruction[self.instruction])
    #     max_distance = -999999
    #     max_W_object = 0
    #     for W_object in range(list(self.object2idx.keys())):
    #         distance = self.w2v_model.wv.similarity(L_object, W_object)
    #         if distance > max_distance:
    #             max_distance = distance
    #             max_W_object = self.object2idx[W_object]
    #     return max_W_object

    def reset(self):
        # house_idx = random.choice(list(self.house_idx_obj.keys()))
        # self.house = self.dataset[int(house_idx)]
        # obj_name = random.choice(self.house_idx_obj[house_idx])
        # self.instruction = random.choice(self.demand_answer[obj_name])
        # self.demand_objs_name = []
        # for obj in self.demand_instruction[self.instruction]:
        #     if int(house_idx) in self.obj_house_idx[obj]:
        #         self.demand_objs_name.append(obj)
        s_time = time.time()

        self.done = False
        self.navigation_success = False
        self.reset_time += 1
        event = None
        if self.current_step == self.args.max_step or "test" in self.args.mode or "gif" in self.args.mode:
            self.house_idx = random.randint(0, 199)
            self.house = self.dataset[int(self.house_idx)]
            self.controller.reset(scene=self.house, renderDepthImage=False, renderInstanceSegmentation=False, snapToGrid=True, visibilityDistance=1.5, gridSize=0.05, width=300, height=300)
            # self.current_house = (self.current_house+1) % 5
            self.all_position = self.get_reachable_position()
            self.current_step = 0
            self.reset_time = 0
            self.object_list = self.get_object_list(self.house)
            self.current_path = []
        event = self.controller.last_event
        while True:
            self.one_of_obj = random.choice(self.object_list)
            self.instruction = random.choice(self.demand_answer[self.one_of_obj])
            self.demand_objs = []
            for obj in self.demand_instruction[self.instruction]:
                for obj_m in event.metadata['objects']:
                    if obj in obj_m["name"]:
                        self.demand_objs.append(deepcopy(obj_m))
            # for name in self.demand_objs_name:
            #     for obj in event.metadata['objects']:
            #         if name in obj["name"]:
            #             self.demand_objs.append(obj)

            if len(self.demand_objs) > 0:
                break
        if len(self.all_position) > 0 and self.reset_time % 5 == 0 or "test" in self.args.mode or "gif" in self.args.mode:
            position = random.choice(self.all_position)
            rotation = {'x': 0, 'y': random.choice([0, 90, 180, 270]), 'z': 0}
            horizon = random.choice([0, 30, -30])
            event = self.teleport(position, rotation, horizon)
        self.min_distance_demand_objs = self.get_min_distance_demand_objs(event)

        self.current_path = []
        self.current_path.append(deepcopy(event.metadata['agent']['position']))

        if "DDN" in self.args.mode:
            with torch.no_grad():
                detr_image = self.transform_detr(image=event.frame.copy() / 255.0)["image"].to(torch.float32).to(self.args.device)
                detr_output = self.detr([detr_image])
            prob = detr_output['pred_logits'].softmax(dim=-1).detach().cpu().numpy()[:, :, 0]
            top_k_idx = torch.tensor(np.argsort(prob, axis=-1)[:, -self.args.top_k:].copy()).to(self.args.device)
            bbox_top_k = torch.index_select(detr_output['pred_boxes'][0], dim=0, index=top_k_idx[0])
            logits_top_k = torch.index_select(detr_output['pred_logits'][0], dim=0, index=top_k_idx[0])
            local_feature = self.crop_from_bounding_box(event.frame, detr_output['pred_boxes'][0], top_k_idx[0])
            del detr_output
            del detr_image
            event.local_feature = {}
            event.local_feature["feature"] = local_feature.cpu().detach().clone()
            event.local_feature["bbox"] = bbox_top_k.cpu().detach().clone()
            event.local_feature["logits"] = logits_top_k.cpu().detach().clone()
        # print("detr time: ", time.time()-start_time)

        event.instruction = self.instruction
        event.instruction_features = torch.tensor(self.instruction_features[self.instruction][1][0][0]).clone()
        event.min_dis = self.min_distance_demand_objs
        # if "GPT" in self.args.mode:
        #     self.target_object = self.get_target_object()
        #     event.target_object = self.target_object
        # event.bbox = detr_output["pred_boxes"].cpu().detach()
        # event.logits = detr_output['pred_logits'].cpu().detach()
        # print("reset time: ", time.time()-s_time)
        self.init_event = deepcopy(event)
        re = deepcopy(event)
        del event
        gc.collect()
        return re

    # @profile()
    def step(self, action_navi, action_select=None):
        if "test" not in self.args.mode:
            self.current_step += 1
        # action: RotateRight RotateLeft LookUp LookDown MoveAhead Done
        s_time = time.time()
        action_navi = self.trans_action(action_navi["action"]["action"]["action"])
        event = None
        if action_navi == "Done" or self.args.max_step <= self.current_step:
            event = self.controller.step(action=action_navi, renderInstanceSegmentation=True)
        else:
            if "Ahead" in action_navi:
                event = self.controller.step(action=action_navi, moveMagnitude=0.25, renderInstanceSegmentation=False)
            else:
                event = self.controller.step(action=action_navi, degrees=30, renderInstanceSegmentation=False)

        # event = self.controller.step(action=self.trans_action(action_navi["action"]["action"]["action"]))
        # print("step_time:", time.time()-s_time)
        obj = self.controller.last_event.metadata['objects']
        reward = self.reward_function(deepcopy(event))
        # reward = -0.001
        info = {}
        done = False
        if action_navi == "Done" or self.args.max_step <= self.current_step:
            done = True
            info["navigation_success"], info["select_success"], info['spl'] = self.success_check(event, action_select)
            if info["navigation_success"]:
                reward = self.args.navigation_success_reward
                # print("navigation success!!")
            if info["select_success"]:
                reward = self.args.select_success_reward
        info["instruction"] = self.instruction
        info["instruction_features"] = torch.tensor(self.instruction_features[self.instruction][1][0][0])
        start = time.time()
        self.current_path.append(deepcopy(event.metadata['agent']['position']))
        if "DDN" in self.args.mode:
            with torch.no_grad():
                detr_image = self.transform_detr(image=event.frame.copy() / 255.0)["image"].to(torch.float32).to(self.args.device)
                detr_output = self.detr([detr_image])
            prob = detr_output['pred_logits'].softmax(dim=-1).detach().cpu().numpy()[:, :, 0]
            top_k_idx = torch.tensor(np.argsort(prob, axis=-1)[:, -self.args.top_k:]).to(self.args.device)
            bbox_top_k = torch.index_select(detr_output['pred_boxes'][0], dim=0, index=top_k_idx[0])
            logits_top_k = torch.index_select(detr_output['pred_logits'][0], dim=0, index=top_k_idx[0])
            local_feature = self.crop_from_bounding_box(event.frame, detr_output['pred_boxes'][0], top_k_idx[0])
            del detr_output
            del detr_image
            event.local_feature = {}
            event.local_feature["feature"] = local_feature.cpu().detach().clone()
            event.local_feature["bbox"] = bbox_top_k.cpu().detach().clone()
            event.local_feature["logits"] = logits_top_k.cpu().detach().clone()
        # print("detr time: ", time.time()-start)

        event.instruction = self.instruction
        event.instruction_features = torch.tensor(self.instruction_features[self.instruction][1][0][0])
        event.min_dis = self.min_distance_demand_objs
        # if "GPT" in self.args.mode:
        #     #self.target_object = self.get_target_object()
        #     event.target_object = self.target_object
        # event.bbox = detr_output["pred_boxes"].cpu().detach()
        # event.logits = detr_output['pred_logits'].cpu().detach()
        # print()
        # print("step time: {}  action: {} ".format(time.time()-s_time, action_navi))
        re = deepcopy(event)
        del event
        gc.collect()
        return re, reward, done, info

    def trans_action(self, action):
        action_idx = {0: 'MoveAhead', 1: 'RotateLeft', 2: 'RotateRight', 3: 'LookUp', 4: 'LookDown', 5: 'Done'}
        return action_idx[int(action)]

    def crop_from_bounding_box(self, image, bbox, top_k_idx):
        crop_start_time = time.time()
        local_image_encoder_preprocess = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        crop_cur = []
        img = torch.tensor(image.copy()).permute(2, 0, 1).to(torch.float32).to(self.args.device)
        for idx_top in range(self.args.top_k):

            bbox_idx = top_k_idx[idx_top]
            x = math.floor(bbox[bbox_idx, 0] * 300)
            y = math.floor(bbox[bbox_idx, 1] * 300)
            w = math.ceil(bbox[bbox_idx, 2] * 300 + 0.5)
            h = math.ceil(bbox[bbox_idx, 3] * 300 + 0.5)
            crop_1 = torchvision.transforms.functional.crop(img, y, x, h, w)
            crop_1_PIL = torchvision.transforms.ToPILImage()(crop_1)
            crop_cur.append(deepcopy(local_image_encoder_preprocess(crop_1_PIL)))

        crop_cur = torch.stack(crop_cur, dim=0)
        # print("crop time: ", time.time()-crop_start_time)
        return crop_cur

    def get_min_distance_demand_objs(self, event):
        min_dis = 999999999
        for obj in self.demand_objs:
            # path, plan = self.get_shortest_path_from_agent_to_object(obj, event)
            # # print(path)
            # if path is not False:
            #     # path_position = []
            #     # for po in path:
            #     #     path_position.append(po[0])
            #     min_dis = min(min_dis, self.get_path_distance(path))
            agent_pos = event.metadata["agent"]["position"]
            obj_pos = obj["position"]
            dis = np.sqrt((agent_pos['x'] - obj_pos['x'])**2 + (agent_pos['z'] - obj_pos['z'])**2)
            min_dis = min(min_dis, dis)
        return min_dis

    def get_object_list(self, house):
        obj_list = set()
        for obj in house["objects"]:
            obj_list.add(obj["id"].split("|")[0])
            if "children" in obj.keys():
                for obj_children in obj["children"]:
                    if obj_children["id"].split("|")[1] == "surface" and obj_children["id"].split("|")[0] in self.demand_answer.keys():
                        obj_list.add(obj_children["id"].split("|")[0])
        final_list = []
        for obj in obj_list:
            if obj in self.demand_answer.keys():
                final_list.append(obj)
        return final_list

    def reward_function(self, event=None):
        min_dis = 99999999
        for obj in self.demand_objs:
            # path, plan = self.get_shortest_path_from_agent_to_object(obj, event)
            # if path is not False:
            # path_position = []
            # for po in path:
            #     path_position.append(po[0])
            # min_dis = min(min_dis, self.get_path_distance(path))
            agent_pos = event.metadata["agent"]["position"]
            obj_pos = obj["position"]
            dis = np.sqrt((agent_pos['x'] - obj_pos['x'])**2 + (agent_pos['z'] - obj_pos['z'])**2)
            min_dis = min(min_dis, dis)
        reward = self.min_distance_demand_objs - min_dis
        self.min_distance_demand_objs = min_dis
        if reward >= 1000:
            t = 1
        return reward

    def success_check(self, event, action_select):
        if self.args.max_step <= self.current_step:
            return False, False, 0
        navigation_success = False
        select_success = False
        candidate_demand_objects = []
        candidate_demand_objects_bbox = []
        spl = 0
        for demand_object in self.demand_objs:
            agent_position = event.metadata["agent"]["position"]
            agent_pos = event.metadata["agent"]["position"]
            obj_pos = demand_object["position"]
            dis = np.sqrt((agent_pos['x'] - obj_pos['x'])**2 + (agent_pos['z'] - obj_pos['z'])**2)
            bbox = self.get_bounding_box(event, demand_object["objectId"])
            if event.instance_detections2D == None:
                print("warning! detection is none")
            if dis <= 1.5 and bbox is not None:
                navigation_success = True
                shortest_path, plan = self.get_shortest_path_from_agent_to_object(demand_object, self.init_event)
                if shortest_path is not False:
                    spl = compute_single_spl(self.current_path, shortest_path, True)
                else:
                    spl = 1
                # spl = 0
                candidate_demand_objects.append(deepcopy(demand_object))
                candidate_demand_objects_bbox.append(deepcopy(bbox))
        if action_select is not None:
            for candidate_demand_object, bbox in zip(candidate_demand_objects, candidate_demand_objects_bbox):

                if bbox is not None:
                    IoU = self.IoU(bbox, action_select)
                else:
                    Iou = 1
                if IoU > 0.5:
                    select_success = True
                    break
        return navigation_success, select_success, spl

    def get_bounding_box(self, event, objectId):
        if event.instance_detections2D == None:
            return None
        return event.instance_detections2D.get(objectId)

    def IoU(self, a, b):

        (x0_1, y0_1, x1_1, y1_1) = a
        (x0_2, y0_2, x1_2, y1_2) = b
        # get the overlap rectangle
        overlap_x0 = max(x0_1, x0_2)
        overlap_y0 = max(y0_1, y0_2)
        overlap_x1 = min(x1_1, x1_2)
        overlap_y1 = min(y1_1, y1_2)
        # check if there is an overlap
        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0
        # if yes, calculate the ratio of the overlap to each ROI size and the unified size
        size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        assert size_1 >= 0
        assert size_2 >= 0
        size_intersection = (overlap_x1 - overlap_x0) * \
            (overlap_y1 - overlap_y0)
        assert size_intersection >= 0
        size_union = size_1 + size_2 - size_intersection
        return size_intersection / size_union

    def get_shortest_path_from_agent_to_object(self, object, event):
        try:
            # path, plan,_,_ = get_shortest_path_to_object(controller=self.controller,
            #                                          object_id=object["name"],
            #                                          start_position=event.metadata["agent"]["position"],
            #                                          start_rotation=event.metadata["agent"]["rotation"],
            #                                          return_plan=True,
            #                                          goal_distance=1.5,
            #                                          h_angles=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])

            path = get_shortest_path_to_object(
                controller=self.controller,
                object_id=object["name"],
                initial_position=event.metadata["agent"]["position"],
                initial_rotation=event.metadata["agent"]["rotation"],
            )
            return path, False
        except Exception as e:
            print(str(e))
            return False,False
            
            

    def get_path_distance(self, path):
        return path_distance(path)

    def get_spl(self, current_path, shortest_path, is_success):
        return compute_single_spl(path=current_path, shortest_path=shortest_path, successful_path=is_success)

    def get_top_down_frame(self, event):
        # Setup the top-down camera
        if "Ours" in self.args.mode:
            features=deepcopy(event.local_feature)
        event = self.controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
        pose = copy.deepcopy(event.metadata["actionReturn"])
        
        bounds = event.metadata["sceneBounds"]["size"]
        max_bound = max(bounds["x"], bounds["z"])

        pose["fieldOfView"] = 50
        pose["position"]["y"] += 1.1 * max_bound
        pose["orthographic"] = False
        pose["farClippingPlane"] = 50
        del pose["orthographicSize"]

        # add the camera to the scene
        event = self.controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )
        # top_down_frame = event.third_party_camera_frames[-1]
        event.top_down=event.third_party_camera_frames[-1]
        event.instruction = self.instruction
        if "Ours" in self.args.mode:
            event.local_feature=features
        return event

    def get_reachable_position(self):
        event = self.controller.step(action="GetReachablePositions")
        return event.metadata["actionReturn"]

    def teleport(self, position, rotation, horizon):
        event = self.controller.step(action="Teleport", position=position, rotation=rotation, horizon=horizon)
        return event

    def random_materials(self, event):
        event = self.controller.step(action="RandomizeMaterials")

    def seed(self, rank):
        random.seed(rank)
        np.random.seed(rank)

    def close(self):
        pass


class Human_Demand_Env_GPT():

    def __init__(self, args=None):
        self.args = args
        # mode: train, val, test
        if args.dataset_mode == "debug":
            args.dataset_mode = "train"
            self.dataset = self.load_dataset()["train"]
        else:
            self.dataset = self.load_dataset()[args.dataset_mode]
        self.mode = self.args.mode
        self.image_size = (300, 300)
        self.demand_answer = self.load_answer()
        self.demand_instruction = self.load_instruction()
        self.obj_house_idx, self.house_idx_obj = self.load_obj_house_idx()
        self.instruction_features = self.load_instuction_features()
        self.current_step = args.max_step
        self.reset_time = 0
        self.observation_space = (self.image_size[0], self.image_size[1], 4)
        self.action_space = 6
        self.number_of_episodes = args.epoch
        self.controller = Controller(scene=self.dataset[0], renderDepthImage=False, gridSize=0.05, snapToGrid=True, visibilityDistance=1.5, width=300, height=300)
        # self.controller.start()
        # self.local_image_encoder, self.local_image_encoder_preprocess = clip.load(
        #     "ViT-B/32")
        self.local_image_encoder_preprocess = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        for ins in self.demand_instruction:
            if ins not in self.instruction_features.keys():
                print("{} not in instruction_features".format(ins))
        # del self.local_image_encoder

        self.semantic_model, self.semantic_preprocess = clip.load("ViT-B/32")
        self.semantic_model = self.semantic_model.to(args.device).eval()
        self.object2idx = self.load_object2idx()

    def load_object2idx(self):
        with open(self.args.path2dataset + "/" + "obj2idx" + ".json", 'r', encoding='utf8') as fp:
            obj2idx = json.load(fp)
        return obj2idx

    def load_instuction_features(self):
        with open(self.args.path2dataset + "/" + "instruction_bert_features" + ".json", 'r', encoding='utf8') as fp:
            instuction_features = json.load(fp)
        return instuction_features

    def load_answer(self):
        with open(self.args.path2dataset + "answer_small.json", 'r', encoding='utf8') as fp:
            answer = json.load(fp)
        return answer

    def load_obj_house_idx(self):
        with open(self.args.path2dataset + "/" + "obj_house_idx_{}".format(self.args.dataset_mode) + ".json", 'r', encoding='utf8') as fp:
            obj_house_idx = json.load(fp)
        with open(self.args.path2dataset + "/" + "house_idx_obj_{}".format(self.args.dataset_mode) + ".json", 'r', encoding='utf8') as fp:
            house_idx_obj = json.load(fp)
        return obj_house_idx, house_idx_obj

    def load_instruction(self):
        with open(self.args.path2dataset + "/" + "instruction_small.json".format(self.args.dataset_mode), 'r', encoding='utf8') as fp:
            instruction = json.load(fp)
        return instruction

    def load_dataset(self) -> prior.DatasetDict:
        """Load the houses dataset."""
        print("[AI2-THOR WARNING] There has been an update to ProcTHOR-10K that must be used with AI2-THOR version 5.0+. To use the new version of ProcTHOR-10K, please update AI2-THOR to version 5.0+ by running:\n"
              "    pip install --upgrade ai2thor\n"
              "Alternatively, to downgrade to the old version of ProcTHOR-10K, run:\n"
              '   prior.load_dataset("procthor-10k", revision="ab3cacd0fc17754d4c080a3fd50b18395fae8647")')
        data = {}
        dic = {"train": 10000, "val": 1000, "test": 1000}
        for split, size in [("train", 10_000), ("val", 1_000), ("test", 1_000)]:
            with gzip.open(f"dataset/{split}.jsonl.gz", "r") as f:
                houses = [line for line in tqdm(f, total=size, desc=f"Loading {split}")]
            data[split] = LazyJsonDataset(data=houses, dataset="procthor-dataset", split=split)
        return prior.DatasetDict(**data)

    @torch.no_grad()
    def get_target_object(self):
        L_object = random.choice(self.demand_instruction[self.instruction])
        L_text_tokens = clip.tokenize([L_object]).to(self.args.device)
        L_semantic_features = self.semantic_model.encode_text(L_text_tokens).float()
        L_semantic_features = L_semantic_features / L_semantic_features.norm(dim=-1, keepdim=True)

        W_text_tokens = clip.tokenize(list(self.object2idx.keys())).to(self.args.device)
        W_semantic_features = self.semantic_model.encode_text(W_text_tokens).float()
        W_semantic_features = W_semantic_features / W_semantic_features.norm(dim=-1, keepdim=True)
        similarity = W_semantic_features.cpu().numpy() @ L_semantic_features.cpu().numpy().T

        max_W_object = list(self.object2idx.keys())[np.argmax(similarity)]
        return max_W_object

    def reset(self):
        # house_idx = random.choice(list(self.house_idx_obj.keys()))
        # self.house = self.dataset[int(house_idx)]
        # obj_name = random.choice(self.house_idx_obj[house_idx])
        # self.instruction = random.choice(self.demand_answer[obj_name])
        # self.demand_objs_name = []
        # for obj in self.demand_instruction[self.instruction]:
        #     if int(house_idx) in self.obj_house_idx[obj]:
        #         self.demand_objs_name.append(obj)
        s_time = time.time()

        self.done = False
        self.navigation_success = False
        self.reset_time += 1
        event = self.controller.last_event
        if self.current_step == self.args.max_step:
            self.house_idx = random.randint(0, 199)
            self.house = self.dataset[int(self.house_idx)]
            self.controller.reset(scene=self.house, renderDepthImage=False, renderInstanceSegmentation=False, snapToGrid=True, visibilityDistance=1.5, gridSize=0.05, width=300, height=300)
            # self.current_house = (self.current_house+1) % 5
            self.all_position = self.get_reachable_position()
            self.current_step = 0
            self.reset_time = 0
            self.object_list = self.get_object_list(self.house)
            self.current_path = []
            event = self.controller.last_event
            while True:
                self.one_of_obj = random.choice(self.object_list)
                self.instruction = random.choice(self.demand_answer[self.one_of_obj])
                self.demand_objs = []
                for obj in self.demand_instruction[self.instruction]:
                    for obj_m in event.metadata['objects']:
                        if obj in obj_m["name"]:
                            self.demand_objs.append(deepcopy(obj_m))
                # for name in self.demand_objs_name:
                #     for obj in event.metadata['objects']:
                #         if name in obj["name"]:
                #             self.demand_objs.append(obj)

                if len(self.demand_objs) > 0:
                    break

            position = random.choice(self.all_position)
            rotation = {'x': 0, 'y': random.choice([0, 90, 180, 270]), 'z': 0}
            horizon = random.choice([0, 30, -30])
            event = self.teleport(position, rotation, horizon)
            self.init_event = deepcopy(event)

        self.min_distance_demand_objs = self.get_min_distance_demand_objs(event)

        self.current_path.append(deepcopy(event.metadata['agent']['position']))

        event.instruction = self.instruction
        event.instruction_features = torch.tensor(self.instruction_features[self.instruction][1][0][0]).clone()
        event.min_dis = self.min_distance_demand_objs
        self.target_object = self.get_target_object()
        event.target_object_name = self.target_object
        event.target_object = self.object2idx[self.target_object]
        # event.bbox = detr_output["pred_boxes"].cpu().detach()
        # event.logits = detr_output['pred_logits'].cpu().detach()
        # print("reset time: ", time.time()-s_time)
        re = deepcopy(event)
        del event
        gc.collect()
        return re

    # @profile()
    def step(self, action_navi, action_select=None):
        self.current_step += 1
        # action: RotateRight RotateLeft LookUp LookDown MoveAhead Done
        s_time = time.time()
        action_navi = self.trans_action(action_navi["action"]["action"]["action"])
        event = None
        if action_navi == "Done":
            event = self.controller.step(action=action_navi, renderInstanceSegmentation=True)
        else:
            if "Ahead" in action_navi:
                event = self.controller.step(action=action_navi, moveMagnitude=0.25, renderInstanceSegmentation=False)
            else:
                event = self.controller.step(action=action_navi, degrees=30, renderInstanceSegmentation=False)

        # event = self.controller.step(action=self.trans_action(action_navi["action"]["action"]["action"]))
        # print("step_time:", time.time()-s_time)
        obj = self.controller.last_event.metadata['objects']
        reward = self.reward_function(deepcopy(event))
        # reward = -0.001
        info = {}
        done = False
        if action_navi == "Done" or self.args.max_step < self.current_step:
            done = True
            info["navigation_success"], info["select_success"], info['spl'] = self.success_check(event, action_select)
            if info["navigation_success"]:
                reward = self.args.navigation_success_reward
                # print("navigation success!!")
            if info["select_success"]:
                reward = self.args.select_success_reward
        info["instruction"] = self.instruction
        info["instruction_features"] = torch.tensor(self.instruction_features[self.instruction][1][0][0])
        start = time.time()
        self.current_path.append(deepcopy(event.metadata['agent']['position']))
        if "Ours" in self.args.mode:
            with torch.no_grad():
                detr_image = self.transform_detr(image=event.frame.copy() / 255.0)["image"].to(torch.float32).to(self.args.device)
                detr_output, hs = self.detr([detr_image])
            prob = detr_output['pred_logits'].softmax(dim=-1).detach().cpu().numpy()[:, :, 0]
            top_k_idx = torch.tensor(np.argsort(prob, axis=-1)[:, -self.args.top_k:]).to(self.args.device)
            bbox_top_k = torch.index_select(detr_output['pred_boxes'][0], dim=0, index=top_k_idx[0])
            logits_top_k = torch.index_select(detr_output['pred_logits'][0], dim=0, index=top_k_idx[0])
            local_feature = self.crop_from_bounding_box(event.frame, detr_output['pred_boxes'][0], top_k_idx[0])
            del detr_output
            del hs
            del detr_image
            event.local_feature = {}
            event.local_feature["feature"] = local_feature.cpu().detach().clone()
            event.local_feature["bbox"] = bbox_top_k.cpu().detach().clone()
            event.local_feature["logits"] = logits_top_k.cpu().detach().clone()
        # print("detr time: ", time.time()-start)

        event.instruction = self.instruction
        event.instruction_features = torch.tensor(self.instruction_features[self.instruction][1][0][0])
        event.min_dis = self.min_distance_demand_objs
        event.target_object_name = self.target_object
        event.target_object = self.object2idx[self.target_object]

        re = deepcopy(event)
        del event
        gc.collect()
        return re, reward, done, info

    def trans_action(self, action):
        action_idx = {0: 'MoveAhead', 1: 'RotateLeft', 2: 'RotateRight', 3: 'LookUp', 4: 'LookDown', 5: 'Done'}
        return action_idx[int(action)]

    def crop_from_bounding_box(self, image, bbox, top_k_idx):
        crop_start_time = time.time()
        local_image_encoder_preprocess = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        crop_cur = []
        img = torch.tensor(image.copy()).permute(2, 0, 1).to(torch.float32).to(self.args.device)
        for idx_top in range(self.args.top_k):

            bbox_idx = top_k_idx[idx_top]
            x = math.floor(bbox[bbox_idx, 0] * 300)
            y = math.floor(bbox[bbox_idx, 1] * 300)
            w = math.ceil(bbox[bbox_idx, 2] * 300 + 0.5)
            h = math.ceil(bbox[bbox_idx, 3] * 300 + 0.5)
            crop_1 = torchvision.transforms.functional.crop(img, y, x, h, w)
            crop_1_PIL = torchvision.transforms.ToPILImage()(crop_1)
            crop_cur.append(deepcopy(local_image_encoder_preprocess(crop_1_PIL)))

        crop_cur = torch.stack(crop_cur, dim=0)
        # print("crop time: ", time.time()-crop_start_time)
        return crop_cur

    def get_min_distance_demand_objs(self, event):
        min_dis = 999999999
        for obj in self.demand_objs:
            # path, plan = self.get_shortest_path_from_agent_to_object(obj, event)
            # # print(path)
            # if path is not False:
            #     # path_position = []
            #     # for po in path:
            #     #     path_position.append(po[0])
            #     min_dis = min(min_dis, self.get_path_distance(path))
            agent_pos = event.metadata["agent"]["position"]
            obj_pos = obj["position"]
            dis = np.sqrt((agent_pos['x'] - obj_pos['x'])**2 + (agent_pos['z'] - obj_pos['z'])**2)
            min_dis = min(min_dis, dis)
        return min_dis

    def get_object_list(self, house):
        obj_list = set()
        for obj in house["objects"]:
            obj_list.add(obj["id"].split("|")[0])
            if "children" in obj.keys():
                for obj_children in obj["children"]:
                    if obj_children["id"].split("|")[1] == "surface" and obj_children["id"].split("|")[0] in self.demand_answer.keys():
                        obj_list.add(obj_children["id"].split("|")[0])
        for obj in obj_list:
            if obj not in self.demand_answer.keys():
                print(obj)
        return list(obj_list)

    def reward_function(self, event=None):
        min_dis = 99999999
        for obj in self.demand_objs:
            # path, plan = self.get_shortest_path_from_agent_to_object(obj, event)
            # if path is not False:
            # path_position = []
            # for po in path:
            #     path_position.append(po[0])
            # min_dis = min(min_dis, self.get_path_distance(path))
            agent_pos = event.metadata["agent"]["position"]
            obj_pos = obj["position"]
            dis = np.sqrt((agent_pos['x'] - obj_pos['x'])**2 + (agent_pos['z'] - obj_pos['z'])**2)
            min_dis = min(min_dis, dis)
        reward = self.min_distance_demand_objs - min_dis
        self.min_distance_demand_objs = min_dis
        if reward >= 1000:
            t = 1
        return reward

    def success_check(self, event, action_select):
        if self.args.max_step < self.current_step:
            return False, False, 0
        navigation_success = False
        select_success = False
        candidate_demand_objects = []
        spl = 0
        for demand_object in self.demand_objs:
            agent_position = event.metadata["agent"]["position"]
            agent_pos = event.metadata["agent"]["position"]
            obj_pos = demand_object["position"]
            dis = np.sqrt((agent_pos['x'] - obj_pos['x'])**2 + (agent_pos['z'] - obj_pos['z'])**2)
            bbox = self.get_bounding_box(event, demand_object["objectId"])
            if event.instance_detections2D == None:
                print("warning! detection is none")
            if dis <= 1.5 and bbox is not None:
                navigation_success = True
                shortest_path, plan = self.get_shortest_path_from_agent_to_object(demand_object, self.init_event)
                if shortest_path is not False:
                    spl = compute_single_spl(self.current_path, shortest_path, True)
                else:
                    spl = 1
                # spl = 0
                candidate_demand_objects.append(deepcopy(demand_object))
        for candidate_demand_object in candidate_demand_objects:
            IoU = self.IoU(self.get_bounding_box(event, candidate_demand_object["name"]), action_select)

            if IoU > 0.5:
                select_success = True
                break
        return navigation_success, select_success, spl

    def get_bounding_box(self, event, objectId):
        if event.instance_detections2D == None:
            return None
        return event.instance_detections2D.get(objectId)

    def IoU(self, a, b):

        (x0_1, y0_1, x1_1, y1_1) = a
        (x0_2, y0_2, x1_2, y1_2) = b
        # get the overlap rectangle
        overlap_x0 = max(x0_1, x0_2)
        overlap_y0 = max(y0_1, y0_2)
        overlap_x1 = min(x1_1, x1_2)
        overlap_y1 = min(y1_1, y1_2)
        # check if there is an overlap
        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0
        # if yes, calculate the ratio of the overlap to each ROI size and the unified size
        size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        assert size_1 >= 0
        assert size_2 >= 0
        size_intersection = (overlap_x1 - overlap_x0) * \
            (overlap_y1 - overlap_y0)
        assert size_intersection >= 0
        size_union = size_1 + size_2 - size_intersection
        return size_intersection / size_union

    def get_shortest_path_from_agent_to_object(self, object, event):
        try:
            # path, plan = get_shortest_path_to_object(controller=self.controller,
            #                                          object_id=object["name"],
            #                                          start_position=event.metadata["agent"]["position"],
            #                                          start_rotation=event.metadata["agent"]["rotation"],
            #                                          return_plan=True,
            #                                          goal_distance=1.5,
            #                                          h_angles=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])

            path = get_shortest_path_to_object(
                controller=self.controller,
                object_id=object["name"],
                initial_position=event.metadata["agent"]["position"],
                initial_rotation=event.metadata["agent"]["rotation"],
            )
            return path, None
        except:
            return False, False

    def get_path_distance(self, path):
        return path_distance(path)

    def get_spl(self, current_path, shortest_path, is_success):
        return compute_single_spl(path=current_path, shortest_path=shortest_path, successful_path=is_success)

    def get_top_down_frame(self, event):
        # Setup the top-down camera
        event = self.controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
        pose = copy.deepcopy(event.metadata["actionReturn"])

        bounds = event.metadata["sceneBounds"]["size"]
        max_bound = max(bounds["x"], bounds["z"])

        pose["fieldOfView"] = 50
        pose["position"]["y"] += 1.1 * max_bound
        pose["orthographic"] = False
        pose["farClippingPlane"] = 50
        del pose["orthographicSize"]

        # add the camera to the scene
        event = self.controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )
        top_down_frame = event.third_party_camera_frames[-1]
        return Image.fromarray(top_down_frame)

    def get_reachable_position(self):
        event = self.controller.step(action="GetReachablePositions")
        return event.metadata["actionReturn"]

    def teleport(self, position, rotation, horizon):
        event = self.controller.step(action="Teleport", position=position, rotation=rotation, horizon=horizon)
        return event

    def random_materials(self, event):
        event = self.controller.step(action="RandomizeMaterials")

    def seed(self, rank):
        random.seed(rank)
        np.random.seed(rank)

    def close(self):
        pass


if __name__ == "__main__":

    args = parse_arguments()
    # env = Human_Demand_Env(args)
    # env.reset()
    # env.step("Done")
    # args_parallel = [copy.deepcopy(args) for i in range(args.workers)]

    # if args.mode == "train":
    #     l = 4000
    # else:
    #     l = 1000
    # args.workers = 1
    # for i in range(args.workers):
    #     args_parallel[i].start = (l//args.workers)*i
    #     args_parallel[i].end = (l//args.workers)*(i+1)
    # env_classes = [Human_Demand_Env] * args.workers
    # envs = VectorEnv(make_env_fn=make_env_fn, env_fn_args=tuple(zip(args_parallel, env_classes, range(args.workers))), auto_reset_done=False)
    # envs.close()
    env = Human_Demand_Env(args)
    env.reset()
    t = 1
