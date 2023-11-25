import os
import numpy as np
import pandas as pd
from datetime import datetime
import time
import random
from tqdm.autonotebook import tqdm
import argparse

# Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from transformers import DetrForObjectDetection
class DETRModel(nn.Module):
    def __init__(self, num_classes, num_queries):
        super(DETRModel, self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        # self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False)
        self.model = torch.hub.load('/home/whc/.cache/torch/hub/facebookresearch_detr_main', 'detr_resnet50', pretrained=False, source="local")
        # self.model=DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        # checkpoint = torch.load(
        #     "./pretrained_model/detr-r50-e632da11.pth", map_location="cpu")
        # self.model.load_state_dict(checkpoint["model"])
       
        self.in_features = self.model.class_embed.in_features

        self.model.class_embed = nn.Linear(in_features=self.in_features, out_features=self.num_classes)
        self.model.num_queries = self.num_queries

    def forward(self, images):
        return self.model(images)
