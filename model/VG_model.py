
import torch
from torch import nn
import numpy as np
import random
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from PIL import Image
import torch.nn.functional as F
import time

def get_gloabal_pos_embedding(size_feature_map, c_pos_embedding):
    mask = torch.ones(1, size_feature_map, size_feature_map)

    y_embed = mask.cumsum(1, dtype=torch.float32)
    x_embed = mask.cumsum(2, dtype=torch.float32)

    dim_t = torch.arange(c_pos_embedding, dtype=torch.float32)
    dim_t = 10000 ** (2 * (dim_t // 2) / c_pos_embedding)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    return pos

class VG_model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=1024, nhead=8, batch_first=True, dropout=0.1), num_layers=6)
        self.proj_mlp = nn.Linear(2048, 1024)
        self.mlp_head = nn.Linear(1024, args.top_k)
        self.resnet_feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
        self.resnet = ResNetForImageClassification.from_pretrained("microsoft/resnet-18").to(args.device)
        self.global_conv = nn.Conv2d(512, 256, 1)
        self.global_pos_embedding = get_gloabal_pos_embedding(7, 128)
        self.crop_mlp = nn.Linear(256 + 4 + 2, 1024)
        self.global_mlp = nn.Sequential(nn.Linear(256 * 49, 4096), nn.ReLU(), nn.Linear(4096, 1024))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1024))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img, crop_feature, instruction_feature):
        bs = img.shape[0]
        num_crop = crop_feature.shape[1]
        image_PIL = [Image.fromarray(np.uint8(img[idx])) for idx in range(bs)]
        image_pre = self.resnet_feature_extractor(image_PIL, return_tensors="pt")
        global_feature = self.resnet(image_pre["pixel_values"].to(self.args.device),return_dict=True,output_hidden_states=True)["hidden_states"][-1].unsqueeze(dim=1).detach()
        global_feature = global_feature.squeeze(dim=1)
        image_embedding = F.relu(self.global_conv(global_feature))
        image_embedding = image_embedding + self.global_pos_embedding.repeat([bs, 1, 1, 1]).to(self.args.device)
        image_embedding = image_embedding.reshape(bs, -1)
        x = torch.cat((self.cls_token.repeat(bs, 1, 1), self.crop_mlp(crop_feature), self.global_mlp(
            image_embedding).unsqueeze(1), instruction_feature.unsqueeze(1).to(self.args.device).float()), dim=1)
        x = self.encoder(x)
        return self.mlp_head(x[:, 0, :])
