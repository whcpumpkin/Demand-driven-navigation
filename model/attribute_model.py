import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
# from pytorch_transformers.modeling_bert import (
#     BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
#     BertPreTrainedModel, PretrainedConfig
# )


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


class linear_attribute_model(nn.Module):
    def __init__(self, args, input_dim=1536, output_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024, True)
        self.fc2 = nn.Linear(1024, 512, True)
        self.fc3 = nn.Linear(512, 256, True)
        self.fc4 = nn.Linear(256, output_dim, True)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x



class attention_attribute_model(nn.Module):
    def __init__(self, args, input_dim=1536, output_dim=512, num_heads=1):
        super().__init__()
        embed_dim = 1536
        self.args = args
        self.multihead_attn_list = nn.ModuleList()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=input_dim, nhead=1, batch_first=True, dropout=0.1), num_layers=self.args.attention_layer_num)
        self.fc1 = nn.Linear(input_dim, 512, True)
        self.fc2 = nn.Linear(512, output_dim, True)

    def forward(self, query, key=None, value=None):
        x = self.encoder(query)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--mode', type=str, default="test", help='train, val, test contrastive')
    parser.add_argument('--title', type=str, default="debuging", help='train, val, test')
    parser.add_argument('--is_depth', type=bool, default=True, help="use depth or not")
    parser.add_argument('--logger', type=bool, default=True, help="use logger")
    parser.add_argument('--max_step', type=int, default=200, help="max steps in an episode")
    parser.add_argument('--epoch', type=int, default=100000, help="max steps in an episode")
    parser.add_argument('--object_bound', type=int, default=1, help="max steps in an episode")
    parser.add_argument('--navigation_success_reward', type=float, default=200.0, help="navigation success reward")
    parser.add_argument('--select_success_reward', type=float, default=400.0, help="select success reward")
    parser.add_argument('--path2answer', type=str, default="./dataset/answer.json", help='path to task json')
    parser.add_argument('--path2dataset', type=str, default="./dataset/", help='path to dataset')
    parser.add_argument('--path2instruction', type=str, default="./dataset/instruction.json", help='path to task json')
    parser.add_argument('--path2instruction_bert_features', type=str, default="./dataset/instruction_bert_features.json", help='path to task json')
    parser.add_argument('--path2LGO_features', type=str, default="./dataset/LGO_features.json", help='path to task json')
    parser.add_argument('--path2saved_checkpoints', type=str, default="saved_checkpoints", help='path to saved_checkpoints')
    parser.add_argument('--path2logs', type=str, default="logs", help='path to logs')
    parser.add_argument('--work-dir', type=str, default='./debugs/', help='Work directory, including: tensorboard log dir, log txt, trained models',)
    parser.add_argument('--save-model-dir', default='debugs', help='folder to save trained navigation',)
    parser.add_argument('--workers', type=int, default=1, help='parallel size',)
    parser.add_argument('--collect_data', type=bool, default=True, help="collect data")
    parser.add_argument('--start', type=int, default=0, help='parallel size',)
    parser.add_argument('--end', type=int, default=10000, help='parallel size',)

    # contrastive learning args
    parser.add_argument('--mini_batch_size', type=int, default=64, help='mini batch size',)
    parser.add_argument('--large_batch_size', type=int, default=128, help='large batch size',)
    parser.add_argument('--batch_size', type=int, default=64, help='batch size',)
    parser.add_argument('--dataloader_worker', type=int, default=4, help='dataloader_worker')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient_accumulation_steps')
    parser.add_argument('--logging_steps', type=int, default=50, help='logging_steps')
    parser.add_argument('--save_steps', type=int, default=200, help='save_steps')

    # model
    parser.add_argument('--attention_layer_num', type=int, default=6, help='batch size',)
    args = parser.parse_args()
    batch_size = 32
    ins_embedding_size = 1024
    ins_emb = torch.randn(batch_size, ins_embedding_size)  # bx768
    obj_embedding_size = 512
    objects_features = torch.randn(batch_size, obj_embedding_size)  # bx768
    query = torch.cat((ins_emb, objects_features), dim=1)
    query = query.unsqueeze(1)
    model = attention_attribute_model(args)
    attn_output = model(query, query, query)
    t = -1
    # config = BertConfig(
    #     hidden_size=1536,
    #     num_hidden_layers=4,
    #     num_attention_heads=12,
    #     type_vocab_size=2)
    # mmt = MMT(config, context_2d=None, mmt_mask=None)  # fusion transformer initialization
    # # fusion
    # # mmt_results = mmt(
    # #     txt_emb=ins_emb,
    # #     txt_mask=txt_mask,
    # #     obj_emb=obj_mmt_in,
    # #     obj_mask=obj_mask,
    # #     obj_num=obj_num
    # # )
    # ins_mask = torch.ones(ins_embedding_size)
    # obj_mask = torch.ones(obj_embedding_size)
    # s = mmt_results = mmt(
    #     txt_emb=ins_emb,
    #     txt_mask=ins_mask,
    #     obj_emb=objects_features,
    #     obj_mask=obj_mask,
    #     obj_num=1
    # )
