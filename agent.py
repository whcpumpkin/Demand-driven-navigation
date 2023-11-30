import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import attribute_model
from model import models_mae
from torchvision import transforms
from model.attribute_model import attention_attribute_model

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        def init(module, weight_init, bias_init, gain=1):
            weight_init(module.weight.data, gain=gain)
            bias_init(module.bias.data)
            return module
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)
    
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)
    
class fusion_model(nn.Module):

    def __init__(self, args, attribute_feature_dim=512, instruction_feature_dim=1024, global_image_feature_dim=1024, depth_dim=0):
        super().__init__()
        self.args = args
        self.proj = nn.Linear(instruction_feature_dim + global_image_feature_dim+depth_dim, attribute_feature_dim)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=attribute_feature_dim, nhead=8, batch_first=True, dropout=0.1), num_layers=self.args.fusion_encoder_layer_num)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=attribute_feature_dim, nhead=8, batch_first=True, dropout=0.1), num_layers=self.args.fusion_decoder_layer_num)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, attribute_feature, instruction_feature, global_image_feature, depth_image=None):
        kv = self.encoder(attribute_feature)
        query = self.proj(torch.cat((instruction_feature, global_image_feature), dim=-1))
        y = self.decoder(query, kv)
        return y

class BaseModel(torch.nn.Module):

    def __init__(self, args, instruction_feature_dim=1024, global_image_feature_dim=1024):
        super().__init__()
        self.args = args
        self.action_space = 6

        self.global_image_encoder = models_mae.__dict__["mae_vit_large_patch16"](norm_pix_loss=True)
        checkpoint = torch.load("./pretrained_model/mae_pretrain_model.pth", map_location='cpu')
        self.global_image_encoder.load_state_dict(checkpoint['model'])
        self.global_image_encoder = self.global_image_encoder.to(args.device)
        self.transform_global = transforms.Compose([transforms.ToPILImage(), transforms.Resize(args.input_size, interpolation=3), transforms.CenterCrop(224),
                                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])



        self.attribute_model = attention_attribute_model(args)

        attribute_model_ckpt = torch.load("./pretrained_model/attribute_model2.pt", map_location=torch.device('cpu'))
        self.attribute_model.load_state_dict(attribute_model_ckpt['model_state_dict'])
        self.attribute_model = self.attribute_model.to(self.args.device)
        self.attribute_mlp = nn.Linear(args.attribute_feature_dim + 6, args.attribute_feature_dim).to(self.args.device)
            


        # --------fusion model--------pretrained in supervised learning----------
        self.fusion_model = fusion_model(args, attribute_feature_dim=args.attribute_feature_dim, depth_dim=0).to(self.args.device)

    def forward(self, inputs):
        # bs*100*512
        local_image_feature = inputs['local_feature']['features'].to(self.args.device)

        bs = local_image_feature.shape[0]
        local_image_num = local_image_feature.shape[1]

        # bs*1024
        instruction_feature = inputs["instruction"].to(self.args.device).float()
        # bs*100*1024
        instruction_feature_rep = instruction_feature.unsqueeze(dim=1).repeat(1, local_image_num, 1).to(self.args.device)
        # 12800*1*1536
        
        attr_input = torch.cat((instruction_feature_rep.reshape(bs * local_image_num, -1), local_image_feature.reshape(bs * local_image_num, -1)), dim=-1).unsqueeze(dim=1)
            # 12800*512
        attr_feature = self.attribute_model(attr_input).squeeze(dim=1)
        # bs*100*512
        attr_feature = attr_feature.reshape(bs, local_image_num, -1)

        attr_feature = self.attribute_mlp(torch.cat((attr_feature, inputs['local_feature']['bboxes'].to(
            self.args.device).squeeze(dim=1), inputs['local_feature']['logits'].to(self.args.device).squeeze(dim=1)), dim=-1))
        # bs*1024

        global_image_feature = self.global_image_encoder(inputs['image'].to(self.args.device), pre_train=False)
        # bs*1*256
        # bs*1*1024
        fusion_instruction = instruction_feature.unsqueeze(dim=1)
        # bs*1*1024
        global_image_feature = global_image_feature.unsqueeze(dim=1)
        # bs*1*512
        vis_feature = self.fusion_model(attr_feature, fusion_instruction, global_image_feature)

        return vis_feature

    def forward_batch(self, inputs):

        detr_query_num = 100
        local_image_feature = inputs.crop.to(self.args.device).reshape(-1, 512)  # (400,512)

        instruction_features = inputs.instruction_features.to(self.args.device)  # (4,1024)
        instruction_features_rep = instruction_features.unsqueeze(dim=1).repeat(1, detr_query_num, 1).reshape(-1, 1024)  # (400,1024)

        attr_input = torch.cat((instruction_features_rep, local_image_feature), dim=-1).unsqueeze(dim=1)  # (400,1,1536)
        attr_feature = self.attribute_model(attr_input).squeeze(dim=1)  # (400,128)
        attr_feature = attr_feature.reshape(-1, detr_query_num, attr_feature.shape[-1])  # (4,100,128)

        # global_image = inputs.frame.to(self.args.device)  # (4, 3, 224, 224)
        global_image = [self.transform_global(np.array(inputs.frame[i], dtype=np.uint8)) for i in range(inputs.frame.shape[0])]
        global_image = torch.stack(global_image, dim=0).to(self.args.device)
        global_image_feature = self.global_image_encoder(global_image, pre_train=False)  # (4,1024)

        # depth = self.depth_resnet_feature_extractor(inputs["depth"], return_tensors="pt")
        depth = inputs.depth_frame.to(self.args.device) / 255.0
        depth_feature = self.depth_mlp(self.depth_cnn(depth.unsqueeze(dim=1).to(self.args.device))).unsqueeze(dim=1)

        fusion_instruction = instruction_features.unsqueeze(dim=1)
        global_image_feature = global_image_feature.unsqueeze(dim=1)
        vis_feature = self.fusion_model(attr_feature, fusion_instruction, global_image_feature, depth_feature)

        return vis_feature

    def forward_pre(self, inputs):
        # bs*100*512
        local_image_feature = inputs['crop'][:, :, :512].to(self.args.device)

        bs = local_image_feature.shape[0]
        local_image_num = local_image_feature.shape[1]

        # bs*1024
        instruction_feature = inputs["instruction_feature"].to(self.args.device).float()
        # bs*100*1024
        instruction_feature_rep = instruction_feature.unsqueeze(dim=1).repeat(1, local_image_num, 1).to(self.args.device)
        # 12800*1*1536
        attr_input = torch.cat((instruction_feature_rep.reshape(bs * local_image_num, -1), local_image_feature.reshape(bs * local_image_num, -1)), dim=-1).unsqueeze(dim=1)
        # 12800*512
        attr_feature = self.attribute_model(attr_input).squeeze(dim=1)
        # bs*100*512
        attr_feature = attr_feature.reshape(bs, local_image_num, -1)

        attr_feature = self.attribute_mlp(torch.cat((attr_feature, inputs['crop'][:, :, 512:].to(self.args.device).squeeze(dim=1)), dim=-1))
        # bs*1024

        global_image_feature = self.global_image_encoder(inputs['image'].to(self.args.device), pre_train=False)


        # bs*1*256
 
        # bs*1*1024
        fusion_instruction = instruction_feature.unsqueeze(dim=1)
        # fusion_instruction = instruction_feature
        # bs*1*1024
        global_image_feature = global_image_feature.unsqueeze(dim=1)
        # bs*1*512
        vis_feature = self.fusion_model(attr_feature, fusion_instruction, global_image_feature)

        return vis_feature



class Agent(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.visual_model = BaseModel(args)
        self.lstm_layer_num = 2
        self.action_space = 6
        self.lstm_input_sz = args.attribute_feature_dim + args.action_embedding_dim
        self.embed_action = nn.Linear(self.action_space, args.action_embedding_dim)
        self.lstm = nn.LSTM(self.lstm_input_sz, args.rnn_hidden_state_dim, self.lstm_layer_num, batch_first=True)
        self.action = Categorical(args.rnn_hidden_state_dim, self.action_space)
        self.use_clipped_value_loss = True

    def forward(self, inputs):
        if self.args.mode == "pretrain":
            return self.forward_pre(inputs)
        else:
            return self.forward_train(inputs)

    def forward_pre(self, inputs, other_inputs=None):
        features = self.visual_model.forward_pre(inputs).squeeze(dim=1)
        seq_len = features.shape[0]
        prev_action_embed = self.embed_action(other_inputs['prev_action'].to(self.args.device))
        hx = other_inputs["prev_hidden_h"].to(self.args.device).contiguous()
        cx = other_inputs["prev_hidden_c"].to(self.args.device).contiguous()
        action_dis = []
        for i in range(seq_len):
            t_feature = torch.cat((features[i].unsqueeze(dim=0).unsqueeze(dim=0), prev_action_embed), dim=-1)
            output, (hx, cx) = self.lstm(t_feature.contiguous(), (hx, cx))
            action_out = self.action(output)
            action_dis.append(action_out.logits)
            prev_action_embed = self.embed_action(action_out.logits)
        action_dis = torch.stack(action_dis)
        return action_dis


    def evaluate_actions(self, inputs, other_inputs, masks, action):
        value, action_out, action_dist, hx, cx = self.forward_train(inputs, other_inputs)

        action_log_probs = action_dist.log_probs(action_out)
        action_dist_entropy = action_dist.entropy().mean()

        return value, action_log_probs, action_dist_entropy, hx, cx

    def act(self, inputs, other_inputs, deterministic=False):
        value, action_out, action_dist, hx, cx = self.forward_train(inputs, other_inputs, deterministic=deterministic)
        action_log_probs = action_dist.log_probs(action_out)
        action_dist_entropy = action_dist.entropy().mean()
        return value, action_out, action_log_probs, hx, cx, action_dist

    def get_value(self, inputs, other_inputs):
        value, action_out, action_dist, hx, cx = self.forward_train(inputs, other_inputs)
        return value

    def evaluate_batch(self, inputs, other_inputs, masks, action, pre_train_Q=False):
        num_envs_per_batch = self.args.workers // self.args.num_mini_batch
        seq_len = int(other_inputs['prev_action'].shape[0] / num_envs_per_batch)
        value, action_out, action_dist, hx, cx = self.forward_batch(inputs, other_inputs, masks, pre_train_Q)

        action_log_probs = action_dist.log_probs(action.to(self.args.device))
        action_dist_entropy = action_dist.entropy().mean()

        return value, action_log_probs, action_dist_entropy, hx, cx

    def forward_batch(self, inputs, other_inputs, masks, pre_train_Q=False):
        num_envs_per_batch = self.args.workers // self.args.num_mini_batch
        seq_len = int(other_inputs['prev_action'].shape[0] / num_envs_per_batch)

        features = self.visual_model.forward(inputs)
        prev_action_embed = self.embed_action(other_inputs['prev_action'].to(self.args.device)).unsqueeze(dim=1)
        t_feature = torch.cat((features, prev_action_embed), dim=-1).reshape(num_envs_per_batch, seq_len, self.lstm_input_sz)
        seq_len = t_feature.shape[1]
        masks = masks.reshape(num_envs_per_batch, seq_len)

        has_zeros = ((masks[:, 1:] == 0.0).any(dim=0).nonzero().squeeze().cpu())
        if has_zeros.dim() == 0:
            # Deal with scalar
            has_zeros = [has_zeros.item() + 1]
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()
        if isinstance(has_zeros, list) is False:
            print(has_zeros)
            print(masks.shape)
            print(masks)
        has_zeros = [0] + has_zeros + [seq_len]
        outputs = []

        hx = other_inputs["prev_hidden_h"].permute(1, 0, 2).to(self.args.device).contiguous()
        cx = other_inputs["prev_hidden_c"].permute(1, 0, 2).to(self.args.device).contiguous()
        for i in range(len(has_zeros) - 1):
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]
            output, (hx, cx) = self.lstm(t_feature.contiguous()[:, start_idx:end_idx, :], (hx * masks[:, start_idx].reshape(1, -1, 1), cx * masks[:, start_idx].reshape(1, -1, 1)))
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        # x = output.reshape([1, self.args.rnn_hidden_state_dim])
        # action_dist = self.action(output).reshape(seq_len*self.args.workers, -1)
        action_dist = self.action(outputs.reshape(seq_len * num_envs_per_batch, -1))
        action_out = action_dist.sample()
        value = self.critic_2(F.relu(self.critic_1(outputs))).reshape(seq_len * num_envs_per_batch, -1)
        return value, action_out, action_dist, hx, cx




class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)




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
    parser.add_argument('--work-dir', type=str, default='./debugs/', help='Work directory, including: tensorboard log dir, log txt, trained models')
    parser.add_argument('--save-model-dir', default='debugs', help='folder to save trained navigation')
    parser.add_argument('--workers', type=int, default=1, help='parallel size')
    parser.add_argument('--collect_data', type=bool, default=True, help="collect data")
    parser.add_argument('--start', type=int, default=0, help='parallel size')
    parser.add_argument('--end', type=int, default=10000, help='parallel size')

    # contrastive learning args
    parser.add_argument('--mini_batch_size', type=int, default=64, help='mini batch size')
    parser.add_argument('--large_batch_size', type=int, default=128, help='large batch size')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--dataloader_worker', type=int, default=4, help='dataloader_worker')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient_accumulation_steps')
    parser.add_argument('--logging_steps', type=int, default=50, help='logging_steps')
    parser.add_argument('--save_steps', type=int, default=200, help='save_steps')

    # model
    parser.add_argument('--attention_layer_num', type=int, default=6, help='batch size')
    args = parser.parse_args()
