import numpy as np
from a2c_ppo_acktr.utils import init
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import model, attribute_model
from model.model import BaseModel, PreTrainedVisualTransformer_Object, PreTrainedVisualTransformer_Demand, ZSONModel
from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian


class Object:
    pass


class Agent(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.visual_model = BaseModel(args)
        self.lstm_layer_num = 2
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

    def forward_train(self, inputs, other_inputs, deterministic=False):
        if self.args.mode == "train_VTN" or self.args.mode == "train_VTN_demand" or self.args.mode == "test_VTN" or self.args.mode == "test_VTN_demand" or self.args.mode == "test_VTN_GPT":
            features = self.visual_model.forward(inputs)["visual_reps"].reshape(self.args.workers, -1, 3136)
        elif self.args.mode == "train_ZSON" or self.args.mode == "test_ZSON":
            features = self.visual_model.forward(inputs).reshape(self.args.workers, -1, self.lstm_input_sz - self.args.action_embedding_dim)
        else:
            features = self.visual_model.forward(inputs)
        prev_action_embed = self.embed_action(other_inputs['prev_action'].to(self.args.device))
        t_feature = torch.cat((features, prev_action_embed), dim=-1)
        output, (hx, cx) = self.lstm(t_feature, (other_inputs["prev_hidden_h"].to(self.args.device), other_inputs["prev_hidden_c"].to(self.args.device)))
        # x = output.reshape([1, self.args.rnn_hidden_state_dim])
        action_dist = self.action(output)
        if deterministic:
            action_out = action_dist.mode()
        else:
            action_out = action_dist.sample()
        value = self.critic_2(F.relu(self.critic_1(output)))
        return value, action_out, action_dist, hx, cx

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
        if self.args.mode == "train_VTN" or self.args.mode == "train_VTN_demand" or self.args.mode == "test_VTN" or self.args.mode == "test_VTN_demand":
            features = self.visual_model.forward(inputs)["visual_reps"].reshape(-1, 3136)
        elif self.args.mode == "train_ZSON" or self.args.mode == "test_ZSON":
            features = self.visual_model.forward(inputs).reshape(-1, self.lstm_input_sz - self.args.action_embedding_dim)
        else:
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

    def update(self, rollouts, pre_train_Q=False):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.args.ppo_epoch):
            data_generator = rollouts.recurrent_generator(advantages, self.args.num_mini_batch)

            for sample in data_generator:
                rgb_batch, depth_batch, instruction_feature_batch, crop_batch, bbox_batch, logits_batch,\
                    recurrent_hidden_states_batch, actions_batch, \
                    value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, prev_action_log_batch = sample
                old_action_log_probs_batch = old_action_log_probs_batch.to(self.args.device)
                adv_targ = adv_targ.to(self.args.device)
                masks_batch = masks_batch.to(self.args.device)
                return_batch = return_batch.to(self.args.device)
                value_preds_batch = value_preds_batch.to(self.args.device)
                # Reshape to do in a single forward pass for all steps
                if self.args.mode == "train_VTN" or self.args.mode == "train_VTN_demand":
                    inputs = {}
                    inputs['global_feature'] = rgb_batch.clone().detach().to(self.args.device)
                    inputs['local_feature'] = {}
                    inputs['local_feature']["features"] = crop_batch.to(self.args.device)
                    inputs["local_feature"]["bboxes"] = bbox_batch.to(self.args.device)
                    inputs["local_feature"]["indicator"] = instruction_feature_batch.unsqueeze(dim=1).repeat(1, 100, 1).to(self.args.device)
                    inputs["local_feature"]["scores"] = logits_batch.to(self.args.device)
                    other_inputs = {}
                    other_inputs['prev_action'] = prev_action_log_batch.to(self.args.device)
                    recurrent_hidden_states_batch = recurrent_hidden_states_batch.reshape(-1, 4, 1024)
                    other_inputs["prev_hidden_h"] = recurrent_hidden_states_batch[:, :2, :].to(self.args.device)
                    other_inputs["prev_hidden_c"] = recurrent_hidden_states_batch[:, 2:, :].to(self.args.device)
                    values, action_log_probs, dist_entropy, _, _ = self.evaluate_batch(inputs, other_inputs, masks_batch, actions_batch)
                elif self.args.mode == "train_ZSON":
                    inputs = {}
                    inputs["images"] = rgb_batch.clone().detach().to(self.args.device)
                    inputs["depths"] = depth_batch.clone().detach().to(self.args.device)
                    inputs["semantic_features"] = instruction_feature_batch.to(self.args.device)
                    other_inputs = {}
                    other_inputs['prev_action'] = prev_action_log_batch.to(self.args.device)
                    recurrent_hidden_states_batch = recurrent_hidden_states_batch.reshape(-1, 4, 1024)
                    other_inputs["prev_hidden_h"] = recurrent_hidden_states_batch[:, :2, :].to(self.args.device)
                    other_inputs["prev_hidden_c"] = recurrent_hidden_states_batch[:, 2:, :].to(self.args.device)
                    values, action_log_probs, dist_entropy, _, _ = self.evaluate_batch(inputs, other_inputs, masks_batch, actions_batch)
                else:
                    inputs = {}
                    inputs["rgb"] = rgb_batch.clone().detach().to(self.args.device)
                    inputs["instruction"] = instruction_feature_batch.clone().detach().to(self.args.device)
                    inputs['local_feature'] = {}
                    inputs['local_feature']["features"] = crop_batch.clone().detach().to(self.args.device)
                    inputs["local_feature"]["bboxes"] = bbox_batch.clone().detach().to(self.args.device)
                    inputs['local_feature']['logits'] = logits_batch.clone().detach().to(self.args.device)
                    other_inputs = {}
                    other_inputs['prev_action'] = prev_action_log_batch.clone().detach()
                    recurrent_hidden_states_batch = recurrent_hidden_states_batch.reshape(-1, 4, 1024)
                    other_inputs["prev_hidden_h"] = recurrent_hidden_states_batch[:, :2, :].clone().detach()
                    other_inputs["prev_hidden_c"] = recurrent_hidden_states_batch[:, 2:, :].clone().detach()
                    values, action_log_probs, dist_entropy, _, _ = self.evaluate_batch(inputs, other_inputs, masks_batch, actions_batch, pre_train_Q)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.args.clip_param, self.args.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.args.value_loss_coef + action_loss - dist_entropy * self.args.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.args.ppo_epoch * self.args.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):

    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(x[start_idx:end_idx], hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


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
