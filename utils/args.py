import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--mode', type=str, default="train", help='train, val, test contrastive')
    parser.add_argument('--dataset_mode', type=str, default="train", help='train, val, test')
    parser.add_argument('--workers', type=int, default=1, help='parallel size')
    parser.add_argument('--path2dataset', type=str, default="./dataset/", help='path to dataset')
    parser.add_argument('--path2saved_checkpoints', type=str, default="saved_checkpoints", help='path to saved_checkpoints')
    parser.add_argument('--path2logs', type=str, default="logs", help='path to logs')
    parser.add_argument('--path2instruction_bert_features', type=str, default="./dataset/instruction_bert_features_check.json", help='path to task json')
    parser.add_argument('--path2LGO_features', type=str, default="./dataset/LGO_features.json", help='path to task json')
    parser.add_argument('--navigation_success_reward', type=float, default=30.0, help="navigation success reward")
    parser.add_argument('--select_success_reward', type=float, default=60.0, help="select success reward")
    parser.add_argument('--device', type=str, default="cuda:0", help='device to load model')
    parser.add_argument('--start', type=int, default=0, help='preprocess start index')
    parser.add_argument('--end', type=int, default=10000, help='preprocess end index')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--top_k', type=int, default=16, help='top k region')
    parser.add_argument('--epoch', type=int, default=100, help="max steps in an episode")
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--rnn_hidden_state_dim', default=1024, type=int, help='images input size')
    parser.add_argument('--action_embedding_dim', default=16, type=int, help='images input size')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--pretrain_lr_drop', default=10, type=int)
    parser.add_argument('--max_norm', default=0.1, type=float, help='images input size')
    parser.add_argument('--eval_path', type=str, default="saved_checkpoints", help='path to logs')
    parser.add_argument('--eval_ckpt', type=int, default=-1, help='path to logs')
    parser.add_argument('--seen_instruction', type=int, default=1, help='path to logs')
    parser.add_argument('--max_step', type=int, default=100, help="max steps in an episode")
    
    # contrastive learning args
    parser.add_argument('--mini_batch_size', type=int, default=64, help='mini batch size')
    parser.add_argument('--large_batch_size', type=int, default=128, help='large batch size')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--dataloader_worker', type=int, default=4, help='dataloader_worker')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient_accumulation_steps')
    parser.add_argument('--logging_steps', type=int, default=50, help='logging_steps')
    parser.add_argument('--save_steps', type=int, default=200, help='save_steps')
    parser.add_argument('--contrastive_lr', type=float, default=1e-5)
    parser.add_argument('--attention_layer_num', default=6, type=int)
    parser.add_argument('--attribute_feature_dim', default=512, type=int, help='attribute_feature_dim')
    
    # fusion_model
    parser.add_argument('--fusion_encoder_layer_num', default=6, type=int, help='images input size')
    parser.add_argument('--fusion_decoder_layer_num', default=6, type=int, help='images input size')
    
    args = parser.parse_args()

    return args
