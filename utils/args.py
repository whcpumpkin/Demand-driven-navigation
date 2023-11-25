import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--mode', type=str, default="train", help='train, val, test contrastive')
    parser.add_argument('--dataset_mode', type=str, default="train", help='train, val, test')
    parser.add_argument('--workers', type=int, default=1, help='parallel size')
    parser.add_argument('--path2dataset', type=str, default="./dataset/", help='path to dataset')
    parser.add_argument('--device', type=str, default="cuda:0", help='device to load model')
    parser.add_argument('--start', type=int, default=0, help='preprocess start index')
    parser.add_argument('--end', type=int, default=10000, help='preprocess end index')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--top_k', type=int, default=16, help='top k region')
    args = parser.parse_args()

    return args
