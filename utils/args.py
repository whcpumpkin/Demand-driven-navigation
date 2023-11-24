import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument('--workers', type=int, default=1, help='parallel size')
    
    args = parser.parse_args()

    return args
