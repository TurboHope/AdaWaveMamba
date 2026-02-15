import argparse
import torch
import random
import numpy as np
from exp.exp_main import Exp_Main

def main():
    parser = argparse.ArgumentParser(description='AdaWave-Mamba Final Run')

    # Basic Config
    parser.add_argument('--model', type=str, default='AdaWaveMamba')
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=2024)

    # Data Loader Config
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='数据集文件名')
    parser.add_argument('--root_path', type=str, default='./data/', help='数据集目录')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--enc_in', type=int, default=7, help='输入通道数 (CI模式下仅用于RevIN初始化)')

    # Model Config (CI + Mamba)
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
    parser.add_argument('--num_levels', type=int, default=3, help='Wavelet levels')
    
    # Optimization
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0005)

    args = parser.parse_args()

    # Fix Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('Args in experiment:')
    print(args)

    exp = Exp_Main(args)
    
    # 1. Train
    exp.train()

    # 2. Test
    exp.test()

if __name__ == "__main__":
    main()