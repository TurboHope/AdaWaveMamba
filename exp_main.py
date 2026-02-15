import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
import matplotlib.pyplot as plt

# [创新点 B] 频域损失函数 (保持不变)
class FrequencyLoss(nn.Module):
    def __init__(self, weight=0.1): 
        super(FrequencyLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.weight = weight

    def forward(self, pred, true):
        loss_time = self.mse(pred, true)
        pred_fft = torch.fft.rfft(pred, dim=1)
        true_fft = torch.fft.rfft(true, dim=1)
        loss_freq = torch.mean(torch.abs(pred_fft - true_fft))
        return loss_time + self.weight * loss_freq

class Exp_Main:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        print(f"Model Parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        if not os.path.exists('results'):
            os.makedirs('results')
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            print(f"Use GPU: {torch.cuda.get_device_name(0)}")
            return torch.device('cuda:0')
        else:
            print("Use CPU")
            return torch.device('cpu')

    def _build_model(self):
        # [核心修复] 动态导入：根据参数决定加载哪个模型文件
        # 这样跑 CNN 时就不会去加载 Mamba，反之亦然
        model_name = self.args.model
        
        if model_name == 'PaperAdaWaveNet':
            print("Loading CNN-based Benchmark: PaperAdaWaveNet...")
            from models import PaperAdaWaveNet
            model = PaperAdaWaveNet.Model(self.args)
            
        elif model_name == 'AdaWaveMamba':
            print("Loading Our Model: AdaWaveMamba (ASF Version)...")
            from models import AdaWaveMamba
            model = AdaWaveMamba.Model(self.args)
            
        else:
            raise ValueError(f"Unknown model name: {model_name}")
            
        return model

    def _get_data(self, flag):
        if not hasattr(self.args, 'root_path'):
            self.args.root_path = './data/'
        
        shuffle_flag = True if flag == 'train' else False
        
        data_dict = {
            'ETTh1.csv': Dataset_ETT_hour,
            'ETTh2.csv': Dataset_ETT_hour,
            'ETTm1.csv': Dataset_ETT_minute,
            'ETTm2.csv': Dataset_ETT_minute,
            'weather.csv': Dataset_Custom,
            'traffic.csv': Dataset_Custom,
            'electricity.csv': Dataset_Custom,
            'exchange_rate.csv': Dataset_Custom,
        }
        
        data_name = self.args.data_path if hasattr(self.args, 'data_path') else 'ETTh1.csv'
        if data_name not in data_dict:
             print(f"Warning: {data_name} not found in dict, using Custom.")
             Data = Dataset_Custom
        else:
             Data = data_dict[data_name]

        dataset = Data(
            root_path=self.args.root_path,
            data_path=data_name,
            flag=flag,
            size=[self.args.seq_len, 0, self.args.pred_len],
            features='M',
            target='OT'
        )
        print(f"Dataset {flag}: {len(dataset)} samples")
        return dataset, shuffle_flag

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)

    def _select_criterion(self):
        # 对比实验通常只用 MSE 以示公平
        if self.args.model == 'PaperAdaWaveNet':
            return nn.MSELoss()
        else:
            # 我们的模型可以用高级 Loss
            return FrequencyLoss(weight=0.01)

    def train(self):
        train_data, shuffle_flag = self._get_data(flag='train')
        train_loader = DataLoader(
            train_data, 
            batch_size=self.args.batch_size, 
            shuffle=shuffle_flag, 
            num_workers=0, 
            drop_last=True
        )
        
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        print(f">>> Start Training {self.args.model} on {self.args.data_path} >>>")
        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_loss = []
            
            steps = len(train_loader)
            for i, (batch_x, batch_y) in enumerate(train_loader):
                optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                outputs = self.model(batch_x)
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
                
                if (i+1) % 100 == 0:
                     print(f"\tEpoch: {epoch+1}, Step: {i+1}/{steps}, Loss: {loss.item():.5f}")
            
            avg_loss = np.average(epoch_loss)
            print(f"Epoch: {epoch+1}, Average Loss: {avg_loss:.7f}")
        
        save_path = f'checkpoints/{self.args.data_path}_{self.args.model}_checkpoint.pth'
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        return self.model

    def test(self):
        test_data, shuffle_flag = self._get_data(flag='test')
        test_loader = DataLoader(test_data, batch_size=self.args.batch_size, shuffle=shuffle_flag, num_workers=0)
        
        criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        
        self.model.eval()
        total_loss = []
        total_mae = []
        
        preds = []
        trues = []
        
        print(">>> Start Testing >>>")
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                mae = mae_criterion(outputs, batch_y)
                
                total_loss.append(loss.item())
                total_mae.append(mae.item())
                
                if i < 5: 
                    pred = outputs.detach().cpu().numpy()
                    true = batch_y.detach().cpu().numpy()
                    preds.append(pred)
                    trues.append(true)
        
        avg_loss = np.average(total_loss)
        avg_mae = np.average(total_mae)
        print(f"Test MSE: {avg_loss:.7f}")
        print(f"Test MAE: {avg_mae:.7f}")

        # 画图
        input_data = preds[0][0, :, -1]
        target_data = trues[0][0, :, -1]

        plt.figure(figsize=(12, 6))
        plt.plot(target_data, label='GroundTruth', linewidth=2, color='black')
        plt.plot(input_data, label='Prediction', linewidth=2, color='red', linestyle='--')
        plt.title(f'{self.args.model} - MSE: {avg_loss:.4f}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/vis_{self.args.model}_{self.args.data_path}.png')
        plt.close()
        
        return avg_loss