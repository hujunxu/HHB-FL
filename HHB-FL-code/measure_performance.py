# measure_performance.py

import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import pickle


class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=128, output_size=2):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, 64)
        self.out = nn.Linear(64, output_size)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_layer_size).to(x.device)
        c0 = torch.zeros(2, x.size(0), self.hidden_layer_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.out(out)
        return out


# 加载测试数据
def load_test_data(file_path):
    data = pd.read_csv(file_path)
    X_test = data[['longitude', 'latitude', 'Length', 'Width', 'Draft']].values
    y_test = data[['target_longitude', 'target_latitude']].values
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    return torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)


# 记录时间和通信成本
def record_performance():
    num_participants = 10
    data_dir = r"E:\桌面\实验\Client_data"
    output_dir = r"E:\桌面\实验\performance_evaluation"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    initialization_times = []
    encryption_times = []
    upload_times = []
    decryption_times = []
    aggregation_times = []

    for participant_id in range(1, num_participants + 1):
        model_path = os.path.join(data_dir, f"local_trained_lstm_model_participant_{participant_id}.pth")
        test_path = os.path.join(data_dir, f"test_data_participant_{participant_id}.csv")

        if not os.path.exists(model_path) or not os.path.exists(test_path):
            print(f"缺少文件: {model_path} 或 {test_path}")
            continue

        model = LSTMModel().to(device)

        # 记录初始化时间
        start_time = time.time()
        model.load_state_dict(torch.load(model_path))
        initialization_time = time.time() - start_time
        initialization_times.append(initialization_time)

        # 加载测试数据
        X_test, y_test = load_test_data(test_path)

        # 记录加密时间
        start_time = time.time()
        # 假设加密操作在这里进行
        encryption_time = time.time() - start_time
        encryption_times.append(encryption_time)

        # 记录上传时间
        start_time = time.time()
        # 假设上传操作在这里进行
        upload_time = time.time() - start_time
        upload_times.append(upload_time)

        # 记录解密时间
        start_time = time.time()
        # 假设解密操作在这里进行
        decryption_time = time.time() - start_time
        decryption_times.append(decryption_time)

        # 记录聚合时间
        start_time = time.time()
        # 假设聚合操作在这里进行
        aggregation_time = time.time() - start_time
        aggregation_times.append(aggregation_time)

    # 保存记录的时间
    performance_data = {
        'Initialization': initialization_times,
        'Encryption': encryption_times,
        'Upload': upload_times,
        'Decryption': decryption_times,
        'Aggregation': aggregation_times
    }

    with open(os.path.join(output_dir, 'performance_data.pkl'), 'wb') as f:
        pickle.dump(performance_data, f)

    print("Performance data recorded successfully.")


if __name__ == "__main__":
    record_performance()
