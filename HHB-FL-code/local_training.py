import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import logging
import os

# 配置日志记录
logging.basicConfig(level=logging.INFO)


# 定义LSTM模型类
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=128, output_size=2):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=2, batch_first=True)
        # 定义全连接层
        self.fc = nn.Linear(hidden_layer_size, 64)
        self.out = nn.Linear(64, output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(2, x.size(0), self.hidden_layer_size).to(x.device)
        c0 = torch.zeros(2, x.size(0), self.hidden_layer_size).to(x.device)
        # 通过LSTM层
        out, _ = self.lstm(x, (h0, c0))
        # 通过全连接层
        out = self.fc(out[:, -1, :])
        # 最后一层输出
        out = self.out(out)
        return out


# 定义训练函数
def train(model, train_loader, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0  # 初始化累积损失
        for inputs, targets in train_loader:
            # 将数据移动到计算设备（CPU或GPU）
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            running_loss += loss.item()  # 累加损失
        # 记录每个epoch的平均损失
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")


if __name__ == "__main__":
    num_participants = 10  # 参与者数量
    data_dir = r"E:\桌面\实验\Client_data"  # 数据目录
    initial_model_path = os.path.join(data_dir, "initialized_global_lstm_model.pth")  # 初始模型路径

    if not os.path.exists(initial_model_path):
        logging.error(f"初始化模型文件不存在: {initial_model_path}")
    else:
        # 遍历每个参与者
        for participant_id in range(1, num_participants + 1):
            train_path = os.path.join(data_dir, f"train_data_participant_{participant_id}.csv")

            if not os.path.exists(train_path):
                logging.warning(f"缺少文件: {train_path}")
                continue

            try:
                # 加载训练数据
                data = pd.read_csv(train_path)
            except Exception as e:
                logging.error(f"加载数据失败: {e}")
                continue

            try:
                # 处理训练数据
                X_train = data[['longitude', 'latitude', 'Length', 'Width', 'Draft']].values
                y_train = data[['target_longitude', 'target_latitude']].values
                X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  # 调整输入形状
            except Exception as e:
                logging.error(f"数据处理失败: {e}")
                continue

            # 选择计算设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = LSTMModel().to(device)  # 初始化模型

            try:
                # 加载初始模型状态
                model.load_state_dict(torch.load(initial_model_path))
            except Exception as e:
                logging.error(f"加载模型失败: {e}")
                continue

            # 定义损失函数和优化器
            criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

            # 创建数据集和数据加载器
            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                          torch.tensor(y_train, dtype=torch.float32))
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            start_time = time.time()  # 记录开始时间
            logging.info(f"参与者 {participant_id} 开始训练模型...")
            train(model, train_loader, criterion, optimizer)  # 训练模型
            logging.info(f"参与者 {participant_id} 的模型训练完成，用时 {time.time() - start_time:.2f} 秒")

            model_save_path = os.path.join(data_dir, f"local_trained_lstm_model_participant_{participant_id}.pth")
            try:
                # 保存训练后的模型
                torch.save(model.state_dict(), model_save_path)
                logging.info(f"参与者 {participant_id} 的模型已保存到 {model_save_path}")
            except Exception as e:
                logging.error(f"保存模型失败: {e}")
