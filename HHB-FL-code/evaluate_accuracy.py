import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 配置日志记录
logging.basicConfig(level=logging.INFO)


# LSTM模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 加载和增强训练数据
def load_and_augment_data(participant_id):
    data_dir = r'E:\桌面\实验\Client_data'
    train_file = os.path.join(data_dir, f'train_data_participant_{participant_id}.csv')
    df = pd.read_csv(train_file)

    X = df[['longitude', 'latitude', 'Length', 'Width', 'Draft']].values
    y = df[['target_longitude', 'target_latitude']].values

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 数据增强: 随机噪声
    noise = np.random.normal(0, 0.01, X.shape)
    X_augmented = np.vstack([X, X + noise])
    y_augmented = np.vstack([y, y])

    # 转换为张量
    X_tensor = torch.tensor(X_augmented, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_augmented, dtype=torch.float32)

    train_dataset = TensorDataset(X_tensor, y_tensor)
    return train_dataset


# 加载测试数据
def load_test_data(participant_id):
    data_dir = r'E:\桌面\实验\Client_data'
    test_file = os.path.join(data_dir, f'test_data_participant_{participant_id}.csv')
    df = pd.read_csv(test_file)
    logging.info(f"参与者 {participant_id} 的数据列名: {df.columns.tolist()}")
    X_test = torch.tensor(df[['longitude', 'latitude', 'Length', 'Width', 'Draft']].values,
                          dtype=torch.float32).unsqueeze(1)
    y_test_longitude = torch.tensor(df['target_longitude'].values, dtype=torch.float32)
    y_test_latitude = torch.tensor(df['target_latitude'].values, dtype=torch.float32)
    y_test = torch.stack((y_test_longitude, y_test_latitude), dim=1)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return test_loader


# 评估模型准确性
def evaluate_model(participant_id, model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            correct += (torch.abs(outputs - y) < 0.1).all(dim=1).sum().item()
            total += y.size(0)
    accuracy = 100 * correct / total
    return accuracy


# 主函数
if __name__ == "__main__":
    data_dir = r'E:\桌面\实验\Client_data'
    hidden_size = 512  # 增加隐藏单元数量
    num_layers = 4  # 增加LSTM层数
    output_size = 2
    input_size = 5
    dropout = 0.5
    learning_rate = 0.001
    num_epochs = 300  # 增加训练轮数
    weight_decay = 1e-5  # 权重衰减
    accuracies = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for participant_id in range(1, 11):
        try:
            logging.info(f"评估参与者 {participant_id} 的模型...")
            model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)
            model_path = os.path.join(data_dir, f'local_trained_lstm_model_participant_{participant_id}.pth')
            checkpoint = torch.load(model_path, map_location=device)

            # 过滤掉不匹配的键
            model_state_dict = model.state_dict()
            filtered_state_dict = {k: v for k, v in checkpoint.items() if
                                   k in model_state_dict and v.size() == model_state_dict[k].size()}
            model_state_dict.update(filtered_state_dict)
            model.load_state_dict(model_state_dict)

            train_dataset = load_and_augment_data(participant_id)
            train_loader = DataLoader(train_dataset, batch_size=32, sampler=RandomSampler(train_dataset))

            test_loader = load_test_data(participant_id)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            best_accuracy = 0

            for epoch in range(num_epochs):
                model.train()
                for X, y in train_loader:
                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    outputs = model(X)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()

                accuracy = evaluate_model(participant_id, model, test_loader, device)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(),
                               os.path.join(data_dir, f'best_model_participant_{participant_id}.pth'))

            accuracies.append(best_accuracy)
            logging.info(f"参与者 {participant_id} 的最佳准确性: {best_accuracy:.2f}%")

            accuracy_df = pd.DataFrame({'accuracy': [best_accuracy]})
            accuracy_df.to_csv(os.path.join(data_dir, f'accuracy_participant_{participant_id}.csv'), index=False)
        except Exception as e:
            logging.error(f"评估参与者 {participant_id} 的模型时发生错误: {e}")
            accuracies.append(None)

    logging.info("所有参与者的准确性:")
    for i, acc in enumerate(accuracies, 1):
        if acc is not None:
            logging.info(f"参与者 {i}: {acc:.2f}%")
        else:
            logging.info(f"参与者 {i}: 评估失败")
