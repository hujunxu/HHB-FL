import torch
import torch.nn as nn
import os
import logging

logging.basicConfig(level=logging.INFO)


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


if __name__ == "__main__":
    # 定义模型保存路径
    model_save_path = r"E:\桌面\实验\Client_data\initialized_global_lstm_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # 初始化模型
        model = LSTMModel().to(device)

        # 确保目录存在
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        # 保存初始化模型
        torch.save(model.state_dict(), model_save_path)
        logging.info(f"初始化的全局模型已保存到 {model_save_path}")
    except Exception as e:
        logging.error(f"保存模型失败: {e}")
