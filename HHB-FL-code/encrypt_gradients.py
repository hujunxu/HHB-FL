import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import tenseal as ts
import logging
import os
from phe import paillier
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from multiprocessing import cpu_count  # 导入 cpu_count

logging.basicConfig(level=logging.INFO)

# 定义LSTM模型类
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

# 添加拉普拉斯噪声函数，确保噪声和张量在同一设备上
def add_laplace_noise(tensor, scale=1.0):
    noise = torch.distributions.laplace.Laplace(0, scale).sample(tensor.size()).to(tensor.device)
    return tensor + noise

# 使用Paillier加密权重
def encrypt_weight_with_paillier(weight, public_key):
    return public_key.encrypt(weight)

def encrypt_weights_with_paillier(weights, public_key, max_workers):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(encrypt_weight_with_paillier, w, public_key) for w in weights]
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Encrypting weights with Paillier"):
            results.append(future.result())
    return results

# 使用CKKS加密偏置
def encrypt_biases_with_ckks(biases, context):
    return ts.ckks_vector(context, biases.cpu().numpy())

# 自定义 CKKSVector 序列化函数
def serialize_ckksvector(vector):
    return vector.serialize()

# 自定义 CKKSVector 反序列化函数
def deserialize_ckksvector(serialized_vector, context):
    return ts.ckks_vector_from(context, serialized_vector)

if __name__ == "__main__":
    num_participants = 10  # 参与者数量
    data_dir = r"E:\桌面\实验\Client_data"  # 数据目录

    # 设置Paillier加密
    public_key, private_key = paillier.generate_paillier_keypair()

    # 设置CKKS全同态加密
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=16384, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2 ** 40
    context.generate_galois_keys()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logging.info("Using CUDA (GPU) for training")
    else:
        logging.info("Using CPU for training")

    max_workers = max(1, cpu_count() // 2)  # 减少并行处理的数量

    for participant_id in range(1, num_participants + 1):
        logging.info(f"Processing participant {participant_id}")
        model_path = os.path.join(data_dir, f"local_trained_lstm_model_participant_{participant_id}.pth")

        if not os.path.exists(model_path):
            logging.warning(f"缺少文件: {model_path}")
            continue

        model = LSTMModel().to(device)

        # 加载参与者训练好的模型
        model.load_state_dict(torch.load(model_path, map_location=device))

        for param in model.parameters():
            param.requires_grad = True

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        loss_fn = nn.MSELoss()

        # 模拟一个训练步骤以获取梯度
        train_path = os.path.join(data_dir, f"train_data_participant_{participant_id}.csv")
        data = pd.read_csv(train_path)

        X_train = data[['longitude', 'latitude', 'Length', 'Width', 'Draft']].values
        y_train = data[['target_longitude', 'target_latitude']].values
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)  # 增加 num_workers

        model.train()
        logging.info(f"开始训练参与者 {participant_id} 的模型...")
        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        logging.info(f"参与者 {participant_id} 的模型训练完成")

        encrypted_gradients = {}

        # 对每个参数（权重和偏置）进行加密
        for name, param in model.named_parameters():
            if 'weight' in name:
                logging.info(f"Encrypting weights for {name}")
                # 对权重加入拉普拉斯噪声，然后进行Paillier加密
                noisy_weights = add_laplace_noise(param.grad)
                encrypted_weights = encrypt_weights_with_paillier(noisy_weights.view(-1).tolist(), public_key, max_workers)
                encrypted_gradients[name] = encrypted_weights
            elif 'bias' in name:
                logging.info(f"Encrypting biases for {name}")
                # 对偏置直接进行CKKS全同态加密
                encrypted_biases = encrypt_biases_with_ckks(param.grad, context)
                encrypted_gradients[name] = serialize_ckksvector(encrypted_biases)

        # 分开保存加密的权重和偏置
        encrypted_weights_path = os.path.join(data_dir, f"encrypted_weights_participant_{participant_id}.pkl")
        encrypted_biases_path = os.path.join(data_dir, f"encrypted_biases_participant_{participant_id}.pkl")

        with open(encrypted_weights_path, 'wb') as f:
            pickle.dump({k: v for k, v in encrypted_gradients.items() if isinstance(v, list)}, f)

        with open(encrypted_biases_path, 'wb') as f:
            pickle.dump({k: v for k, v in encrypted_gradients.items() if isinstance(v, bytes)}, f)

        logging.info(f"参与者 {participant_id} 的加密权重已保存到 {encrypted_weights_path}")
        logging.info(f"参与者 {participant_id} 的加密偏置已保存到 {encrypted_biases_path}")
