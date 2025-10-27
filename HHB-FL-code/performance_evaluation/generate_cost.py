import os
import pickle
import pandas as pd
import numpy as np

# 设置路径
data_dir = r'E:\桌面\实验\Client_data'

# 加载现有的时间数据文件
encryption_times = np.load(os.path.join(data_dir, 'encryption_times.npy'))
with open(os.path.join(data_dir, 'decryption_times.pkl'), 'rb') as f:
    decryption_times = pickle.load(f)
with open(os.path.join(data_dir, 'aggregation_times.pkl'), 'rb') as f:
    aggregation_times = pickle.load(f)

# 假设通信开销与这些时间成比例关系
# 例如，通信开销 = 加密时间 + 解密时间 + 聚合时间
# 并转换为KB
num_participants = len(encryption_times)
communication_costs = []

total_decryption_time = decryption_times["total_decryption_time"]  # 总解密时间
decryption_time_per_participant = total_decryption_time / num_participants  # 平均到每个参与者的解密时间

for i in range(num_participants):
    encryption_time_ms = encryption_times[i] * 1000  # 假设加密时间单位为秒，转换为毫秒
    decryption_time_ms = decryption_time_per_participant * 1000  # 假设解密时间单位为秒，转换为毫秒
    aggregation_time_ms = aggregation_times[i] * 1000  # 假设聚合时间单位为秒，转换为毫秒

    # 计算通信开销，假设每毫秒时间开销为1KB
    communication_cost = encryption_time_ms + decryption_time_ms + aggregation_time_ms
    communication_costs.append(communication_cost)

    # 保存为CSV文件
    df = pd.DataFrame({
        'communication_cost': [communication_cost]  # 仅一条通信开销数据
    })
    df.to_csv(os.path.join(data_dir, f'communication_cost_participant_{i+1}.csv'), index=False)

print(f"已生成 {num_participants} 个通信开销文件，每个文件包含1条数据。")
