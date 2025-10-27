import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle

# 列出 Client_data 目录中的所有文件
data_dir = r'E:\桌面\实验\Client_data'
print("Client_data directory contents:")
print(os.listdir(data_dir))

# 加载加密时间数据
encryption_times = np.load(os.path.join(data_dir, 'encryption_times.npy'))

# 加载解密时间数据
with open(os.path.join(data_dir, 'decryption_times.pkl'), 'rb') as f:
    decryption_times = pickle.load(f)

# 加载聚合时间数据
with open(os.path.join(data_dir, 'aggregation_times.pkl'), 'rb') as f:
    aggregation_times = pickle.load(f)

# 加载通信开销数据
communication_costs = []
for i in range(1, 11):
    communication_data_path = os.path.join(data_dir, f'communication_cost_participant_{i}.csv')
    if not os.path.exists(communication_data_path):
        print(f"File not found: {communication_data_path}")
        communication_costs.append(0)
        continue
    communication_data = pd.read_csv(communication_data_path)
    print(f"Participant {i} - Communication Data Columns: {communication_data.columns}")
    print(communication_data.head())  # 打印前几行数据用于检查实际列名
    # 修改此行以匹配实际列名
    if 'communication_cost' in communication_data.columns:
        communication_cost = communication_data['communication_cost'].sum()  # 修改列名为实际的列名
    else:
        print(f"'communication_cost' column not found in {communication_data_path}")
        communication_cost = 0
    communication_costs.append(communication_cost)

# 确保所有时间数据是相同长度的
num_participants = len(encryption_times)

# 转换所有时间为ms
encryption_times_ms = encryption_times * 1000  # 假设encryption_times单位是秒
decryption_times_ms = np.array(decryption_times["total_decryption_time"]) * 1000  # 假设解密时间单位是秒
aggregation_times_ms = np.array(aggregation_times) * 1000  # 假设聚合时间单位是秒

# 计算每个参与者的总开销时间
total_times_ms = encryption_times_ms + decryption_times_ms + aggregation_times_ms

# 绘制总开销时间图表
plt.figure(figsize=(10, 6))
participants = range(1, num_participants + 1)
plt.plot(participants, total_times_ms, marker='o', linestyle='-', label='Total Time')
plt.xlabel('Participants')
plt.ylabel('Time (ms)')
plt.title('Total Time per Participant')
plt.legend()
output_dir = r'E:\桌面\实验\performance_evaluation'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'total_time_plot.png'))
plt.show()

# 导出总开销时间数据到CSV文件
total_time_df = pd.DataFrame({
    'Participant': participants,
    'Total Time (ms)': total_times_ms
})
total_time_df.to_csv(os.path.join(output_dir, 'total_time.csv'), index=False)

# 计算总通信开销
total_communication_costs = np.array(communication_costs)  # 假设communication_cost单位是KB

# 检查total_communication_costs的长度
print(f"Total Communication Costs: {total_communication_costs}")
print(f"Length of Total Communication Costs: {len(total_communication_costs)}")

# 确保total_communication_costs的长度与participants匹配
if len(total_communication_costs) != num_participants:
    print("Mismatch in number of participants and communication costs. Please check the data.")
else:
    # 绘制总通信开销图表
    plt.figure(figsize=(10, 6))
    plt.plot(participants, total_communication_costs, marker='o', linestyle='-', label='Total Communication Cost')
    plt.xlabel('Participants')
    plt.ylabel('Communication Cost (KB)')
    plt.title('Total Communication Cost per Participant')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'total_communication_cost_plot.png'))
    plt.show()

    # 导出总通信开销数据到CSV文件
    communication_cost_df = pd.DataFrame({
        'Participant': participants,
        'Total Communication Cost (KB)': total_communication_costs
    })
    communication_cost_df.to_csv(os.path.join(output_dir, 'total_communication_cost.csv'), index=False)
