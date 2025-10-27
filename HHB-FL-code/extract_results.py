import pandas as pd
import re
import os
import logging

logging.basicConfig(level=logging.INFO)

# 存储评估结果的列表
results = []

# 获取所有参与者的日志文件
log_files = [file for file in os.listdir('.') if file.startswith('participant_') and file.endswith('.log')]

# 检查是否找到了日志文件
if not log_files:
    logging.error("未找到参与者的日志文件")
else:
    logging.info(f"找到的日志文件: {log_files}")

# 从日志文件中提取评估结果
for log_file in log_files:
    with open(log_file, 'r', encoding='utf-8') as file:
        log_content = file.read()
        participant_id_match = re.search(r'PARTICIPANT_ID=(\d+)', log_content)
        mse_match = re.search(r'模型测试损失: ([\d\.]+)', log_content)

        if participant_id_match and mse_match:
            participant_id = int(participant_id_match.group(1))
            mse = float(mse_match.group(1))
            results.append({'Participant ID': participant_id, 'MSE': mse})
        else:
            logging.warning(f"日志文件 {log_file} 中未找到匹配的评估结果")

# 将结果保存到DataFrame中
df = pd.DataFrame(results)

# 检查是否成功提取到结果
if df.empty:
    logging.error("未能从日志文件中提取评估结果")
else:
    # 将结果保存到CSV文件中
    df.to_csv('evaluation_results.csv', index=False)
    logging.info("评估结果已保存到 evaluation_results.csv")

print("评估结果已保存到 evaluation_results.csv")
