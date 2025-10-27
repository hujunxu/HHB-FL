import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import time
import os

logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    """
    加载数据
    """
    start_time = time.time()
    logging.info(f"开始加载数据 {file_path}...")
    data = pd.read_csv(file_path)
    logging.info(f"数据加载完成，用时 {time.time() - start_time:.2f} 秒")
    return data

def preprocess_data(data):
    """
    准备测试数据
    """
    logging.info("开始准备测试数据...")
    X = data[['longitude', 'latitude', 'Length', 'Width', 'Draft']].values
    y = data[['longitude', 'latitude']].shift(-1).ffill().values
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    logging.info(f"测试数据准备完成，X 形状: {X.shape}, y 形状: {y.shape}")
    return X, y

def evaluate_model(model, X, y):
    """
    评估模型
    """
    logging.info("开始模型评估...")
    loss = model.evaluate(X, y)
    logging.info(f"模型评估完成，损失: {loss}")
    return loss

if __name__ == "__main__":
    data_path = "test_data.csv"
    model_path = "local_trained_ais_lstm_model.h5"

    logging.info(f"数据路径: {data_path}")
    logging.info(f"模型路径: {model_path}")

    if not os.path.exists(data_path):
        logging.error(f"数据文件 {data_path} 不存在")
        exit(1)

    if not os.path.exists(model_path):
        logging.error(f"模型文件 {model_path} 不存在")
        exit(1)

    data = load_data(data_path)
    X, y = preprocess_data(data)

    model = tf.keras.models.load_model(model_path)
    loss = evaluate_model(model, X, y)
    logging.info(f"模型测试损失: {loss}")
