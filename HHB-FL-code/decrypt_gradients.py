import os
import pickle
import time
import logging
from tqdm import tqdm
from phe import paillier
from concurrent.futures import ThreadPoolExecutor, as_completed
import tenseal as ts

# 配置日志记录
logging.basicConfig(level=logging.INFO)

# 加载并检查加密数据
data_dir = r'E:\桌面\实验\Client_data'
weights_files = [os.path.join(data_dir, f'encrypted_weights_participant_{i}.pkl') for i in range(1, 11)]
biases_files = [os.path.join(data_dir, f'encrypted_biases_participant_{i}.pkl') for i in range(1, 11)]

encrypted_weights = []
encrypted_biases = []

# 验证权重数据
for participant_id, weights_file in enumerate(weights_files, 1):
    with open(weights_file, 'rb') as wf:
        weights = pickle.load(wf)
        logging.info(f"Participant {participant_id} - 权重类型: {type(weights)}, 数据内容: {weights[:3]}")  # 打印前三个元素用于检查
        encrypted_weights.append(weights)

# 验证偏置数据
for participant_id, biases_file in enumerate(biases_files, 1):
    with open(biases_file, 'rb') as bf:
        biases = pickle.load(bf)
        logging.info(f"Participant {participant_id} - 偏置类型: {type(biases)}, 数据内容: {biases[:3]}")  # 打印前三个元素用于检查
        encrypted_biases.append(biases)

# 验证数据类型
def validate_encrypted_data(data, expected_type):
    for participant_id, item in enumerate(data, 1):
        for grad in item:
            if isinstance(grad, list):
                for g in grad:
                    if not isinstance(g, expected_type):
                        logging.error(f"Participant {participant_id} - Expected {expected_type}, got {type(g)}")
                        exit()
            elif not isinstance(grad, expected_type):
                logging.error(f"Participant {participant_id} - Expected list of {expected_type}, got {type(grad)}")
                exit()

validate_encrypted_data(encrypted_weights, paillier.EncryptedNumber)
validate_encrypted_data(encrypted_biases, paillier.EncryptedNumber)

print("数据验证完成。")

# 解密Paillier加密的梯度
def decrypt_single_gradient(grad, private_key):
    decrypted = []
    for g in grad:
        if isinstance(g, paillier.EncryptedNumber):
            decrypted.append(private_key.decrypt(g))
        else:
            logging.error(f"Expected EncryptedNumber, got {type(g)}")
            raise TypeError(f"Expected EncryptedNumber, got {type(g)}")
    return decrypted

def decrypt_paillier(encrypted_gradients, private_key):
    decrypted_gradients = []
    for grad in encrypted_gradients:
        if isinstance(grad, list):
            decrypted_gradients.append(decrypt_single_gradient(grad, private_key))
        else:
            decrypted_gradients.append(private_key.decrypt(grad))
    return decrypted_gradients

# 解密Paillier加密的偏置
def decrypt_paillier_biases(encrypted_biases, private_key):
    decrypted_biases = []
    for bias in encrypted_biases:
        if isinstance(bias, list):
            decrypted_biases.append(decrypt_single_gradient(bias, private_key))
        else:
            decrypted_biases.append(private_key.decrypt(bias))
    return decrypted_biases

if __name__ == "__main__":
    # 加载Paillier私钥
    paillier_private_key_file = os.path.join(data_dir, 'paillier_private_key.pkl')
    with open(paillier_private_key_file, 'rb') as key_file:
        paillier_private_key = pickle.load(key_file)

    # 加载CKKS上下文
    ckks_context_file = os.path.join(data_dir, 'ckks_context.pkl')
    ckks_context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    with open(ckks_context_file, 'rb') as context_file:
        ckks_context.load(context_file.read())

    # 解密时间记录
    decryption_times = {
        "paillier_decryption_time": [],
        "ckks_decryption_time": []
    }

    # 总时间记录
    start_total_time = time.time()

    # 解密权重
    for participant_id, weights in enumerate(encrypted_weights, 1):
        start_time = time.time()
        logging.info(f"开始解密Participant {participant_id}的Paillier加密的权重梯度...")
        try:
            decrypted_weights = decrypt_paillier(weights, paillier_private_key)
            elapsed_time = time.time() - start_time
            decryption_times["paillier_decryption_time"].append(elapsed_time)
            logging.info(f"Participant {participant_id} - 解密Paillier加密的权重梯度完成，用时 {elapsed_time:.2f} 秒")
        except Exception as e:
            logging.error(f"Participant {participant_id} - 解密Paillier加密的权重梯度时发生错误: {e}")
            exit()

    # 解密偏置
    for participant_id, biases in enumerate(encrypted_biases, 1):
        start_time = time.time()
        logging.info(f"开始解密Participant {participant_id}的CKKS加密的偏置梯度...")
        try:
            decrypted_biases = decrypt_paillier_biases(biases, paillier_private_key)
            elapsed_time = time.time() - start_time
            decryption_times["ckks_decryption_time"].append(elapsed_time)
            logging.info(f"Participant {participant_id} - 解密CKKS加密的偏置梯度完成，用时 {elapsed_time:.2f} 秒")
        except Exception as e:
            logging.error(f"Participant {participant_id} - 解密CKKS加密的偏置梯度时发生错误: {e}")
            exit()

    # 总时间记录
    total_elapsed_time = time.time() - start_total_time
    decryption_times["total_decryption_time"] = total_elapsed_time
    logging.info(f"解密过程总时间：{total_elapsed_time:.2f} 秒")

    # 保存解密后的梯度
    with open(os.path.join(data_dir, 'decrypted_weights.pkl'), 'wb') as wf:
        pickle.dump(decrypted_weights, wf)
    with open(os.path.join(data_dir, 'decrypted_biases.pkl'), 'wb') as bf:
        pickle.dump(decrypted_biases, bf)
    logging.info("解密后的梯度已保存")

    # 保存解密时间
    with open(os.path.join(data_dir, 'decryption_times.pkl'), 'wb') as f:
        pickle.dump(decryption_times, f)
    logging.info("解密时间已保存")
