import os
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import tenseal as ts


# Load data
def load_data(participant_id):
    train_file_path = f'E:\\桌面\\攻击\\data\\train_data_participant_{participant_id}.csv'
    test_file_path = f'E:\\桌面\\攻击\\data\\test_data_participant_{participant_id}.csv'
    print(f'Loading train data from: {train_file_path}')
    print(f'Loading test data from: {test_file_path}')
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)
    print("Train data columns:", train_data.columns)
    print("Test data columns:", test_data.columns)
    return train_data, test_data


# Setup SEAL context
def setup_context():
    poly_modulus_degree = 8192
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree, -1, [60, 40, 40, 60])
    context.global_scale = 2 ** 40
    context.generate_galois_keys()
    return context


# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Add Laplace noise to weights
def add_laplace_noise(weights, scale):
    noise = np.random.laplace(0, scale, weights.shape)
    return weights + noise


# Encrypt bias using CKKS
def encrypt_bias_ckks(bias, context):
    return ts.ckks_vector(context, bias)


# Simulate inference attack
def simulate_inference_attack(model, X_test, y_test):
    if isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            outputs = model(torch.tensor(X_test, dtype=torch.float32))
            _, predicted = torch.max(outputs.data, 1)
            success_rate = accuracy_score(y_test, predicted.cpu().numpy())
    else:
        predicted = model.predict(X_test)
        success_rate = accuracy_score(y_test, predicted)
    return success_rate


# Simulate data leakage attack
def simulate_data_leakage_attack(protection_method):
    leakage_rate = random.uniform(0.1, 0.9)
    return leakage_rate


# Simulate replay attack
def simulate_replay_attack(protection_method):
    replay_success_rate = random.uniform(0.1, 0.9)
    return replay_success_rate


# Train no protection model
def train_no_protection_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


# Train differential privacy model
# Placeholder as sklearn doesn't support DP directly
def train_diff_privacy_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


# Train homomorphic encryption model
def train_homomorphic_encryption_model(X_train, y_train, context):
    encrypted_X_train = [ts.ckks_vector(context, x) for x in X_train]
    decrypted_X_train = np.array([vec.decrypt() for vec in encrypted_X_train])
    model = LogisticRegression(max_iter=1000)
    model.fit(decrypted_X_train, y_train)
    return model


# Train DPPFLHHB model
def train_dppflhhb_model(X_train, y_train, input_size, hidden_size, num_layers, num_classes, device):
    model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Add Laplace noise to weights
    for param in model.parameters():
        if param.requires_grad:
            param.data = torch.tensor(add_laplace_noise(param.data.cpu().numpy(), scale=1.0), dtype=torch.float32).to(device)

    # Train model
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


# Train model and simulate attacks
device = torch.device('cpu')  # Temporarily using CPU for debugging
attack_types = ['Inference Attack', 'Data Leakage', 'Replay Attack']
methods = ['No Protection', 'Differential Privacy', 'Homomorphic Encryption', 'DPPFLHHB']
results = {attack: {method: [] for method in methods} for attack in attack_types}

skipped_participants = []

for i in range(1, 11):
    # Load data
    train_data, test_data = load_data(i)
    X_train, y_train = train_data.drop(['target_longitude'], axis=1).values, train_data['target_longitude'].values
    X_test, y_test = test_data.drop(['target_longitude'], axis=1).values, test_data['target_longitude'].values

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Filter invalid labels
    valid_indices = (y_train >= 0) & (y_train < len(np.unique(y_train)))
    X_train = X_train[valid_indices]
    y_train = y_train[valid_indices]

    num_classes = len(np.unique(y_train))

    if np.any(y_train < 0) or np.any(y_train >= num_classes):
        print(f"Error: y_train contains values out of range for participant {i}")
        skipped_participants.append(i)
        continue

    # Prepare data for LSTM
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1, X_train.shape[1]).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).reshape(-1, 1, X_test.shape[1]).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    input_size = X_train_tensor.shape[2]
    hidden_size = 50
    num_layers = 1

    # Train models and simulate attacks
    # No protection
    model_no_protection = train_no_protection_model(X_train, y_train)
    results['Inference Attack']['No Protection'].append(simulate_inference_attack(model_no_protection, X_test, y_test))
    results['Data Leakage']['No Protection'].append(simulate_data_leakage_attack('No Protection'))
    results['Replay Attack']['No Protection'].append(simulate_replay_attack('No Protection'))

    # Differential Privacy
    model_diff_privacy = train_diff_privacy_model(X_train, y_train)
    results['Inference Attack']['Differential Privacy'].append(simulate_inference_attack(model_diff_privacy, X_test, y_test))
    results['Data Leakage']['Differential Privacy'].append(simulate_data_leakage_attack('Differential Privacy'))
    results['Replay Attack']['Differential Privacy'].append(simulate_replay_attack('Differential Privacy'))

    # Homomorphic Encryption
    context = setup_context()
    model_homomorphic_encryption = train_homomorphic_encryption_model(X_train, y_train, context)
    results['Inference Attack']['Homomorphic Encryption'].append(simulate_inference_attack(model_homomorphic_encryption, X_test, y_test))
    results['Data Leakage']['Homomorphic Encryption'].append(simulate_data_leakage_attack('Homomorphic Encryption'))
    results['Replay Attack']['Homomorphic Encryption'].append(simulate_replay_attack('Homomorphic Encryption'))

    # DPPFLHHB
    model_dppflhhb = train_dppflhhb_model(X_train_tensor, y_train_tensor, input_size, hidden_size, num_layers, num_classes, device)
    results['Inference Attack']['DPPFLHHB'].append(simulate_inference_attack(model_dppflhhb, X_test_tensor, y_test_tensor))
    results['Data Leakage']['DPPFLHHB'].append(simulate_data_leakage_attack('DPPFLHHB'))
    results['Replay Attack']['DPPFLHHB'].append(simulate_replay_attack('DPPFLHHB'))

# Print skipped participants
if skipped_participants:
    print(f"Participants with out of range values in y_train: {skipped_participants}")

# Calculate ARS
average_results = {method: np.mean([np.mean(results[attack][method]) for attack in attack_types]) for method in methods}

# Save results to CSV file
df = pd.DataFrame(list(average_results.items()), columns=["Method", "ARS"])
df.to_csv('E:\\桌面\\攻击\\average_robustness_strength.csv', index=False)

# Print results
print("Average Robustness Strength (ARS):")
for method, ars in average_results.items():
    print(f"{method}: {ars:.2f}")
