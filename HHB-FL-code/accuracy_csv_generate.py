import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 128).to(x.device)
        c_0 = torch.zeros(2, x.size(0), 128).to(x.device)
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(lstm_out[:, -1, :])
        return out


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=['target_longitude', 'target_latitude']).values
    y_train = train_df[['target_longitude', 'target_latitude']].values
    X_test = test_df.drop(columns=['target_longitude', 'target_latitude']).values
    y_test = test_df[['target_longitude', 'target_latitude']].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


def train_model(train_loader, test_loader, input_size, hidden_size, output_size, learning_rate=0.001, num_epochs=100):
    model = LSTMModel(input_size, hidden_size, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)

        model.eval()
        accuracy = evaluate_model(model, test_loader)
        epoch_accuracies.append(accuracy)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return model, epoch_losses, epoch_accuracies


def evaluate_model(model, test_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.unsqueeze(1))
            predicted = torch.round(outputs)
            correct += (predicted.round() == labels.round()).sum().item()
            total += labels.numel()

    accuracy = 100 * correct / total
    return accuracy


def plot_high_accuracy_participant(accuracies, save_path):
    best_idx = np.argmax([max(acc) for acc in accuracies])
    best_accuracy = accuracies[best_idx]

    filtered_epochs = np.where(np.array(best_accuracy) > 80)[0]

    plt.figure()
    if len(filtered_epochs) > 0:
        start_epoch = filtered_epochs[0]
        epochs = range(start_epoch, len(best_accuracy))
        smoothed_accuracy = pd.Series(best_accuracy[start_epoch:]).rolling(window=5).mean()
        plt.plot(epochs, smoothed_accuracy, label=f'Participant {best_idx + 1}', linestyle='-', linewidth=2.0)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('High Accuracy over 80% for Best Participant')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


base_path = "E:/桌面/实验/Client_data"
save_path = "E:/桌面/实验/performance_evaluation/high_accuracy_participant.png"
accuracy_save_path = "E:/桌面/实验/Client_data/accuracy"
epoch_accuracies = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

accuracies = []

if not os.path.exists(accuracy_save_path):
    os.makedirs(accuracy_save_path)

for i in range(1, 11):
    train_path = os.path.join(base_path, f"train_data_participant_{i}.csv")
    test_path = os.path.join(base_path, f"test_data_participant_{i}.csv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_loader, test_loader = load_data(train_path, test_path)
        model, epoch_losses, epoch_acc = train_model(train_loader, test_loader, input_size=5, hidden_size=128,
                                                     output_size=2, num_epochs=100)
        accuracy = evaluate_model(model, test_loader)
        print(f"Participant {i} Accuracy: {accuracy:.2f}%")
        accuracies.append(epoch_acc)

        # Save the accuracies for the participant
        accuracy_file_path = os.path.join(accuracy_save_path, f"participant_{i}_accuracy.csv")
        pd.DataFrame(epoch_acc).to_csv(accuracy_file_path, index=False)
    else:
        print(f"未找到参与者 {i} 的数据文件")

plot_high_accuracy_participant(accuracies, save_path)
