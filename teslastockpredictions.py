

# First we need to unzip file

!unzip tesla_stock.zip

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

# Lets load the dataset

data = pd.read_csv('tsla_raw_data.csv')
prices = data['close'].values.astype(float)

# Let visualize the tesla stock prices
plt.figure(figsize=(15, 5))
plt.plot(prices)
plt.title('Tesla stock prices history')
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()

# We need to preprocess the data

scaler = MinMaxScaler(feature_range=(-1, 1))
prices_normalized = scaler.fit_transform(prices.reshape(-1, 1))

train_size = int(len(prices_normalized) * 0.9)
train_data = prices_normalized[:train_size]
test_data = prices_normalized[train_size:]

def sliding_windows(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data) - seq_length -1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)


    return np.array(xs), np.array(ys)

seq_length = 10

X_train, y_train = sliding_windows(train_data, seq_length)
X_test, y_test = sliding_windows(test_data, seq_length)


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Lets define the Model

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _= self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(1, 64, 2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Lets train the model

epochs = 100
for epoch in range(epochs):
  outputs = model(X_train)
  optimizer.zero_grad()
  loss = criterion(outputs, y_train)
  loss.backward()
  optimizer.step()
  if epoch % 10 == 0:
      print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}')

# Lets test this model

model.eval()
test_outputs = model(X_test).detach().numpy()
y_test = scaler.inverse_transform(y_test)
test_outputs = scaler.inverse_transform(test_outputs)

# lets visualize the results

plt.figure(figsize=(15, 5))
plt.plot(y_test, label="True Prices")
plt.plot(test_outputs, label="Predicted Prices")
plt.legend()
plt.title("Tesla stock Predictions")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()
