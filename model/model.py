import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class CNNModel(BaseModel):
    def __init__(self, input_dim=1, hidden_dim=16, output_dim=10, kernel_size=5, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=kernel_size)
        self.conv2_drop = nn.Dropout2d(dropout)
        self.conv3 = nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=kernel_size)
        self.conv3_drop = nn.Dropout2d(dropout)
        self.fc_layer = nn.Linear(hidden_dim*4*3*3, output_dim)  # fully connected layer
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))  # [BATCH_SIZE, hidden_dim, 24, 24]
        x = F.relu(self.conv2(x))  # [BATCH_SIZE, hidden_dim*2, 20, 20]
        x = F.max_pool2d(x, 2)     # [BATCH_SIZE, hidden_dim*2, 10, 10]
        x = self.conv2_drop(x)
        x = F.relu(self.conv3(x))  # [BATCH_SIZE, hidden_dim*4, 6, 6]
        x = F.max_pool2d(x, 2)     # [BATCH_SIZE, hidden_dim*4, 3, 3]
        x = self.conv3_drop(x)
        x = x.view(x.size(0), -1)  # [BATCH_SIZE, hidden_dim*4*3*3]
        x = self.fc_layer(x)       # [BATCH_SIZE, output_dim]
        x = self.softmax(x)        # [BATCH_SIZE, output_dim]
        return x

class RNNModel(BaseModel):
    def __init__(self, input_dim=28, hidden_dim=128, output_dim=10, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x의 형태: [batch_size, channel, height, width]
        x = x.squeeze(1)  # [batch_size, height, width]
        x = x.permute(0, 2, 1)  # [batch_size, width, height]
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return F.log_softmax(out, dim=1)

class LSTMModel(BaseModel):
    def __init__(self, input_dim=28, hidden_dim=128, output_dim=10, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x의 형태: [batch_size, channel, height, width]
        x = x.squeeze(1)  # [batch_size, height, width]
        x = x.permute(0, 2, 1)  # [batch_size, width, height]
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return F.log_softmax(out, dim=1)
    
class GRUModel(BaseModel):
    def __init__(self, input_dim=28, hidden_dim=128, output_dim=10, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x의 형태: [batch_size, channel, height, width]
        # print("??")
        x = x.squeeze(1)  # [batch_size, height, width]
        x = x.permute(0, 2, 1)  # [batch_size, width, height]
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return F.log_softmax(out, dim=1)
    
def get_model(model_type, **kwargs):
    if model_type == "CNNModel":
        return CNNModel(**kwargs)
    elif model_type == "RNNModel":
        return RNNModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
