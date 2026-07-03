from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import joblib

app = FastAPI()

SAMPLING_LEN = 600
NUM_CHANNELS = 6

class SelfAttention(torch.nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.query = torch.nn.Linear(feature_dim, feature_dim)
        self.key = torch.nn.Linear(feature_dim, feature_dim)
        self.value = torch.nn.Linear(feature_dim, feature_dim)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (x.shape[-1] ** 0.5))
        return torch.matmul(attn, V)

class DSCS(torch.nn.Module):
    def __init__(self, input_length, num_channels=6):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(num_channels, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool1d(2)
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.attention = SelfAttention(feature_dim=64)
        self.bn = torch.nn.BatchNorm1d(128)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(128, 256)
        self.output = torch.nn.Linear(256, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.global_pool(x)
        features = x.squeeze(-1)
        attended = self.attention(features)
        combined = torch.cat([features, attended], dim=1)
        out = self.bn(combined.unsqueeze(-1)).squeeze(-1)
        out = self.dropout(out)
        out = torch.relu(self.fc(out))
        out = self.output(out)
        return out

model = DSCS(input_length=SAMPLING_LEN, num_channels=NUM_CHANNELS)
model.load_state_dict(torch.load('dscs_fall_model_sisfall.pth', map_location='cpu'))
model.eval()
scaler = joblib.load('scaler_sisfall.pkl')

class InputData(BaseModel):
    values: list[float]

@app.get('/')
def root():
    return {'status': 'ok', 'shape': [SAMPLING_LEN, NUM_CHANNELS]}

@app.post('/predict')
def predict(data: InputData):
    arr = np.array(data.values, dtype=float)
    if arr.size != SAMPLING_LEN * NUM_CHANNELS:
        return {'error': f'Expected {SAMPLING_LEN * NUM_CHANNELS} values, got {arr.size}'}
    x = arr.reshape(1, SAMPLING_LEN, NUM_CHANNELS)
    x = scaler.transform(x.reshape(-1, NUM_CHANNELS)).reshape(1, SAMPLING_LEN, NUM_CHANNELS)
    x = torch.tensor(x, dtype=torch.float32).permute(0, 2, 1)
    with torch.no_grad():
        out = model(x)
        pred = int(out.argmax(dim=1).item())
        prob = torch.softmax(out, dim=1)[0, pred].item()
    return {'prediction': pred, 'label': 'Fall' if pred == 1 else 'ADL', 'confidence': round(prob, 4)}