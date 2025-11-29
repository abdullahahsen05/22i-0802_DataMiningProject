import numpy as np
import torch
import os
import pandas as pd
import ast
from torch.utils.data import Dataset

def get_available_channels(data_dir='data', limit=4):
    """
    Scans the data/train folder and returns the first 'limit' channel IDs.
    """
    train_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(train_dir):
        return ['P-1'] # Fallback
    
    # List all .npy files
    files = [f for f in os.listdir(train_dir) if f.endswith('.npy')]
    # Remove extension to get IDs (e.g., "P-1.npy" -> "P-1")
    channels = [f.replace('.npy', '') for f in files]
    
    # Sort to ensure consistency (P-1, P-2, etc.)
    channels.sort()
    return channels[:limit]

def load_smap_data(data_dir='data', channel_id='P-1'):
    """
    Loads specific channel data and labels.
    """
    # 1. Load the raw .npy files
    train_path = os.path.join(data_dir, 'train', f'{channel_id}.npy')
    test_path = os.path.join(data_dir, 'test', f'{channel_id}.npy')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Could not find {train_path}")

    train_data = np.load(train_path)
    test_data = np.load(test_path)
    
    # 2. Load the labels
    csv_path = 'labeled_anomalies.csv'
    test_labels = np.zeros(len(test_data)) 
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        row = df[df['chan_id'] == channel_id]
        
        if not row.empty:
            anomaly_str = row.iloc[0]['anomaly_sequences']
            try:
                anomalies = ast.literal_eval(anomaly_str)
                for start, end in anomalies:
                    start, end = int(start), int(end)
                    test_labels[start:min(end, len(test_labels))] = 1
            except:
                pass
    
    return train_data, test_data, test_labels

class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size=100):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        return self.data[idx : idx + self.window_size]

def geometric_masking(x, mask_ratio=0.2, p_geom=0.3):
    B, L, D = x.shape
    mask = torch.ones_like(x)
    for b in range(B):
        idx = 0
        while idx < L:
            if np.random.rand() < mask_ratio:
                length = np.random.geometric(p_geom)
                end = min(idx + length, L)
                mask[b, idx:end, :] = 0
                idx = end
            else:
                idx += 1
    return x * mask, mask