import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Added for plotting
from sklearn.metrics import precision_recall_fscore_support

from model import TransformerGenerator, Discriminator, contrastive_loss
from utils import TimeSeriesDataset, load_smap_data, geometric_masking, get_available_channels

# --- Configuration ---
WINDOW_SIZE = 100
BATCH_SIZE = 64
EPOCHS = 20        # 20 Epochs for best results
LR = 0.0005

# Find channels automatically
target_channels = get_available_channels(limit=4)
print(f"🎯 Targeting Channels: {target_channels}")

results_summary = []

for channel_id in target_channels:
    print(f"\n" + "="*40)
    print(f"🚀 STARTING CHANNEL: {channel_id}")
    print("="*40)

    # 1. Load Data
    try:
        train_raw, test_raw, test_labels = load_smap_data(channel_id=channel_id)
    except Exception as e:
        print(f"❌ Skipping {channel_id}: {e}")
        continue

    # Normalize
    train_mean = np.mean(train_raw, axis=0)
    train_std = np.std(train_raw, axis=0) + 1e-6
    train_data = (train_raw - train_mean) / train_std
    test_data = (test_raw - train_mean) / train_std

    loader = DataLoader(TimeSeriesDataset(train_data, WINDOW_SIZE), batch_size=BATCH_SIZE, shuffle=True)

    # 2. Setup Models
    feat_dim = train_raw.shape[1]
    gen = TransformerGenerator(feat_dim)
    disc = Discriminator(feat_dim, WINDOW_SIZE)
    
    opt_g = optim.Adam(gen.parameters(), lr=LR)
    opt_d = optim.Adam(disc.parameters(), lr=LR)
    mse = nn.MSELoss()
    bce = nn.BCELoss()

    # 3. Training Loop
    print(f"   Training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        gen.train()
        total_loss = 0
        
        for x in loader:
            masked_x, _ = geometric_masking(x)
            
            # Train Discriminator
            opt_d.zero_grad()
            real_out = disc(x)
            fake_out = disc(gen(masked_x)[0].detach())
            loss_d = (bce(real_out, torch.ones_like(real_out)) + 
                      bce(fake_out, torch.zeros_like(fake_out))) / 2
            loss_d.backward()
            opt_d.step()
            
            # Train Generator
            opt_g.zero_grad()
            recon, feat_recon = gen(masked_x)
            _, feat_real = gen(x)
            
            loss_g = mse(recon, x) + 0.1 * bce(disc(recon), torch.ones_like(real_out)) + 0.1 * contrastive_loss(feat_recon, feat_real)
            loss_g.backward()
            opt_g.step()
            total_loss += loss_g.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

    # 4. Evaluation & Plotting
    print("   Evaluating & Generating Plot...")
    gen.eval()
    test_loader = DataLoader(TimeSeriesDataset(test_data, WINDOW_SIZE), batch_size=1, shuffle=False)
    errors = []

    with torch.no_grad():
        for x in test_loader:
            recon, _ = gen(x)
            e = torch.mean((x - recon)**2).item()
            errors.append(e)

    errors = np.array(errors)
    # Threshold: Mean + 2 STD
    threshold = np.mean(errors) + 2 * np.std(errors)
    preds = (errors > threshold).astype(int)

    # --- PLOTTING SECTION ---
    plt.figure(figsize=(12, 6))
    plt.plot(errors, label='Reconstruction Error', color='blue', linewidth=1)
    plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
    
    # Highlight Ground Truth Anomalies
    gt = test_labels[WINDOW_SIZE:]
    preds = preds[:len(gt)]
    
    max_val = np.max(errors) if len(errors) > 0 else 1
    plt.fill_between(range(len(gt)), 0, max_val, where=(gt==1), color='green', alpha=0.3, label='Actual Anomaly')
    
    plt.title(f"Anomaly Detection: {channel_id} (F1-Score calculation)")
    plt.xlabel("Time Step")
    plt.ylabel("Error")
    plt.legend(loc='upper left')
    
    # Save plot automatically
    plot_filename = f"plot_{channel_id}.png"
    plt.savefig(plot_filename)
    plt.close() # Close to free memory
    print(f"   🎨 Saved graph to {plot_filename}")

    # Calculate Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(gt, preds, average='binary', zero_division=0)
    
    results_summary.append({
        'Channel': channel_id,
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4),
        'Threshold': round(threshold, 4)
    })
    print(f"   ✅ F1-Score: {f1:.4f}")

# --- Final Report ---
print("\n" + "#"*40)
print("📊 FINAL RESULTS TABLE")
print("#"*40)

if len(results_summary) > 0:
    df = pd.DataFrame(results_summary)
    avg_row = pd.DataFrame([{
        'Channel': 'AVERAGE',
        'Precision': round(df['Precision'].mean(), 4),
        'Recall': round(df['Recall'].mean(), 4),
        'F1-Score': round(df['F1-Score'].mean(), 4),
        'Threshold': '-'
    }])
    df_final = pd.concat([df, avg_row], ignore_index=True)
    
    print(df_final.to_string(index=False))
    df_final.to_csv("evaluation_metrics.csv", index=False)
    print("\n💾 Results saved to: evaluation_metrics.csv")
    print("   (Check your folder for the .png plots!)")
else:
    print("❌ No results found.")