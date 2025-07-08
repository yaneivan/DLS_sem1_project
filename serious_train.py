import os
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet34
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import math

from models import get_recognition_model
from defenitions import ArcFaceLoss

CONFIG = {
    "data_path": "./data/celeba_aligned_top_50000",
    "checkpoint_path": "./models/train_serious.pth",
    "model_dir": "./models",
    
    "embedding_size": 512,
    "epochs": 50,
    "batch_size": 128,
    "num_workers": 8,
    
    "lr": 1e-3,
    "arcface_margin": 0.5,
    "arcface_scale": 64.0,

    "save_frequency": 1000, # Сохранять каждые N шагов (батчей)
}

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = torchvision.datasets.ImageFolder(root=CONFIG['data_path'])
    n_classes = len(dataset.classes)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataset.dataset.transform = transform_train
    val_dataset.dataset.transform = transform_val
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=CONFIG['num_workers'], 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'], 
        pin_memory=True
    )
    
    model = get_recognition_model(embedding_size=CONFIG['embedding_size']).to(device)
    loss_fn = ArcFaceLoss(
        num_classes=n_classes, 
        embedding_size=CONFIG['embedding_size'],
        margin=CONFIG['arcface_margin'],
        scale=CONFIG['arcface_scale']
    ).to(device)
    
    optimizer = optim.AdamW(
        itertools.chain(model.parameters(), loss_fn.parameters()), 
        lr=CONFIG['lr']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)

    start_epoch = 0
    global_step = 0
    best_val_accuracy = 0.0
    
    if os.path.exists(CONFIG['checkpoint_path']):
        print(f"Resuming from checkpoint: {CONFIG['checkpoint_path']}")
        checkpoint = torch.load(CONFIG['checkpoint_path'], map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        best_val_accuracy = checkpoint['best_val_accuracy']
        
    else:
        print("Starting from scratch.")

    for epoch in range(start_epoch, CONFIG['epochs']):
        model.train()
        loss_fn.train()
        
        total_train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for imgs, labels in progress_bar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            embeddings = model(imgs)
            loss = loss_fn(embeddings, labels)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            global_step += 1
            progress_bar.set_postfix(loss=loss.item())

            if global_step % CONFIG['save_frequency'] == 0:
                checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'loss_fn_state_dict': loss_fn.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_accuracy': best_val_accuracy,
                }
                torch.save(checkpoint, CONFIG['checkpoint_path'])
                print(f"\nCheckpoint saved at step {global_step}")

        model.eval()
        loss_fn.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                embeddings = F.normalize(model(imgs), p=2, dim=1)
                W_norm = F.normalize(loss_fn.W, p=2, dim=1)
                logits = torch.mm(embeddings, W_norm.T) * loss_fn.s
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        avg_train_loss = total_train_loss / len(train_loader)
        
        print(f"\nEpoch {epoch+1} done. Avg Loss: {avg_train_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"New best accuracy: {best_val_accuracy:.2f}%. Saving model.")
            
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'loss_fn_state_dict': loss_fn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_accuracy': best_val_accuracy,
            }
            torch.save(checkpoint, CONFIG['checkpoint_path'])
        
        scheduler.step()

    print("Training finished.")