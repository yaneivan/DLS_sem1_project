import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import MSELoss
from tqdm import tqdm
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import numpy as np
from accelerate import Accelerator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.relu(x + residual)

class HourglassBlock(nn.Module):
    def __init__(self, in_feautres=5, num_features=8):
        super().__init__()
        self.res1 = ResidualBlock(in_feautres, num_features)
        self.res2 = ResidualBlock(num_features, num_features)
        self.res3 = ResidualBlock(num_features, num_features)
        self.res4 = ResidualBlock(num_features, num_features)
        self.resMid = ResidualBlock(num_features, num_features)
        self.conv1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.MaxPool2d(2, 2)
        self.resMid1 = ResidualBlock(num_features, num_features)
        self.resMid2 = ResidualBlock(num_features, num_features)
        self.upconv1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.upconv2 = nn.Upsample(scale_factor=2, mode="nearest")
    def forward(self, x):
        x1 = self.res1(x)
        x = self.conv1(x1)
        x2 = self.res2(x)
        x = self.conv2(x2)
        x = self.resMid(x)
        x1 = self.resMid1(x1)
        x2 = self.resMid2(x2)
        x = self.upconv1(x)
        x += x2
        x = self.res3(x)
        x = self.upconv2(x)
        x += x1
        x = self.res4(x)
        return x

class StackedHourglass(nn.Module):
    def __init__(self, num_outputs, feature_size = 8):
        super().__init__()
        self.first_conv = nn.Conv2d(3, feature_size, kernel_size=1, stride=1, padding=0)
        self.c1 = nn.Conv2d(feature_size, num_outputs, kernel_size=1, stride=1, padding=0)
        self.h1 = HourglassBlock( in_feautres=feature_size, num_features=feature_size)
        self.c2 = nn.Conv2d(feature_size, num_outputs, kernel_size=1, stride=1, padding=0)
        self.h2 = HourglassBlock(in_feautres=feature_size, num_features=feature_size)
        self.c3 = nn.Conv2d(feature_size, num_outputs, kernel_size=1, stride=1, padding=0)
        self.h3 = HourglassBlock(in_feautres=feature_size, num_features=feature_size)
        self.c4 = nn.Conv2d(feature_size, num_outputs, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        outputs = []
        x1 = self.first_conv(x)
        x = self.c1(x1)
        outputs.append(x)
        x1 = self.h1(x1)
        x = self.c2(x1)
        outputs.append(x)
        x1 = self.h2(x1)
        x = self.c3(x1)
        outputs.append(x)
        x1 = self.h3(x1)
        x = self.c4(x1)
        outputs.append(x)
        return outputs

class CelebALightweightDataset(Dataset):
    def __init__(self, root='./data', split='train', transform=None):
        self.original_dataset = torchvision.datasets.CelebA(root=root, split=split, target_type="landmarks", download=True)
        self.transform = transform
    def __len__(self):
        return len(self.original_dataset)
    def __getitem__(self, idx):
        img, landmarks_raw = self.original_dataset[idx]
        if self.transform:
            img = self.transform(img)
        landmarks = landmarks_raw.view(5, 2).float()
        landmarks[:, 0] /= 178
        landmarks[:, 1] /= 218
        return img, landmarks

def batch_landmarks_to_heatmaps(landmarks, output_size, sigma=2, device='cpu'):
    B, N, _ = landmarks.shape
    H, W = output_size
    landmarks_scaled = landmarks * torch.tensor([W, H], device=device, dtype=torch.float32).view(1, 1, 2)
    xx, yy = torch.meshgrid(torch.arange(W, device=device, dtype=torch.float32), torch.arange(H, device=device, dtype=torch.float32), indexing='xy')
    xx_b = xx.view(1, 1, H, W).expand(B, N, -1, -1)
    yy_b = yy.view(1, 1, H, W).expand(B, N, -1, -1)
    x_b = landmarks_scaled[:, :, 0].view(B, N, 1, 1)
    y_b = landmarks_scaled[:, :, 1].view(B, N, 1, 1)
    heatmaps = torch.exp(-((yy_b - y_b)**2 + (xx_b - x_b)**2) / (2 * sigma**2))
    return heatmaps

def save_progress_image(images, true_landmarks, pred_heatmaps, epoch, image_size, accelerator):
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    images = images.cpu()
    pred_heatmaps = pred_heatmaps.cpu()
    true_landmarks = true_landmarks.cpu()

    num_images_to_show = min(4, images.shape[0])
    fig, axes = plt.subplots(2, num_images_to_show, figsize=(num_images_to_show * 4, 8))
    fig.suptitle(f'Результаты после эпохи {epoch + 1}', fontsize=16)

    for i in range(num_images_to_show):
        ax = axes[0, i]
        ax.imshow(images[i].permute(1, 2, 0))
        true_pts = true_landmarks[i] * torch.tensor([image_size, image_size])
        ax.scatter(true_pts[:, 0], true_pts[:, 1], s=30, c='lime', marker='o', edgecolors='black')
        ax.set_title(f'Истинные точки (Img {i+1})')
        ax.axis('off')

        pred_coords = []
        for j in range(pred_heatmaps.shape[1]):
            hm = pred_heatmaps[i, j]
            y, x = np.unravel_index(np.argmax(hm), hm.shape)
            pred_coords.append([x, y])
        pred_coords = np.array(pred_coords)

        ax = axes[1, i]
        ax.imshow(images[i].permute(1, 2, 0))
        ax.scatter(pred_coords[:, 0], pred_coords[:, 1], s=35, c='red', marker='x')
        ax.set_title(f'Предсказания (Img {i+1})')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, f"epoch_{epoch+1:03d}_progress.png")
    plt.savefig(save_path)
    plt.close(fig)
    accelerator.print(f"Картинка с прогрессом сохранена в {save_path}")


def main():
    accelerator = Accelerator(log_with="tensorboard", project_dir="logs")
    accelerator.print("Accelerator инициализирован.")

    IMAGE_SIZE = 128
    BATCH_SIZE = 128
    NUM_EPOCHS = 40
    LEARNING_RATE = 1e-3
    CHECKPOINT_PATH = 'final_model.pth'

    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    train_dataset = CelebALightweightDataset(split='train', transform=transform)
    valid_dataset = CelebALightweightDataset(split='valid', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    accelerator.print("Датасеты и загрузчики данных созданы.")

    model = StackedHourglass(num_outputs=5, feature_size=32)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    accelerator.print("Веса модели загружены")

    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * NUM_EPOCHS, eta_min=1e-4)
    criterion = MSELoss()

    model, optimizer, scheduler, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, valid_loader
    )
    accelerator.print("Все компоненты подготовлены.")

    def compute_loss(outputs, targets):
        total_loss = 0
        for output in outputs:
            if output.shape[-2:] != targets.shape[-2:]:
                output = transforms.functional.resize(output, targets.shape[-2:], antialias=True)
            total_loss += criterion(output, targets)
        return total_loss

    accelerator.print("=== НАЧАЛО ОБУЧЕНИЯ ===")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Эпоха {epoch+1} [Train]', disable=not accelerator.is_main_process)
        for images, landmarks in progress_bar:
            heatmaps = batch_landmarks_to_heatmaps(landmarks, (IMAGE_SIZE, IMAGE_SIZE), sigma=2, device=accelerator.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = compute_loss(outputs, heatmaps)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        visualization_batch = None
        with torch.no_grad():
            progress_bar_val = tqdm(valid_loader, desc=f'Эпоха {epoch+1} [Valid]', disable=not accelerator.is_main_process)
            for i, (images, landmarks) in enumerate(progress_bar_val):
                if i == 0:
                    visualization_batch = (images.clone(), landmarks.clone())
                
                heatmaps = batch_landmarks_to_heatmaps(landmarks, (IMAGE_SIZE, IMAGE_SIZE), sigma=2, device=accelerator.device)
                outputs = model(images)
                loss = compute_loss(outputs, heatmaps)
                total_val_loss += loss.item()

        viz_images, viz_landmarks = visualization_batch
        with torch.no_grad():
            viz_pred_heatmaps = model(viz_images)[-1]
        gathered_images, gathered_landmarks, gathered_heatmaps = accelerator.gather_for_metrics(
            (viz_images, viz_landmarks, viz_pred_heatmaps)
        )
        if accelerator.is_main_process:
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(valid_loader)
            
            accelerator.print(f'Эпоха {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
            
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), CHECKPOINT_PATH)
            accelerator.print(f'Чекпоинт сохранен в {CHECKPOINT_PATH}')
            save_progress_image(
                gathered_images, gathered_landmarks, gathered_heatmaps,
                epoch, IMAGE_SIZE, accelerator
            )
        accelerator.wait_for_everyone()
    accelerator.print("=== ОБУЧЕНИЕ ЗАВЕРШЕНО! ===")


if __name__ == "__main__":
    main()