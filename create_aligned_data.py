import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw 
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from models import StackedHourglass


model = StackedHourglass(num_outputs=5, feature_size=32).to(device)
model.load_state_dict(torch.load('models/working_aligner.pth'))

# =====================
N_TOP_PEOPLE = 50000
MODEL_INPUT_SIZE = 128 
ALIGNED_OUTPUT_SIZE = (256, 256) 
VISUALIZE_EVERY_N_IMAGES = 100
# =====================

print("Этап 1: Подготовка метаданных...")
celeba_root = './data/celeba'
img_dir = os.path.join(celeba_root, 'img_align_celeba')
aligned_root = f'./data/celeba_aligned_top_{N_TOP_PEOPLE}'
debug_root = f'./data/celeba_aligned_top_{N_TOP_PEOPLE}_debug'

os.makedirs(aligned_root, exist_ok=True)
os.makedirs(debug_root, exist_ok=True)

identity_df = pd.read_csv(os.path.join(celeba_root, 'identity_CelebA.txt'), sep=' ', header=None, names=['image', 'identity'])
partition_df = pd.read_csv(os.path.join(celeba_root, 'list_eval_partition.txt'), sep=' ', header=None, names=['image', 'partition'])
df = pd.merge(identity_df, partition_df, on='image')

print(f"Фильтрация датасета для выбора топ-{N_TOP_PEOPLE} людей...")
identity_counts = df['identity'].value_counts()
top_n_identities = identity_counts.nlargest(N_TOP_PEOPLE).index
df_filtered = df[df['identity'].isin(top_n_identities)].copy().reset_index(drop=True)
print(f"Выбрано топ-{N_TOP_PEOPLE} людей. Общее количество изображений для обработки: {len(df_filtered)}")
print("-" * 30)

print("Этап 2: Определение кастомного датасета...")
class FaceProcessingDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.df, self.root_dir, self.transform = dataframe, root_dir, transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['image'])
        pil_image = Image.open(img_path).convert('RGB')
        tensor_image = self.transform(pil_image) if self.transform else pil_image
        return tensor_image, pil_image, str(row['identity']), row['image']

print("Этап 3: Определение вспомогательных функций...")

def custom_collate_fn(batch):
    tensors, pil_images, identities, filenames = [], [], [], []
    for item in batch:
        tensors.append(item[0])
        pil_images.append(item[1])
        identities.append(item[2])
        filenames.append(item[3])
    return torch.stack(tensors, 0), pil_images, identities, filenames

def are_landmarks_plausible(coords):
    try:
        le_x, le_y = coords[0]; re_x, re_y = coords[1]
        n_x, n_y = coords[2]; lm_x, lm_y = coords[3]; rm_x, rm_y = coords[4]
        if not (le_x < n_x < re_x and le_x < re_x and lm_x < rm_x): return False
        avg_eye_y = (le_y + re_y) / 2.0; avg_mouth_y = (lm_y + rm_y) / 2.0
        if not (avg_eye_y < n_y < avg_mouth_y): return False
    except IndexError: return False
    return True

def align_face(image_pil, src_points, output_size=(256, 256)):
    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h_orig, w_orig, _ = image_bgr.shape; w_out, h_out = output_size
    dst_points = np.float32([[w_out*0.35,h_out*0.3],[w_out*0.65,h_out*0.3],[w_out*0.5,h_out*0.55],[w_out*0.35,h_out*0.7],[w_out*0.65,h_out*0.7]])
    src_points_scaled = src_points.copy().astype(np.float32)
    src_points_scaled[:, 0] *= (w_orig / MODEL_INPUT_SIZE); src_points_scaled[:, 1] *= (h_orig / MODEL_INPUT_SIZE)
    M, _ = cv2.estimateAffine2D(src_points_scaled, dst_points, method=cv2.RANSAC)
    if M is None: return cv2.resize(image_bgr, output_size)
    return cv2.warpAffine(image_bgr, M, output_size, borderMode=cv2.BORDER_REPLICATE)

def create_debug_image(original_pil, pred_coords):
    """Рисует предсказанные точки на копии исходного изображения."""
    img_with_dots = original_pil.copy()
    draw = ImageDraw.Draw(img_with_dots)
    
    w_orig, h_orig = original_pil.size
    scale_x = w_orig / MODEL_INPUT_SIZE
    scale_y = h_orig / MODEL_INPUT_SIZE

    cross_size = 4
    for x, y in pred_coords:
        scaled_x = x * scale_x
        scaled_y = y * scale_y
        draw.line([(scaled_x - cross_size, scaled_y - cross_size), (scaled_x + cross_size, scaled_y + cross_size)], fill="red", width=2)
        draw.line([(scaled_x - cross_size, scaled_y + cross_size), (scaled_x + cross_size, scaled_y - cross_size)], fill="red", width=2)
        
    return img_with_dots
    
print("\nЭтап 4: Запуск процесса обработки...")
transform_for_model = transforms.Compose([transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)), transforms.ToTensor()])
processing_dataset = FaceProcessingDataset(df_filtered, img_dir, transform=transform_for_model)
processing_loader = DataLoader(processing_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

model.eval()
images_processed = 0
skipped_count = 0

with torch.no_grad():
    for tensor_batch, pil_batch, identity_batch, filename_batch in tqdm(processing_loader, desc="Aligning Faces"):
        tensor_batch = tensor_batch.to(device)
        pred_heatmaps_list = model(tensor_batch)
        pred_heatmaps = pred_heatmaps_list[-1]
        
        for i in range(tensor_batch.shape[0]):
            pil_image = pil_batch[i]; identity = identity_batch[i]; filename = filename_batch[i]

            pred_coords = []
            heatmap_h, heatmap_w = pred_heatmaps.shape[2], pred_heatmaps.shape[3]
            for k in range(pred_heatmaps.shape[1]):
                hm = pred_heatmaps[i, k].cpu().numpy()
                y, x = np.unravel_index(np.argmax(hm), (heatmap_h, heatmap_w))
                scaled_x = x * (MODEL_INPUT_SIZE / heatmap_w)
                scaled_y = y * (MODEL_INPUT_SIZE / heatmap_h)
                pred_coords.append([scaled_x, scaled_y])
            pred_coords = np.array(pred_coords)
            
            if not are_landmarks_plausible(pred_coords):
                skipped_count += 1
                continue

            aligned_image_cv2 = align_face(pil_image, pred_coords, output_size=ALIGNED_OUTPUT_SIZE)
            
            debug_image_pil = create_debug_image(pil_image, pred_coords)

            person_dir_main = os.path.join(aligned_root, identity)
            os.makedirs(person_dir_main, exist_ok=True)
            output_path_main = os.path.join(person_dir_main, filename)
            cv2.imwrite(output_path_main, aligned_image_cv2)
            
            person_dir_debug = os.path.join(debug_root, identity)
            os.makedirs(person_dir_debug, exist_ok=True)
            
            output_path_debug_aligned = os.path.join(person_dir_debug, filename)
            cv2.imwrite(output_path_debug_aligned, aligned_image_cv2)
            
            base_name, ext = os.path.splitext(filename)
            debug_filename = f"{base_name}_debug{ext}"
            output_path_debug_dots = os.path.join(person_dir_debug, debug_filename)
            debug_image_pil.save(output_path_debug_dots)
            images_processed += 1

print("\n" + "="*30)
print("             Обработка завершена!             ")
print("="*30)
print(f"Всего изображений для обработки: {len(df_filtered)}")
print(f"Успешно обработано и сохранено:  {images_processed}")
print(f"Пропущено из-за неверных точек:   {skipped_count}")
print(f"Результаты сохранены в папку: '{aligned_root}'")
print(f"Дебаг-версии сохранены в папку: '{debug_root}'")