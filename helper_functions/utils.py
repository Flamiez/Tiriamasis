import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .object_models import ParsedImage
from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt

class PersonReIDDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_training=True, max_frame_gap=50):
        self.root_dir = root_dir
        self.is_training = is_training
        self.parsed_images = []
        self.max_frame_gap = max_frame_gap
        self.sequences_by_person = {}
        
        # Filename example 0005_c4s1_002993_01.jpg

        for filename in os.listdir(root_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                parts = filename.split(sep="_")

                parsed_image = ParsedImage(image_path=os.path.join(root_dir, filename),
                                           person_id=int(parts[0]),
                                           camera_id=parts[1],
                                           frame_id=int(parts[2]))
                self.parsed_images.append(parsed_image)
        
        self.parsed_images = np.array(self.parsed_images)
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        self._split_sequences_by_frame_gap()
    
    def _split_sequences_by_frame_gap(self):
        """
        Split sequences when frame gap exceeds max_frame_gap.
        Keep only sequences with >=10 frames.
        """
        sequences = []
        current_seq = []

        for img in self.parsed_images:
            if not current_seq:
                current_seq.append(img)
                continue

            last = current_seq[-1]
            same_person_cam = (img.person_id == last.person_id and img.camera_id == last.camera_id)
            frame_gap = img.frame_id - last.frame_id

            if same_person_cam and frame_gap <= self.max_frame_gap:
                current_seq.append(img)
            else:
                if len(current_seq) >= 10:
                    sequences.append(current_seq)
                current_seq = [img]

        if current_seq and len(current_seq) >= 10:
            sequences.append(current_seq)

        self.parsed_images = [img for seq in sequences for img in seq]

        for seq in sequences:
            pid = seq[0].person_id
            cam = seq[0].camera_id
            self.sequences_by_person.setdefault(pid, {}).setdefault(cam, []).append(seq)

        
    def __len__(self):
        return len(self.parsed_images)
    
    def __getitem__(self, idx):
        parsed_image = self.parsed_images[idx]
        image = Image.open(parsed_image.image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'person_id': parsed_image.person_id,
            'camera_id': parsed_image.camera_id,
            'frame_id': parsed_image.frame_id,
            'image_path': parsed_image.image_path
        }


def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    person_ids = torch.tensor([item['person_id'] for item in batch])
    camera_ids = [item['camera_id'] for item in batch]
    frame_ids = torch.tensor([item['frame_id'] for item in batch])
    paths = [item['image_path'] for item in batch]
    
    return {
        'image': images,
        'person_id': person_ids,
        'camera_id': camera_ids,
        'frame_id': frame_ids,
        'image_path': paths
    }

def extract_features(dataset, model, device='cuda'):
    batch_size = 32
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    features = []
    labels = []
    camera_ids = []
    frame_ids = []
    
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            images = batch['image'].to(device)
            batch_features = model(images)
            batch_features = batch_features.cpu().numpy()
            
            features.append(batch_features)
            labels.extend(batch['person_id'].numpy())
            camera_ids.extend(batch['camera_id'].numpy())
            frame_ids.extend(batch['frame_id'].numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    camera_ids = np.array(camera_ids)
    frame_ids = np.array(frame_ids)
    
    return features, labels, camera_ids, frame_ids

def prepare_data(data_dir, model, is_training=False):

    dataset = PersonReIDDataset(data_dir, is_training=is_training)
    features, labels, camera_ids = extract_features(dataset, model)
    return features, labels, camera_ids


class Market1501Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.img_paths = [os.path.join(data_dir, img) for img in os.listdir(data_dir) if img.endswith('.jpg')]
        
        self.labels = []
        for img in os.listdir(data_dir):
            if img.endswith('.jpg'):
                person_id = img.split('_')[0]
                self.labels.append(int(person_id) if person_id != '-1' else -1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def display_sequences(sequences):
    for i, seq in enumerate(sequences):
        fig, axes = plt.subplots(1, len(seq), figsize=(3 * len(seq), 3))

        for j, parsed_img in enumerate(seq):
            img = Image.open(parsed_img.image_path).convert('RGB')
            img = np.array(img)
            axes[j].imshow(img)
            axes[j].set_title(f"PID {parsed_img.person_id}\n{parsed_img.camera_id}\nF{parsed_img.frame_id}")
            axes[j].axis("off")
        plt.show()

def get_image_palette(image, mask, n_colors=5):
    masked_pixels = image[mask > 0]
    if len(masked_pixels) == 0:
        return np.zeros((n_colors, 3), dtype=np.uint8)
    
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(masked_pixels)
    colors = np.clip(kmeans.cluster_centers_.astype(np.uint8), 0, 255)
    return colors

def plot_palette(colors, ax):
    palette = np.zeros((50, 300, 3), dtype=np.uint8)
    step = 300 // len(colors)
    for i, color in enumerate(colors):
        palette[:, i * step:(i + 1) * step, :] = color
    ax.imshow(palette)
    ax.axis("off")

def display_img_mask_palette(images_list, masks_list):
    for i, (images, masks) in enumerate(zip(images_list, masks_list)):
        fig, axes = plt.subplots(3, len(images), figsize=(3 * len(images), 8))

        for j, (image, mask) in enumerate(zip(images, masks)):
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            green_overlay = np.zeros_like(mask_rgb)
            green_overlay[:, :, 1] = mask
            colors = get_image_palette(image, mask, n_colors=5)
            axes[0, j].imshow(image)
            axes[0, j].set_title(f"{j+1}")
            axes[1, j].imshow(cv2.addWeighted(image, 1, (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (0, 255, 0)).astype(np.uint8), 0.5, 0))
            plot_palette(colors, axes[2, j])

            for k in range(3):
                axes[k, j].axis("off")

        plt.tight_layout()
        plt.show()

def display_masks(images_list, masks_list):
    for i, (images, masks) in enumerate(zip(images_list, masks_list)):
        fig, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3))
        for j, (image, mask) in enumerate(zip(images, masks)):
            axes[j].imshow(cv2.addWeighted(image, 1, (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (0, 255, 0)).astype(np.uint8), 0.5, 0))
            axes[j].axis("off")
        plt.show()

def get_person_masks_for_sequence(sequence, model):
    images = []
    masks = []

    for item in sequence:
        image = cv2.cvtColor(cv2.imread(item.image_path), cv2.COLOR_BGR2RGB)
        images.append(image)
        mask_combined = get_image_mask(image, model)
        masks.append(mask_combined)

    return images, masks

def get_image_mask(image, model):
    results = model(image)
    result = results[0]

    if result.masks is not None:
        mask_combined = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for m in result.masks.data.cpu().numpy():
            mask_resized = cv2.resize(m, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_combined = np.maximum(mask_combined, mask_resized.astype(np.uint8))
    else:
        mask_combined = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    return mask_combined



# Find person in images sequence and then rank them based by most recent sequences found, If many sequences which are found lets say when using 4 consecutive images, sort them by most recent ones and by similarity assigning weights of importance to both



