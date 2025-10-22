from torch.utils.data import Dataset
import os
from PIL import Image
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .object_models import ParsedImage


class PersonReIDDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_training=True, max_frame_gap=50):
        self.root_dir = root_dir
        self.is_training = is_training
        self.parsed_images = []
        self.max_frame_gap = max_frame_gap
        self.sequences_by_person = {}  # {person_id: {camera_id: [ [ParsedImage,...], ... ]}}
        
        
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
        Also store them in a hashmap: sequences_by_person[person_id][camera_id] = [sequence1, sequence2, ...]
        """
        sequences = []
        current_seq = []

        for img in self.parsed_images:
            if not current_seq:
                current_seq.append(img)
                continue

            last = current_seq[-1]
            same_person_cam = (
                img.person_id == last.person_id and img.camera_id == last.camera_id
            )
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

def extract_features(dataset, model, device='cuda'):
    batch_size = 0
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
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


# Find person in images sequence and then rank them based by most recent sequences found, If many sequences which are found lets say when using 4 consecutive images, sort them by most recent ones and by similarity assigning weights of importance to both



