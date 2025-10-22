from torch.utils.data import Dataset
import os
from PIL import Image

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


# Find person in images sequence and then rank them based by most recent sequences found, If many sequences which are found using 4 consecutive images, sort them by most recent ones and by similarity assigning weights of importance to both



