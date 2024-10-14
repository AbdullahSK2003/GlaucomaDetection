from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch

class GlaucomaDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = os.path.join(root_dir, mode)
        self.transform = transform
        self.classes = ['NRG', 'RG']
        self.image_paths = []
        self.labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, img_name))
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders(data_dir, batch_size=32):
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = GlaucomaDataset(data_dir, transform=train_transform, mode='train')
    val_dataset = GlaucomaDataset(data_dir, transform=val_transform, mode='validation')
    test_dataset = GlaucomaDataset(data_dir, transform=val_transform, mode='test')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader