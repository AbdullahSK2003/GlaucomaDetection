from torchvision import transforms
import torch
import random
import numpy as np
from PIL import ImageFilter, ImageOps

class CustomGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class AdaptiveHistogramEqualization:
    def __call__(self, img):
        return ImageOps.equalize(img)

class RandomGammaCorrection:
    def __init__(self, gamma_range=(0.8, 1.2)):
        self.gamma_range = gamma_range
    
    def __call__(self, img):
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        return transforms.functional.adjust_gamma(img, gamma)

def get_enhanced_augmentation_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),  # Changed radius to kernel_size
        transforms.RandomApply([AdaptiveHistogramEqualization()], p=0.3),
        RandomGammaCorrection(gamma_range=(0.8, 1.2)),
        transforms.RandomApply([
            transforms.Lambda(lambda x: ImageOps.autocontrast(x))
        ], p=0.3),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.05
        ),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomApply([CustomGaussianNoise(0., 0.01)], p=0.2)
    ])

class AugmentationWrapper:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.dataset)

class EnhancedAugmentationWrapper(AugmentationWrapper):
    def __init__(self, dataset, transform, mixup_prob=0.2, mixup_alpha=0.2):
        super().__init__(dataset, transform)
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
    
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        
        # Apply mixup augmentation with probability
        if random.random() < self.mixup_prob:
            # Get another random sample
            idx2 = random.randint(0, len(self.dataset) - 1)
            image2, label2 = super().__getitem__(idx2)
            
            # Generate mixup weight
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            
            # Perform mixup
            image = lam * image + (1 - lam) * image2
            label = torch.tensor([label, label2, lam], dtype=torch.float32)
        
        return image, label