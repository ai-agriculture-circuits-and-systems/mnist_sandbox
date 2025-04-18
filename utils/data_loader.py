import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import numpy as np
import os
from torchvision import transforms

class MNISTDataset(Dataset):
    def __init__(self, data_path, transform=None):
        if data_path.endswith('.mat'):
            data = sio.loadmat(data_path)
            self.images = data['x'].reshape(-1, 28, 28).astype(np.float32) / 255.0
            self.labels = data['y'].ravel()
        else:  # .npy format
            self.images = np.load(data_path).astype(np.float32) / 255.0
            # For test data, we expect labels in a separate file with _labels suffix
            base_path = os.path.splitext(data_path)[0]  # Remove extension
            label_path = f"{base_path}_labels.npy"
            if not os.path.exists(label_path):
                # If _labels suffix doesn't exist, try test_labels.npy in the same directory
                dir_path = os.path.dirname(data_path)
                label_path = os.path.join(dir_path, 'test_labels.npy')
            self.labels = np.load(label_path)
            
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            # Apply transform and ensure we get a tensor with shape [channels, height, width]
            image = self.transform(image)
        else:
            # If no transform, convert to tensor with shape [channels, height, width]
            image = torch.FloatTensor(image).unsqueeze(0)
            
        return image, torch.LongTensor([label])[0]

class DataLoaderFactory:
    @staticmethod
    def get_data_loaders(train_path, test_path, batch_size=32, num_workers=4, image_size=224):
        # Create default transform that resizes images to the specified size
        default_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()  # This will convert to [channels, height, width]
        ])
        
        train_dataset = MNISTDataset(train_path, transform=default_transform)
        test_dataset = MNISTDataset(test_path, transform=default_transform)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return train_loader, test_loader 