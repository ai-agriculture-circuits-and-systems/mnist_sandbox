import os
import numpy as np
import struct
import matplotlib.pyplot as plt
from pathlib import Path

def read_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        # Read header
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # Read images
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
    return images

def read_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        # Read header
        magic, num_labels = struct.unpack('>II', f.read(8))
        # Read labels
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def create_test_dataset(num_images=100):
    # Create test data directory if it doesn't exist
    test_data_dir = Path('data/test_data')
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Read MNIST data
    train_images = read_mnist_images('data/MNIST/raw/train-images-idx3-ubyte')
    train_labels = read_mnist_labels('data/MNIST/raw/train-labels-idx1-ubyte')
    
    # Randomly select 100 images
    indices = np.random.choice(len(train_images), num_images, replace=False)
    test_images = train_images[indices]
    test_labels = train_labels[indices]
    
    # Save the subset
    np.save(test_data_dir / 'test_images.npy', test_images)
    np.save(test_data_dir / 'test_labels.npy', test_labels)
    
    # Save some sample visualizations
    plt.figure(figsize=(10, 10))
    for i in range(min(25, num_images)):
        plt.subplot(5, 5, i + 1)
        plt.imshow(test_images[i], cmap='gray')
        plt.title(f'Label: {test_labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(test_data_dir / 'sample_images.png')
    plt.close()
    
    print(f"Created test dataset with {num_images} images")
    print(f"Data saved in {test_data_dir}")
    print(f"Sample visualization saved as {test_data_dir}/sample_images.png")

if __name__ == '__main__':
    create_test_dataset(100) 