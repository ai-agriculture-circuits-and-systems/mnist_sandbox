import torch
import pytest
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.model_factory import ModelFactory
import time

# Global variables for dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to standard size
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@pytest.fixture(scope="session")
def train_loader():
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    return DataLoader(train_dataset, batch_size=64, shuffle=True)

@pytest.fixture(scope="session")
def test_loader():
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    return DataLoader(test_dataset, batch_size=64, shuffle=False)

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_evaluate_model(model_name, train_loader, test_loader, device):
    print(f"\n{'='*50}")
    print(f"Testing model: {model_name}")
    print(f"{'='*50}")
    
    # Create model
    model = ModelFactory.create_model(model_name)
    print(f"Model created: {model.get_model_name()}")
    
    # Training setup
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print(f"\nTraining {model_name} for 1 epoch...")
    start_time = time.time()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'[{model_name}] Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluation
    print(f"\nEvaluating {model_name}...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"{'='*50}\n")
    
    return accuracy

@pytest.mark.parametrize("model_name", ModelFactory.get_available_models())
def test_model(model_name, train_loader, test_loader, device):
    """Test a single model's training and evaluation."""
    try:
        accuracy = train_and_evaluate_model(model_name, train_loader, test_loader, device)
        assert accuracy > 0, f"Model {model_name} achieved 0% accuracy"
        assert accuracy <= 100, f"Model {model_name} achieved impossible accuracy > 100%"
    except Exception as e:
        pytest.fail(f"Error testing {model_name}: {str(e)}") 