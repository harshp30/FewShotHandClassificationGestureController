import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from model import HybridCNNTransformer

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, image_dir, labels_path, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.images = []
        self.labels = []
        
        with open(labels_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = parts[0]
                    label = int(parts[1])  # Updated to handle integer labels
                    self.images.append(filename)
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Use long type for class labels
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, label

# Hyperparameters
batch_size = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
test_image_dir = '/Users/harsh/Desktop/HandGestureController/training_data/test/images'
test_labels_path = '/Users/harsh/Desktop/HandGestureController/training_data/test/labels.txt'
model_save_path = '/Users/harsh/Desktop/HandGestureController/models/model.pth'

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = CustomDataset(test_image_dir, test_labels_path, transforms=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(set(test_dataset.labels))  # Determine number of classes from the dataset
model = HybridCNNTransformer(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_save_path))
model.eval()

# Evaluation function
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

if __name__ == "__main__":
    accuracy = evaluate_model(model, test_loader)
    print(f'Test Accuracy: {accuracy:.2f}%')
