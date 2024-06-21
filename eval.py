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
        """
        Initialize the custom dataset.

        Parameters:
        image_dir (str): Directory containing the images.
        labels_path (str): Path to the labels file.
        transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.transforms = transforms
        self.images = []
        self.labels = []
        
        # Read the labels file
        with open(labels_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = parts[0]
                    label = int(parts[1])  # Convert label to integer
                    self.images.append(filename)
                    self.labels.append(label)
    
    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Parameters:
        idx (int): Index of the sample to retrieve.

        Returns:
        tuple: (image, label) where image is the transformed image and label is the corresponding label.
        """
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Convert label to tensor
        
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

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create dataset and dataloader for testing
test_dataset = CustomDataset(test_image_dir, test_labels_path, transforms=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Determine the number of classes
num_classes = len(set(test_dataset.labels))

# Initialize the model
model = HybridCNNTransformer(num_classes=num_classes).to(device)
# Load the trained model parameters
model.load_state_dict(torch.load(model_save_path))
model.eval()

# Evaluation function
def evaluate_model(model, test_loader):
    """
    Evaluate the model on the test dataset.

    Parameters:
    model (nn.Module): The trained model to evaluate.
    test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
    float: The accuracy of the model on the test dataset.
    """
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
    # Evaluate the model and print the test accuracy
    accuracy = evaluate_model(model, test_loader)
    print(f'Test Accuracy: {accuracy:.2f}%')
