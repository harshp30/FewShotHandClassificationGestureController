import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
from model import HybridResNetTransformer

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
                    label = int(parts[1])
                    self.images.append(filename)
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, label

# Hyperparameters
batch_size = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
path = ''
test_image_dir = f'{path}/training_data/test/images'
test_labels_path = f'{path}/training_data/test/labels.txt'
model_save_path = f'{path}/models/model.pth'

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
model = HybridResNetTransformer(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_save_path))
model.eval()

# Evaluation function
def evaluate_model(model, test_loader):
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    accuracy = np.mean(np.array(all_labels) == np.array(all_predictions)) * 100
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    return accuracy, precision, recall, f1, conf_matrix

if __name__ == "__main__":
    accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, test_loader)
    
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-Score: {f1:.2f}')
    print("Confusion Matrix:")
    print(conf_matrix)
