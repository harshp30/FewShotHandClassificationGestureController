import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from model import HybridCNNTransformer
from PIL import Image
import time

# Paths
model_save_path = '/Users/harsh/Desktop/HandGestureController/models/model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class mapping
class_mapping = {
    0: 'dorsal left',
    1: 'dorsal right',
    2: 'palmar left',
    3: 'palmar right',
}

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Initialize the model
num_classes = len(class_mapping)  # Ensure this matches your trained model
model = HybridCNNTransformer(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

# Function to perform inference on a single image
def predict_image(image_path, model, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
    return predicted.item(), probabilities

# Function to play music using osascript
def play_music():
    os.system("osascript -e 'tell application \"Music\" to play'")

# Function to pause music using osascript
def pause_music():
    os.system("osascript -e 'tell application \"Music\" to pause'")

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

start_time = time.time()
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Display the resulting frame
        cv2.imshow('Live Video', frame)

        # Capture and classify a snapshot every 3 seconds
        if time.time() - start_time >= 3:
            # Save the frame for debugging
            debug_image_path = 'debug_frame.jpg'
            cv2.imwrite(debug_image_path, frame)
            
            # Reload and display the debug frame to check correctness
            debug_image = cv2.imread(debug_image_path)
            cv2.imshow('Debug Frame', debug_image)
            
            # Perform prediction on the saved frame
            prediction_index, probabilities = predict_image(debug_image_path, model, transform, device)
            prediction_class = class_mapping[prediction_index]
            prediction_prob = probabilities[0][prediction_index].item()
            print(f'Predicted Class: {prediction_class}')
            print('Probabilities:')
            for idx, prob in enumerate(probabilities[0]):
                print(f'{class_mapping[idx]}: {prob.item():.4f}')
            
            # Control music based on prediction and probability threshold
            if prediction_prob > 0.7:
                if prediction_class == 'palmar right':
                    play_music()
                elif prediction_class in ['dorsal right']:
                    pause_music()
            
            start_time = time.time()  # Reset the start time

        # Press 'q' on the keyboard to exit the live video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
