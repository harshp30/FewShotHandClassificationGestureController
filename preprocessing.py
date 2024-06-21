import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
input_data_path = '/Users/harsh/Desktop/HandGestureController/data/Hands'
labels_file_path = '/Users/harsh/Desktop/HandGestureController/data/HandInfo.csv'
output_data_path = '/Users/harsh/Desktop/HandGestureController/training_data/'

# Ensure directories exist
def ensure_dirs(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

# Function to balance and split data
def balance_and_split_data(input_data_path, labels_file_path, output_data_path):
    # Ensure training, validation, and testing directories exist
    ensure_dirs([
        os.path.join(output_data_path, 'train', 'images'),
        os.path.join(output_data_path, 'val', 'images'),
        os.path.join(output_data_path, 'test', 'images')
    ])
    
    # Read labels from CSV file
    df = pd.read_csv(labels_file_path)

    # Ensure only 30 samples per class
    balanced_df = df.groupby('aspectOfHand').apply(lambda x: x.sample(n=30, random_state=42)).reset_index(drop=True)

    # Create a dictionary to map each unique class to an integer
    class_mapping = {label: idx for idx, label in enumerate(balanced_df['aspectOfHand'].unique())}
    
    # Map the classes to their respective integer values
    balanced_df['mappedLabel'] = balanced_df['aspectOfHand'].map(class_mapping)

    # Split into training (20), validation (5), and testing (5) sets
    train_df, temp_df = train_test_split(balanced_df, test_size=0.3333, stratify=balanced_df['aspectOfHand'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['aspectOfHand'], random_state=42)

    # Function to save data
    def save_data(df, split):
        set_dir = os.path.join(output_data_path, split, 'images')
        labels_path = os.path.join(output_data_path, split, 'labels.txt')
        with open(labels_path, 'w') as f:
            for index, row in df.iterrows():
                image_name = row['imageName']
                label = row['mappedLabel']
                
                # Copy image
                shutil.copy(os.path.join(input_data_path, image_name), os.path.join(set_dir, image_name))
                
                # Save label
                f.write(f'{image_name} {label}\n')

    # Save the split data
    save_data(train_df, 'train')
    save_data(val_df, 'val')
    save_data(test_df, 'test')

    print(f'Data split into training, validation, and testing sets and saved to {output_data_path}.')

# Execute the data balancing and splitting
balance_and_split_data(input_data_path, labels_file_path, output_data_path)
