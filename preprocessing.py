'''
Extract and cleanup data and split into train, val, test 
Also assure only 20 images per class in training split for low-shot training
'''

# Import libraries
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
path = ''
input_data_path = f'{path}/data/Hands'
labels_file_path = f'{path}/data/HandInfo.csv'
output_data_path = f'{path}/training_data/'

# Ensure directories exist
def ensure_dirs(paths):
    """
    Ensure that the directories in the provided list of paths exist.
    If a directory does not exist, create it.

    Parameters:
    paths (list): List of directory paths to check/create.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)

# Function to balance and split data
def balance_and_split_data(input_data_path, labels_file_path, output_data_path):
    """
    Balance the dataset by ensuring an equal number of samples per class,
    then split the dataset into training, validation, and test sets.

    Parameters:
    input_data_path (str): Path to the input image data.
    labels_file_path (str): Path to the CSV file containing image labels.
    output_data_path (str): Path to the output directory where the split data will be saved.
    """
    # Ensure training, validation, and testing directories exist
    ensure_dirs([
        os.path.join(output_data_path, 'train', 'images'),
        os.path.join(output_data_path, 'val', 'images'),
        os.path.join(output_data_path, 'test', 'images')
    ])
    
    # Read labels from CSV file
    df = pd.read_csv(labels_file_path)

    # Ensure only 30 samples per class
    # Group by 'aspectOfHand' and sample 30 instances from each group
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
        """
        Save the split data into the appropriate directories and create a labels file.

        Parameters:
        df (DataFrame): DataFrame containing the data to save.
        split (str): The data split ('train', 'val', or 'test') being saved.
        """
        set_dir = os.path.join(output_data_path, split, 'images')
        labels_path = os.path.join(output_data_path, split, 'labels.txt')
        with open(labels_path, 'w') as f:
            for index, row in df.iterrows():
                image_name = row['imageName']
                label = row['mappedLabel']
                
                # Copy image to the appropriate directory
                shutil.copy(os.path.join(input_data_path, image_name), os.path.join(set_dir, image_name))
                
                # Save label to the labels file
                f.write(f'{image_name} {label}\n')

    # Save the split data
    save_data(train_df, 'train')
    save_data(val_df, 'val')
    save_data(test_df, 'test')

    print(f'Data split into training, validation, and testing sets and saved to {output_data_path}.')

if __name__ == '__main__':
    # Execute the data balancing and splitting
    balance_and_split_data(input_data_path, labels_file_path, output_data_path)
