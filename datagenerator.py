# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 22:17:57 2023

@author: Avinash
"""

import os
import shutil
from sklearn.model_selection import train_test_split

# Set up directories
data_dir = 'C:/Users/Avinash/OneDrive/Desktop/Projects/Major Project/COVID-19'
train_dir = 'C:/Users/Avinash/OneDrive/Desktop/Projects/Major Project/COVID-19/train'
test_dir = 'C:/Users/Avinash/OneDrive/Desktop/Projects/Major Project/COVID-19/test'

# Create train and test directories
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Get subdirectories in data directory
subdirs = ['COVID','non-COVID']

# Loop through subdirectories and split into train and test sets
for subdir in subdirs:
    subdir_path = os.path.join(data_dir, subdir)
    train_subdir_path = os.path.join(train_dir, subdir)
    test_subdir_path = os.path.join(test_dir, subdir)
    
    if not os.path.exists(train_subdir_path):
        os.makedirs(train_subdir_path)
    if not os.path.exists(test_subdir_path):
        os.makedirs(test_subdir_path)
    
    # Split files in subdirectory into train and test sets
    files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    
    # Move train files to train directory
    for f in train_files:
        src = os.path.join(subdir_path, f)
        dst = os.path.join(train_subdir_path, f)
        shutil.copy(src, dst)
    
    # Move test files to test directory
    for f in test_files:
        src = os.path.join(subdir_path, f)
        dst = os.path.join(test_subdir_path, f)
        shutil.copy(src, dst)
