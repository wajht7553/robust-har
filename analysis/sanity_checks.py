
import os
import sys
import json
import hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import dataset_loader



# load raw processed data
X = np.load('dataset/processed_acc_gyr/X.npy')
y = np.load('dataset/processed_acc_gyr/y.npy')
with open('dataset/processed_acc_gyr/subject_index.json','r') as f:
    subj_idx = json.load(f)
print('X.shape, y.shape, unique labels:', X.shape, y.shape, np.unique(y))

# Check whether any label/subject id looks like a channel (range per channel)
print('per-channel min/max:')
print(X.min(axis=(0,1)), X.max(axis=(0,1)))

# Quick check whether any channel equals labels (possible leak)
for c in range(X.shape[2]):
    # compare flattened channel values to labels (requires alignment to windows)
    ch_vals = X[:,:,c].mean(axis=1)  # per-window summary
    if np.any(np.isin(ch_vals, np.unique(y))):
        print('channel', c, 'contains values overlapping label set (possible leak)')

# Use LOSO splitter to check for exact duplicates between train/test per fold
splitter = dataset_loader.LOSOSplitter('dataset/processed_acc_gyr')
for subj in splitter.subjects[:]:
    X_train, y_train, X_test, y_test = splitter.get_train_test_split(subj)
    def hashes(Xarr):
        return set(hashlib.md5(r.tobytes()).hexdigest() for r in Xarr.reshape(len(Xarr),-1))
    h_train = hashes(X_train)
    h_test = hashes(X_test)
    overlap = len(h_train & h_test)
    if overlap:
        print(f'Overlap for {subj}: {overlap} identical windows')
    else:
        print(f'No exact overlap for {subj}')

