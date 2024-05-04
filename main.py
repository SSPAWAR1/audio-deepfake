#mfcc and random forest
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define paths to the "fake" and "real" folders
fake_folder = '/Users/shaunak/Desktop/audio deepfake/data/fake'
real_folder = '/Users/shaunak/Desktop/audio deepfake/data/real'

# Function to extract MFCC features from an audio file
def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return np.mean(mfcc, axis=1)

# Initialize lists to store features and labels
features, labels = [], []

# Process audio files from the "fake" folder
for filename in os.listdir(fake_folder):
    if filename.endswith(".wav"):
        audio_path = os.path.join(fake_folder, filename)
        feature = extract_features(audio_path)
        features.append(feature)
        labels.append(1)  # 1 for fake

# Process audio files from the "real" folder
for filename in os.listdir(real_folder):
    if filename.endswith(".wav"):
        audio_path = os.path.join(real_folder, filename)
        feature = extract_features(audio_path)
        features.append(feature)
        labels.append(0)  # 0 for real

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
