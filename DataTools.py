import json
import os
import numpy as np
import cv2

from GestureSVM import GestureSVM, SVMLinear


def GetDataLabels(filepath):
    """Get the data and labels for a gesture (dislike, like, ok, etc) and a data type (train/val/test)"""
    if not os.path.exists(filepath):
        raise RuntimeError(f"No file found at {filepath}")
    X = []  # Features (hand landmarks)
    Y = []  # Labels

    with open(filepath, "r") as f:
        json_data = json.load(f)

    for entryId, data in json_data.items():
        bboxes = data["bboxes"]
        labels = data["labels"]
        hand_landmarks = data["hand_landmarks"]

        # Ensure the number of bboxes, labels, and hand_landmarks match
        if len(bboxes) != len(labels) or len(bboxes) != len(hand_landmarks):
            print(f"Skipping entry {entryId}: Mismatched lengths.")
            continue

        # Process each hand in the entry
        for i in range(len(hand_landmarks)):
            landmarks = hand_landmarks[i]
            label = labels[i]

            # Skip if no landmarks are provided (e.g., empty list)
            if not landmarks:
                continue

            # Flatten the 21 (x, y) coordinates into a single vector
            flat_landmarks = []
            for x, y in landmarks:
                flat_landmarks.append(x)
                flat_landmarks.append(y)

            # Each hand should have 21 landmarks (42 features)
            if len(flat_landmarks) != 42:
                print(f"Skipping hand in {entryId}: Expected 42 features, got {len(flat_landmarks)}.")
                continue

            X.append(flat_landmarks)
            Y.append(label)

    return X, Y

def FullDataLabels():
    """Get all training and test data and labels from the full HaGRID dataset"""
    trainingFiles = [f for f in os.listdir("HaGRID/train") if f.endswith('.json')]
    testingFiles = [f for f in os.listdir("HaGRID/test") if f.endswith('.json')]

    trainingData = []
    trainingLabels = []
    testingData =[]
    testingLabels = []

    for filepath in trainingFiles:
        X, Y = GetDataLabels(f"HaGRID/train/{filepath}")
        print(f"Getting Data Label {filepath}")
        for x in X:
            trainingData.append(x)
        for y in Y:
            trainingLabels.append(y)

    for filepath in testingFiles:
        print(f"Getting Test Label {filepath}")
        X, Y = GetDataLabels(f"HaGRID/test/{filepath}")
        for x in X:
            testingData.append(x)
        for y in Y:
            testingLabels.append(y)
    print("Done collecting data")
    return trainingData, trainingLabels, testingData, testingLabels

#Methods for training on Images
def LoadImagesFromFolder(folder, label, img_size=(64, 64)):
    """Load images from a folder and preprocess them"""
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if not img_path.endswith(('.png', '.jpg', '.jpeg')):
            continue
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        if img is None:
            continue
        img = cv2.resize(img, img_size)  # Resize to fixed size
        img = img.flatten()  # Flatten into a 1D vector
        images.append(img)
        labels.append(label)
    return images, labels

def PrepareDatasetImages(train_folder, test_folder, img_size=(64, 64)):
    """Prepare the dataset by loading and preprocessing images"""
    # Load training data
    train_images, train_labels = LoadImagesFromFolder(train_folder, label=1, img_size=img_size)

    # Load test data
    test_images, test_labels = LoadImagesFromFolder(test_folder, label=1, img_size=img_size)

    # Convert to numpy arrays
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    return train_images, train_labels, test_images, test_labels

if __name__ == '__main__':
    trainingData, trainingLabels, testingData, testingLabels = FullDataLabels()
    SVMLinear(trainingData, trainingLabels, testingData, testingLabels)

