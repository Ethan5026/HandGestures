import json
import os
import numpy as np

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

    return np.array(X), np.array(Y)

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
        for x in X:
            trainingData.append(x)
        for y in Y:
            trainingLabels.append(y)

    for filepath in testingFiles:
        X, Y = GetDataLabels(f"HaGRID/test/{filepath}")
        for x in X:
            testingData.append(x)
        for y in Y:
            testingLabels.append(y)

    return trainingData, trainingLabels, testingData, testingLabels

if __name__ == '__main__':
    trainingData, trainingLabels, testingData, testingLabels = FullDataLabels()
    SVMLinear(trainingData, trainingLabels, testingData, testingLabels)

