from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import torch

class TabNetGestureClassifier:
    def __init__(self, max_epochs=100, batch_size=1024):
        self.model = TabNetClassifier(device_name='auto')
        self.label_encoder = LabelEncoder()
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.trained = False

    def train(self, trainingData, trainingLabels):
        print("Encoding labels...")
        y_encoded = self.label_encoder.fit_transform(trainingLabels)
        trainingData = trainingData.astype(np.float32)

        X_train, X_valid, y_train, y_valid = train_test_split(
            trainingData, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        print(f"Starting training on {len(X_train)} samples...")

        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_name=['valid'],
            eval_metric=['accuracy'],
            max_epochs=self.max_epochs,
            patience=10,
            batch_size=self.batch_size,
            virtual_batch_size=128,
            drop_last=False,
            from_unsupervised=None
        )
        self.trained = True

    def predict(self, sampleData):
        if not self.trained:
            raise ValueError("Model has not been trained yet.")

        if isinstance(sampleData, list):
            sampleData = np.array(sampleData)

        sampleData = sampleData.astype(np.float32)
        preds = self.model.predict(sampleData)
        decoded = self.label_encoder.inverse_transform(preds)
        return decoded

    def test(self, testingData, testingLabels):
        print("Testing...")
        testingData = testingData.astype(np.float32)
        y_true = self.label_encoder.transform(testingLabels)
        preds = self.model.predict(testingData)
        decoded_preds = self.label_encoder.inverse_transform(preds)

        acc = accuracy_score(testingLabels, decoded_preds)
        report = classification_report(testingLabels, decoded_preds, digits=4)

        print(f"Accuracy: {acc:.4f}")
        print(report)
        return {"accuracy": acc, "report": report, "predictions": decoded_preds}
