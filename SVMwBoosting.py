import os
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
from GestureSVM import GestureSVM
from sklearn.preprocessing import LabelEncoder
import pickle
from collections import Counter

class SVMwBoosting:
    """Boosting ensemble of GestureSVMs with manual label handling."""

    def __init__(self, n_estimators=5, model=None, class_labels=None):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []
        self.trained = False
        self.class_labels = class_labels  # Manually provide class labels if needed

        if model and os.path.exists(model):
            self.load_model(model)

    def load_model(self, model_path):
        """Load model from pickle file (without LabelEncoder)."""
        with open(model_path, "rb") as f:
            self.models, self.alphas = pickle.load(f)
            self.n_estimators = len(self.models)
            self.trained = True

    def train(self, trainingData, trainingLabels):
        """Train the boosting SVM ensemble (with manual label handling)."""
        # Get unique class labels and sort them
        self.class_labels = sorted(list(set(trainingLabels)))

        n_samples = len(trainingLabels)
        weights = np.ones(n_samples) / n_samples  # Initialize weights equally

        for i in range(self.n_estimators):
            print(f"Training SVM {i + 1}/{self.n_estimators}")

            # Create a bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples)
            X_boot = trainingData[indices]
            y_boot = [trainingLabels[i] for i in indices]

            # Train a new GestureSVM on the bootstrap sample
            svm = GestureSVM()
            svm.train(X_boot, y_boot)

            # Predictions on the training data
            predictions = svm.predict(trainingData)

            # Compute error rate (manually compare labels)
            err = np.mean([1 if p != t else 0 for p, t in zip(predictions, trainingLabels)])

            # Calculate model weight (alpha)
            alpha = 0.5 * np.log((1 - err) / max(err, 1e-10))

            # Update sample weights
            weights *= np.exp(alpha * np.array([1 if p != t else 0 for p, t in zip(predictions, trainingLabels)]))
            weights /= np.sum(weights)  # Normalize weights

            # Store the model and alpha
            self.models.append(svm)
            self.alphas.append(alpha)

        self.trained = True

    def predict(self, testData):
        """Predict using weighted voting (handles string labels)."""
        if not self.trained:
            raise RuntimeError("Model must be trained before predicting.")

        # Get predictions from each model (raw string labels)
        predictions = np.array([model.predict(testData) for model in self.models])

        # If class_labels are known, use weighted voting
        if self.class_labels:
            # Convert predictions to indices for weighted voting
            pred_indices = np.array([
                [self.class_labels.index(p) for p in preds]
                for preds in predictions
            ])
            weighted_votes = np.dot(np.array(self.alphas).reshape(-1, 1), pred_indices)
            final_indices = np.argmax(weighted_votes, axis=0)
            return [self.class_labels[i] for i in final_indices]
        else:
            # Fallback: Majority voting using np.unique (works with strings)
            return [self._mode(preds) for preds in predictions.T]

    def _mode(self, arr):
        """Custom mode function for string labels."""
        counts = Counter(arr)
        return max(counts.items(), key=lambda x: x[1])[0]

    def test(self, testData, testLabels):
        """Evaluate the model (manual label comparison)."""
        predictions = self.predict(testData)

        accuracy = np.mean([1 if p == t else 0 for p, t in zip(predictions, testLabels)])
        confusion = confusion_matrix(testLabels, predictions, labels=self.class_labels)
        report = classification_report(testLabels, predictions, output_dict=True)

        self.testResults = {
            "accuracy": accuracy,
            "confusionMatrix": confusion,
            "predictions": predictions,
            "classificationReport": report
        }
        return self.testResults

    def export(self, modelName):
        """Export models and alphas (without LabelEncoder)."""
        with open(f"{modelName}.pkl", "wb") as f:
            pickle.dump((self.models, self.alphas), f)

    def graph(self, filename=None):
        """Visualize performance metrics."""
        if not hasattr(self, 'testResults'):
            raise RuntimeError("Run test() first.")

        plt.figure(figsize=(12, 6))
        sns.heatmap(
            self.testResults["confusionMatrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_labels,
            yticklabels=self.class_labels
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        if filename:
            os.makedirs("visuals", exist_ok=True)
            plt.savefig(f"visuals/{filename}.png")
            plt.close()
        else:
            plt.show()