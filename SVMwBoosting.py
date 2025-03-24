import os
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
from GestureSVM import GestureSVM

class SVMwBoosting:
    """Boosting ensemble of GestureSVMs using scikit-learn."""

    def __init__(self, n_estimators=5):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []
        self.trained = False

    def train(self, trainingData, trainingLabels):
        """Train the boosting SVM ensemble on training data."""
        self.trained = True
        n_samples = len(trainingLabels)
        weights = np.ones(n_samples) / n_samples  # Initialize weights equally

        for i in range(self.n_estimators):
            print(f"Training SVM {i + 1}/{self.n_estimators}")

            # Create a bootstrap sample
            X_boot, y_boot = resample(trainingData, trainingLabels)

            # Train a new GestureSVM on the bootstrap sample
            svm = GestureSVM()

            print("Computing Sample Weight")
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_boot)

            print("Fitting GestureSVM model")
            svm.train(X_boot, y_boot)

            # Predictions on the training data
            print("Predicting")
            predictions = svm.predict(trainingData)

            # Compute error rate
            err = np.sum(weights * (predictions != trainingLabels)) / np.sum(weights)

            # Calculate model weight (alpha), preventing division by zero
            alpha = 0.5 * np.log((1 - err) / max(err, 1e-10))

            # Update sample weights
            weights *= np.exp(-alpha * trainingLabels * predictions)
            weights /= np.sum(weights)  # Normalize weights

            # Store the model and alpha
            print("Store models")
            self.models.append(svm)
            self.alphas.append(alpha)

    def predict(self, testData):
        """Predict using the ensemble with weighted voting."""
        if not self.trained:
            raise RuntimeError("BoostingSVM must be trained before predicting.")

        predictions = np.array([model.predict(testData) for model in self.models])
        weighted_votes = np.dot(np.array(self.alphas), predictions)
        final_predictions = np.sign(weighted_votes)
        return final_predictions

    def test(self, testData, testLabels):
        """Evaluate the boosting SVM ensemble."""
        predictions = self.predict(testData)
        accuracy = accuracy_score(testLabels, predictions)
        confusionMatrix = confusion_matrix(testLabels, predictions)
        classificationReport = classification_report(testLabels, predictions, output_dict=True)

        self.testResults = {
            "accuracy": accuracy,
            "confusionMatrix": confusionMatrix,
            "predictions": predictions.tolist(),
            "classificationReport": classificationReport
        }
        return self.testResults

    def export(self, modelName):
        """Export each SVM in the ensemble to a file"""
        for i, model in enumerate(self.models):
            model.export(f"{modelName}_{i}")

    def graph(self, filename=None):
        """Visualize performance metrics."""
        if not self.testResults:
            raise RuntimeError("Run test() first.")

        plt.figure(figsize=(12, 6))
        sns.heatmap(self.testResults["confusionMatrix"], annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        if filename:
            os.makedirs("visuals", exist_ok=True)
            plt.savefig(f"visuals/{filename}.png")
            print(f"Graph saved as visuals/{filename}.png")
        else:
            plt.show()

        plt.close()

def BoostingSVMTrain(trainingData, trainingLabels, testData, testLabels, n_estimators=10):
    """Train, test, and graph a Boosting SVM ensemble using GestureSVMs."""
    boosting_svm = SVMwBoosting(n_estimators=n_estimators)
    boosting_svm.train(trainingData, trainingLabels)
    boosting_svm.test(testData, testLabels)
    boosting_svm.graph()
    return boosting_svm
