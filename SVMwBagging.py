import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from GestureSVM import GestureSVM
import seaborn as sns
import pickle

class SVMwBagging:
    """A Bagging ensemble of SVMs for gesture detection"""

    def __init__(self, n_estimators=5, model=None):
        """
        Create a Bagging ensemble of SVMs.

        Parameters:
        - n_estimators: Number of SVMs in the ensemble.
        - model: Path to a pre-trained model (optional).
        """
        self.n_estimators = n_estimators
        self.models = []
        self.trained = False

        if model and os.path.exists(model):  # Load from file if model exists
            with open(model, "rb") as f:
                self.models = pickle.load(f)
                self.n_estimators = len(self.models)
                self.trained = True

    def train(self, trainingData, trainingLabels):
        """Train each SVM in the ensemble on a bootstrap sample of the training data"""
        self.trained = True
        for i in range(self.n_estimators):
            print(f"Iteration {i} of {self.n_estimators}")
            # Create a bootstrap sample
            X_boot, y_boot = resample(trainingData, trainingLabels)

            # Train a new SVM on the bootstrap sample
            svm = GestureSVM()
            svm.train(X_boot, y_boot)
            self.models.append(svm)

    def predict(self, testData):
        """Predict data using the ensemble of SVMs"""
        if not self.trained:
            raise RuntimeError("BaggingSVM must be trained on data before predicting, call train() first")

        # Get predictions from each model
        #print(len(self.models))
        predictions = np.array([model.predict(testData) for model in self.models])

        # Convert string labels to numerical if needed
        if predictions.dtype.kind in ['U', 'S']:  # If predictions are strings
            unique_labels = np.unique(predictions)
            label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
            predictions = np.vectorize(label_to_num.get)(predictions)

        # Aggregate predictions (majority voting)
        final_predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions.astype(int)  # Ensure integer type
        )

        # Convert back to original labels if needed
        if 'label_to_num' in locals():
            num_to_label = {v: k for k, v in label_to_num.items()}
            final_predictions = np.vectorize(num_to_label.get)(final_predictions)

        return final_predictions

    def test(self, testData, testLabels):
        """Test the ensemble of SVMs on test data"""
        if not self.trained:
            raise RuntimeError("BaggingSVM must be trained on data before testing, call train() first")

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
        """Export the entire ensemble to a file"""
        with open(f"{modelName}.pkl", "wb") as f:
            pickle.dump(self.models, f)

    def graph(self, filename=None):
        """Create a full report on the BaggingSVM. Can save the report to visuals directory if filename is specified"""
        if self.testResults is None:
            raise RuntimeError("No test results available. Run test() first.")

        # Set up the figure with multiple subplots
        plt.figure(figsize=(15, 10))
        sns.set(style="whitegrid")

        # 1. Confusion Matrix Heatmap
        plt.subplot(2, 2, 1)
        sns.heatmap(self.testResults["confusionMatrix"], annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # 2. Classification Metrics Bar Plot
        plt.subplot(2, 2, 2)
        metrics = self.testResults["classificationReport"]
        classes = [k for k in metrics.keys() if k not in ("accuracy", "macro avg", "weighted avg")]
        precision = [metrics[c]["precision"] for c in classes]
        recall = [metrics[c]["recall"] for c in classes]
        f1 = [metrics[c]["f1-score"] for c in classes]
        x = np.arange(len(classes))
        width = 0.2
        plt.bar(x - width, precision, width, label="Precision", color="lightcoral")
        plt.bar(x, recall, width, label="Recall", color="lightgreen")
        plt.bar(x + width, f1, width, label="F1-Score", color="lightblue")
        plt.xticks(x, classes)
        plt.title("Classification Metrics by Class")
        plt.xlabel("Class")
        plt.ylabel("Score")
        plt.legend()

        # Adjust layout and add overall title
        plt.suptitle(f"BaggingSVM Performance Report (Accuracy: {self.testResults['accuracy']:.2f})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save to file if filename is provided
        if filename:
            visuals_dir = "visuals"
            if not os.path.exists(visuals_dir):
                os.makedirs(visuals_dir)
            filepath = os.path.join(visuals_dir, f"{filename}.png")
            plt.savefig(filepath)
            print(f"Graphs saved to {filepath}")
        else:
            plt.show()

        plt.close()

def BaggingSVMLinear(trainingData, trainingLabels, testData, testLabels, n_estimators=10):
    """Create, train, test, and graph a BaggingSVM given training and test data"""
    bagging_svm = SVMwBagging(n_estimators=n_estimators)
    bagging_svm.train(trainingData=np.array(trainingData), trainingLabels=np.array(trainingLabels))
    bagging_svm.test(testData=np.array(testData), testLabels=np.array(testLabels))
    bagging_svm.graph()
    return bagging_svm