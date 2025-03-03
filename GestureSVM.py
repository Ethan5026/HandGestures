import os

import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC, LinearSVC
import seaborn as sns

class GestureSVM:
    """A SVM for gesture detection"""


    def __init__(self, model=None):
        """Create the SVM with a specified kernel (default 'linear'). Set model to import a pre-trained model"""

        self.svm = LinearSVC()
        if model:
            self.trained = True
            self.svm = joblib.load(model)

    def train(self, trainingData, trainingLabels):
        """Train the SVM on training data"""
        self.trained = True
        self.svm.fit(trainingData, trainingLabels)


    def predict(self, testData):
        if self.trained == False:
            raise RuntimeError("SVM must be trained on data before predicting, call train() first")
        """Predicts data using the SVM"""
        return self.svm.predict(testData)

    def test(self, testData, testLabels):
        if self.trained == False:
            raise RuntimeError("SVM must be trained on data before tested, call train() first")
        score = self.svm.score(testData, testLabels)
        predictions = self.svm.predict(testData)
        confusionMatrix = confusion_matrix(testLabels, testLabels)
        decisionFunction = self.svm.decision_function(testData).tolist()
        classificationReport = classification_report(testLabels, predictions, output_dict=True)
        featureWeights = self.svm.coef_
        self.testResults = {
            "accuracy": score,
            "confusionMatrix": confusionMatrix,
            "featureWeights": featureWeights,
            "predictions": predictions.tolist(),
            "decisionFunction": decisionFunction,
            "classificationReport": classificationReport
        }
        return self.testResults

    def export(self, modelName):
        joblib.dump(self.svm, f"models/{modelName}.model")

    def graph(self, filename=None):
        """Creates a full report on the SVM. Can save the report to visuals directory if filename is specified"""
        if self.testResults is None:
            raise RuntimeError("No test results available. Run test() first.")

        # Set up the figure with multiple subplots
        plt.figure(figsize=(15, 10))
        sns.set(style="whitegrid")

        # 1. Confusion Matrix Heatmap
        plt.subplot(2, 3, 1)
        sns.heatmap(self.testResults["confusionMatrix"], annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")


        # 3. Feature Weights Bar Plot (only for linear kernel)
        if self.testResults["featureWeights"] is not None:
            plt.subplot(2, 3, 3)
            feature_weights = self.testResults["featureWeights"][0]  # Assuming binary classification
            plt.bar(range(len(feature_weights)), feature_weights)
            plt.title("Feature Weights (Linear Kernel)")
            plt.xlabel("Feature Index")
            plt.ylabel("Weight")

        # 4. Decision Function Histogram
        plt.subplot(2, 3, 4)
        plt.hist(self.testResults["decisionFunction"], bins=20, color="skyblue", edgecolor="black")
        plt.title("Decision Function Distribution")
        plt.xlabel("Distance to Hyperplane")
        plt.ylabel("Frequency")

        # 5. Classification Metrics Bar Plot
        plt.subplot(2, 3, 5)
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
        plt.suptitle(f"SVM Performance Report (Accuracy: {self.testResults['accuracy']:.2f})", fontsize=16)
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

def SVMLinear(trainingData, trainingLabels, testData, testLabels):
    """Create, train, test, and graph a svm given training and test data"""
    svm = GestureSVM()
    svm.train(trainingData=np.array(trainingData) , trainingLabels=np.array(trainingLabels))
    svm.test(testData=np.array(testData) , testLabels=np.array(testLabels))
    svm.graph()
    return svm
