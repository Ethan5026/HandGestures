import os
import numpy as np
import joblib
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


class GestureCNN:
    """A CNN for gesture detection"""

    def __init__(self, model=None):
        """Create the CNN. Set model to import a pre-trained model"""
        self.trained = False
        self.testResults = None
        self.label_to_index = {}

        if model:
            self.trained = True
            self.model = load_model(model)
            # Load metadata if available
            metadata_path = model.replace('.h5', '_metadata.pkl')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.input_shape = metadata.get('input_shape')
                self.num_classes = metadata.get('num_classes')
                self.classes = metadata.get('classes')
                self.label_to_index = metadata.get('label_to_index', {})
        else:
            self._build_model()

    def _build_model(self, input_shape=None):
        """Build the CNN architecture

        If input_shape is None, it will be set during training
        """
        self.model = None  # Will be initialized during training when we know input shape
        self.input_shape = input_shape

    def _prepare_data(self, data, labels=None):
        """Reshape data for CNN and one-hot encode labels if provided"""
        # Assuming data comes as a 2D array, we need to reshape it for CNN
        # The exact reshape will depend on the nature of your gesture data

        if self.input_shape is None:
            # Try to infer a reasonable shape from the data
            n_samples = data.shape[0]
            n_features = data.shape[1]

            # If perfect square, make it a square image with 1 channel
            side_length = int(np.sqrt(n_features))
            if side_length * side_length == n_features:
                reshaped_data = data.reshape(n_samples, side_length, side_length, 1)
            else:
                # Try to make a reasonable rectangle with sufficient dimensions for CNN
                # We need at least 5x5 for two pooling layers
                min_dimension = 5
                # Calculate a width that's at least the min_dimension
                width = max(min_dimension, int(np.sqrt(n_features)))
                height = max(min_dimension, n_features // width + (1 if n_features % width > 0 else 0))

                # Pad the data to fit the rectangle
                padded_data = np.zeros((n_samples, width * height))
                padded_data[:, :n_features] = data
                reshaped_data = padded_data.reshape(n_samples, height, width, 1)

            self.input_shape = reshaped_data.shape[1:]
            print(f"Inferred input shape: {self.input_shape}")

            # Now that we know the input shape, build the model
            self._create_cnn_architecture()

            return reshaped_data
        else:
            # Use existing input shape
            n_samples = data.shape[0]
            height, width, channels = self.input_shape

            # Check if reshaping is needed
            if data.shape[1] != height * width * channels:
                print(f"Warning: Input data shape {data.shape} doesn't match expected shape for CNN input {(n_samples, height, width, channels)}")
                # Try to maintain compatibility by padding or truncating
                expected_features = height * width * channels
                if data.shape[1] < expected_features:
                    # Pad with zeros
                    padded_data = np.zeros((n_samples, expected_features))
                    padded_data[:, :data.shape[1]] = data
                    return padded_data.reshape(n_samples, height, width, channels)
                else:
                    # Truncate
                    return data[:, :expected_features].reshape(n_samples, height, width, channels)
            else:
                return data.reshape(n_samples, height, width, channels)

    def _create_cnn_architecture(self):
        """Create the CNN architecture based on the input shape"""
        # Get the input dimensions
        height, width, channels = self.input_shape

        # Determine how many convolutional and pooling layers we can use
        # based on the input dimensions
        model_layers = []

        # Start with input layer
        model_layers.append(Conv2D(32, (3, 3), activation='relu', padding='same',
                              input_shape=self.input_shape))

        # Only add pooling if the dimension is larger than the pool size
        if height > 2 and width > 2:
            model_layers.append(MaxPooling2D((2, 2)))

            # We can add more conv+pool layers if dimensions allow
            if height > 4 and width > 4:
                model_layers.append(Conv2D(64, (3, 3), activation='relu', padding='same'))
                model_layers.append(MaxPooling2D((2, 2)))

                if height > 8 and width > 8:
                    model_layers.append(Conv2D(128, (3, 3), activation='relu', padding='same'))
                    model_layers.append(MaxPooling2D((2, 2)))

        # Always add flatten and dense layers
        model_layers.append(Flatten())
        model_layers.append(Dense(128, activation='relu'))
        model_layers.append(Dropout(0.5))
        model_layers.append(Dense(self.num_classes, activation='softmax'))

        # Create the sequential model
        self.model = Sequential(model_layers)

        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print(self.model.summary())

    def train(self, trainingData, trainingLabels, epochs=20, batch_size=32, validation_split=0.2):
        """Train the CNN on training data"""
        # Get unique classes and create a mapping from string labels to integers
        self.classes = np.unique(trainingLabels)
        self.num_classes = len(self.classes)
        self.label_to_index = {label: i for i, label in enumerate(self.classes)}

        # Convert string labels to numeric indices
        numeric_labels = np.array([self.label_to_index[label] for label in trainingLabels])

        # Reshape data and convert labels to one-hot encoding
        X_train = self._prepare_data(np.array(trainingData))
        y_train = to_categorical(numeric_labels, num_classes=self.num_classes)

        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )

        self.trained = True
        return self.history

    def predict(self, testData):
        """Predicts data using the CNN"""
        if self.trained == False:
            raise RuntimeError("CNN must be trained on data before predicting, call train() first")

        X_test = self._prepare_data(np.array(testData))
        predictions_proba = self.model.predict(X_test)
        predictions_indices = np.argmax(predictions_proba, axis=1)

        # Convert numeric predictions back to original label strings
        index_to_label = {i: label for label, i in self.label_to_index.items()}
        predictions_labels = [index_to_label[idx] for idx in predictions_indices]

        return predictions_labels

    def test(self, testData, testLabels):
        """Test the CNN and return results"""
        if self.trained == False:
            raise RuntimeError("CNN must be trained on data before tested, call train() first")

        # Convert string labels to numeric indices
        numeric_testLabels = np.array([self.label_to_index[label] for label in testLabels])

        X_test = self._prepare_data(np.array(testData))
        y_test_cat = to_categorical(numeric_testLabels, num_classes=self.num_classes)

        # Evaluate the model
        loss, accuracy = self.model.evaluate(X_test, y_test_cat, verbose=0)

        # Get predictions
        predictions_proba = self.model.predict(X_test)
        predictions_indices = np.argmax(predictions_proba, axis=1)

        # Convert numeric predictions back to original labels
        index_to_label = {i: label for label, i in self.label_to_index.items()}
        predictions_labels = [index_to_label[idx] for idx in predictions_indices]

        # Calculate confusion matrix and classification report
        confusionMatrix = confusion_matrix(testLabels, predictions_labels)
        classificationReport = classification_report(testLabels, predictions_labels, output_dict=True)

        self.testResults = {
            "accuracy": accuracy,
            "loss": loss,
            "confusionMatrix": confusionMatrix,
            "predictions": predictions_labels,
            "predictionsProba": predictions_proba.tolist(),
            "classificationReport": classificationReport
        }

        return self.testResults

    def export(self, modelName):
        """Save the trained model to disk"""
        if not os.path.exists("models"):
            os.makedirs("models")
        self.model.save(f"models/{modelName}.h5")

        # Also save the input shape, number of classes, and label mapping for later loading
        metadata = {
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "classes": self.classes,
            "label_to_index": self.label_to_index
        }
        joblib.dump(metadata, f"models/{modelName}_metadata.pkl")

    def graph(self, filename=None):
        """Creates a full report on the CNN. Can save the report to visuals directory if filename is specified"""
        if self.testResults is None:
            raise RuntimeError("No test results available. Run test() first.")

        # Set up the figure with multiple subplots
        plt.figure(figsize=(15, 12))
        sns.set(style="whitegrid")

        # 1. Confusion Matrix Heatmap
        plt.subplot(2, 2, 1)
        sns.heatmap(self.testResults["confusionMatrix"], annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # 2. Training History (if available)
        if hasattr(self, 'history'):
            plt.subplot(2, 2, 2)
            plt.plot(self.history.history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in self.history.history:
                plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()

            plt.subplot(2, 2, 3)
            plt.plot(self.history.history['loss'], label='Training Loss')
            if 'val_loss' in self.history.history:
                plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()

        # 3. Classification Metrics Bar Plot
        plt.subplot(2, 2, 4)
        metrics = self.testResults["classificationReport"]
        classes = [k for k in metrics.keys() if k not in ("accuracy", "macro avg", "weighted avg")]
        precision = [metrics[c]["precision"] for c in classes]
        recall = [metrics[c]["recall"] for c in classes]
        f1 = [metrics[c]["f1-score"] for c in classes]

        x = np.arange(len(classes))
        width = 0.25
        plt.bar(x - width, precision, width, label="Precision", color="lightcoral")
        plt.bar(x, recall, width, label="Recall", color="lightgreen")
        plt.bar(x + width, f1, width, label="F1-Score", color="lightblue")
        plt.xticks(x, classes)
        plt.title("Classification Metrics by Class")
        plt.xlabel("Class")
        plt.ylabel("Score")
        plt.legend()

        # Adjust layout and add overall title
        plt.suptitle(f"CNN Performance Report (Accuracy: {self.testResults['accuracy']:.2f})", fontsize=16)
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


def CNNModel(trainingData, trainingLabels, testData, testLabels, epochs=20):
    """Create, train, test, and graph a CNN given training and test data"""
    cnn = GestureCNN()
    cnn.train(trainingData=np.array(trainingData), trainingLabels=np.array(trainingLabels), epochs=epochs)
    cnn.test(testData=np.array(testData), testLabels=np.array(testLabels))
    cnn.graph()
    return cnn