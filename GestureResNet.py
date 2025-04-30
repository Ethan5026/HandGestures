import os
import numpy as np
import joblib
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import to_categorical


class GestureResNet:
    """A ResNet50-based model for gesture detection"""

    def __init__(self, model=None):
        """Create the ResNet50 model. Set model to import a pre-trained model"""
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
            self.input_shape = None

    def _build_model(self):
        """Build the ResNet50 architecture with custom classification head"""
        # ResNet50 expects 3 channels and minimum 32x32 pixels
        # We'll ensure our input meets these requirements

        # Create the base pre-trained model
        base_model = ResNet50(weights='imagenet', include_top=False,
                             input_shape=self.input_shape)

        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False

        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)

        # Create the final model
        self.model = Model(inputs=base_model.input, outputs=predictions)

        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print(self.model.summary())

    def _prepare_data(self, data):
        """Reshape data for ResNet50 and preprocess it"""
        # ResNet50 expects 3-channel images of at least 32x32 pixels
        # We need to reshape and potentially upscale the data

        if self.input_shape is None:
            # Try to infer a reasonable shape from the data
            n_samples = data.shape[0]
            n_features = data.shape[1]

            # ResNet50 requires minimum dimensions of 32x32
            min_size = 32

            # Calculate dimensions that maintain aspect ratio but meet minimum size
            side_length = int(np.sqrt(n_features))
            if side_length * side_length == n_features:
                # Perfect square - upscale to at least 32x32
                target_size = max(min_size, side_length)
                # Create a square shape
                height = width = target_size
            else:
                # Not a perfect square - create a rectangle
                width = max(min_size, int(np.sqrt(n_features)))
                height = max(min_size, n_features // width + (1 if n_features % width > 0 else 0))

                # Ensure both dimensions meet the minimum
                if width < min_size:
                    width = min_size
                if height < min_size:
                    height = min_size

            # Initialize the reshaped data with 3 channels (RGB)
            reshaped_data = np.zeros((n_samples, height, width, 3))

            # Reshape original data to single channel
            if side_length * side_length == n_features:
                # If perfect square, reshape directly
                single_channel = data.reshape(n_samples, side_length, side_length, 1)
            else:
                # Otherwise pad and reshape
                padded_data = np.zeros((n_samples, width * height))
                padded_data[:, :min(n_features, width * height)] = data[:, :min(n_features, width * height)]
                single_channel = padded_data.reshape(n_samples, height, width, 1)

            # If we need to upscale, use resize
            if side_length < min_size:
                # Upscale using tensorflow's resize
                single_channel_upscaled = tf.image.resize(
                    single_channel,
                    (height, width)
                ).numpy()

                # Copy the single channel to all 3 RGB channels
                reshaped_data[:, :, :, 0] = single_channel_upscaled[:, :, :, 0]  # R
                reshaped_data[:, :, :, 1] = single_channel_upscaled[:, :, :, 0]  # G
                reshaped_data[:, :, :, 2] = single_channel_upscaled[:, :, :, 0]  # B
            else:
                # Copy the single channel to all 3 RGB channels
                reshaped_data[:, :, :, 0] = single_channel[:, :, :, 0]  # R
                reshaped_data[:, :, :, 1] = single_channel[:, :, :, 0]  # G
                reshaped_data[:, :, :, 2] = single_channel[:, :, :, 0]  # B

            self.input_shape = reshaped_data.shape[1:]
            print(f"Inferred input shape for ResNet50: {self.input_shape}")

            # Preprocess the data for ResNet50
            processed_data = preprocess_input(reshaped_data)

            return processed_data
        else:
            # Use existing input shape
            n_samples = data.shape[0]
            height, width, channels = self.input_shape

            # Check if reshaping is needed
            if data.shape[1] != height * width:
                print(f"Warning: Input data shape {data.shape} doesn't match expected shape")
                # Try to maintain compatibility by padding or truncating
                expected_features = height * width
                if data.shape[1] < expected_features:
                    # Pad with zeros
                    padded_data = np.zeros((n_samples, expected_features))
                    padded_data[:, :data.shape[1]] = data
                    single_channel = padded_data.reshape(n_samples, height, width, 1)
                else:
                    # Truncate
                    single_channel = data[:, :expected_features].reshape(n_samples, height, width, 1)
            else:
                single_channel = data.reshape(n_samples, height, width, 1)

            # Create 3-channel data by copying the single channel
            reshaped_data = np.zeros((n_samples, height, width, 3))
            reshaped_data[:, :, :, 0] = single_channel[:, :, :, 0]  # R
            reshaped_data[:, :, :, 1] = single_channel[:, :, :, 0]  # G
            reshaped_data[:, :, :, 2] = single_channel[:, :, :, 0]  # B

            # Preprocess the data for ResNet50
            processed_data = preprocess_input(reshaped_data)

            return processed_data

    def train(self, trainingData, trainingLabels, epochs=20, batch_size=32, validation_split=0.2, fine_tune=False):
        """Train the ResNet50 model on training data"""
        # Get unique classes and create a mapping from string labels to integers
        self.classes = np.unique(trainingLabels)
        self.num_classes = len(self.classes)
        self.label_to_index = {label: i for i, label in enumerate(self.classes)}

        # Convert string labels to numeric indices
        numeric_labels = np.array([self.label_to_index[label] for label in trainingLabels])

        # Reshape and preprocess data
        X_train = self._prepare_data(np.array(trainingData))

        # Convert labels to one-hot encoding
        y_train = to_categorical(numeric_labels, num_classes=self.num_classes)

        # Build the model if not already built
        if self.model is None:
            self._build_model()

        # Fine-tune the model if requested (unfreeze some layers)
        if fine_tune and hasattr(self, 'model'):
            # Unfreeze the last few layers of the ResNet50 base
            for layer in self.model.layers[0].layers[-10:]:  # Unfreeze last 10 layers
                layer.trainable = True

            # Recompile with a lower learning rate
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower learning rate
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("Fine-tuning last 10 layers of ResNet50")

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
        """Predicts data using the ResNet50 model"""
        if self.trained == False:
            raise RuntimeError("ResNet50 model must be trained before predicting, call train() first")

        X_test = self._prepare_data(np.array(testData))
        predictions_proba = self.model.predict(X_test)
        predictions_indices = np.argmax(predictions_proba, axis=1)

        # Convert numeric predictions back to original label strings
        index_to_label = {i: label for label, i in self.label_to_index.items()}
        predictions_labels = [index_to_label[idx] for idx in predictions_indices]

        return predictions_labels

    def test(self, testData, testLabels):
        """Test the ResNet50 model and return results"""
        if self.trained == False:
            raise RuntimeError("ResNet50 model must be trained before testing, call train() first")

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
        """Creates a full report on the ResNet50 model. Can save the report to visuals directory if filename is specified"""
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
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.title("Classification Metrics by Class")
        plt.xlabel("Class")
        plt.ylabel("Score")
        plt.legend()

        # Adjust layout and add overall title
        plt.suptitle(f"ResNet50 Performance Report (Accuracy: {self.testResults['accuracy']:.2f})", fontsize=16)
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


def ResNetModel(trainingData, trainingLabels, testData, testLabels, epochs=20, fine_tune=False):
    """Create, train, test, and graph a ResNet50 model given training and test data"""
    resnet = GestureResNet()
    resnet.train(trainingData=np.array(trainingData), trainingLabels=np.array(trainingLabels),
                epochs=epochs, fine_tune=fine_tune)
    resnet.test(testData=np.array(testData), testLabels=np.array(testLabels))
    resnet.graph()
    return resnet