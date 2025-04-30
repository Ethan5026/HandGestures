import os
import numpy as np
import joblib
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Flatten,
                                    Dense, Dropout, BatchNormalization,
                                    GlobalAveragePooling1D, Reshape)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class Gesture1DCNN:
    """A 1D CNN optimized for hand landmark gesture recognition"""

    def __init__(self, model_path=None):
        """Initialize the model.

        Args:
            model_path: Path to a pre-trained model (optional)
        """
        self.trained = False
        self.test_results = None
        self.label_to_index = {}
        self.index_to_label = {}
        self.input_shape = None
        self.model = None
        self.history = None

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path):
        """Load a pre-trained model and its metadata"""
        self.model = load_model(model_path)
        self.trained = True

        # Load metadata
        metadata_path = model_path.replace('.h5', '_metadata.pkl')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.input_shape = metadata['input_shape']
            self.num_classes = metadata['num_classes']
            self.classes = metadata['classes']
            self.label_to_index = metadata['label_to_index']
            self.index_to_label = {v:k for k,v in self.label_to_index.items()}
            print(f"Loaded model with input shape {self.input_shape} "
                  f"and {self.num_classes} classes")

    def _build_model(self):
        """Build the 1D CNN architecture for landmark data"""
        self.model = Sequential([
            Input(shape=self.input_shape),

            # Add channel dimension (batch_size, 42) -> (batch_size, 42, 1)
            Reshape((self.input_shape[0], 1)),

            # First Conv Block
            Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),

            # Second Conv Block
            Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),

            # Global pooling instead of Flatten for variable length
            GlobalAveragePooling1D(),

            # Classifier
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print("1D CNN Architecture:")
        self.model.summary()

    def train(self, trainingData, trainingLabels,
              epochs=20, batch_size=32, validation_split=0.2):
        """Train the 1D CNN on landmark data"""

        # Prepare labels
        self.classes = np.unique(trainingLabels)
        self.num_classes = len(self.classes)
        self.label_to_index = {label:i for i,label in enumerate(self.classes)}
        self.index_to_label = {i:label for label,i in self.label_to_index.items()}

        # Convert labels to numeric
        y_train = np.array([self.label_to_index[label] for label in trainingLabels])
        y_train = to_categorical(y_train, num_classes=self.num_classes)

        # Prepare input data
        X_train = np.array(trainingData, dtype=np.float32)

        # Set input shape if not already set
        if self.input_shape is None:
            self.input_shape = (X_train.shape[1],)
            print(f"Input shape set to: {self.input_shape}")

        # Build model if not already built
        if self.model is None:
            self._build_model()

        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', save_best_only=True)
        ]

        # Train
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        self.trained = True
        return self.history

    def predict(self, data):
        """Make predictions on new landmark data"""
        if not self.trained:
            raise RuntimeError("Model must be trained first")

        X = np.array(data, dtype=np.float32)
        if X.shape[1] != self.input_shape[0]:
            raise ValueError(f"Expected {self.input_shape[0]} features, got {X.shape[1]}")

        proba = self.model.predict(X, verbose=0)
        pred_indices = np.argmax(proba, axis=1)
        return [self.index_to_label[idx] for idx in pred_indices]

    def evaluate(self, test_data, test_labels):
        """Evaluate model performance"""
        if not self.trained:
            raise RuntimeError("Model must be trained first")

        # Prepare data
        X_test = np.array(test_data, dtype=np.float32)
        y_test = np.array([self.label_to_index[label] for label in test_labels])
        y_test_cat = to_categorical(y_test, num_classes=self.num_classes)

        # Evaluate
        loss, accuracy = self.model.evaluate(X_test, y_test_cat, verbose=0)
        predictions = self.predict(X_test)

        # Calculate metrics
        cm = confusion_matrix(test_labels, predictions)
        cr = classification_report(test_labels, predictions, output_dict=True)

        self.test_results = {
            'accuracy': accuracy,
            'loss': loss,
            'confusion_matrix': cm,
            'classification_report': cr,
            'predictions': predictions
        }

        return self.test_results

    def export(self, model_name):
        """Save model and metadata"""
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{model_name}.h5"
        metadata_path = f"models/{model_name}_metadata.pkl"

        self.model.save(model_path)
        joblib.dump({
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'classes': self.classes,
            'label_to_index': self.label_to_index
        }, metadata_path)

        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")

    def plot_results(self, filename=None):
        """Visualize training and evaluation results"""
        if not self.test_results:
            raise ValueError("No test results available. Run evaluate() first")

        plt.figure(figsize=(15, 10))

        # Plot training history
        if self.history:
            plt.subplot(2, 2, 1)
            plt.plot(self.history.history['accuracy'], label='Train')
            if 'val_accuracy' in self.history.history:
                plt.plot(self.history.history['val_accuracy'], label='Validation')
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()

            plt.subplot(2, 2, 2)
            plt.plot(self.history.history['loss'], label='Train')
            if 'val_loss' in self.history.history:
                plt.plot(self.history.history['val_loss'], label='Validation')
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()

        # Plot confusion matrix
        plt.subplot(2, 2, 3)
        sns.heatmap(self.test_results['confusion_matrix'],
                    annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.classes,
                    yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Plot classification metrics
        plt.subplot(2, 2, 4)
        cr = self.test_results['classification_report']
        classes = [c for c in cr.keys() if c not in ['macro avg', 'weighted avg', 'accuracy']]
        metrics = ['precision', 'recall', 'f1-score']

        for i, metric in enumerate(metrics):
            plt.bar(
                np.arange(len(classes)) + i*0.25,
                [cr[c][metric] for c in classes],
                width=0.25,
                label=metric
            )

        plt.xticks(np.arange(len(classes)) + 0.25, classes)
        plt.legend()
        plt.title('Classification Metrics')

        plt.tight_layout()

        if filename:
            os.makedirs("visuals", exist_ok=True)
            path = f"visuals/{filename}.png"
            plt.savefig(path)
            print(f"Saved visualization to {path}")
        else:
            plt.show()
        plt.close()


def train_1dcnn_model(train_data, train_labels, test_data, test_labels,
                      model_name='gesture_1dcnn', epochs=20):
    """Complete training pipeline for the 1D CNN"""
    model = Gesture1DCNN()

    print("Training 1D CNN model...")
    model.train(train_data, train_labels, epochs=epochs)

    print("\nEvaluating model...")
    results = model.evaluate(test_data, test_labels)

    print(f"\nTest Accuracy: {results['accuracy']:.4f}")
    print(f"Test Loss: {results['loss']:.4f}")

    print("\nSaving model...")
    model.export(model_name)

    print("\nGenerating visualizations...")
    model.plot_results(model_name)

    return model