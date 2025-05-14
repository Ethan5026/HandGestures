# Live Hand Gesture Classifier

This repository contains a collection of machine learning models for recognizing hand gestures using the [HaGRID dataset](https://github.com/hukenovs/hagrid), real-time webcam input, and various classification architectures including CNNs, MLPs, GNNs, TabNet, SVMs, and DenseNets.

---
### **View our documentation and presentation documents**
- ```Hand Gesture Presentation.pdf```
- ```Hand Gesture Documentation.pdf```

---

## Project Structure

```
.
├── DataTools.py              # Data parsing and preprocessing from HaGRID JSON files
├── LiveTest.py               # Real-time gesture detection with MediaPipe and webcam
├── GestureSVM.py             # Linear SVM gesture classifier
├── GestureMLPClassifier.py   # MLP (dense neural network) classifier
├── GestureDenseNet.py        # 1D DenseNet for gesture recognition
├── GestureDenseNet2D.py      # 2D DenseNet variant
├── Gesture1DCNN.py           # 1D CNN for landmark vector input
├── GestureCNN.py             # 2D CNN for image-based input
├── GestureGNN.py             # Graph Neural Network using MediaPipe topology
├── GestureResNet.py          # Transfer learning with ResNet50
├── TabNet.py                 # Tabular model using TabNet
├── SVMwBagging.py            # Ensemble of SVMs with bagging
├── SVMwBoosting.py           # Ensemble of SVMs with boosting
├── models/                   # Saved model files
└── visuals/                  # Auto-generated evaluation plots
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ethan5026/HandGestures.git
   cd HandGestures
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

> ⚠️ If using `torch-geometric`, follow [official installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for your PyTorch/CUDA version.

---
## Live Gesture Detection

To launch the webcam-based gesture recognition:

```bash
python LiveTest.py
```

- Press `L` to load a saved model (`.model` or `.pkl` file).
- Or train a new DenseNet Convolutional Neural Network model on the HaGRID dataset and use it live.
- Press `q` to exit the webcam window.

---


## Available Models

- ✅ Linear SVM and Bagging/Boosting Ensembles
- ✅ MLPClassifier (fully-connected NN)
- ✅ DenseNet (1D and 2D)
- ✅ 1D CNN for flattened landmark input
- ✅ 2D CNN for hand images
- ✅ ResNet50 (transfer learning)
- ✅ TabNet (tabular deep learning)
- ✅ GNN using MediaPipe landmark graph

---


## 📁 Dataset

This project is designed to work with the [HaGRID dataset](https://github.com/hukenovs/hagrid). Place the JSON label files inside:

```
HaGRID/
├── train/
│   └── *.json
├── test/
│   └── *.json
```

You can also load and preprocess image data using `PrepareDatasetImages()` in `DataTools.py`.

---

## Example Usage

```python
from GestureMLPClassifier import MLPClassifier
from DataTools import FullDataLabels

X_train, y_train, X_test, y_test = FullDataLabels()
model = MLPClassifier()
model.train(X_train, y_train)
model.test(X_test, y_test)
```

---

## Model Evaluation

Most model classes have `.graph()` methods that visualize:

- Confusion Matrix
- Classification Report
- Training History (if applicable)

Output is saved in the `visuals/` folder.

---

## Saving & Loading

Most models support saving via:

```python
model.export("MyModelName")
```

And loading via:

```python
model = GestureCNN(model="models/MyModelName.h5")
```

---

## ✍Authors

- Developed by Ethan Gruening and Owen Harty
- Built for research and experimentation in gesture recognition.

---

## Acknowledgements

- [HaGRID Dataset](https://github.com/hukenovs/hagrid)
- [MediaPipe](https://google.github.io/mediapipe/)
- [PyTorch](https://pytorch.org/)
- [TensorFlow](https://www.tensorflow.org/)