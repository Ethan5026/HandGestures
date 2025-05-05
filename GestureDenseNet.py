import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report

class DenseBlock(nn.Module):
    #Collects all the inputs of the previous DenseBlocks
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Conv1d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1)
            )
    #concatenates features from each layer
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            out = F.relu(out)
            features.append(out)
        return torch.cat(features, dim=1)

class TransitionLayer(nn.Module):
    #Set up the convolution and average pooling for the next DenseBlock
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool1d(kernel_size=2)

    #Do the convolution and average pooling for the next DenseBlock
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

from sklearn.preprocessing import LabelEncoder

class GestureDenseNet:
    #Setup up the Loss function and using the cpu
    def __init__(self, learning_rate=0.001):
        self.model = None  # Will be built after knowing num_classes
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = None  # Will also be initialized after model is built
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_encoder = LabelEncoder()
        self.learning_rate = learning_rate

    #the model for one iteration of DenseNet
    def build_model(self, num_classes):
        model = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            DenseBlock(16, growth_rate=12, num_layers=3),
            TransitionLayer(52, 32),
            DenseBlock(32, growth_rate=8, num_layers=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(48, num_classes)
        )
        return model

    #encode the classes, set up the model and iterate through the epochs, outputting the minimized loss
    def train(self, trainingData, trainingLabels, epochs=10, batch_size=64):
        y_encoded = self.label_encoder.fit_transform(trainingLabels)
        num_classes = len(self.label_encoder.classes_)

        if self.model is None:
            self.model = self.build_model(num_classes)
            self.model.to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        X = torch.tensor(np.array(trainingData), dtype=torch.float32).reshape(-1, 2, 21).to(self.device)
        y = torch.tensor(y_encoded, dtype=torch.long).to(self.device)

        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in dataloader:
                self.optimizer.zero_grad()
                output = self.model(xb)
                loss = self.loss_fn(output, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

    #predict a value with sample data
    def predict(self, testData):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(np.array(testData), dtype=torch.float32).reshape(-1, 2, 21).to(self.device)
            logits = self.model(X)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return self.label_encoder.inverse_transform(preds)

    #collect all predictions with the test data and compare them with the labels. Output accuracy and the report
    def test(self, testData, testLabels):
        y_pred = self.predict(testData)
        y_true = np.array(testLabels)
        accuracy = np.mean(y_pred == y_true)
        report = classification_report(y_true, y_pred, digits=4)
        print(f"Accuracy: {accuracy:.4f}")
        print(report)
        return accuracy, report
