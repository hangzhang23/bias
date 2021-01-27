import torch
import torch.nn as nn
import torch.nn.functional as F

"""
autoencoder structure:
encoder:168-84-42-10
decoder:168-84-42-10
"""
class MyAutoencoder(nn.Module):
    def __init__(self):
        super(MyAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(168, 84),
            nn.Tanh(),
            nn.Linear(84, 42),
            nn.Tanh(),
            nn.Linear(42, 10)
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 42),
            nn.Tanh(),
            nn.Linear(42, 84),
            nn.Tanh(),
            nn.Linear(84, 168),
            nn.ReLU()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

"""
MLP classifier model structure
10-20-1
"""
class Classifier_1(nn.Module):
    def __init__(self):
        super(Classifier_1, self).__init__()
        self.hidden_layer = nn.Linear(10, 20)
        self.predict_layer = nn.Linear(20, 1)

    def forward(self, x):
        hidden_result = self.hidden_layer(x)
        relu_result = F.relu(hidden_result)
        predict_result = self.predict_layer(relu_result)
        return torch.sigmoid(predict_result)

