import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd

# Hyperparameters
num_epochs = 150
learningRate = 7e-05
decay_weight = 1e-2

# Path to the CSV data file
csv_path = "data.csv"

# Load data from CSV file
tab = pd.read_csv(csv_path, sep=";")

# Split data into training and validation sets
tab_train = tab.iloc[500:, :].reset_index()
tab_val = tab.iloc[:500, :].reset_index()

# Create data loaders for training and validation
train_dl = t.utils.data.DataLoader(
    ChallengeDataset(tab_train, "train"), batch_size=64, shuffle=True
)
val_dl = t.utils.data.DataLoader(
    ChallengeDataset(tab_val, "val"), batch_size=64, shuffle=True
)

# Initialize the ResNet model, loss function, optimizer, and trainer
myModel = model.ResNet()
crit = t.nn.BCELoss()
optim = t.optim.Adam(myModel.parameters(), lr=learningRate, weight_decay=decay_weight)
trainer = Trainer(
    myModel,
    crit=crit,
    optim=optim,
    cuda=False,
    train_dl=train_dl,
    val_test_dl=val_dl,
    early_stopping_patience=10,
)

# Train the model and get the training and validation loss
res = trainer.fit(num_epochs)
res = t.tensor(res, device="cpu")

# Save the trained model in ONNX format
trainer.save_onnx("checkpoint_{:03d}.onnx".format(trainer.best_epoch))
