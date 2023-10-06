# Deep Learning ResNet for Binary Classification

This repository contains a Python implementation of a deep learning model based on the ResNet architecture for binary classification tasks. The model is designed to classify data into two categories and includes features such as data loading, training, and model evaluation.

## Getting Started

### Prerequisites

To run this project, you will need:

- Python 3.x
- PyTorch (for deep learning)
- pandas (for data handling)
- scikit-learn (for evaluation metrics)
- matplotlib (for visualization)

You can install these dependencies using pip:

pip install torch pandas scikit-learn matplotlib


### Dataset

The dataset used for this project is stored in a CSV file (`data.csv`). It is split into training and validation sets for model training and evaluation.

### Training

To train the model, execute the `train.py` script:

python train.py

This script initializes the ResNet model, specifies hyperparameters, and trains the model on the provided dataset. Training progress, including loss and metrics, is displayed during training.

### Model Checkpoints

The best-performing model checkpoint is saved in ONNX format as `checkpoint_{best_epoch}.onnx`, where `{best_epoch}` represents the epoch with the lowest validation loss during training.

## Results

The project provides insights into the following:

- Data preprocessing and loading using PyTorch's DataLoader.
- Training a ResNet-based deep learning model for binary classification.
- Early stopping to prevent overfitting.
- Saving the best model checkpoint in ONNX format.
- Evaluation using metrics like binary cross-entropy loss and F1-score.

## Acknowledgments

This project was inspired by the ResNet architecture and leverages PyTorch for deep learning tasks. Special thanks to the authors of ResNet for their contributions to deep learning.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

