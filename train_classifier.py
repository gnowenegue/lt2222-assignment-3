import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os


class ClassifierDataset(Dataset):
    def __init__(self, embeddings_path, dataset_path, label_to_index_mapping):
        # load the sentence embeddings from the numpy file
        self.embeddings = np.load(embeddings_path).astype(np.float32)

        # load the dataset to get the corresponding label index
        dataframe = pd.read_csv(dataset_path, sep='\t')
        self.labels = [
            label_to_index_mapping[category] for category in dataframe['category']
        ]

        # verify that the number of embeddings matches the number of labels
        if len(self.embeddings) != len(self.labels):
            raise ValueError(f"mismatch: {len(self.embeddings)} embeddings VS {len(self.labels)} labels")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]


class Classifier(nn.Module):
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        super(Classifier, self).__init__()

        # define a neural network with one hidden layer and relu non-linearity
        self.model = nn.Sequential(
            nn.Linear(input_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, output_layer_size)
        )

    def forward(self, x):
        return self.model(x)


def load_labels(labels_path):
    with open(labels_path, 'r', encoding='utf-8') as file:
        labels = [line.strip() for line in file if line.strip()]

    # create a mapping from label to index
    return {label: i for i, label in enumerate(labels)}


def train_model(
        model,
        train_loader,
        dev_loader,
        criterion,
        optimizer,
        epochs,
        device
):
    # track performance
    history = {
        'train_loss': [],
        'train_acc': [],
        'dev_loss': [],
        'dev_acc': []
    }

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)

            # backward pass
            loss = criterion(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # calculate statistics
            # 1. total penalty for this batch (average loss * number of sentences)
            running_loss += loss.item() * inputs.size(0)

            # 2. the model's predictions
            _, predicted = torch.max(outputs.data, 1)

            # 3. total questions that have been asked
            total_samples += labels.size(0)

            # 4. right answers the model got
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        print(f"epoch {epoch + 1}/{epochs}: train loss: {epoch_loss:.4f} | train acc: {epoch_acc:.4f}")

        # run validation if dev loader is provided
        if dev_loader:
            model.eval()

            dev_running_loss = 0.0
            dev_correct_predictions = 0
            dev_total_samples = 0

            with torch.no_grad():
                for inputs, labels in dev_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # calculate statistics
                    dev_running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    dev_total_samples += labels.size(0)
                    dev_correct_predictions += (predicted == labels).sum().item()

            dev_epoch_loss = dev_running_loss / dev_total_samples
            dev_epoch_acc = dev_correct_predictions / dev_total_samples
            history['dev_loss'].append(dev_epoch_loss)
            history['dev_acc'].append(dev_epoch_acc)
            print(f"dev loss: {dev_epoch_loss:.4f} | dev acc: {dev_epoch_acc:.4f}")

    return history


def plot_performance(history, output_path):
    # plot accuracy and loss curves for training and validation
    epochs = range(1, len(history['train_acc']) + 1)

    plt.figure(figsize=(12, 5))

    # plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], label='train acc')
    if history['dev_acc']:
        plt.plot(epochs, history['dev_acc'], label='dev acc')
    plt.title('accuracy over epochs')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    # plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], label='train loss')
    if history['dev_loss']:
        plt.plot(epochs, history['dev_loss'], label='dev loss')
    plt.title('loss over epochs')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"performance plot saved to {output_path}")


def main():
    arg_parser = argparse.ArgumentParser(
        description="Part 3: Train a feed-forward model for multiclass classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # dataset arguments
    arg_parser.add_argument(
        "--train_data",
        type=str,
        default="dataset/train.tsv",
        metavar="<path>",
        help="path to the training tsv file"
    )
    arg_parser.add_argument(
        "--train_embeddings",
        type=str,
        default="embedding/train.npy",
        metavar="<path>",
        help="path to the training embeddings file"
    )
    arg_parser.add_argument(
        "--dev_data",
        type=str,
        default="dataset/dev.tsv",
        metavar="<path>",
        help="optional: path to the development tsv file (example: dataset/dev.tsv)"
    )
    arg_parser.add_argument(
        "--dev_embeddings",
        type=str,
        metavar="<path>",
        help="optional: path to the development embeddings file (example: embedding/dev.npy)"
    )
    arg_parser.add_argument(
        "-l",
        "--labels",
        type=str,
        default="dataset/labels.txt",
        metavar="<path>",
        help="path to the labels file"
    )

    # model parameters
    arg_parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100,
        metavar="<number>",
        help="number of training epochs"
    )
    arg_parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        metavar="<number>",
        help="batch size for training"
    )
    arg_parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.001,
        metavar="<number>",
        help="learning rate for the optimizer"
    )
    arg_parser.add_argument(
        "-hs",
        "--hidden_size",
        type=int,
        default=64,
        metavar="<number>",
        help="number of neurons in the hidden layer"
    )

    # output arguments
    arg_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="model/classifier.pth",
        metavar="<path>",
        help="where to save the trained model"
    )
    arg_parser.add_argument(
        "-p",
        "--plot",
        type=str,
        metavar="<path>",
        help="optional: where to save the performance plot (example: evaluation_plot.png)"
    )

    arguments = arg_parser.parse_args()

    # 1. load label mapping and dataset
    label_to_index_mapping = load_labels(arguments.labels)
    train_dataset = ClassifierDataset(
        arguments.train_embeddings, arguments.train_data, label_to_index_mapping
    )
    train_loader = DataLoader(
        train_dataset, batch_size=arguments.batch_size, shuffle=True
    )

    dev_loader = None
    if arguments.dev_data and arguments.dev_embeddings:
        dev_dataset = ClassifierDataset(
            arguments.dev_embeddings, arguments.dev_data, label_to_index_mapping
        )
        dev_loader = DataLoader(
            dev_dataset, batch_size=arguments.batch_size, shuffle=False
        )

    # 2. initialize the model, loss function, and optimizer
    input_layer_size = train_dataset.embeddings.shape[1]
    output_layer_size = len(label_to_index_mapping)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Classifier(input_layer_size, arguments.hidden_size, output_layer_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=arguments.learning_rate)

    # 3. train the model
    print(f"starting training for {arguments.epochs} epochs...")
    history = train_model(
        model, train_loader, dev_loader, criterion, optimizer, arguments.epochs, device
    )

    # 4. save the model
    print(f"saving the model to {arguments.output}...")
    os.makedirs(os.path.dirname(arguments.output), exist_ok=True)
    torch.save(model.state_dict(), arguments.output)
    print(f"successfully saved the model to {arguments.output}")

    # 5. plot performance if requested
    if arguments.plot:
        plot_performance(history, arguments.plot)


if __name__ == "__main__":
    main()
