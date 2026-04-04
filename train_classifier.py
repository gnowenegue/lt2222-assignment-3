import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import Classifier, ClassifierDataset, load_labels


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

        print(f"epoch {epoch + 1}/{epochs}:")
        print(f"    train loss: {epoch_loss:.4f} | train acc: {epoch_acc:.4f}")

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
            print(f"    dev loss: {dev_epoch_loss:.4f} | dev acc: {dev_epoch_acc:.4f}")

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
    print(f"✅ successfully save performance plot to {output_path}")


def main():
    arg_parser = argparse.ArgumentParser(
        description="Part 3: Train a feed-forward model for multiclass classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # dataset arguments
    arg_parser.add_argument(
        "--train",
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
        "--dev",
        type=str,
        default="dataset/dev.tsv",
        metavar="<path>",
        help="path to the development tsv file (example: dataset/dev.tsv)"
    )
    arg_parser.add_argument(
        "--dev_embeddings",
        type=str,
        metavar="<path>",
        help="path to the development embeddings file (example: embedding/dev.npy)"
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
        help="where to save the performance plot (example: evaluation_plot.png)"
    )

    args = arg_parser.parse_args()

    # 1. load label mapping and dataset
    label_to_index_mapping = load_labels(args.labels)
    train_dataset = ClassifierDataset(
        args.train_embeddings, args.train, label_to_index_mapping
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    dev_loader = None
    if args.dev and args.dev_embeddings:
        dev_dataset = ClassifierDataset(
            args.dev_embeddings, args.dev, label_to_index_mapping
        )
        dev_loader = DataLoader(
            dev_dataset, batch_size=args.batch_size, shuffle=False
        )

    # 2. initialize the model, loss function, and optimizer
    input_layer_size = train_dataset.embeddings.shape[1]
    output_layer_size = len(label_to_index_mapping)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Classifier(input_layer_size, args.hidden_size, output_layer_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 3. train the model
    print(f"starting training for {args.epochs} epochs...")
    history = train_model(
        model, train_loader, dev_loader, criterion, optimizer, args.epochs, device
    )

    # 4. save the model
    print(f"saving the model to {args.output}...")
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"✅ successfully saved the model to {args.output}")

    # 5. plot performance if requested
    if args.plot:
        # create the output directory if it doesn't exist
        plot_dir = os.path.dirname(args.plot)
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)

        plot_performance(history, args.plot)


if __name__ == "__main__":
    main()
