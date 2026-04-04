import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from utils import Classifier, ClassifierDataset, load_labels


def evaluate_model(model, data_loader, device):
    model.eval()

    all_predictions = []
    all_actual_labels = []

    # disable gradient calculations
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # make predictions (forward pass)
            outputs = model(inputs)

            # find the category with the highest score
            _, predicted = torch.max(outputs.data, 1)

            # store the results for evaluation
            all_predictions.extend(predicted.cpu().numpy())
            all_actual_labels.extend(labels.cpu().numpy())

    return all_actual_labels, all_predictions


def main():
    arg_parser = argparse.ArgumentParser(
        description="Part 4: Evaluate the model on the test dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # input arguments
    arg_parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="dataset/test.tsv",
        metavar="<path>",
        help="path to the test dataset file"
    )
    arg_parser.add_argument(
        "-e",
        "--embeddings",
        type=str,
        default="embedding/test.npy",
        metavar="<path>",
        help="path to the test embeddings file"
    )
    arg_parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="model/classifier.pth",
        metavar="<path>",
        help="path to the saved classifier model file"
    )
    arg_parser.add_argument(
        "-l",
        "--labels",
        type=str,
        default="dataset/labels.txt",
        metavar="<path>",
        help="path to the labels file"
    )

    # parameter arguments
    arg_parser.add_argument(
        "-hs",
        "--hidden_size",
        type=int,
        default=64,
        metavar="<number>",
        help="number of neurons in the hidden layer (must match trained model)"
    )
    arg_parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=128,
        metavar="<number>",
        help="batch size for evaluation"
    )

    # output arguments
    arg_parser.add_argument(
        "-cm",
        "--confusion_matrix",
        type=str,
        metavar="<path>",
        help="where to save the confusion matrix plot (e.g., confusion_matrix.png)"
    )

    args = arg_parser.parse_args()

    # set the processing device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load label mapping
    label_to_index_mapping = load_labels(args.labels)
    index_to_label_list = list(label_to_index_mapping.keys())

    # load the dataset and data loader
    test_dataset = ClassifierDataset(args.embeddings, args.input, label_to_index_mapping)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # initialize and load the trained model
    input_size = test_dataset.embeddings.shape[1]
    output_size = len(label_to_index_mapping)

    model = Classifier(input_size, args.hidden_size, output_size)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # perform evaluation
    actual_labels, predicted_labels = evaluate_model(model, test_loader, device)

    # print final metrics
    print("\n--- Topic Classification Evaluation Results ---")
    print(classification_report(actual_labels, predicted_labels, target_names=index_to_label_list))

    # create and plot confusion matrix
    if args.confusion_matrix:
        # create the output directory if it doesn't exist
        output_dir = os.path.dirname(args.confusion_matrix)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        matrix = confusion_matrix(actual_labels, predicted_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=index_to_label_list,
            yticklabels=index_to_label_list
        )
        plt.title('Confusion Matrix: Predicted vs Actual Topics')
        plt.xlabel('predicted topic')
        plt.ylabel('actual topic')
        plt.savefig(args.confusion_matrix)
        print(f"Confusion Matrix saved to: {args.confusion_matrix}")


if __name__ == "__main__":
    main()
