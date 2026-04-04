import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def tokenize(sentence):
    """
    splits a sentence into tokens.
    chinese characters are treated as individual units, 
    while latin words and numbers are kept whole.
    """

    # use a regular expression to keep Latin words/numbers whole and split Chinese into characters
    return re.findall(r"[a-zA-Z0-9]+|[^\s]", str(sentence))


def load_labels(labels_path):
    """
    reads a file of labels and returns a mapping from name to index.
    """

    with open(labels_path, 'r', encoding='utf-8') as file:
        labels = [line.strip() for line in file if line.strip()]

    # create a mapping from label to index
    return {label: i for i, label in enumerate(labels)}


class Classifier(torch.nn.Module):
    """
    a feed-forward neural network for topic classification.
    consists of one hidden layer with relu non-linearity.
    """

    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        super(Classifier, self).__init__()

        # define a neural network with one hidden layer and relu non-linearity
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_layer_size, hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_size, output_layer_size)
        )

    def forward(self, x):
        return self.model(x)


class ClassifierDataset(Dataset):
    """
    a pytorch dataset for topic classification.
    loads sentence embeddings and their corresponding category labels.
    """

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
