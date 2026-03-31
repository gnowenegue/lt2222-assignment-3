import argparse
import pandas as pd
import numpy as np
from gensim.models import FastText
import os

from train_fasttext import tokenize


def calculate_mean_sentence_embedding(tokens, model):
    # convert each token to its vector
    vectors = [
        model.wv[token] for token in tokens
    ]

    # handle empty case
    if not vectors:
        return np.zeros(model.vector_size)

    # calculate the average vector for the sentence
    return np.mean(vectors, axis=0)


def convert_dataset_to_embedding_matrix(dataset, model):
    dataframe = pd.read_csv(dataset, sep='\t')

    print(f"generating embeddings for {len(dataframe)} sentences...")

    vectors = []
    for sentence in dataframe['text']:
        tokens = tokenize(sentence)
        current_sentence_vector = calculate_mean_sentence_embedding(tokens, model)
        vectors.append(current_sentence_vector)

    return np.array(vectors)


def main():
    arg_parser = argparse.ArgumentParser(
        description="Part 2b: Generate averaged sentence embeddings using FastText model."
    )

    # configuration for the input dataset and the pre-trained model
    arg_parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        metavar="<path>",
        help="path to the input tsv file (e.g., dataset/train.tsv)"
    )
    arg_parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="model/fasttext.model",
        metavar="<path>",
        help="path to the trained FastText model (default: model/fasttext.model)"
    )
    arg_parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        metavar="<path>",
        help="where to save the resulting NumPy matrix (e.g., embeddings/train.npy)"
    )

    arguments = arg_parser.parse_args()

    # 1. load the pre-trained FastText model
    print(f"loading the FastText model from {arguments.model}...")
    fasttext_model = FastText.load(arguments.model)

    # 2. process the entire dataset to create the embedding matrix
    embedding_matrix = convert_dataset_to_embedding_matrix(arguments.input, fasttext_model)

    # 3. create the output folder and save the matrix to disk
    print(f"saving the embedding matrix to {arguments.output}...")
    os.makedirs(os.path.dirname(arguments.output), exist_ok=True)
    np.save(arguments.output, embedding_matrix)
    print(f"successfully saved the embedding matrix to {arguments.output}")


if __name__ == "__main__":
    main()
