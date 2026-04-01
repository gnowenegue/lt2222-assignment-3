import argparse
import pandas as pd
from gensim.models import FastText
import os
import re


def tokenize(sentence):
    # use a regular expression to keep Latin words/numbers whole and split Chinese into characters
    # this treats each Chinese character as a word while preserving English terms and numbers
    return re.findall(r"[a-zA-Z0-9]+|[^\s]", str(sentence))


def load_datasets(datasets):
    dataframes = [
        pd.read_csv(dataset, sep='\t') for dataset in datasets
    ]
    return pd.concat(dataframes)


def train_fasttext_model(
        sentences,
        vector_size,
        context_window_size,
        minimum_character_count
):
    fasttext_model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=context_window_size,
        min_count=minimum_character_count,
        workers=4,
        sg=1
    )
    return fasttext_model


def main():
    arg_parser = argparse.ArgumentParser(
        description="Part 2a: Train FastText word embeddings on Chinese Wikipedia sentences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # arguments for data paths and output
    arg_parser.add_argument(
        "--train",
        type=str,
        default="dataset/train.tsv",
        metavar="<path>",
        help="path to the training tsv file"
    )
    arg_parser.add_argument(
        "--dev",
        type=str,
        default="dataset/dev.tsv",
        metavar="<path>",
        help="path to the development tsv file"
    )
    arg_parser.add_argument(
        "--test",
        type=str,
        default="dataset/test.tsv",
        metavar="<path>",
        help="path to the testing tsv file"
    )
    arg_parser.add_argument(
        "-d",
        "--dimension",
        type=int,
        default=100,
        metavar="<number>",
        help="number of dimensions for the embeddings"
    )
    arg_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="model/fasttext.model",
        metavar="<path>",
        help="where to save the trained model file"
    )

    # parameters for model fine-tuning
    arg_parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        metavar="<number>",
        help="size of the context window"
    )
    arg_parser.add_argument(
        "--min_count",
        type=int,
        default=1,
        metavar="<number>",
        help="minimum frequency for a character to be included"
    )

    arguments = arg_parser.parse_args()

    # 1. load the datasets and combine everything
    print("loading the datasets...")
    combined_dataframes = load_datasets([arguments.train, arguments.dev, arguments.test])

    # print(combined_dataframes)

    # 2. tokenized the sentences
    print("tokenizing the sentences...")
    tokenized_sentences_list = [
        tokenize(sentence) for sentence in combined_dataframes['text']
    ]

    # 3. train the FastText model
    print(f"training the FastText model with dimension {arguments.dimension}...")
    fasttext_model = train_fasttext_model(
        tokenized_sentences_list,
        arguments.dimension,
        arguments.window_size,
        arguments.min_count,
    )

    # 4. save the FastText model
    print(f"saving the FastText model to {arguments.output}...")
    os.makedirs(os.path.dirname(arguments.output), exist_ok=True)
    fasttext_model.save(arguments.output)
    print(f"successfully saved the FastText model to {arguments.output}")


#

if __name__ == "__main__":
    main()
