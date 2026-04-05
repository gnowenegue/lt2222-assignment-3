import argparse
import os

import pandas as pd
from gensim.models import FastText

from utils import tokenize


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
        min_count=minimum_character_count,  # ignores all words with total frequency lower than this
        workers=4,
        sg=1  # skip-gram
    )
    return fasttext_model


def main():
    arg_parser = argparse.ArgumentParser(
        description="Part 2a: Train FastText word embeddings on Chinese Wikipedia sentences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # data paths arguments
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
        help="path to the development/validation tsv file"
    )
    arg_parser.add_argument(
        "--test",
        type=str,
        default="dataset/test.tsv",
        metavar="<path>",
        help="path to the testing tsv file"
    )

    # model fine-tuning arguments
    arg_parser.add_argument(
        "-d",
        "--dimension",
        type=int,
        default=100,
        metavar="<number>",
        help="number of dimensions for the embeddings"
    )
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

    # output arguments
    arg_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="model/fasttext.model",
        metavar="<path>",
        help="where to save the trained model file"
    )
    args = arg_parser.parse_args()

    # 1. load the datasets and combine everything
    print("loading the datasets...")
    combined_dataframes = load_datasets([args.train, args.dev, args.test])

    # 2. tokenize the sentences
    print("tokenizing the sentences...")
    tokenized_sentences = [
        tokenize(sentence) for sentence in combined_dataframes['text']
    ]

    # 3. train the FastText model
    print(
        f"training the FastText model with dimension {args.dimension}, context window {args.window_size} and minimum count {args.min_count}..."
    )
    fasttext_model = train_fasttext_model(
        tokenized_sentences,
        args.dimension,
        args.window_size,
        args.min_count,
    )

    # 4. save the FastText model
    print(f"saving the FastText model to {args.output}...")
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fasttext_model.save(args.output)
    print(f"✅ successfully saved the FastText model to {args.output}")


if __name__ == "__main__":
    main()
