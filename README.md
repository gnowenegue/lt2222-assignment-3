# Assignment 3: neural topic classification for Simplified Chinese

This project implements a multiclass topic classification pipeline for Simplified Chinese Wikipedia sentences. It uses **FastText** for character-based word embeddings and **PyTorch** for a feed-forward neural network for classification.

## Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install uv:**
   This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   *Or visit [the official installation page](https://docs.astral.sh/uv/getting-started/installation/) for other methods.*

3. **Install dependencies:**

   ```bash
   uv sync
   ```

   This will create a virtual environment and install all required libraries (torch, gensim, pandas, matplotlib, etc.).

## Usage

### Running the Pipeline

The entire process from training word embeddings to final evaluation is run via a shell script:

```bash
./run_pipeline.sh
```

*Note: The script uses `uv run` to ensure everything runs within the virtual environment.*

### Manual Execution

If you wish to run each script individually with full explicit parameters, please refer to [cli_commands.md](cli_commands.md).

## Data Strategy (Latin characters and numbers)

To handle Latin characters (English words) and numbers, we use a regular expression (`r"[a-zA-Z0-9]+|[^\s]"`) to keep them whole as single tokens while splitting the surrounding Chinese text into individual characters. This preserves the semantic meaning of English words and numbers while maintaining the character-based approach for Chinese.
