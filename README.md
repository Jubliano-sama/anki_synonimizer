# Offline Synonym-Interference Detector

> **Tagline:** *Maintain clarity in your vocabulary flashcards by automatically detecting and addressing semantically similar definitions.*

This repository includes **`main.py`**, an offline, configurable tool designed to identify pairs of vocabulary cards with definitions that are excessively similar, potentially causing interference during spaced-repetition learning. Originally optimized for Anki® CSV/TXT exports, this tool can handle various tabular data formats.

---

## Features

| Capability                      | Description                                                                                              |
| ------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Self-comparison**             | Detect interference within a single file.                                                                |
| **Cross-comparison**            | Identify similarities between two separate files.                                                        |
| Sentence-transformer embeddings | Default model: `all-MiniLM-L6-v2`, customizable with the `-m` flag.                                      |
| Fast ANN search                 | Utilizes the FAISS HNSW indexing method (parameters `M`, `efConstruction`, and `efSearch` configurable). |
| Dual similarity thresholds      | Filters based on cosine similarity and Jaccard index.                                                    |
| Extensive CLI customization     | Adjust separators, headers, markers, batching, hardware use, and randomness.                             |
| Robust error handling           | Provides user-friendly warnings and guidance for out-of-memory issues.                                   |
| Reproducible results            | Deterministic outcomes achievable by specifying a random seed.                                           |

---

## Installation

```bash
# Step 1: Clone the repository
git clone https://github.com/your-org/anki-interference.git
cd anki-interference

# Step 2: Set up a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Step 3: Install dependencies
pip install pandas numpy sentence-transformers faiss-cpu beautifulsoup4 torch psutil
```

> **Python version ≥3.10 is required** (specified in `.python-version`). GPU acceleration is optional, with automatic detection of `faiss-gpu`. Replace `faiss-cpu` with `faiss-gpu` if you wish to leverage GPU support.


---

## Quick Start

### Self-comparison

```bash
python main.py vocab.csv \
       --id-col 0 \
       --def-col 1
```

### Cross-comparison

```bash
python main.py new_list.csv \
       -r master_deck.csv \
       --cosine-t 0.9 \
       --jaccard-t 0.25 \
       -k 20
```

The output is saved as **`interference.csv`** by default, which can be modified with the `-o` option. The generated file always includes headers for consistent downstream processing.

---

## Command-Line Interface (CLI)

```text
Positional Arguments:
  input_file              Path to the CSV or TXT file to analyze.

Optional Arguments:
  -r, --reference-file    Path to a reference file for cross-comparison.
  -o, --output            Output file name (default: interference.csv).

Pre-processing Options:
  --sep                   Field separator (auto-detected if omitted).
  --no-header             Specify if input files do not have headers.
  --example-marker TEXT   Ignore text after this marker (default: "Example:").

Model and Search Parameters:
  -m, --model NAME        Sentence-transformer model (default: all-MiniLM-L6-v2).
  -k, --num-neighbors N   Maximum neighbors per query (default: 10).
  --hnsw-m N              HNSW index parameter M (default: 32).
  --hnsw-efc N            HNSW index efConstruction parameter (default: 64).
  --hnsw-efs N            HNSW index efSearch parameter (default: 64).

Similarity Thresholds:
  -ct, --cosine-t T       Cosine similarity threshold (default: 0.85).
  -jt, --jaccard-t λ      Jaccard similarity threshold (default: 0.30).

System Options:
  --force-cpu             Force the use of CPU, disabling CUDA.
  --batch-size N          Batch size for embedding (default: 64).
  --seed N                Random seed for reproducibility (default: 42).
```

For a complete list of options, run `python main.py -h`.

---

## Internal Architecture

```
main.py
 ├─ Config              # Configuration parameters
 ├─ Preprocessor        # Text cleaning and preprocessing
 ├─ ModelLoader         # Embedding model management
 ├─ ANNWrapper          # FAISS HNSW index integration
 ├─ find_interference() # Main logic pipeline
 └─ main()              # CLI interface and orchestration
```

The process flow is streamlined, using functional transformations with pandas, NumPy, and FAISS.

---

## Algorithm Overview

1. **Load and preprocess**: Clean definitions, removing HTML and content following a specified marker.
2. **Embed definitions**: Transform definitions into vector embeddings.
3. **Build index**: Construct a searchable HNSW graph of reference vectors.
4. **Search index**: Query nearest neighbors using embeddings.
5. **Apply filters**: Retain pairs meeting cosine and Jaccard similarity thresholds.
6. **Generate output**: Produce a deduplicated results file.

Complexity: Approximately O(N log N) for indexing and O(Q log N) per query.

---

## Troubleshooting

| Issue               | Cause                                     | Solution                                                    |
| ------------------- | ----------------------------------------- | ----------------------------------------------------------- |
| CUDA out of memory  | High memory consumption                   | Reduce `--batch-size` or enable `--force-cpu`.              |
| Tokenization errors | Incorrect separator specified             | Explicitly set `--sep ","` or `--sep "\t"`.                 |
| Empty results       | Stringent thresholds or insufficient data | Adjust similarity thresholds or verify preprocessing steps. |

Use `LOGLEVEL=DEBUG python main.py` to enable detailed logging for troubleshooting.

---

## Contributing

Contributions are welcome. Please ensure code adheres to style guidelines (`ruff`, `black`) and maintains existing documentation standards. Substantive changes should include relevant tests.

---
