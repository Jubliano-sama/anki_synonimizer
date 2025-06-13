#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
interference.py

Offline Synonym-Interference Detector for Anki® Vocabulary Lists.

Identifies pairs of cards with semantically similar definitions using
sentence embeddings and Jaccard similarity.

Modes:
1. Self-comparison: Compares items within a single input file.
2. Cross-comparison: Compares items from an input file against a reference file.

Uses the word/term from the specified column as the identifier in the output.
Optionally ignores text after a specified marker (default: 'Example:').
Handles CSV and TXT files, auto-detecting format where possible.
"""

import argparse
import logging
import os
import re # Import regex module
import sys
import time
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set

# --- Dependency Check ---
try:
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    import faiss
    from bs4 import BeautifulSoup
    import torch # Often needed by sentence-transformers & faiss
except ImportError as e:
    missing_module = str(e).split("'")[1]
    print(f"Error: Missing required library '{missing_module}'.")
    print("Please install dependencies using:")
    print("pip install -r requirements.txt")
    print("\nRequired packages: pandas, numpy, sentence-transformers, faiss-cpu (or faiss-gpu), beautifulsoup4, torch, psutil (optional)")
    sys.exit(1)

# Optional import for memory check
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: 'psutil' not found. Cannot perform memory checks before loading models.")
    print("Install with: pip install psutil")


# --- Configuration ---

@dataclass
class Config:
    """Configuration parameters for the interference detection process."""
    input_file: str # File containing items to check (queries)
    reference_file: Optional[str] = None # Optional file to compare against (indexed)
    output_file: str = "interference.csv"
    id_column: str = "0" # Column containing the word/term itself
    definition_column: str = "1"
    model_name: str = "all-MiniLM-L6-v2"
    cosine_threshold: float = 0.85
    jaccard_threshold: float = 0.30
    num_neighbors: int = 10 # Max neighbors to find per query
    batch_size: int = 64
    hnsw_m: int = 32
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 64
    force_cpu: bool = False
    seed: int = 42
    # File parsing options (applied to both input and reference file if applicable)
    input_separator: Optional[str] = None
    input_has_header: Optional[bool] = None
    input_comment_char: str = '#'
    # Preprocessing options
    example_marker: str = "Example:"

    # Internal fields, not CLI args
    device: str = field(init=False, default="cpu")
    is_cross_comparison: bool = field(init=False, default=False)

    def __post_init__(self):
        """Determine device and comparison mode after initialization."""
        if not self.force_cpu and torch.cuda.is_available():
            self.device = "cuda"
            logging.info("CUDA available, using GPU.")
        else:
            self.device = "cpu"
            if self.force_cpu: logging.info("Forcing CPU usage.")
            else: logging.info("CUDA not available or forced CPU, using CPU.")

        self.is_cross_comparison = self.reference_file is not None
        if self.is_cross_comparison:
            logging.info(f"Cross-comparison mode enabled: Comparing '{self.input_file}' against '{self.reference_file}'.")
        else:
            logging.info(f"Self-comparison mode enabled: Comparing items within '{self.input_file}'.")


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- Constants ---
# Output columns for self-comparison
SELF_COL_WORD1 = "Word1"
SELF_COL_WORD2 = "Word2"
SELF_COL_DEF1 = "Definition1"
SELF_COL_DEF2 = "Definition2"
SELF_COL_COSINE = "CosineSimilarity"
SELF_COL_JACCARD = "JaccardSimilarity"
SELF_OUTPUT_COLUMNS = [
    SELF_COL_WORD1, SELF_COL_WORD2, SELF_COL_DEF1, SELF_COL_DEF2,
    SELF_COL_COSINE, SELF_COL_JACCARD
]

# Output columns for cross-comparison
CROSS_COL_QUERY_WORD = "Word_Query"
CROSS_COL_REF_WORD = "Word_Reference"
CROSS_COL_QUERY_DEF = "Definition_Query"
CROSS_COL_REF_DEF = "Definition_Reference"
CROSS_COL_COSINE = "CosineSimilarity"
CROSS_COL_JACCARD = "JaccardSimilarity"
CROSS_OUTPUT_COLUMNS = [
    CROSS_COL_QUERY_WORD, CROSS_COL_REF_WORD, CROSS_COL_QUERY_DEF, CROSS_COL_REF_DEF,
    CROSS_COL_COSINE, CROSS_COL_JACCARD
]


# --- Core Components ---

class Preprocessor:
    """Handles text cleaning operations."""
    @staticmethod
    def clean_text(html_text: str, example_marker: Optional[str] = None) -> str:
        """Strips HTML, normalizes whitespace, removes text after marker."""
        if pd.isna(html_text): return ""
        if not isinstance(html_text, str): html_text = str(html_text)
        try:
            soup = BeautifulSoup(html_text, "html.parser")
            text = soup.get_text(); text = ' '.join(text.split()); base_cleaned_text = text.strip()
        except Exception as e:
            logging.warning(f"Error cleaning HTML: {e}. Original: '{html_text[:50]}...'")
            base_cleaned_text = ' '.join(str(html_text).split()).strip()
        if example_marker and base_cleaned_text:
            marker_pattern_core = re.escape(example_marker.strip())
            marker_pattern = rf"\s*{marker_pattern_core}\s*:?\s*"
            parts = re.split(marker_pattern, base_cleaned_text, maxsplit=1, flags=re.IGNORECASE)
            return parts[0].strip()
        return base_cleaned_text

    @staticmethod
    def calculate_jaccard(text1: str, text2: str) -> float:
        """Calculates Jaccard similarity."""
        tokens1 = set(text1.lower().split()); tokens2 = set(text2.lower().split())
        intersection = len(tokens1.intersection(tokens2)); union = len(tokens1.union(tokens2))
        if union == 0: return 1.0 if intersection == 0 else 0.0
        return float(intersection) / union

class ModelLoader:
    """Loads and wraps the sentence embedding model."""
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name; self.device = device
        self.model: Optional[SentenceTransformer] = None; self._load_model()
    def _estimate_model_memory_usage(self) -> Optional[int]: # Simplified
        known_sizes = {"all-MiniLM-L6-v2": 80,"paraphrase-multilingual-MiniLM-L12-v2": 400,"all-mpnet-base-v2": 420,"all-distilroberta-v1": 290}
        return known_sizes.get(self.model_name, 500) * 1024 * 1024 # Default 500MB if unknown
    def _check_available_memory(self): # Simplified
        if not HAS_PSUTIL: return
        est_size = self._estimate_model_memory_usage(); avail_mem = psutil.virtual_memory().available
        req_buf = est_size * 1.5 + 500 * 1024 * 1024
        logging.info(f"Est. model mem: {est_size/(1024**2):.1f}MB. Available: {avail_mem/(1024**2):.1f}MB.")
        if avail_mem < req_buf: print("\nWARNING: Low available memory detected.", file=sys.stderr)
    def _load_model(self):
        logging.info(f"Loading model '{self.model_name}' onto '{self.device}'...")
        self._check_available_memory()
        try: self.model = SentenceTransformer(self.model_name, device=self.device); logging.info("Model loaded.")
        except Exception as e: logging.error(f"Could not load model '{self.model_name}'. Error: {e}"); sys.exit(1)
    def encode(self, texts: List[str], batch_size: int) -> np.ndarray:
        if self.model is None: raise RuntimeError("Model not loaded.")
        logging.info(f"Encoding {len(texts)} texts (batch size {batch_size})...")
        t0 = time.time()
        try:
            embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
            logging.info(f"Embedding finished in {time.time() - t0:.2f}s.")
            return embeddings.astype(np.float32)
        except Exception as e:
            logging.error(f"Embedding error: {e}")
            if "CUDA out of memory" in str(e): logging.error("CUDA OOM. Reduce --batch-size or use --force-cpu.")
            sys.exit(1)
    @property
    def embedding_dim(self) -> int:
        if self.model is None: raise RuntimeError("Model not loaded.")
        return self.model.get_sentence_embedding_dimension()

class ANNWrapper:
    """Wraps the FAISS HNSW index."""
    def __init__(self, config: Config):
        self.config = config; self.index: Optional[faiss.IndexHNSWFlat] = None; self.dimension: Optional[int] = None
    def build_index(self, embeddings: np.ndarray):
        if embeddings.ndim != 2: raise ValueError("Embeddings must be 2D.")
        if embeddings.shape[0] == 0: logging.warning("Empty embeddings. Index empty."); self.dimension, self.index = None, None; return
        self.dimension = embeddings.shape[1]
        logging.info(f"Building FAISS HNSW index (M={self.config.hnsw_m}, efC={self.config.hnsw_ef_construction}) for {embeddings.shape[0]} vectors (dim={self.dimension})...")
        t0 = time.time()
        self.index = faiss.IndexHNSWFlat(self.dimension, self.config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.config.hnsw_ef_construction
        try: self.index.add(embeddings); logging.info(f"Index built in {time.time() - t0:.2f}s. Trained: {self.index.is_trained}")
        except Exception as e: logging.error(f"FAISS index build error: {e}"); sys.exit(1)
    def search(self, query_embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None or self.dimension is None: logging.warning("Index empty. Cannot search."); return np.array([[]]*query_embeddings.shape[0]), np.array([[]]*query_embeddings.shape[0], dtype=np.int64)
        if query_embeddings.shape[1] != self.dimension: raise ValueError(f"Query dim ({query_embeddings.shape[1]}) != index dim ({self.dimension}).")
        logging.info(f"Searching for top {k} neighbors (efS={self.config.hnsw_ef_search})...")
        t0 = time.time()
        self.index.hnsw.efSearch = self.config.hnsw_ef_search
        try: scores, indices = self.index.search(query_embeddings, k); logging.info(f"Search completed in {time.time() - t0:.2f}s."); return scores, indices
        except Exception as e: logging.error(f"FAISS search error: {e}"); sys.exit(1)

# --- Helper Function for Data Preparation ---

def prepare_dataframe(df: pd.DataFrame, config: Config, is_reference: bool = False) -> Tuple[pd.DataFrame, str, str]:
    """Selects columns, handles duplicates, returns prepared df and column names."""
    file_type = "Reference" if is_reference else "Query/Input"
    word_col_ref = config.id_column
    def_col_ref = config.definition_column

    word_col_name = f"_word_{file_type.lower()}"
    def_col_name = f"_def_{file_type.lower()}"

    # Select columns by configured name or index
    selected_data = {}
    try:
        def get_selector(ref_str, col_list):
            try: idx = int(ref_str); return col_list[idx]
            except ValueError:
                if ref_str not in col_list: raise KeyError(f"Column name '{ref_str}' not found.")
                return ref_str
            except IndexError: raise IndexError(f"Index {ref_str} out of bounds.")
            except KeyError as e: raise KeyError(f"Column '{ref_str}' invalid: {e}")

        actual_columns = list(df.columns)
        word_selector = get_selector(word_col_ref, actual_columns)
        def_selector = get_selector(def_col_ref, actual_columns)
        logging.info(f"{file_type} - Resolved Word column: '{word_col_ref}' -> '{word_selector}'")
        logging.info(f"{file_type} - Resolved Definition column: '{def_col_ref}' -> '{def_selector}'")
        if word_selector == def_selector:
            logging.error(f"{file_type} - Word and Definition columns cannot be the same ('{word_selector}').")
            sys.exit(1)
        selected_data[word_col_name] = df[word_selector]
        selected_data[def_col_name] = df[def_selector]
    except (IndexError, KeyError) as e:
         logging.error(f"Error selecting columns for {file_type} file: {e}")
         logging.error(f"Check --id-col ('{word_col_ref}') / --def-col ('{def_col_ref}') against actual columns: {actual_columns}")
         sys.exit(1)

    prepared_df = pd.DataFrame(selected_data)

    # Handle duplicates for processing ID (only needed if self-comparing or if ref file has duplicates)
    prepared_df[word_col_name] = prepared_df[word_col_name].astype(str)
    if prepared_df[word_col_name].isnull().any() or not prepared_df[word_col_name].is_unique:
        logging.warning(f"{file_type} - Word column '{word_col_name}' has nulls/duplicates. Using internal index for processing uniqueness.")
        prepared_df = prepared_df.reset_index()
        internal_id_col_name = f"_internal_id_{file_type.lower()}"
        prepared_df.rename(columns={'index': internal_id_col_name}, inplace=True)
        processing_id_col = internal_id_col_name
    else:
        processing_id_col = word_col_name # Word itself is unique enough

    # Add processing ID column if it's different from word column
    if processing_id_col != word_col_name:
         prepared_df[processing_id_col] = prepared_df[internal_id_col_name]

    return prepared_df, word_col_name, def_col_name, processing_id_col


# --- Core Detection Function ---

def find_interference(query_df: pd.DataFrame, config: Config, reference_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Core function. Handles both self-comparison and cross-comparison.
    """
    output_columns = CROSS_OUTPUT_COLUMNS if config.is_cross_comparison else SELF_OUTPUT_COLUMNS
    mode = "Cross-Comparison" if config.is_cross_comparison else "Self-Comparison"
    logging.info(f"Starting interference detection ({mode} mode)...")

    # --- 1. Prepare Query Data ---
    query_df_prep, query_word_col, query_def_col, query_proc_id_col = prepare_dataframe(query_df, config, is_reference=False)

    work_query_df = query_df_prep[[query_proc_id_col, query_word_col, query_def_col]].copy()
    original_query_count = len(work_query_df)
    work_query_df[query_def_col] = work_query_df[query_def_col].astype(str)
    work_query_df.dropna(subset=[query_def_col], inplace=True)
    work_query_df = work_query_df[work_query_df[query_def_col].str.strip() != '']
    if len(work_query_df) < original_query_count: logging.warning(f"Query - Dropped {original_query_count - len(work_query_df)} rows with missing/empty definitions.")
    if len(work_query_df) == 0: logging.warning("Query - No valid definitions found. Exiting."); return pd.DataFrame(columns=output_columns)

    query_words = work_query_df[query_word_col].tolist()
    query_definitions_raw = work_query_df[query_def_col].tolist()

    # --- 2. Preprocess Query Definitions ---
    logging.info(f"Query - Preprocessing {len(query_definitions_raw)} definitions...")
    preprocessor = Preprocessor()
    query_definitions_cleaned = [preprocessor.clean_text(d, config.example_marker) for d in query_definitions_raw]
    query_cleaned_map = {i: text for i, text in enumerate(query_definitions_cleaned)}

    # Filter empty after cleaning
    query_valid_indices = [i for i, text in query_cleaned_map.items() if text]
    if len(query_valid_indices) < len(query_definitions_cleaned):
        logging.warning(f"Query - Dropped {len(query_definitions_cleaned) - len(query_valid_indices)} entries empty after cleaning.")
        if not query_valid_indices: logging.warning("Query - No valid definitions remaining."); return pd.DataFrame(columns=output_columns)
        query_words = [query_words[i] for i in query_valid_indices]
        query_definitions_cleaned = [query_cleaned_map[i] for i in query_valid_indices]
        query_cleaned_map = {i: text for i, text in enumerate(query_definitions_cleaned)} # Re-map 0..N-1

    # --- 3. Embed Query Definitions ---
    model_loader = ModelLoader(config.model_name, config.device)
    query_embeddings = model_loader.encode(query_definitions_cleaned, config.batch_size)
    if query_embeddings.shape[0] != len(query_definitions_cleaned): logging.error("Query embedding/definition count mismatch."); sys.exit(1)

    # --- 4. Prepare Reference Data (if applicable) ---
    if config.is_cross_comparison and reference_df is not None:
        ref_df_prep, ref_word_col, ref_def_col, _ = prepare_dataframe(reference_df, config, is_reference=True) # Don't need ref processing ID here

        work_ref_df = ref_df_prep[[ref_word_col, ref_def_col]].copy() # Only need word and def
        original_ref_count = len(work_ref_df)
        work_ref_df[ref_def_col] = work_ref_df[ref_def_col].astype(str)
        work_ref_df.dropna(subset=[ref_def_col], inplace=True)
        work_ref_df = work_ref_df[work_ref_df[ref_def_col].str.strip() != '']
        if len(work_ref_df) < original_ref_count: logging.warning(f"Reference - Dropped {original_ref_count - len(work_ref_df)} rows with missing/empty definitions.")
        if len(work_ref_df) == 0: logging.warning("Reference - No valid definitions found. Cannot perform comparison."); return pd.DataFrame(columns=output_columns)

        ref_words = work_ref_df[ref_word_col].tolist()
        ref_definitions_raw = work_ref_df[ref_def_col].tolist()

        logging.info(f"Reference - Preprocessing {len(ref_definitions_raw)} definitions...")
        ref_definitions_cleaned = [preprocessor.clean_text(d, config.example_marker) for d in ref_definitions_raw]
        ref_cleaned_map = {i: text for i, text in enumerate(ref_definitions_cleaned)}

        ref_valid_indices = [i for i, text in ref_cleaned_map.items() if text]
        if len(ref_valid_indices) < len(ref_definitions_cleaned):
            logging.warning(f"Reference - Dropped {len(ref_definitions_cleaned) - len(ref_valid_indices)} entries empty after cleaning.")
            if not ref_valid_indices: logging.warning("Reference - No valid definitions remaining."); return pd.DataFrame(columns=output_columns)
            ref_words = [ref_words[i] for i in ref_valid_indices]
            ref_definitions_cleaned = [ref_cleaned_map[i] for i in ref_valid_indices]
            ref_cleaned_map = {i: text for i, text in enumerate(ref_definitions_cleaned)}

        # Embed Reference Definitions
        reference_embeddings = model_loader.encode(ref_definitions_cleaned, config.batch_size)
        if reference_embeddings.shape[0] != len(ref_definitions_cleaned): logging.error("Reference embedding/definition count mismatch."); sys.exit(1)

        # Data for indexing and lookup
        index_embeddings = reference_embeddings
        lookup_words = ref_words
        lookup_defs_cleaned = ref_cleaned_map
        logging.info(f"Using {len(lookup_words)} reference items for index.")

    else: # Self-comparison mode
        index_embeddings = query_embeddings
        lookup_words = query_words
        lookup_defs_cleaned = query_cleaned_map
        logging.info(f"Using {len(lookup_words)} input items for index (self-comparison).")

    # --- 5. Index Construction ---
    ann_wrapper = ANNWrapper(config)
    ann_wrapper.build_index(index_embeddings)

    # --- 6. Neighbour Retrieval & Filtering ---
    # In self-compare, need k+1 to discard self. In cross-compare, just need k.
    num_search_neighbors = config.num_neighbors + (0 if config.is_cross_comparison else 1)
    # Ensure k is not more than the number of items in the index
    num_search_neighbors = min(num_search_neighbors, len(lookup_words))

    if num_search_neighbors <= (0 if config.is_cross_comparison else 1):
        logging.warning("Not enough items in index to find neighbors.")
        interfering_pairs_data = []
    else:
        scores, indices = ann_wrapper.search(query_embeddings, num_search_neighbors)
        logging.info("Filtering potential pairs...")
        interfering_pairs_data = []
        # Use 0..N-1 indices relative to the *query* list for the outer loop
        processed_pairs_self_indices: Set[Tuple[int, int]] = set() # Only used in self-compare

        for i in range(len(query_definitions_cleaned)): # Iterate through queries
            query_word_i = query_words[i]
            query_def_i = query_cleaned_map[i]

            # Iterate through neighbors found for query i
            # Neighbor indices refer to the *indexed* data (lookup_*)
            start_neighbor_idx = 1 if not config.is_cross_comparison else 0 # Skip self in self-compare
            for j in range(start_neighbor_idx, num_search_neighbors):
                neighbor_idx_in_lookup = indices[i, j]

                # Validate neighbor index against the lookup list size
                if neighbor_idx_in_lookup < 0 or neighbor_idx_in_lookup >= len(lookup_words):
                    continue

                # Additional self-check needed if self-comparing (in case index 0 wasn't self)
                if not config.is_cross_comparison and neighbor_idx_in_lookup == i:
                    continue

                # Prevent duplicate pairs (A,B) and (B,A) only in self-comparison mode
                if not config.is_cross_comparison:
                    pair_key = tuple(sorted((i, neighbor_idx_in_lookup)))
                    if pair_key in processed_pairs_self_indices:
                        continue

                cosine_similarity = scores[i, j]

                # Filter 1: Cosine Similarity
                if cosine_similarity >= config.cosine_threshold:
                    lookup_word_neighbor = lookup_words[neighbor_idx_in_lookup]
                    lookup_def_neighbor = lookup_defs_cleaned[neighbor_idx_in_lookup]

                    # Filter 2: Jaccard Similarity
                    jaccard_similarity = preprocessor.calculate_jaccard(query_def_i, lookup_def_neighbor)

                    if jaccard_similarity >= config.jaccard_threshold:
                        if config.is_cross_comparison:
                            pair_data = {
                                CROSS_COL_QUERY_WORD: query_word_i,
                                CROSS_COL_REF_WORD: lookup_word_neighbor,
                                CROSS_COL_QUERY_DEF: query_def_i,
                                CROSS_COL_REF_DEF: lookup_def_neighbor,
                                CROSS_COL_COSINE: cosine_similarity,
                                CROSS_COL_JACCARD: jaccard_similarity,
                            }
                        else: # Self-comparison
                            pair_data = {
                                SELF_COL_WORD1: query_word_i,
                                SELF_COL_WORD2: lookup_word_neighbor,
                                SELF_COL_DEF1: query_def_i,
                                SELF_COL_DEF2: lookup_def_neighbor,
                                SELF_COL_COSINE: cosine_similarity,
                                SELF_COL_JACCARD: jaccard_similarity,
                            }
                            processed_pairs_self_indices.add(pair_key) # Mark self-pair done

                        interfering_pairs_data.append(pair_data)


    # --- 7. Result Generation ---
    logging.info(f"Found {len(interfering_pairs_data)} interfering pairs matching criteria.")
    if not interfering_pairs_data:
        return pd.DataFrame(columns=output_columns)

    results_df = pd.DataFrame(interfering_pairs_data)
    results_df.sort_values(by=CROSS_COL_COSINE, ascending=False, inplace=True) # Use consistent cosine col name
    results_df[CROSS_COL_COSINE] = results_df[CROSS_COL_COSINE].astype(float)
    results_df[CROSS_COL_JACCARD] = results_df[CROSS_COL_JACCARD].astype(float)

    logging.info("Interference detection process completed.")
    return results_df[output_columns] # Ensure correct column order


# --- Main Execution Block (CLI) ---

def main():
    """Main function to handle command-line execution."""
    parser = argparse.ArgumentParser(
        description="Offline Synonym-Interference Detector for Anki Vocabulary Lists.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input Files
    parser.add_argument(
        "input_file", type=str, help="Path to the primary input file (queries).",
    )
    parser.add_argument(
        "-r", "--reference-file", type=str, default=None,
        help="Optional path to a second file to compare against (reference/index). If omitted, compares input_file against itself.",
    )

    # File parsing options (applied to both files if -r is used)
    parser.add_argument(
        "--sep", dest="input_separator", type=str, default=None,
        help="Field separator. Default: Auto-detect (, or \\t).",
    )
    parser.add_argument(
        "--no-header", dest="input_has_header", action="store_false", default=None,
        help="Specify if input file(s) lack a header. Default: Auto-detect.",
    )
    parser.add_argument(
        "--comment", dest="input_comment_char", type=str, default="#",
        help="Character indicating comment lines.",
    )

    # Preprocessing options
    parser.add_argument(
        "--example-marker", dest="example_marker", type=str, default="Example:",
        help="Ignore text after this marker in definitions (case-insensitive).",
    )

    # Core processing options
    parser.add_argument(
        "-o", "--output", dest="output_file", type=str, default="interference.csv",
        help="Path for the output CSV file.",
    )
    parser.add_argument(
        "--id-col", dest="id_column", type=str, default="0",
        help="Name or 0-based index of the column containing the word/term.",
    )
    parser.add_argument(
        "--def-col", dest="definition_column", type=str, default="1",
        help="Name or 0-based index of the column containing the definition.",
    )
    parser.add_argument(
        "-m", "--model", dest="model_name", type=str, default="sentence-transformers/all-mpnet-base-v2",
        help="Sentence-transformer model name or path.",
    )
    parser.add_argument(
        "-ct", "--cosine-t", dest="cosine_threshold", type=float, default=-10,
        help="Minimum cosine similarity (τ) threshold.",
    )
    parser.add_argument(
        "-jt", "--jaccard-t", dest="jaccard_threshold", type=float, default=-10,
        help="Minimum Jaccard similarity (λ) threshold.",
    )
    parser.add_argument(
        "-k", "--num-neighbors", dest="num_neighbors", type=int, default=512,
        help="Max number of neighbors to find per query item.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Embedding batch size.",
    )
    parser.add_argument(
        "--hnsw-m", type=int, default=512, help="HNSW index parameter M.",
    )
    parser.add_argument(
        "--hnsw-efc", dest="hnsw_ef_construction", type=int, default=512,
        help="HNSW index parameter efConstruction.",
    )
    parser.add_argument(
        "--hnsw-efs", dest="hnsw_ef_search", type=int, default=64,
        help="HNSW index parameter efSearch.",
    )
    parser.add_argument(
        "--force-cpu", action="store_true", help="Force CPU usage.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed.",
    )

    args = parser.parse_args()

    # --- Initialize Config (triggers __post_init__) ---
    config = Config(**vars(args))

    # --- Auto-detect file parsing settings ---
    # Apply same logic to both input and reference file for simplicity
    def get_parsing_params(filename):
        ext = os.path.splitext(filename)[1].lower()
        sep = config.input_separator
        has_header = config.input_has_header
        if sep is None: sep = '\t' if ext in ['.txt', '.tsv'] else ','
        else: sep = sep.replace('\\t', '\t')
        if has_header is None: has_header = False if ext in ['.txt', '.tsv'] else True
        return sep, has_header

    input_sep, input_has_hdr = get_parsing_params(config.input_file)
    ref_sep, ref_has_hdr = get_parsing_params(config.reference_file) if config.is_cross_comparison else (None, None)
    # Log detected/used settings
    logging.info(f"Input file parsing: sep='{input_sep}', header={input_has_hdr}, comment='{config.input_comment_char}'")
    if config.is_cross_comparison:
        logging.info(f"Reference file parsing: sep='{ref_sep}', header={ref_has_hdr}, comment='{config.input_comment_char}'")


    # --- Set Seed ---
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    if config.device == "cuda": torch.cuda.manual_seed_all(config.seed)

    logging.info(f"Final Configuration used: {config}")

    # --- Load Data ---
    start_time_total = time.time()
    def load_data(filepath, sep, has_header, comment_char, file_label):
        logging.info(f"Loading {file_label} data from: {filepath}")
        try:
            read_args = {"sep": sep, "comment": comment_char if comment_char else None,
                         "header": 0 if has_header else None, "keep_default_na": False,
                         "na_values": [''], "dtype": str}
            df = pd.read_csv(filepath, **read_args)
            logging.info(f"Loaded {len(df)} rows and {len(df.columns)} columns from {file_label}.")
            if df.empty: logging.warning(f"{file_label} file loaded but resulted in an empty DataFrame.")
            return df
        except FileNotFoundError: logging.error(f"{file_label} file not found: {filepath}"); sys.exit(1)
        except Exception as e:
            logging.error(f"Error reading {file_label} file '{filepath}': {e}")
            if "Error tokenizing data" in str(e): logging.error("Check --sep argument.")
            sys.exit(1)

    # Load query data
    query_df_raw = load_data(config.input_file, input_sep, input_has_hdr, config.input_comment_char, "Input/Query")
    if query_df_raw.empty and not config.is_cross_comparison: # Only exit if self-comparing and input is empty
         logging.error("Input file is empty. Cannot perform self-comparison.")
         sys.exit(1)

    # Load reference data if needed
    reference_df_raw = None
    if config.is_cross_comparison:
        reference_df_raw = load_data(config.reference_file, ref_sep, ref_has_hdr, config.input_comment_char, "Reference")
        if reference_df_raw.empty:
             logging.error("Reference file is empty. Cannot perform cross-comparison.")
             sys.exit(1)
        if query_df_raw.empty: # Allow empty query if reference exists
             logging.warning("Query file is empty, but reference file loaded. Output will be empty.")
             # Create empty output and exit cleanly later

    # --- Run Interference Detection ---
    try:
        # Pass both DFs (reference might be None)
        results_df = find_interference(query_df_raw, config, reference_df_raw)
    except Exception as e:
        logging.exception("An unexpected error occurred during interference detection.")
        sys.exit(1)

    # --- Save Results ---
    output_columns = CROSS_OUTPUT_COLUMNS if config.is_cross_comparison else SELF_OUTPUT_COLUMNS
    if not results_df.empty:
        logging.info(f"Saving {len(results_df)} interference pairs to: {config.output_file}")
        try:
            results_df.to_csv(config.output_file, index=False, encoding='utf-8')
        except IOError as e:
            logging.error(f"Error writing output file '{config.output_file}': {e}")
            sys.exit(1)
    else:
        logging.info("No interference pairs found or processing resulted in no valid data.")
        try: # Create empty file with correct headers
             pd.DataFrame(columns=output_columns).to_csv(config.output_file, index=False, encoding='utf-8')
             logging.info(f"Created empty output file with headers: {config.output_file}")
        except IOError as e:
            logging.error(f"Error writing empty output file '{config.output_file}': {e}")

    end_time_total = time.time()
    logging.info(f"Total execution time: {end_time_total - start_time_total:.2f} seconds.")
    logging.info("Script finished.")

if __name__ == "__main__":
    main()