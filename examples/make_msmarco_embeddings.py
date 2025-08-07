#!/usr/bin/env python3
"""
MS MARCO Passage Dataset Embeddings Generator

This script downloads the MS MARCO Passage TREC-DL 2019 validation split,
generates embeddings using sentence-transformers/all-MiniLM-L6-v2,
and outputs two files:
- msmarco_passages.bin: f32 vectors (dim 384) in binary format
- msmarco_passages.tsv: <id>\t<raw text> metadata

The output format is compatible with the DiskANN-Rust CLI for index building.
"""

import argparse
import os
import struct
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import numpy as np
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install required packages:")
    print("pip install datasets sentence-transformers tqdm numpy")
    sys.exit(1)


def download_msmarco_dataset() -> Tuple[List[str], List[str]]:
    """Download MS MARCO Passage TREC-DL 2019 validation split."""
    print("Downloading MS MARCO Passage TREC-DL 2019 validation split...")
    
    # Load the dataset from HuggingFace
    dataset = load_dataset(
        "ms_marco", 
        "v1.1",
        split="validation",
        trust_remote_code=True
    )
    
    # Extract passages and IDs
    passages = []
    passage_ids = []
    
    print(f"Processing {len(dataset)} passages...")
    for i, item in enumerate(tqdm(dataset)):
        # Use the passage text and create a simple ID
        passage_text = item['passages']['passage_text'][0] if item['passages']['passage_text'] else ""
        if passage_text.strip():  # Only include non-empty passages
            passages.append(passage_text.strip())
            passage_ids.append(str(i))
    
    print(f"Collected {len(passages)} valid passages")
    return passages, passage_ids


def generate_embeddings(passages: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Generate embeddings for passages using sentence-transformers."""
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print("Generating embeddings...")
    embeddings = model.encode(
        passages,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False  # Keep raw embeddings
    )
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings


def save_binary_vectors(embeddings: np.ndarray, output_path: str):
    """Save embeddings in DiskANN binary format."""
    print(f"Saving binary vectors to {output_path}")
    
    num_vectors, dimension = embeddings.shape
    
    with open(output_path, 'wb') as f:
        # Write header: [num_vectors: u32][dimension: u32]
        f.write(struct.pack('<I', num_vectors))  # Little-endian uint32
        f.write(struct.pack('<I', dimension))   # Little-endian uint32
        
        # Write vectors as f32 values
        for vector in tqdm(embeddings, desc="Writing vectors"):
            for value in vector:
                f.write(struct.pack('<f', float(value)))  # Little-endian float32
    
    print(f"Saved {num_vectors} vectors of dimension {dimension}")


def save_metadata(passages: List[str], passage_ids: List[str], output_path: str):
    """Save passage metadata in TSV format."""
    print(f"Saving metadata to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for passage_id, passage_text in tqdm(zip(passage_ids, passages), desc="Writing metadata"):
            # Escape tabs and newlines in the passage text
            clean_text = passage_text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            f.write(f"{passage_id}\t{clean_text}\n")
    
    print(f"Saved metadata for {len(passages)} passages")


def main():
    parser = argparse.ArgumentParser(description="Generate MS MARCO passage embeddings for DiskANN")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=".", 
        help="Output directory for generated files"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model to use"
    )
    parser.add_argument(
        "--max-passages",
        type=int,
        default=None,
        help="Maximum number of passages to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file paths
    vectors_path = output_dir / "msmarco_passages.bin"
    metadata_path = output_dir / "msmarco_passages.tsv"
    
    # Check if files already exist
    if vectors_path.exists() and metadata_path.exists():
        print(f"Output files already exist:")
        print(f"  {vectors_path}")
        print(f"  {metadata_path}")
        response = input("Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return
    
    try:
        # Download dataset
        passages, passage_ids = download_msmarco_dataset()
        
        # Limit passages if requested (for testing)
        if args.max_passages and args.max_passages < len(passages):
            print(f"Limiting to first {args.max_passages} passages for testing")
            passages = passages[:args.max_passages]
            passage_ids = passage_ids[:args.max_passages]
        
        # Generate embeddings
        embeddings = generate_embeddings(passages, args.model)
        
        # Save outputs
        save_binary_vectors(embeddings, str(vectors_path))
        save_metadata(passages, passage_ids, str(metadata_path))
        
        print("\nSuccess! Generated files:")
        print(f"  Vectors: {vectors_path} ({vectors_path.stat().st_size:,} bytes)")
        print(f"  Metadata: {metadata_path} ({metadata_path.stat().st_size:,} bytes)")
        print(f"  Vector count: {len(embeddings)}")
        print(f"  Vector dimension: {embeddings.shape[1]}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()