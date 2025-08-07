#!/usr/bin/env python3
"""
STSB Semantic Search Demo - Dataset Preparation

Downloads the STSB dataset and generates embeddings for DiskANN indexing.
Creates a corpus from both sentences in each pair for a richer search experience.
"""

import argparse
import struct
import time
from pathlib import Path
from typing import List, Tuple
import sys

try:
    import numpy as np
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install required packages:")
    print("pip install datasets sentence-transformers numpy tqdm")
    sys.exit(1)


def download_stsb_dataset() -> Tuple[List[str], List[str]]:
    """Download STSB dataset and extract unique sentences."""
    print("Downloading STSB dataset from HuggingFace...")
    dataset = load_dataset("sentence-transformers/stsb")
    
    # Collect all unique sentences from train, validation, and test splits
    sentences = set()
    
    for split in ['train', 'validation', 'test']:
        print(f"Processing {split} split...")
        for example in tqdm(dataset[split]):
            sentences.add(example['sentence1'])
            sentences.add(example['sentence2'])
    
    # Convert to sorted list for consistent ordering
    sentences = sorted(list(sentences))
    
    print(f"Total unique sentences: {len(sentences)}")
    
    # Also create some interesting query examples from the dataset
    queries = [
        "A man is playing a guitar",
        "The weather is perfect for a picnic",
        "Scientists made an important discovery",
        "The economy is showing signs of recovery",
        "Children are having fun at the playground"
    ]
    
    return sentences, queries


def generate_embeddings(sentences: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Generate embeddings for all sentences using sentence-transformers."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Generating embeddings for {len(sentences)} sentences...")
    embeddings = model.encode(
        sentences,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalize for cosine similarity
    )
    
    return embeddings


def save_binary_vectors(embeddings: np.ndarray, output_path: Path):
    """Save embeddings in binary format for DiskANN."""
    num_vectors, dimension = embeddings.shape
    
    print(f"Saving {num_vectors} vectors of dimension {dimension} to {output_path}")
    
    with open(output_path, 'wb') as f:
        # Write header: num_vectors (u32) and dimension (u32)
        f.write(struct.pack('II', num_vectors, dimension))
        
        # Write vectors as f32
        for vector in embeddings:
            f.write(vector.astype(np.float32).tobytes())
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Binary file size: {file_size_mb:.2f} MB")


def save_metadata(sentences: List[str], output_path: Path):
    """Save sentence metadata in TSV format."""
    print(f"Saving metadata to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, sentence in enumerate(sentences):
            # Escape tabs and newlines in text
            clean_sentence = sentence.replace('\t', ' ').replace('\n', ' ')
            f.write(f"{idx}\t{clean_sentence}\n")
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Metadata file size: {file_size_mb:.2f} MB")


def save_sample_queries(queries: List[str], embeddings: np.ndarray, model_name: str, output_dir: Path):
    """Generate and save embeddings for sample queries."""
    print(f"\nGenerating embeddings for {len(queries)} sample queries...")
    
    model = SentenceTransformer(model_name)
    query_embeddings = model.encode(
        queries,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Save queries as binary vectors
    query_path = output_dir / "stsb_queries.bin"
    save_binary_vectors(query_embeddings, query_path)
    
    # Save query text
    query_text_path = output_dir / "stsb_queries.txt"
    with open(query_text_path, 'w', encoding='utf-8') as f:
        for query in queries:
            f.write(query + '\n')
    
    print(f"Sample queries saved to {query_text_path}")
    
    return query_embeddings


def print_statistics(sentences: List[str], embeddings: np.ndarray):
    """Print dataset statistics."""
    print("\n" + "="*50)
    print("STSB Dataset Statistics")
    print("="*50)
    print(f"Total sentences: {len(sentences)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Average sentence length: {np.mean([len(s.split()) for s in sentences]):.1f} words")
    print(f"Min sentence length: {min(len(s.split()) for s in sentences)} words")
    print(f"Max sentence length: {max(len(s.split()) for s in sentences)} words")
    
    # Sample sentences
    print("\nSample sentences:")
    for i in range(min(5, len(sentences))):
        print(f"  {i+1}. {sentences[i][:80]}...")


def main():
    parser = argparse.ArgumentParser(description="Prepare STSB dataset for DiskANN indexing")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for files")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence transformer model to use")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Download and process dataset
    start_time = time.time()
    sentences, queries = download_stsb_dataset()
    
    # Generate embeddings
    embeddings = generate_embeddings(sentences, args.model)
    
    # Save outputs
    save_binary_vectors(embeddings, output_dir / "stsb_vectors.bin")
    save_metadata(sentences, output_dir / "stsb_metadata.tsv")
    query_embeddings = save_sample_queries(queries, embeddings, args.model, output_dir)
    
    # Print statistics
    print_statistics(sentences, embeddings)
    
    elapsed = time.time() - start_time
    print(f"\nTotal processing time: {elapsed:.1f} seconds")
    
    print("\n" + "="*50)
    print("Next steps:")
    print("="*50)
    print("1. Build DiskANN index:")
    print(f"   cd ../DiskANNInRust")
    print(f"   ./target/release/diskann build -i ../examples/stsb_vectors.bin -o ../examples/stsb.index")
    print("\n2. Run interactive search:")
    print(f"   cd ../examples")
    print(f"   python stsb_search.py --index stsb.index --metadata stsb_metadata.tsv")


if __name__ == "__main__":
    main()