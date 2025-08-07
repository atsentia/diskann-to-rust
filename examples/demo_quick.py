#!/usr/bin/env python3
"""
Quick demo of the text search functionality using simple synthetic data.
This script demonstrates the complete pipeline without requiring large downloads.
"""

import struct
import tempfile
import subprocess
import os
from pathlib import Path

def create_demo_embeddings():
    """Create simple demo embeddings and metadata."""
    print("Creating demo embeddings...")
    
    # Simple synthetic 384-dimensional vectors
    import random
    random.seed(42)
    
    # Sample passages
    passages = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn patterns from data.",
        "Python is a popular programming language known for its simplicity and powerful libraries.",
        "The restaurant serves delicious Italian food with fresh ingredients and authentic recipes.",
        "Climate change is one of the most pressing challenges facing humanity in the 21st century.",
        "Basketball is a team sport played with two teams of five players each on a rectangular court."
    ]
    
    # Generate random vectors (normally these would come from sentence transformers)
    vectors = []
    for i in range(len(passages)):
        vector = [random.gauss(0, 1) for _ in range(384)]
        # Make similar concepts closer in vector space
        if "machine learning" in passages[i].lower() or "artificial intelligence" in passages[i].lower():
            # AI/ML related - bias towards positive values in first dimensions
            for j in range(50):
                vector[j] += 2.0
        elif "food" in passages[i].lower() or "restaurant" in passages[i].lower():
            # Food related - bias towards positive values in different dimensions  
            for j in range(50, 100):
                vector[j] += 2.0
                
        vectors.append(vector)
    
    return vectors, passages

def save_demo_data(vectors, passages, output_dir):
    """Save demo vectors and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save vectors in DiskANN format
    vectors_file = output_dir / "demo_vectors.bin" 
    with open(vectors_file, 'wb') as f:
        # Header
        f.write(struct.pack('<I', len(vectors)))  # num_vectors
        f.write(struct.pack('<I', len(vectors[0])))  # dimension
        
        # Vectors
        for vector in vectors:
            for value in vector:
                f.write(struct.pack('<f', value))
    
    # Save metadata
    metadata_file = output_dir / "demo_metadata.tsv"
    with open(metadata_file, 'w') as f:
        for i, passage in enumerate(passages):
            f.write(f"{i}\t{passage}\n")
    
    print(f"Saved demo data to {output_dir}")
    return str(vectors_file), str(metadata_file)

def build_index(vectors_file, index_file):
    """Build index using DiskANN CLI."""
    print("Building search index...")
    
    diskann_dir = Path(__file__).parent.parent / "DiskANNInRust"
    cmd = [
        "cargo", "run", "--release", "--bin", "diskann", "--",
        "build", 
        "-i", vectors_file,
        "-o", index_file,
        "--max-degree", "32",
        "--search-list-size", "64"
    ]
    
    result = subprocess.run(cmd, cwd=diskann_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Index building failed: {result.stderr}")
        return False
    
    print("Index built successfully!")
    return True

def create_query_vector():
    """Create a simple query vector."""
    # Simulate a query about machine learning
    import random
    random.seed(123)  # Different seed for query
    
    vector = [random.gauss(0, 1) for _ in range(384)]
    # Bias towards AI/ML dimensions to match the pattern
    for j in range(50):
        vector[j] += 1.5
        
    return vector

def save_query(vector, query_file):
    """Save query vector."""
    with open(query_file, 'wb') as f:
        f.write(struct.pack('<I', len(vector)))  # dimension
        for value in vector:
            f.write(struct.pack('<f', value))

def search_index(index_file, query_file):
    """Search the index."""
    print("Searching index...")
    
    diskann_dir = Path(__file__).parent.parent / "DiskANNInRust"
    cmd = [
        "cargo", "run", "--release", "--bin", "diskann", "--",
        "search",
        "-i", index_file,
        "-q", query_file,
        "-k", "3",
        "--beam", "32"
    ]
    
    result = subprocess.run(cmd, cwd=diskann_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Search failed: {result.stderr}")
        return False
    
    print("Search results:")
    print(result.stdout)
    return True

def main():
    """Run the complete demo."""
    print("=== DiskANN Text Search Quick Demo ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 1. Create demo data
        vectors, passages = create_demo_embeddings()
        vectors_file, metadata_file = save_demo_data(vectors, passages, temp_path)
        
        # 2. Build index
        index_file = str(temp_path / "demo.index")
        if not build_index(vectors_file, index_file):
            return 1
        
        # 3. Create and search with query
        query_vector = create_query_vector()
        query_file = str(temp_path / "query.bin")
        save_query(query_vector, query_file)
        
        if not search_index(index_file, query_file):
            return 1
        
        # 4. Show what the passages actually contain
        print("\nDemo passages for reference:")
        with open(metadata_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    print(f"  {parts[0]}: {parts[1]}")
        
        print("\n✓ Demo completed successfully!")
        print("This demonstrates the complete text search pipeline:")
        print("  1. Vector embeddings → Binary format")
        print("  2. Index building with DiskANN") 
        print("  3. Query processing and search")
        print("  4. Result ranking and retrieval")
        
        return 0

if __name__ == "__main__":
    exit(main())