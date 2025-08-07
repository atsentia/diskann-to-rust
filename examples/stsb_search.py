#!/usr/bin/env python3
"""
STSB Interactive Search Demo

Interactive semantic search using DiskANN-Rust CLI.
Allows real-time querying of the STSB sentence corpus.
"""

import argparse
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install required packages:")
    print("pip install sentence-transformers numpy")
    sys.exit(1)


class SearchResult:
    """Search result with ID and distance."""
    def __init__(self, id: int, distance: float):
        self.id = id
        self.distance = distance


class DiskANNSearcher:
    """Wrapper for DiskANN search using Rust CLI."""
    
    def __init__(self, index_path: str, metadata_path: str, diskann_binary: str = None):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        
        # Find DiskANN binary
        if diskann_binary:
            self.diskann_bin = Path(diskann_binary)
        else:
            # Try to find it in common locations
            possible_paths = [
                Path("../DiskANNInRust/target/release/diskann"),
                Path("../DiskANNInRust/target/debug/diskann"),
                Path("./diskann"),
            ]
            for path in possible_paths:
                if path.exists():
                    self.diskann_bin = path.resolve()
                    break
            else:
                print("ERROR: Could not find DiskANN binary!")
                print("Please build it with: cd ../DiskANNInRust && cargo build --release")
                sys.exit(1)
        
        print(f"Using DiskANN binary: {self.diskann_bin}")
        
        # Verify index exists
        if not self.index_path.exists():
            print(f"ERROR: Index file not found: {self.index_path}")
            print("Please build the index first using:")
            print(f"  {self.diskann_bin} build -i stsb_vectors.bin -o {self.index_path}")
            sys.exit(1)
        
        # Load metadata
        self.sentences = []
        print(f"Loading metadata from {metadata_path}...")
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    self.sentences.append(parts[1])
        
        print(f"Loaded {len(self.sentences)} sentences")
        
        # Initialize sentence transformer for query encoding
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    def search(self, query: str, k: int = 5, beam: int = 64) -> List[SearchResult]:
        """Search for k nearest neighbors to the query using Rust DiskANN."""
        
        # Encode query
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        
        # Save query to temporary file in DiskANN format
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
            # Write single vector (no header for query)
            # Write dimension first
            tmp.write(struct.pack('I', len(query_vector)))
            # Write vector data
            tmp.write(query_vector.astype(np.float32).tobytes())
            query_file = tmp.name
        
        # Call Rust DiskANN CLI
        cmd = [
            str(self.diskann_bin),
            'search',
            '-i', str(self.index_path),
            '-q', query_file,
            '-k', str(k),
            '--beam', str(beam)
        ]
        
        try:
            # Run search
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse output
            results = self._parse_search_output(result.stdout)
            
        except subprocess.CalledProcessError as e:
            print(f"Search failed: {e.stderr}")
            results = []
        finally:
            # Clean up temp file
            Path(query_file).unlink(missing_ok=True)
        
        return results
    
    def _parse_search_output(self, output: str) -> List[SearchResult]:
        """Parse the search output from Rust CLI."""
        results = []
        
        # Expected format: "ID: <id>, Distance: <distance>"
        for line in output.strip().split('\n'):
            if 'ID:' in line and 'Distance:' in line:
                try:
                    # Parse ID and distance
                    parts = line.split(',')
                    id_part = parts[0].split(':')[1].strip()
                    dist_part = parts[1].split(':')[1].strip()
                    
                    id = int(id_part)
                    distance = float(dist_part)
                    results.append(SearchResult(id, distance))
                except (IndexError, ValueError) as e:
                    # Skip malformed lines
                    continue
        
        return results
    
    def get_sentence(self, id: int) -> str:
        """Get sentence text by ID."""
        if 0 <= id < len(self.sentences):
            return self.sentences[id]
        return ""


def print_results(query: str, results: List[SearchResult], searcher: DiskANNSearcher, search_time: float):
    """Pretty print search results."""
    print("\n" + "="*80)
    print(f"Query: '{query}'")
    print(f"Search time: {search_time*1000:.2f} ms")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        sentence = searcher.get_sentence(result.id)
        similarity = 1 - result.distance  # Convert distance back to similarity
        print(f"\n{i}. [ID: {result.id}] (Similarity: {similarity:.4f})")
        print(f"   {sentence}")
    
    print("\n" + "-"*80)


def run_benchmark_queries(searcher: DiskANNSearcher):
    """Run benchmark queries and collect timing statistics."""
    print("\n" + "="*80)
    print("Running Benchmark Queries")
    print("="*80)
    
    benchmark_queries = [
        "A person is riding a horse",
        "The weather is beautiful today",
        "Scientists discovered a new species",
        "The stock market crashed yesterday",
        "Children are playing in the park"
    ]
    
    total_time = 0
    results_data = []
    
    for query in benchmark_queries:
        start = time.time()
        results = searcher.search(query, k=5)
        elapsed = time.time() - start
        total_time += elapsed
        
        print(f"\nQuery: '{query}'")
        print(f"Time: {elapsed*1000:.2f} ms")
        
        # Store for documentation
        result_sentences = []
        for r in results[:3]:  # Show top 3 for brevity
            sentence = searcher.get_sentence(r.id)
            similarity = 1 - r.distance
            result_sentences.append((sentence, similarity))
            print(f"  → {sentence[:60]}... (sim: {similarity:.3f})")
        
        results_data.append((query, elapsed*1000, result_sentences))
    
    avg_time = (total_time / len(benchmark_queries)) * 1000
    print(f"\n{'='*80}")
    print(f"Average search time: {avg_time:.2f} ms")
    print(f"Total queries: {len(benchmark_queries)}")
    
    return results_data


def interactive_mode(searcher: DiskANNSearcher):
    """Run interactive search mode."""
    print("\n" + "="*80)
    print("Interactive Semantic Search (STSB Corpus)")
    print("="*80)
    print(f"Corpus size: {len(searcher.sentences)} sentences")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'benchmark' to run benchmark queries")
    print("-"*80)
    
    while True:
        try:
            query = input("\nEnter query: ").strip()
            
            if query.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if query.lower() == 'benchmark':
                run_benchmark_queries(searcher)
                continue
            
            if not query:
                print("Please enter a query")
                continue
            
            # Search
            start = time.time()
            results = searcher.search(query, k=5)
            elapsed = time.time() - start
            
            # Display results
            print_results(query, results, searcher, elapsed)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="STSB Interactive Search Demo")
    parser.add_argument("--index", type=str, default="stsb.index",
                        help="Path to DiskANN index file")
    parser.add_argument("--metadata", type=str, default="stsb_metadata.tsv",
                        help="Path to metadata TSV file")
    parser.add_argument("--diskann-binary", type=str, default=None,
                        help="Path to DiskANN Rust binary")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark queries and exit")
    args = parser.parse_args()
    
    # Initialize searcher
    searcher = DiskANNSearcher(args.index, args.metadata, args.diskann_binary)
    
    if args.benchmark:
        results = run_benchmark_queries(searcher)
        # Save results for documentation
        with open("benchmark_results.txt", "w") as f:
            for query, time_ms, results in results:
                f.write(f"Query: {query}\n")
                f.write(f"Time: {time_ms:.2f} ms\n")
                for sentence, sim in results:
                    f.write(f"  - {sentence} (similarity: {sim:.3f})\n")
                f.write("\n")
    else:
        interactive_mode(searcher)


if __name__ == "__main__":
    main()