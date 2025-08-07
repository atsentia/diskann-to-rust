#!/usr/bin/env python3
"""
STSB Disk Index Demo - Memory-Efficient Search

This demo showcases disk-based indexing for larger-than-memory datasets.
The index is built once and stored on disk, then loaded on-demand for search.
"""

import argparse
import json
import mmap
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install required packages:")
    print("pip install sentence-transformers numpy tqdm")
    sys.exit(1)


class DiskIndex:
    """Disk-based index for memory-efficient search."""
    
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True, parents=True)
        
        # Paths for index components
        self.vectors_path = self.index_dir / "vectors.bin"
        self.metadata_path = self.index_dir / "metadata.tsv"
        self.config_path = self.index_dir / "config.json"
        self.graph_path = self.index_dir / "graph.bin"
        
    def exists(self) -> bool:
        """Check if disk index exists."""
        return (self.vectors_path.exists() and 
                self.metadata_path.exists() and 
                self.config_path.exists())
    
    def save_config(self, config: Dict):
        """Save index configuration."""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self) -> Dict:
        """Load index configuration."""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def get_size_mb(self) -> float:
        """Get total index size in MB."""
        total_size = 0
        for path in [self.vectors_path, self.metadata_path, 
                     self.config_path, self.graph_path]:
            if path.exists():
                total_size += path.stat().st_size
        return total_size / (1024 * 1024)


class DiskIndexBuilder:
    """Build disk-based index using DiskANN."""
    
    def __init__(self, index_dir: str, diskann_binary: str = None):
        self.index = DiskIndex(Path(index_dir))
        
        # Find DiskANN binary
        if diskann_binary:
            self.diskann_bin = Path(diskann_binary)
        else:
            possible_paths = [
                Path("../DiskANNInRust/target/release/diskann"),
                Path("../DiskANNInRust/target/debug/diskann"),
            ]
            for path in possible_paths:
                if path.exists():
                    self.diskann_bin = path.resolve()
                    break
            else:
                raise FileNotFoundError("DiskANN binary not found. Build with: cargo build --release")
        
        print(f"Using DiskANN binary: {self.diskann_bin}")
    
    def build(self, vectors_path: str, metadata_path: str, 
              max_degree: int = 32, search_list_size: int = 64, 
              alpha: float = 1.2) -> None:
        """Build disk index from vectors."""
        
        print("\n" + "="*60)
        print("Building Disk Index")
        print("="*60)
        
        # Copy vectors to index directory
        print("1. Copying vectors to index directory...")
        import shutil
        shutil.copy(vectors_path, self.index.vectors_path)
        shutil.copy(metadata_path, self.index.metadata_path)
        
        # Build graph using DiskANN
        print("2. Building graph structure...")
        start_time = time.time()
        
        cmd = [
            str(self.diskann_bin),
            'build',
            '-i', str(self.index.vectors_path),
            '-o', str(self.index.graph_path),
            '--max-degree', str(max_degree),
            '--search-list-size', str(search_list_size),
            '--alpha', str(alpha),
            '--seed', '42'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error building index: {result.stderr}")
            raise RuntimeError("Index building failed")
        
        build_time = time.time() - start_time
        
        # Load vector count and dimension
        with open(self.index.vectors_path, 'rb') as f:
            num_vectors, dimension = struct.unpack('II', f.read(8))
        
        # Save configuration
        config = {
            'num_vectors': num_vectors,
            'dimension': dimension,
            'max_degree': max_degree,
            'search_list_size': search_list_size,
            'alpha': alpha,
            'build_time_seconds': build_time,
            'index_size_mb': self.index.get_size_mb()
        }
        self.index.save_config(config)
        
        print(f"\n✓ Index built successfully in {build_time:.1f} seconds")
        print(f"  - Vectors: {num_vectors:,}")
        print(f"  - Dimensions: {dimension}")
        print(f"  - Index size: {config['index_size_mb']:.1f} MB")
        print(f"  - Location: {self.index.index_dir}")


class DiskIndexSearcher:
    """Memory-efficient searcher using disk index."""
    
    def __init__(self, index_dir: str, diskann_binary: str = None):
        self.index = DiskIndex(Path(index_dir))
        
        if not self.index.exists():
            raise FileNotFoundError(f"Index not found at {index_dir}")
        
        # Load configuration
        self.config = self.index.load_config()
        print(f"Loaded disk index: {self.config['num_vectors']:,} vectors, "
              f"{self.config['dimension']}D, {self.index.get_size_mb():.1f} MB")
        
        # Find DiskANN binary
        if diskann_binary:
            self.diskann_bin = Path(diskann_binary)
        else:
            possible_paths = [
                Path("../DiskANNInRust/target/release/diskann"),
                Path("../DiskANNInRust/target/debug/diskann"),
            ]
            for path in possible_paths:
                if path.exists():
                    self.diskann_bin = path.resolve()
                    break
        
        # Memory-map metadata for efficient access
        self.metadata_map = self._load_metadata_mmap()
        
        # Initialize encoder
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    def _load_metadata_mmap(self) -> Dict[int, str]:
        """Load metadata using memory mapping for efficiency."""
        metadata = {}
        with open(self.index.metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    metadata[int(parts[0])] = parts[1]
        return metadata
    
    def search(self, query: str, k: int = 5, beam: int = 64) -> List[Tuple[int, float, str]]:
        """Search disk index with a query."""
        
        # Encode query
        query_vector = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        
        # Save query to temporary file
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
            tmp.write(struct.pack('I', len(query_vector)))
            tmp.write(query_vector.astype(np.float32).tobytes())
            query_file = tmp.name
        
        try:
            # Search using DiskANN
            cmd = [
                str(self.diskann_bin),
                'search',
                '-i', str(self.index.graph_path),
                '-q', query_file,
                '-k', str(k),
                '--beam', str(beam)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse results
            results = []
            for line in result.stdout.strip().split('\n'):
                if 'ID:' in line and 'Distance:' in line:
                    parts = line.split(',')
                    id_val = int(parts[0].split(':')[1].strip())
                    distance = float(parts[1].split(':')[1].strip())
                    text = self.metadata_map.get(id_val, "")
                    results.append((id_val, distance, text))
            
            return results
            
        finally:
            Path(query_file).unlink(missing_ok=True)
    
    def batch_search(self, queries: List[str], k: int = 5, 
                     beam: int = 64) -> List[List[Tuple[int, float, str]]]:
        """Batch search multiple queries efficiently."""
        
        print(f"Processing {len(queries)} queries...")
        all_results = []
        
        # Encode all queries at once
        query_vectors = self.encoder.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        # Search each query
        for i, (query, vector) in enumerate(zip(queries, query_vectors)):
            # Save query vector
            with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
                tmp.write(struct.pack('I', len(vector)))
                tmp.write(vector.astype(np.float32).tobytes())
                query_file = tmp.name
            
            try:
                cmd = [
                    str(self.diskann_bin),
                    'search',
                    '-i', str(self.index.graph_path),
                    '-q', query_file,
                    '-k', str(k),
                    '--beam', str(beam)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Parse results
                results = []
                for line in result.stdout.strip().split('\n'):
                    if 'ID:' in line and 'Distance:' in line:
                        parts = line.split(',')
                        id_val = int(parts[0].split(':')[1].strip())
                        distance = float(parts[1].split(':')[1].strip())
                        text = self.metadata_map.get(id_val, "")
                        results.append((id_val, distance, text))
                
                all_results.append(results)
                
            finally:
                Path(query_file).unlink(missing_ok=True)
        
        return all_results


def benchmark_disk_index(searcher: DiskIndexSearcher):
    """Run benchmarks on disk index."""
    
    print("\n" + "="*60)
    print("Disk Index Benchmark")
    print("="*60)
    
    queries = [
        "A person is riding a horse",
        "The weather is beautiful today",
        "Scientists discovered a new species",
        "The stock market crashed yesterday",
        "Children are playing in the park",
        "How to cook pasta",
        "The benefits of exercise",
        "Climate change effects",
        "Space exploration missions",
        "Artificial intelligence applications"
    ]
    
    # Warm-up search
    _ = searcher.search("test", k=1)
    
    # Individual searches
    print("\nIndividual search times:")
    total_time = 0
    for query in queries[:5]:
        start = time.time()
        results = searcher.search(query, k=5)
        elapsed = time.time() - start
        total_time += elapsed
        print(f"  '{query[:30]}...': {elapsed*1000:.1f} ms")
    
    avg_time = total_time / 5
    print(f"\nAverage search time: {avg_time*1000:.1f} ms")
    
    # Batch search
    print("\nBatch search (10 queries):")
    start = time.time()
    batch_results = searcher.batch_search(queries, k=5)
    batch_time = time.time() - start
    print(f"  Total time: {batch_time:.2f} seconds")
    print(f"  Per query: {batch_time/len(queries)*1000:.1f} ms")
    
    # Memory usage
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 * 1024)
    print(f"\nMemory usage: {memory_mb:.1f} MB")
    print(f"Index size on disk: {searcher.index.get_size_mb():.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="STSB Disk Index Demo")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build disk index')
    build_parser.add_argument('--vectors', default='stsb_vectors.bin',
                             help='Input vectors file')
    build_parser.add_argument('--metadata', default='stsb_metadata.tsv',
                             help='Metadata file')
    build_parser.add_argument('--index-dir', default='stsb_disk_index',
                             help='Output index directory')
    build_parser.add_argument('--max-degree', type=int, default=32,
                             help='Max degree for graph')
    build_parser.add_argument('--search-list-size', type=int, default=64,
                             help='Search list size for construction')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search disk index')
    search_parser.add_argument('--index-dir', default='stsb_disk_index',
                              help='Index directory')
    search_parser.add_argument('--query', type=str,
                              help='Query text (if not provided, interactive mode)')
    search_parser.add_argument('--k', type=int, default=5,
                              help='Number of results')
    search_parser.add_argument('--benchmark', action='store_true',
                              help='Run benchmarks')
    
    args = parser.parse_args()
    
    if args.command == 'build':
        # Build disk index
        builder = DiskIndexBuilder(args.index_dir)
        builder.build(
            args.vectors,
            args.metadata,
            max_degree=args.max_degree,
            search_list_size=args.search_list_size
        )
        
    elif args.command == 'search':
        # Search disk index
        searcher = DiskIndexSearcher(args.index_dir)
        
        if args.benchmark:
            benchmark_disk_index(searcher)
        elif args.query:
            # Single query
            print(f"\nSearching for: '{args.query}'")
            results = searcher.search(args.query, k=args.k)
            
            print("\nResults:")
            for i, (id_val, distance, text) in enumerate(results, 1):
                similarity = 1 - distance
                print(f"{i}. [ID: {id_val}] (Similarity: {similarity:.3f})")
                print(f"   {text}")
        else:
            # Interactive mode
            print("\nInteractive Search (type 'quit' to exit)")
            print("-" * 40)
            
            while True:
                try:
                    query = input("\nQuery: ").strip()
                    if query.lower() in ['quit', 'exit']:
                        break
                    
                    if not query:
                        continue
                    
                    start = time.time()
                    results = searcher.search(query, k=5)
                    elapsed = time.time() - start
                    
                    print(f"\nResults ({elapsed*1000:.1f} ms):")
                    for i, (id_val, distance, text) in enumerate(results, 1):
                        similarity = 1 - distance
                        print(f"{i}. (Similarity: {similarity:.3f}) {text[:80]}...")
                    
                except KeyboardInterrupt:
                    break
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()