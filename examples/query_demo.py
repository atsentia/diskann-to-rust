#!/usr/bin/env python3
"""
Interactive Text Search Demo

This script demonstrates interactive text search using the DiskANN-Rust library
through the FFI layer. It loads a pre-built index and allows users to enter
natural language queries to find relevant passages.

Usage:
    python query_demo.py --index msmarco.disk.index --metadata msmarco_passages.tsv
"""

import argparse
import ctypes
import struct
import sys
from pathlib import Path
from typing import List, Tuple, Dict

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install required packages:")
    print("pip install sentence-transformers numpy")
    sys.exit(1)


# FFI Structures matching the Rust definitions
class SearchResultC(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint),
        ("distance", ctypes.c_float)
    ]


class DiskAnnError(ctypes.c_int):
    SUCCESS = 0
    INVALID_ARGUMENT = 1
    OUT_OF_MEMORY = 2
    IO_ERROR = 3
    BUILD_ERROR = 4
    SEARCH_ERROR = 5


class DiskAnnFFI:
    """Wrapper for DiskANN FFI operations."""
    
    def __init__(self, library_path: str = None):
        """Load the DiskANN shared library."""
        if library_path is None:
            # Try to find the library in common locations
            possible_paths = [
                "./target/release/libdiskann_ffi.so",
                "./target/debug/libdiskann_ffi.so", 
                "./DiskANNInRust/target/release/libdiskann_ffi.so",
                "./DiskANNInRust/target/debug/libdiskann_ffi.so",
                "libdiskann_ffi.so"
            ]
            
            library_path = None
            for path in possible_paths:
                if Path(path).exists():
                    library_path = path
                    break
            
            if library_path is None:
                raise FileNotFoundError(
                    "Could not find DiskANN FFI library. Please build the library first:\n"
                    "cd DiskANNInRust && cargo build --release"
                )
        
        self.lib = ctypes.CDLL(library_path)
        self._setup_function_signatures()
    
    def _setup_function_signatures(self):
        """Setup ctypes function signatures for FFI calls."""
        # diskann_load_index
        self.lib.diskann_load_index.argtypes = [ctypes.c_char_p]
        self.lib.diskann_load_index.restype = ctypes.c_void_p
        
        # diskann_search
        self.lib.diskann_search.argtypes = [
            ctypes.c_void_p,           # handle
            ctypes.POINTER(ctypes.c_float),  # query
            ctypes.c_uint,             # query_dim
            ctypes.c_uint,             # k
            ctypes.c_uint,             # beam_width
            ctypes.POINTER(SearchResultC),   # results
            ctypes.POINTER(ctypes.c_uint),   # results_len
        ]
        self.lib.diskann_search.restype = ctypes.c_int
        
        # diskann_destroy_index
        self.lib.diskann_destroy_index.argtypes = [ctypes.c_void_p]
        self.lib.diskann_destroy_index.restype = None
        
        # diskann_get_version
        self.lib.diskann_get_version.argtypes = []
        self.lib.diskann_get_version.restype = ctypes.c_char_p
    
    def load_index(self, index_path: str) -> ctypes.c_void_p:
        """Load index from file."""
        return self.lib.diskann_load_index(index_path.encode('utf-8'))
    
    def search(self, handle: ctypes.c_void_p, query: np.ndarray, k: int = 10, beam_width: int = 64) -> List[Tuple[int, float]]:
        """Search for k nearest neighbors."""
        if handle == 0:
            raise ValueError("Invalid index handle")
        
        # Prepare query array
        query_array = (ctypes.c_float * len(query))(*query.astype(np.float32))
        
        # Prepare results array
        results_array = (SearchResultC * k)()
        results_len = ctypes.c_uint()
        
        # Call FFI function
        error = self.lib.diskann_search(
            handle,
            query_array,
            len(query),
            k,
            beam_width,
            results_array,
            ctypes.byref(results_len)
        )
        
        if error != DiskAnnError.SUCCESS:
            raise RuntimeError(f"Search failed with error code: {error}")
        
        # Convert results to Python list
        results = []
        for i in range(min(results_len.value, k)):
            results.append((results_array[i].id, results_array[i].distance))
        
        return results
    
    def destroy_index(self, handle: ctypes.c_void_p):
        """Clean up index handle."""
        if handle != 0:
            self.lib.diskann_destroy_index(handle)
    
    def get_version(self) -> str:
        """Get library version."""
        version_ptr = self.lib.diskann_get_version()
        return ctypes.string_at(version_ptr).decode('utf-8')


def load_metadata(metadata_path: str) -> Dict[str, str]:
    """Load passage metadata from TSV file."""
    metadata = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    passage_id, passage_text = parts
                    metadata[passage_id] = passage_text
    return metadata


def create_simple_demo_index(ffi: DiskAnnFFI) -> ctypes.c_void_p:
    """Create a simple demo index for testing when no real index is available."""
    print("Creating demo index for testing...")
    
    # Simple demo vectors (would normally be loaded from file)
    demo_vectors = np.array([
        [1.0, 0.0, 0.0],  # ID 0
        [0.0, 1.0, 0.0],  # ID 1  
        [0.0, 0.0, 1.0],  # ID 2
        [0.5, 0.5, 0.0],  # ID 3
        [0.0, 0.5, 0.5],  # ID 4
    ], dtype=np.float32)
    
    # Expand to 384 dimensions to match sentence transformer
    full_vectors = np.zeros((5, 384), dtype=np.float32)
    full_vectors[:, :3] = demo_vectors
    
    # Use FFI to build index (this would normally load from file)
    # For now, return a placeholder handle (null) to test the UI
    return ctypes.c_void_p(0)


def main():
    parser = argparse.ArgumentParser(description="Interactive text search demo")
    parser.add_argument(
        "--index",
        type=str,
        help="Path to DiskANN index file"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        help="Path to passage metadata TSV file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model for query encoding"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to return"
    )
    parser.add_argument(
        "--beam",
        type=int,
        default=64,
        help="Beam width for search"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode without requiring real index"
    )
    
    args = parser.parse_args()
    
    # Load sentence transformer for query encoding
    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)
    
    # Load metadata if available
    metadata = {}
    if args.metadata and Path(args.metadata).exists():
        print(f"Loading metadata from {args.metadata}")
        metadata = load_metadata(args.metadata)
        print(f"Loaded metadata for {len(metadata)} passages")
    
    # Initialize FFI
    try:
        ffi = DiskAnnFFI()
        print(f"DiskANN FFI version: {ffi.get_version()}")
    except Exception as e:
        if args.demo:
            print(f"Warning: Could not load FFI library ({e})")
            print("Running in demo mode without actual search")
            ffi = None
        else:
            print(f"Error: Could not load FFI library: {e}")
            print("Build the library first: cd DiskANNInRust && cargo build --release")
            sys.exit(1)
    
    # Load or create index
    index_handle = None
    if args.index and Path(args.index).exists() and ffi:
        print(f"Loading index from {args.index}")
        index_handle = ffi.load_index(args.index)
        if index_handle == 0:
            print("Warning: Failed to load index, using demo mode")
            index_handle = None
    elif args.demo and ffi:
        index_handle = create_simple_demo_index(ffi)
    
    print("\n" + "="*60)
    print("DiskANN Text Search Demo")
    print("="*60)
    print("Enter natural language queries to search for relevant passages.")
    print("Type 'quit' or 'exit' to stop.")
    print("="*60)
    
    try:
        while True:
            # Get user query
            query_text = input("\nQuery: ").strip()
            
            if query_text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query_text:
                continue
            
            try:
                # Encode query
                print("Encoding query...")
                query_embedding = model.encode([query_text], convert_to_numpy=True)[0]
                
                # Search
                if ffi and index_handle:
                    print("Searching index...")
                    results = ffi.search(index_handle, query_embedding, args.k, args.beam)
                else:
                    # Demo mode - simulate results
                    print("Demo mode: simulating search results...")
                    results = [
                        (0, 0.1234),
                        (1, 0.2345), 
                        (2, 0.3456),
                        (3, 0.4567),
                        (4, 0.5678)
                    ][:args.k]
                
                # Display results
                print(f"\nTop {len(results)} results:")
                print("-" * 80)
                
                for rank, (passage_id, distance) in enumerate(results, 1):
                    passage_text = metadata.get(str(passage_id), f"[No metadata for ID {passage_id}]")
                    
                    # Truncate long passages
                    if len(passage_text) > 200:
                        passage_text = passage_text[:197] + "..."
                    
                    print(f"{rank:2d}. ID: {passage_id:6d} | Distance: {distance:.4f}")
                    print(f"    {passage_text}")
                    print()
                
            except Exception as e:
                print(f"Error processing query: {e}")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        # Cleanup
        if ffi and index_handle:
            ffi.destroy_index(index_handle)
        print("Demo finished.")


if __name__ == "__main__":
    main()