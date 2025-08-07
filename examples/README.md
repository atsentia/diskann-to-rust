# DiskANN Text Search Demo Examples

This directory contains Python scripts for the end-to-end text search demonstration using the MS MARCO Passage dataset.

## Files

- `make_msmarco_embeddings.py` - Downloads MS MARCO passages and generates embeddings
- `query_demo.py` - Interactive search interface using DiskANN FFI
- `requirements.txt` - Python dependencies

## Quick Start

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Generate embeddings:
```bash
python make_msmarco_embeddings.py --max-passages 1000  # For testing
```

3. Build DiskANN index:
```bash
cd ../DiskANNInRust
cargo build --release --bin diskann
./target/release/diskann build -i ../examples/msmarco_passages.bin -o ../examples/msmarco.disk.index
```

4. Run interactive search:
```bash
python query_demo.py --index msmarco.disk.index --metadata msmarco_passages.tsv
```

## Demo Mode

For testing without the full setup, run:
```bash
python query_demo.py --demo
```