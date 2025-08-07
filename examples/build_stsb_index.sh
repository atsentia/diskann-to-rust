#!/bin/bash
# Build script for STSB DiskANN Demo
# This script builds the Rust DiskANN binary and creates the search index

set -e  # Exit on error

echo "=================================================="
echo "STSB DiskANN Demo - Build Script"
echo "=================================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Build Rust DiskANN in release mode
echo -e "\n${YELLOW}Step 1: Building DiskANN Rust binary...${NC}"
cd ../DiskANNInRust
cargo build --release --bin diskann
echo -e "${GREEN}✓ DiskANN binary built successfully${NC}"

# Step 2: Generate STSB embeddings if not present
cd ../examples
if [ ! -f "stsb_vectors.bin" ]; then
    echo -e "\n${YELLOW}Step 2: Generating STSB embeddings...${NC}"
    python stsb_demo.py --output-dir .
    echo -e "${GREEN}✓ Embeddings generated successfully${NC}"
else
    echo -e "\n${GREEN}✓ STSB embeddings already exist${NC}"
fi

# Step 3: Build DiskANN index
echo -e "\n${YELLOW}Step 3: Building DiskANN index...${NC}"
echo "Parameters:"
echo "  - Max degree (R): 32"
echo "  - Search list size (L): 64"
echo "  - Alpha: 1.2"
echo ""

START_TIME=$(date +%s)
../DiskANNInRust/target/release/diskann build \
    -i stsb_vectors.bin \
    -o stsb.index \
    --max-degree 32 \
    --search-list-size 64 \
    --alpha 1.2 \
    --seed 42
END_TIME=$(date +%s)
BUILD_TIME=$((END_TIME - START_TIME))

echo -e "${GREEN}✓ Index built in ${BUILD_TIME} seconds${NC}"

# Step 4: Verify index file
if [ -f "stsb.index" ]; then
    INDEX_SIZE=$(du -h stsb.index | cut -f1)
    echo -e "\n${GREEN}✓ Index created successfully (size: ${INDEX_SIZE})${NC}"
else
    echo -e "\n${YELLOW}Warning: Index file not found${NC}"
    exit 1
fi

# Step 5: Run test search
echo -e "\n${YELLOW}Step 4: Running test search...${NC}"

# Create a test query file with a sample vector
python -c "
import struct
import numpy as np
from sentence_transformers import SentenceTransformer

# Generate test query
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
query_vec = model.encode(['A person is walking'], normalize_embeddings=True)[0]

# Save as binary
with open('test_query.bin', 'wb') as f:
    f.write(struct.pack('I', len(query_vec)))
    f.write(query_vec.astype(np.float32).tobytes())
print('Test query saved')
"

# Run test search
echo "Testing search with query: 'A person is walking'"
../DiskANNInRust/target/release/diskann search \
    -i stsb.index \
    -q test_query.bin \
    -k 5 \
    --beam 64

# Clean up test query
rm -f test_query.bin

echo -e "\n${GREEN}=================================================="
echo "Build Complete!"
echo "=================================================="
echo ""
echo "Index Statistics:"
echo "  - Index file: stsb.index (${INDEX_SIZE})"
echo "  - Build time: ${BUILD_TIME} seconds"
echo ""
echo "To run interactive search:"
echo "  python stsb_search.py --index stsb.index --metadata stsb_metadata.tsv"
echo ""
echo "To run benchmarks:"
echo "  python stsb_search.py --benchmark"
echo "==================================================${NC}"