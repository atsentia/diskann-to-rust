# DiskANN-to-Rust Repository

DiskANN-to-Rust contains a mature C++ implementation of DiskANN (scalable approximate nearest neighbor search) and an early-stage Rust port. The C++ DiskANN provides highly optimized vector search algorithms for large-scale datasets with support for memory and disk-based indices.

**ALWAYS reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Bootstrap and Build Process

**CRITICAL: Set timeout to 60+ minutes for build commands. NEVER CANCEL builds as they may take 5-10 minutes.**

Bootstrap dependencies (Ubuntu):
```bash
cd DiskANN
sudo scripts/dev/install-dev-deps-ubuntu.bash
```
- Time: ~2 minutes
- Installs: cmake, g++, libaio-dev, libgoogle-perftools-dev, libunwind-dev, clang-format, libboost-dev, libboost-program-options-dev, libboost-test-dev, libmkl-full-dev

Configure build:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUNIT_TEST=True
```
- Time: ~2 seconds
- Creates build configuration with unit tests enabled

Build the project:
```bash
cmake --build build -- -j
```
- Time: **5-6 minutes** - NEVER CANCEL. Set timeout to 60+ minutes.
- Builds all libraries, CLI tools, and test utilities
- **KNOWN ISSUE**: If build fails with "uint32_t has not been declared" error in ann_exception.h, add `#include <cstdint>` to the header file

Install binaries:
```bash
cmake --install build --prefix="dist"
```
- Time: ~1 second
- Installs executables to dist/bin/ and libraries to dist/lib/

### Testing

Run unit tests:
```bash
cd build && ctest -C Release
```
- Time: ~1 second - NEVER CANCEL. Set timeout to 30+ minutes for large test suites.
- Validates core functionality

### Code Quality

Format code:
```bash
make -C build format
```
- Time: ~1 second
- Applies clang-format to all source files

Check formatting:
```bash
make -C build checkformat
```
- Time: ~1 second
- Validates code formatting without changes
- **ALWAYS** run before committing to pass CI

## Validation

### Manual Testing Workflow
After making changes, **ALWAYS** validate with this complete scenario:

1. Generate test data:
```bash
mkdir -p /tmp/diskann_test
build/apps/utils/rand_data_gen --data_type float --output_file /tmp/diskann_test/test_base.bin -N 1000 -D 32
build/apps/utils/rand_data_gen --data_type float --output_file /tmp/diskann_test/test_query.bin -N 100 -D 32
```

2. Build memory index:
```bash
build/apps/build_memory_index --data_type float --dist_fn l2 --data_path /tmp/diskann_test/test_base.bin --index_path_prefix /tmp/diskann_test/test_index -R 32 -L 50 --alpha 1.2 -T 1
```

3. Compute ground truth:
```bash
build/apps/utils/compute_groundtruth --data_type float --base_file /tmp/diskann_test/test_base.bin --query_file /tmp/diskann_test/test_query.bin --K 10 --gt_file /tmp/diskann_test/test_gt --dist_fn l2
```

4. Search index:
```bash
build/apps/search_memory_index --data_type float --dist_fn l2 --index_path_prefix /tmp/diskann_test/test_index -T 1 --query_file /tmp/diskann_test/test_query.bin --gt_file /tmp/diskann_test/test_gt.bin -K 10 --result_path /tmp/diskann_test/result -L 10 20 30
```

This workflow validates the complete build, index, and search pipeline is functioning correctly.

### Python Extension

Python package installation may fail due to network issues with numpy dependencies:
```bash
pip install . --verbose
```
- **KNOWN ISSUE**: Installation often fails in restricted environments due to numpy download timeouts
- The C++ CLI tools are the primary validated interface

## Project Structure

### Key Directories
- `DiskANN/` - Mature C++ implementation with full feature set
- `DiskANNInRust/` - Minimal Rust port (early development, just README)
- `DiskANN/src/` - Core DiskANN library source code
- `DiskANN/include/` - Header files
- `DiskANN/apps/` - CLI applications and tools
- `DiskANN/apps/utils/` - Utility programs for data conversion and analysis
- `DiskANN/tests/` - Unit tests
- `DiskANN/workflows/` - Documentation for common usage patterns
- `DiskANN/.github/workflows/` - CI/CD pipeline definitions

### Key CLI Tools (in build/apps/ after building)
- `build_memory_index` - Build in-memory ANN index
- `search_memory_index` - Search in-memory index
- `build_disk_index` - Build disk-based index for large datasets
- `search_disk_index` - Search disk-based index
- `compute_groundtruth` - Generate ground truth for evaluation
- `rand_data_gen` - Generate random test data

### Key Utility Tools (in build/apps/utils/)
- `fvecs_to_bin` - Convert FVECS format to binary
- `calculate_recall` - Compute recall metrics
- `vector_analysis` - Analyze vector properties

## Common Tasks

### Building Different Configurations
- Release build: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
- Debug build: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug`
- With unit tests: Add `-DUNIT_TEST=True`
- With Python bindings: Add `-DPYBIND=True`

### Data Format Requirements
- Binary format: 4 bytes (num_points) + 4 bytes (dimensions) + point_data
- Supported types: float (32-bit), int8, uint8
- Distance functions: l2 (Euclidean), mips (max inner product), cosine

### Performance Notes
- Use `-j` flag with cmake --build for parallel compilation
- Index building is CPU intensive and benefits from multiple threads
- Memory requirements scale with dataset size and index parameters

## CI Integration

The repository includes comprehensive GitHub Actions workflows in `DiskANN/.github/workflows/`:
- `push-test.yml` - Main build and test pipeline
- `unit-tests.yml` - Unit test execution
- Format checking and dependency validation

**ALWAYS** run `make -C build checkformat` before committing to ensure CI passes.

## Common Issues

1. **Missing cstdint include**: Add `#include <cstdint>` to ann_exception.h if build fails
2. **Python installation timeout**: Use C++ CLI tools directly instead of Python wrapper
3. **MKL library detection**: Ensure Intel MKL is properly installed via package manager
4. **Build timeouts**: Always use 60+ minute timeouts for build commands

## Reference Commands

Quick reference for frequently used commands:

```bash
# Full build from scratch
cd DiskANN
sudo scripts/dev/install-dev-deps-ubuntu.bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUNIT_TEST=True
cmake --build build -- -j  # NEVER CANCEL - 5-6 minutes
cd build && ctest -C Release

# Code quality
make -C build checkformat
make -C build format

# Basic validation test
mkdir -p /tmp/test && cd /tmp/test
../DiskANN/build/apps/utils/rand_data_gen --data_type float --output_file base.bin -N 1000 -D 32
../DiskANN/build/apps/build_memory_index --data_type float --dist_fn l2 --data_path base.bin --index_path_prefix index -R 32 -L 50
```