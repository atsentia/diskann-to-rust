#!/usr/bin/env python3
"""
Simple integration test for the text search demo.
Tests the basic functionality without requiring heavy dependencies.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

def run_command(cmd, cwd=None, timeout=60):
    """Run a command and return success status."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            cwd=cwd, 
            timeout=timeout,
            capture_output=True, 
            text=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"

def test_cli_build():
    """Test that the CLI builds successfully."""
    print("Testing CLI build...")
    
    diskann_dir = Path(__file__).parent.parent / "DiskANNInRust"
    success, stdout, stderr = run_command(
        "cargo build --bin diskann",
        cwd=diskann_dir,
        timeout=120
    )
    
    if success:
        print("✓ CLI build successful")
        return True
    else:
        print(f"✗ CLI build failed: {stderr}")
        return False

def test_cli_help():
    """Test that the CLI help works."""
    print("Testing CLI help...")
    
    diskann_dir = Path(__file__).parent.parent / "DiskANNInRust"
    success, stdout, stderr = run_command(
        "cargo run --bin diskann -- --help",
        cwd=diskann_dir
    )
    
    if success and "Usage: diskann" in stdout:
        print("✓ CLI help works")
        return True
    else:
        print(f"✗ CLI help failed: {stderr}")
        return False

def test_python_scripts_syntax():
    """Test that Python scripts have valid syntax."""
    print("Testing Python script syntax...")
    
    scripts = [
        "make_msmarco_embeddings.py",
        "query_demo.py"
    ]
    
    for script in scripts:
        script_path = Path(__file__).parent / script
        success, stdout, stderr = run_command(f"python3 -m py_compile {script_path}")
        
        if success:
            print(f"✓ {script} syntax OK")
        else:
            print(f"✗ {script} syntax error: {stderr}")
            return False
    
    return True

def test_file_structure():
    """Test that all required files exist."""
    print("Testing file structure...")
    
    base_dir = Path(__file__).parent
    required_files = [
        "make_msmarco_embeddings.py",
        "query_demo.py", 
        "requirements.txt",
        "README.md"
    ]
    
    for file in required_files:
        file_path = base_dir / file
        if file_path.exists():
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            return False
    
    return True

def test_documentation():
    """Test that documentation files exist."""
    print("Testing documentation...")
    
    base_dir = Path(__file__).parent.parent
    docs = [
        "README.md",
        "DiskANNInRust/README.md",
        "DiskANNInRust/docs/demo_text_embedding.md"
    ]
    
    for doc in docs:
        doc_path = base_dir / doc
        if doc_path.exists():
            print(f"✓ {doc} exists")
        else:
            print(f"✗ {doc} missing")
            return False
    
    return True

def create_test_vectors():
    """Create a small test vector file for testing."""
    print("Creating test vectors...")
    
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    
    vector_file = test_dir / "test_vectors.bin"
    
    try:
        import struct
        
        # Create 3 test vectors of dimension 4
        vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0, 0.0]
        ]
        
        with open(vector_file, 'wb') as f:
            # Write header
            f.write(struct.pack('<I', len(vectors)))  # num_vectors
            f.write(struct.pack('<I', 4))  # dimension
            
            # Write vectors
            for vector in vectors:
                for value in vector:
                    f.write(struct.pack('<f', value))
        
        print(f"✓ Created test vectors: {vector_file}")
        return str(vector_file)
        
    except Exception as e:
        print(f"✗ Failed to create test vectors: {e}")
        return None

def test_cli_with_test_data():
    """Test CLI with simple test data."""
    print("Testing CLI with test data...")
    
    # Create test data
    vector_file = create_test_vectors()
    if not vector_file:
        return False
    
    test_dir = Path(vector_file).parent
    index_file = test_dir / "test_index.bin"
    
    # Build index
    diskann_dir = Path(__file__).parent.parent / "DiskANNInRust"
    build_cmd = f"cargo run --bin diskann -- build -i {vector_file} -o {index_file}"
    
    success, stdout, stderr = run_command(build_cmd, cwd=diskann_dir, timeout=30)
    
    if success:
        print("✓ Index building successful")
        return True
    else:
        print(f"✗ Index building failed: {stderr}")
        return False

def main():
    """Run all tests."""
    print("=== DiskANN Text Search Demo Integration Test ===\n")
    
    tests = [
        test_file_structure,
        test_documentation,
        test_python_scripts_syntax,
        test_cli_build,
        test_cli_help,
        test_cli_with_test_data
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Blank line after each test
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}\n")
    
    print(f"=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All tests passed! Demo is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())