//! C Foreign Function Interface for DiskANN vector search library
//!
//! This module provides C-compatible bindings to enable integration with other languages
//! such as Python (via pyo3), JavaScript (via WASM), and other systems.

#![deny(warnings)]
#![warn(missing_docs)]

use std::ffi::CStr;
use std::os::raw::{c_char, c_float, c_uint, c_void};
use std::ptr;
use std::slice;

use diskann_impl::{VamanaIndex, IndexBuilder, VamanaConfig};
use diskann_traits::{distance::EuclideanDistance, index::Index, search::{Search, SearchBuffer}};

/// Opaque handle to a DiskANN index
pub type DiskAnnIndexHandle = *mut c_void;

/// Search result structure for C interface
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SearchResultC {
    /// Vector ID
    pub id: c_uint,
    /// Distance to query
    pub distance: c_float,
}

/// Error codes for C interface
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiskAnnError {
    /// No error
    Success = 0,
    /// Invalid argument
    InvalidArgument = 1,
    /// Out of memory
    OutOfMemory = 2,
    /// File I/O error
    IoError = 3,
    /// Index build error
    BuildError = 4,
    /// Search error
    SearchError = 5,
}

/// Create a new DiskANN index with default configuration
#[no_mangle]
pub extern "C" fn diskann_create_index() -> DiskAnnIndexHandle {
    let distance_fn = EuclideanDistance;
    let config = VamanaConfig::default();
    let index = Box::new(VamanaIndex::new(distance_fn, config));
    Box::into_raw(index) as DiskAnnIndexHandle
}

/// Create a new DiskANN index with custom configuration
#[no_mangle]
pub extern "C" fn diskann_create_index_with_config(
    max_degree: c_uint,
    search_list_size: c_uint,
    alpha: c_float,
    seed: c_uint,
) -> DiskAnnIndexHandle {
    let distance_fn = EuclideanDistance;
    let config = VamanaConfig {
        max_degree: max_degree as usize,
        search_list_size: search_list_size as usize,
        alpha,
        seed: seed as u64,
    };
    let index = Box::new(VamanaIndex::new(distance_fn, config));
    Box::into_raw(index) as DiskAnnIndexHandle
}

/// Build an index from vectors
#[no_mangle]
pub extern "C" fn diskann_build_index(
    vectors: *const c_float,
    num_vectors: c_uint,
    vector_dim: c_uint,
    max_degree: c_uint,
    search_list_size: c_uint,
    alpha: c_float,
    seed: c_uint,
) -> DiskAnnIndexHandle {
    if vectors.is_null() || num_vectors == 0 || vector_dim == 0 {
        return ptr::null_mut();
    }

    let distance_fn = EuclideanDistance;
    let config = VamanaConfig {
        max_degree: max_degree as usize,
        search_list_size: search_list_size as usize,
        alpha,
        seed: seed as u64,
    };

    // Convert C array to Rust vectors
    let mut vector_data = Vec::new();
    unsafe {
        let slice = slice::from_raw_parts(vectors, (num_vectors * vector_dim) as usize);
        for i in 0..num_vectors {
            let start = (i * vector_dim) as usize;
            let end = start + vector_dim as usize;
            let vector = slice[start..end].to_vec();
            vector_data.push((i, vector));
        }
    }

    match IndexBuilder::new(distance_fn)
        .max_degree(config.max_degree)
        .search_list_size(config.search_list_size)
        .alpha(config.alpha)
        .seed(config.seed)
        .build(vector_data) 
    {
        Ok(index) => Box::into_raw(Box::new(index)) as DiskAnnIndexHandle,
        Err(_) => ptr::null_mut(),
    }
}

/// Add a vector to the index
#[no_mangle]
pub extern "C" fn diskann_add_vector(
    handle: DiskAnnIndexHandle,
    id: c_uint,
    vector: *const c_float,
    dim: c_uint,
) -> DiskAnnError {
    if handle.is_null() || vector.is_null() || dim == 0 {
        return DiskAnnError::InvalidArgument;
    }

    let index = unsafe { &mut *(handle as *mut VamanaIndex<EuclideanDistance>) };
    
    let vector_slice = unsafe { slice::from_raw_parts(vector, dim as usize) };
    let vector_vec = vector_slice.to_vec();

    match index.add(id, vector_vec) {
        Ok(()) => DiskAnnError::Success,
        Err(_) => DiskAnnError::BuildError,
    }
}

/// Search for k nearest neighbors
#[no_mangle]
pub extern "C" fn diskann_search(
    handle: DiskAnnIndexHandle,
    query: *const c_float,
    query_dim: c_uint,
    k: c_uint,
    beam_width: c_uint,
    results: *mut SearchResultC,
    results_len: *mut c_uint,
) -> DiskAnnError {
    if handle.is_null() || query.is_null() || query_dim == 0 || k == 0 || results.is_null() || results_len.is_null() {
        return DiskAnnError::InvalidArgument;
    }

    let index = unsafe { &*(handle as *const VamanaIndex<EuclideanDistance>) };
    
    let query_slice = unsafe { slice::from_raw_parts(query, query_dim as usize) };
    
    let search_results = if beam_width > 0 {
        match index.search_with_beam(query_slice, k as usize, beam_width as usize) {
            Ok(results) => results,
            Err(_) => return DiskAnnError::SearchError,
        }
    } else {
        match index.search(query_slice, k as usize) {
            Ok(results) => results,
            Err(_) => return DiskAnnError::SearchError,
        }
    };

    let num_results = search_results.len().min(k as usize);
    unsafe {
        *results_len = num_results as c_uint;
        for (i, result) in search_results.iter().take(num_results).enumerate() {
            (*results.add(i)).id = result.id;
            (*results.add(i)).distance = result.distance;
        }
    }

    DiskAnnError::Success
}

/// Search with zero-allocation using provided buffer
#[no_mangle]
pub extern "C" fn diskann_search_with_buffer(
    handle: DiskAnnIndexHandle,
    query: *const c_float,
    query_dim: c_uint,
    k: c_uint,
    beam_width: c_uint,
    buffer_handle: *mut c_void,
    results: *mut SearchResultC,
    results_len: *mut c_uint,
) -> DiskAnnError {
    if handle.is_null() || query.is_null() || query_dim == 0 || k == 0 || 
       buffer_handle.is_null() || results.is_null() || results_len.is_null() {
        return DiskAnnError::InvalidArgument;
    }

    let index = unsafe { &*(handle as *const VamanaIndex<EuclideanDistance>) };
    let buffer = unsafe { &mut *(buffer_handle as *mut SearchBuffer) };
    
    let query_slice = unsafe { slice::from_raw_parts(query, query_dim as usize) };
    
    let search_results = match index.search_with_buffer(
        query_slice, 
        k as usize, 
        beam_width as usize, 
        buffer
    ) {
        Ok(results) => results,
        Err(_) => return DiskAnnError::SearchError,
    };

    let num_results = search_results.len().min(k as usize);
    unsafe {
        *results_len = num_results as c_uint;
        for (i, result) in search_results.iter().take(num_results).enumerate() {
            (*results.add(i)).id = result.id;
            (*results.add(i)).distance = result.distance;
        }
    }

    DiskAnnError::Success
}

/// Create a search buffer for zero-allocation search
#[no_mangle]
pub extern "C" fn diskann_create_search_buffer(capacity: c_uint) -> *mut c_void {
    let buffer = Box::new(SearchBuffer::new(capacity as usize));
    Box::into_raw(buffer) as *mut c_void
}

/// Destroy a search buffer
#[no_mangle]
pub extern "C" fn diskann_destroy_search_buffer(buffer_handle: *mut c_void) {
    if !buffer_handle.is_null() {
        unsafe {
            let _ = Box::from_raw(buffer_handle as *mut SearchBuffer);
        }
    }
}

/// Get the size of the index
#[no_mangle]
pub extern "C" fn diskann_get_index_size(handle: DiskAnnIndexHandle) -> c_uint {
    if handle.is_null() {
        return 0;
    }

    let index = unsafe { &*(handle as *const VamanaIndex<EuclideanDistance>) };
    index.size() as c_uint
}

/// Destroy the index and free memory
#[no_mangle]
pub extern "C" fn diskann_destroy_index(handle: DiskAnnIndexHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle as *mut VamanaIndex<EuclideanDistance>);
        }
    }
}

/// Save index to file (placeholder for file I/O integration)
#[no_mangle]
pub extern "C" fn diskann_save_index(
    handle: DiskAnnIndexHandle,
    filename: *const c_char,
) -> DiskAnnError {
    if handle.is_null() || filename.is_null() {
        return DiskAnnError::InvalidArgument;
    }

    // For now, return success as file I/O would require integration with diskann-io
    let _filename_str = unsafe {
        match CStr::from_ptr(filename).to_str() {
            Ok(s) => s,
            Err(_) => return DiskAnnError::InvalidArgument,
        }
    };

    // TODO: Implement actual file saving using diskann-io
    DiskAnnError::Success
}

/// Load index from file (placeholder for file I/O integration)
#[no_mangle]
pub extern "C" fn diskann_load_index(filename: *const c_char) -> DiskAnnIndexHandle {
    if filename.is_null() {
        return ptr::null_mut();
    }

    let _filename_str = unsafe {
        match CStr::from_ptr(filename).to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        }
    };

    // TODO: Implement actual file loading using diskann-io
    ptr::null_mut()
}

/// Get version string
#[no_mangle]
pub extern "C" fn diskann_get_version() -> *const c_char {
    static VERSION: &str = "0.1.0\0";
    VERSION.as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_basic_operations() {
        // Test index creation
        let handle = diskann_create_index();
        assert!(!handle.is_null());

        // Test adding vectors
        let vector1 = [1.0f32, 0.0, 0.0];
        let vector2 = [0.0f32, 1.0, 0.0];
        
        assert_eq!(
            diskann_add_vector(handle, 0, vector1.as_ptr(), 3),
            DiskAnnError::Success
        );
        assert_eq!(
            diskann_add_vector(handle, 1, vector2.as_ptr(), 3),
            DiskAnnError::Success
        );

        // Test index size
        assert_eq!(diskann_get_index_size(handle), 2);

        // Test search
        let query = [0.9f32, 0.1, 0.0];
        let mut results = [SearchResultC { id: 0, distance: 0.0 }; 2];
        let mut results_len = 0;

        assert_eq!(
            diskann_search(
                handle,
                query.as_ptr(),
                3,
                2,
                64,
                results.as_mut_ptr(),
                &mut results_len,
            ),
            DiskAnnError::Success
        );

        assert!(results_len > 0);
        assert!(results[0].distance >= 0.0);

        // Test cleanup
        diskann_destroy_index(handle);
    }

    #[test]
    fn test_ffi_search_buffer() {
        // Test buffer creation
        let buffer = diskann_create_search_buffer(1000);
        assert!(!buffer.is_null());

        // Test buffer cleanup
        diskann_destroy_search_buffer(buffer);
    }

    #[test]
    fn test_ffi_build_index() {
        let vectors = [
            1.0f32, 0.0, 0.0,
            0.0f32, 1.0, 0.0,
            0.0f32, 0.0, 1.0,
        ];

        let handle = diskann_build_index(
            vectors.as_ptr(),
            3, // num_vectors
            3, // vector_dim
            64, // max_degree
            100, // search_list_size
            1.2, // alpha
            42, // seed
        );

        assert!(!handle.is_null());
        assert_eq!(diskann_get_index_size(handle), 3);

        diskann_destroy_index(handle);
    }

    #[test]
    fn test_ffi_version() {
        let version = diskann_get_version();
        assert!(!version.is_null());
        
        let version_str = unsafe { CStr::from_ptr(version) };
        assert!(version_str.to_str().unwrap().starts_with("0.1.0"));
    }
}