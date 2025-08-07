//! Memory-mapped I/O with safety wrappers and cross-platform fallbacks
//!
//! This module provides safe abstractions over memory-mapped files with proper
//! alignment guarantees and lifetime management.

use std::path::Path;
use std::fs::File;
use std::io::Read;
use anyhow::{Result, Context};
use diskann_core::utils;

#[cfg(feature = "mmap")]
use memmap2::Mmap;

/// A safe wrapper around memory-mapped data with alignment guarantees
pub struct SafeMmap {
    #[cfg(feature = "mmap")]
    _mmap: Mmap,
    data: *const u8,
    len: usize,
    alignment: usize,
}

unsafe impl Send for SafeMmap {}
unsafe impl Sync for SafeMmap {}

impl SafeMmap {
    /// Create a new memory-mapped file
    #[cfg(feature = "mmap")]
    pub fn new(file: File) -> Result<Self> {
        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .context("Failed to create memory map")?
        };
        
        let data = mmap.as_ptr();
        let len = mmap.len();
        
        // Check alignment - DiskANN typically requires 4-byte alignment for floats
        let alignment = utils::get_alignment(data as usize);
        if alignment < 4 {
            tracing::warn!("Memory map has poor alignment: {} bytes", alignment);
        }
        
        Ok(Self {
            _mmap: mmap,
            data,
            len,
            alignment,
        })
    }
    
    /// Fallback for platforms without mmap support
    #[cfg(not(feature = "mmap"))]
    pub fn new(_file: File) -> Result<Self> {
        compile_error!("Memory mapping not available on this platform. Use buffered I/O instead.");
    }
    
    /// Get a slice of the mapped data with bounds checking
    pub fn slice(&self, offset: usize, len: usize) -> Result<&[u8]> {
        if offset + len > self.len {
            anyhow::bail!("Slice bounds exceed mapped region: offset={}, len={}, total={}", 
                         offset, len, self.len);
        }
        
        Ok(unsafe {
            std::slice::from_raw_parts(self.data.add(offset), len)
        })
    }
    
    /// Get a typed slice with alignment checking
    pub fn typed_slice<T>(&self, offset: usize, count: usize) -> Result<&[T]> {
        let type_size = std::mem::size_of::<T>();
        let type_align = std::mem::align_of::<T>();
        let byte_len = count * type_size;
        
        // Check alignment
        if (self.data as usize + offset) % type_align != 0 {
            anyhow::bail!("Misaligned access: offset {} is not aligned to {} bytes for type {}", 
                         offset, type_align, std::any::type_name::<T>());
        }
        
        let bytes = self.slice(offset, byte_len)?;
        
        Ok(unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const T, count)
        })
    }
    
    /// Get the total length of the mapped region
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Check if the mapping is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// Get the alignment of the mapped data
    pub fn alignment(&self) -> usize {
        self.alignment
    }
}

/// Memory mapping strategy that can fall back to buffered I/O
pub enum MappingStrategy {
    /// Use memory mapping (zero-copy)
    #[cfg(feature = "mmap")]
    MemoryMapped(SafeMmap),
    /// Use buffered I/O as fallback
    Buffered(Vec<u8>),
}

impl MappingStrategy {
    /// Create the best available mapping strategy for a file
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open file: {}", path.as_ref().display()))?;
        
        #[cfg(feature = "mmap")]
        {
            match SafeMmap::new(file) {
                Ok(mmap) => {
                    tracing::debug!("Using memory-mapped I/O for {}", path.as_ref().display());
                    return Ok(Self::MemoryMapped(mmap));
                }
                Err(e) => {
                    tracing::warn!("Failed to create memory map, falling back to buffered I/O: {}", e);
                }
            }
        }
        
        #[cfg(not(feature = "mmap"))]
        {
            tracing::info!("Memory mapping not available, using buffered I/O");
        }
        
        // Fallback to buffered I/O
        let mut file = File::open(path.as_ref())
            .with_context(|| format!("Failed to reopen file for buffered I/O: {}", path.as_ref().display()))?;
        
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .context("Failed to read file into buffer")?;
        
        Ok(Self::Buffered(buffer))
    }
    
    /// Get a slice of the data
    pub fn slice(&self, offset: usize, len: usize) -> Result<&[u8]> {
        match self {
            #[cfg(feature = "mmap")]
            Self::MemoryMapped(mmap) => mmap.slice(offset, len),
            Self::Buffered(buffer) => {
                if offset + len > buffer.len() {
                    anyhow::bail!("Slice bounds exceed buffer: offset={}, len={}, total={}", 
                                 offset, len, buffer.len());
                }
                Ok(&buffer[offset..offset + len])
            }
        }
    }
    
    /// Get a typed slice with alignment checking
    pub fn typed_slice<T>(&self, offset: usize, count: usize) -> Result<&[T]> {
        match self {
            #[cfg(feature = "mmap")]
            Self::MemoryMapped(mmap) => mmap.typed_slice(offset, count),
            Self::Buffered(buffer) => {
                let type_size = std::mem::size_of::<T>();
                let type_align = std::mem::align_of::<T>();
                let byte_len = count * type_size;
                
                if offset + byte_len > buffer.len() {
                    anyhow::bail!("Typed slice bounds exceed buffer: offset={}, byte_len={}, total={}", 
                                 offset, byte_len, buffer.len());
                }
                
                // Check alignment
                let ptr = buffer.as_ptr() as usize + offset;
                if ptr % type_align != 0 {
                    anyhow::bail!("Misaligned access: offset {} results in misaligned pointer for type {}", 
                                 offset, std::any::type_name::<T>());
                }
                
                Ok(unsafe {
                    std::slice::from_raw_parts(
                        (buffer.as_ptr() as usize + offset) as *const T,
                        count
                    )
                })
            }
        }
    }
    
    /// Get the total length
    pub fn len(&self) -> usize {
        match self {
            #[cfg(feature = "mmap")]
            Self::MemoryMapped(mmap) => mmap.len(),
            Self::Buffered(buffer) => buffer.len(),
        }
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get information about the mapping strategy being used
    pub fn strategy_info(&self) -> &'static str {
        match self {
            #[cfg(feature = "mmap")]
            Self::MemoryMapped(_) => "memory-mapped",
            Self::Buffered(_) => "buffered",
        }
    }
}

/// Emit compile-time warnings when falling back to buffered I/O
#[cfg(not(feature = "mmap"))]
const _: () = {
    #[deprecated = "Memory mapping is not available on this platform. Using buffered I/O fallback."]
    fn _mmap_unavailable_warning() {}
    let _ = _mmap_unavailable_warning;
};

/// Platform-specific information about memory mapping support
pub fn platform_mmap_info() -> &'static str {
    #[cfg(all(feature = "mmap", unix))]
    return "Memory mapping available (Unix)";
    
    #[cfg(all(feature = "mmap", windows))]  
    return "Memory mapping available (Windows)";
    
    #[cfg(all(feature = "mmap", not(any(unix, windows))))]
    return "Memory mapping available (other platform)";
    
    #[cfg(not(feature = "mmap"))]
    return "Memory mapping not available (feature disabled or unsupported platform)";
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_mapping_strategy_creation() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"Hello, world!").unwrap();
        temp_file.flush().unwrap();
        
        let strategy = MappingStrategy::new(temp_file.path()).unwrap();
        assert_eq!(strategy.len(), 13);
        
        let slice = strategy.slice(0, 5).unwrap();
        assert_eq!(slice, b"Hello");
    }
    
    #[test]
    fn test_typed_slice_f32() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        temp_file.write_all(bytemuck::cast_slice(&data)).unwrap();
        temp_file.flush().unwrap();
        
        let strategy = MappingStrategy::new(temp_file.path()).unwrap();
        let f32_slice = strategy.typed_slice::<f32>(0, 4).unwrap();
        
        assert_eq!(f32_slice.len(), 4);
        assert!((f32_slice[0] - 1.0).abs() < f32::EPSILON);
        assert!((f32_slice[3] - 4.0).abs() < f32::EPSILON);
    }
    
    #[test]
    fn test_bounds_checking() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"short").unwrap();
        temp_file.flush().unwrap();
        
        let strategy = MappingStrategy::new(temp_file.path()).unwrap();
        
        // This should fail
        assert!(strategy.slice(0, 10).is_err());
        assert!(strategy.slice(3, 10).is_err());
    }
}