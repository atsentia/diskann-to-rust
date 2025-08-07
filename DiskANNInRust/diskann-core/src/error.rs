//! Error types for the DiskANN core system

#[cfg(not(feature = "std"))]
use alloc::string::String;

/// Core error types for DiskANN operations
#[derive(Debug, Clone)]
pub enum DiskAnnError {
    /// Generic computation error
    Computation(String),
    
    /// Invalid parameter error
    InvalidParameter(String),
    
    /// Memory allocation error
    MemoryAllocation,
}

impl core::fmt::Display for DiskAnnError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DiskAnnError::Computation(msg) => write!(f, "Computation error: {}", msg),
            DiskAnnError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            DiskAnnError::MemoryAllocation => write!(f, "Memory allocation failed"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DiskAnnError {}

/// Result type alias for DiskANN operations
pub type DiskAnnResult<T> = Result<T, DiskAnnError>;