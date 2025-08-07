//! Error types for the DiskANN core system

use thiserror::Error;

/// Core error types for DiskANN operations
#[derive(Error, Debug)]
pub enum DiskAnnError {
    /// Generic computation error
    #[error("Computation error: {0}")]
    Computation(String),
    
    /// Invalid parameter error
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    /// Memory allocation error
    #[error("Memory allocation failed")]
    MemoryAllocation,
}

/// Result type alias for DiskANN operations
pub type DiskAnnResult<T> = Result<T, DiskAnnError>;