//! Distance computation traits

/// Trait for computing distances between vectors
pub trait Distance<T> {
    /// Compute distance between two vectors
    fn distance(&self, a: &[T], b: &[T]) -> f32;
}

/// Euclidean distance implementation
pub struct EuclideanDistance;

impl Distance<f32> for EuclideanDistance {
    fn distance(&self, _a: &[f32], _b: &[f32]) -> f32 {
        // TODO: Implement actual euclidean distance
        0.0
    }
}