//! Distance computation traits

use diskann_core::math::{dot_product, l2_norm_squared};

/// Trait for computing distances between vectors
/// 
/// This trait provides a generic interface for different distance metrics
/// used in nearest neighbor search algorithms.
pub trait Distance<T> {
    /// Compute distance between two vectors
    /// 
    /// # Arguments
    /// * `a` - First vector
    /// * `b` - Second vector
    /// 
    /// # Returns
    /// The distance between the vectors as a non-negative f32 value
    fn distance(&self, a: &[T], b: &[T]) -> f32;
    
    /// Get the name of this distance metric
    fn name(&self) -> &'static str;
    
    /// Check if this distance metric satisfies the triangle inequality
    fn is_metric(&self) -> bool {
        true
    }
}

/// Trait for squared distance computation (avoids sqrt for efficiency)
pub trait SquaredDistance<T> {
    /// Compute squared distance between two vectors
    fn squared_distance(&self, a: &[T], b: &[T]) -> f32;
}

/// Euclidean (L2) distance implementation for f32
#[derive(Debug, Clone, Copy, Default)]
pub struct EuclideanDistance;

impl Distance<f32> for EuclideanDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        self.squared_distance(a, b).sqrt()
    }
    
    fn name(&self) -> &'static str {
        "euclidean"
    }
}

impl SquaredDistance<f32> for EuclideanDistance {
    fn squared_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::INFINITY;
        }
        
        let mut sum = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            let diff = x - y;
            sum += diff * diff;
        }
        sum
    }
}

impl Distance<f64> for EuclideanDistance {
    fn distance(&self, a: &[f64], b: &[f64]) -> f32 {
        self.squared_distance(a, b).sqrt()
    }
    
    fn name(&self) -> &'static str {
        "euclidean"
    }
}

impl SquaredDistance<f64> for EuclideanDistance {
    fn squared_distance(&self, a: &[f64], b: &[f64]) -> f32 {
        if a.len() != b.len() {
            return f32::INFINITY;
        }
        
        let mut sum = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let diff = x - y;
            sum += diff * diff;
        }
        sum as f32
    }
}

/// Manhattan (L1) distance implementation
#[derive(Debug, Clone, Copy, Default)]
pub struct ManhattanDistance;

impl Distance<f32> for ManhattanDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::INFINITY;
        }
        
        let mut sum = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            sum += (x - y).abs();
        }
        sum
    }
    
    fn name(&self) -> &'static str {
        "manhattan"
    }
}

impl Distance<f64> for ManhattanDistance {
    fn distance(&self, a: &[f64], b: &[f64]) -> f32 {
        if a.len() != b.len() {
            return f32::INFINITY;
        }
        
        let mut sum = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            sum += (x - y).abs();
        }
        sum as f32
    }
    
    fn name(&self) -> &'static str {
        "manhattan"
    }
}

/// Cosine distance implementation
#[derive(Debug, Clone, Copy, Default)]
pub struct CosineDistance;

impl Distance<f32> for CosineDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::INFINITY;
        }
        
        if a.is_empty() {
            return 0.0;
        }
        
        let dot = dot_product(a, b);
        let norm_a_sq = l2_norm_squared(a);
        let norm_b_sq = l2_norm_squared(b);
        
        if norm_a_sq == 0.0 || norm_b_sq == 0.0 {
            return 1.0; // Maximum distance for zero vectors
        }
        
        let cosine_sim = dot / (norm_a_sq.sqrt() * norm_b_sq.sqrt());
        
        // Clamp to handle numerical errors
        let cosine_sim = cosine_sim.max(-1.0).min(1.0);
        
        1.0 - cosine_sim
    }
    
    fn name(&self) -> &'static str {
        "cosine"
    }
    
    fn is_metric(&self) -> bool {
        false // Cosine distance doesn't satisfy triangle inequality
    }
}

impl Distance<f64> for CosineDistance {
    fn distance(&self, a: &[f64], b: &[f64]) -> f32 {
        if a.len() != b.len() {
            return f32::INFINITY;
        }
        
        if a.is_empty() {
            return 0.0;
        }
        
        let dot = dot_product(a, b);
        let norm_a_sq = l2_norm_squared(a);
        let norm_b_sq = l2_norm_squared(b);
        
        if norm_a_sq == 0.0 || norm_b_sq == 0.0 {
            return 1.0; // Maximum distance for zero vectors
        }
        
        let cosine_sim = dot / (norm_a_sq.sqrt() * norm_b_sq.sqrt());
        
        // Clamp to handle numerical errors
        let cosine_sim = cosine_sim.max(-1.0).min(1.0);
        
        (1.0 - cosine_sim) as f32
    }
    
    fn name(&self) -> &'static str {
        "cosine"
    }
    
    fn is_metric(&self) -> bool {
        false // Cosine distance doesn't satisfy triangle inequality
    }
}

/// Minkowski distance implementation with configurable p-norm
#[derive(Debug, Clone, Copy)]
pub struct MinkowskiDistance {
    /// The p parameter for the Minkowski distance
    pub p: f32,
}

impl MinkowskiDistance {
    /// Create a new Minkowski distance with the given p parameter
    pub fn new(p: f32) -> Self {
        Self { p }
    }
    
    /// Create Manhattan distance (p=1)
    pub fn manhattan() -> Self {
        Self::new(1.0)
    }
    
    /// Create Euclidean distance (p=2)
    pub fn euclidean() -> Self {
        Self::new(2.0)
    }
}

impl Distance<f32> for MinkowskiDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::INFINITY;
        }
        
        if self.p == 1.0 {
            return ManhattanDistance.distance(a, b);
        } else if self.p == 2.0 {
            return EuclideanDistance.distance(a, b);
        }
        
        let mut sum = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            let diff = (x - y).abs();
            sum += diff.powf(self.p);
        }
        sum.powf(1.0 / self.p)
    }
    
    fn name(&self) -> &'static str {
        "minkowski"
    }
}

impl Distance<f64> for MinkowskiDistance {
    fn distance(&self, a: &[f64], b: &[f64]) -> f32 {
        if a.len() != b.len() {
            return f32::INFINITY;
        }
        
        if self.p == 1.0 {
            return ManhattanDistance.distance(a, b);
        } else if self.p == 2.0 {
            return EuclideanDistance.distance(a, b);
        }
        
        let mut sum = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let diff = (x - y).abs();
            sum += diff.powf(self.p as f64);
        }
        sum.powf(1.0 / self.p as f64) as f32
    }
    
    fn name(&self) -> &'static str {
        "minkowski"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    fn assert_distance_properties<T, D: Distance<T>>(
        distance: &D,
        a: &[T],
        b: &[T],
        c: &[T],
    ) where
        T: Clone + PartialEq,
    {
        // Test non-negativity
        let d_ab = distance.distance(a, b);
        assert!(d_ab >= 0.0, "Distance should be non-negative");
        
        // Test symmetry
        let d_ba = distance.distance(b, a);
        assert!((d_ab - d_ba).abs() < 1e-6, "Distance should be symmetric");
        
        // Test identity of indiscernibles (distance to self is 0)
        let d_aa = distance.distance(a, a);
        assert!(d_aa < 1e-6, "Distance to self should be zero");
        
        // Test triangle inequality (only for metrics)
        if distance.is_metric() {
            let d_ac = distance.distance(a, c);
            let d_bc = distance.distance(b, c);
            assert!(
                d_ac <= d_ab + d_bc + 1e-6,
                "Triangle inequality should hold: d(a,c) <= d(a,b) + d(b,c)"
            );
        }
    }

    #[test]
    fn test_euclidean_distance_f32() {
        let euclidean = EuclideanDistance;
        
        let a = vec![0.0f32, 0.0];
        let b = vec![3.0f32, 4.0];
        let c = vec![1.0f32, 1.0];
        
        let distance = euclidean.distance(&a, &b);
        assert!((distance - 5.0).abs() < 1e-6);
        
        assert_distance_properties(&euclidean, &a, &b, &c);
    }

    #[test]
    fn test_euclidean_distance_f64() {
        let euclidean = EuclideanDistance;
        
        let a = vec![0.0f64, 0.0];
        let b = vec![3.0f64, 4.0];
        let c = vec![1.0f64, 1.0];
        
        let distance = euclidean.distance(&a, &b);
        assert!((distance - 5.0).abs() < 1e-6);
        
        assert_distance_properties(&euclidean, &a, &b, &c);
    }

    #[test]
    fn test_manhattan_distance_f32() {
        let manhattan = ManhattanDistance;
        
        let a = vec![0.0f32, 0.0];
        let b = vec![3.0f32, 4.0];
        let c = vec![1.0f32, 1.0];
        
        let distance = manhattan.distance(&a, &b);
        assert!((distance - 7.0).abs() < 1e-6);
        
        assert_distance_properties(&manhattan, &a, &b, &c);
    }

    #[test]
    fn test_manhattan_distance_f64() {
        let manhattan = ManhattanDistance;
        
        let a = vec![0.0f64, 0.0];
        let b = vec![3.0f64, 4.0];
        let c = vec![1.0f64, 1.0];
        
        let distance = manhattan.distance(&a, &b);
        assert!((distance - 7.0).abs() < 1e-6);
        
        assert_distance_properties(&manhattan, &a, &b, &c);
    }

    #[test]
    fn test_cosine_distance_f32() {
        let cosine = CosineDistance;
        
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let _c = vec![1.0f32, 1.0];
        
        // Perpendicular vectors should have cosine distance of 1
        let distance = cosine.distance(&a, &b);
        assert!((distance - 1.0).abs() < 1e-6);
        
        // Parallel vectors should have cosine distance of 0
        let parallel_distance = cosine.distance(&a, &a);
        assert!(parallel_distance < 1e-6);
        
        // Test properties (but not triangle inequality since cosine is not a metric)
        let d_ab = cosine.distance(&a, &b);
        let d_ba = cosine.distance(&b, &a);
        assert!((d_ab - d_ba).abs() < 1e-6, "Should be symmetric");
        assert!(d_ab >= 0.0, "Should be non-negative");
    }

    #[test]
    fn test_cosine_distance_f64() {
        let cosine = CosineDistance;
        
        let a = vec![1.0f64, 0.0];
        let b = vec![0.0f64, 1.0];
        
        // Perpendicular vectors should have cosine distance of 1
        let distance = cosine.distance(&a, &b);
        assert!((distance - 1.0).abs() < 1e-6);
        
        // Parallel vectors should have cosine distance of 0
        let parallel_distance = cosine.distance(&a, &a);
        assert!(parallel_distance < 1e-6);
    }

    #[test]
    fn test_minkowski_distance_f32() {
        let l1 = MinkowskiDistance::manhattan();
        let l2 = MinkowskiDistance::euclidean();
        
        let a = vec![0.0f32, 0.0];
        let b = vec![3.0f32, 4.0];
        let c = vec![1.0f32, 1.0];
        
        // Test that specialized cases match dedicated implementations
        let manhattan = ManhattanDistance;
        let euclidean = EuclideanDistance;
        
        assert!((l1.distance(&a, &b) - manhattan.distance(&a, &b)).abs() < 1e-6);
        assert!((l2.distance(&a, &b) - euclidean.distance(&a, &b)).abs() < 1e-6);
        
        assert_distance_properties(&l1, &a, &b, &c);
        assert_distance_properties(&l2, &a, &b, &c);
    }

    #[test]
    fn test_minkowski_distance_f64() {
        let l1 = MinkowskiDistance::manhattan();
        let l2 = MinkowskiDistance::euclidean();
        
        let a = vec![0.0f64, 0.0];
        let b = vec![3.0f64, 4.0];
        let c = vec![1.0f64, 1.0];
        
        // Test that specialized cases match dedicated implementations
        let manhattan = ManhattanDistance;
        let euclidean = EuclideanDistance;
        
        assert!((l1.distance(&a, &b) - manhattan.distance(&a, &b)).abs() < 1e-6);
        assert!((l2.distance(&a, &b) - euclidean.distance(&a, &b)).abs() < 1e-6);
        
        assert_distance_properties(&l1, &a, &b, &c);
        assert_distance_properties(&l2, &a, &b, &c);
    }

    #[test]
    fn test_empty_vectors() {
        let euclidean = EuclideanDistance;
        let manhattan = ManhattanDistance;
        let cosine = CosineDistance;
        
        let empty_f32: Vec<f32> = vec![];
        let empty_f64: Vec<f64> = vec![];
        
        assert_eq!(euclidean.distance(&empty_f32, &empty_f32), 0.0);
        assert_eq!(manhattan.distance(&empty_f32, &empty_f32), 0.0);
        assert_eq!(cosine.distance(&empty_f32, &empty_f32), 0.0);
        
        assert_eq!(euclidean.distance(&empty_f64, &empty_f64), 0.0);
        assert_eq!(manhattan.distance(&empty_f64, &empty_f64), 0.0);
        assert_eq!(cosine.distance(&empty_f64, &empty_f64), 0.0);
    }

    #[test]
    fn test_mismatched_dimensions() {
        let euclidean = EuclideanDistance;
        
        let a = vec![1.0f32, 2.0];
        let b = vec![1.0f32, 2.0, 3.0];
        
        let distance = euclidean.distance(&a, &b);
        assert_eq!(distance, f32::INFINITY);
    }

    #[test]
    fn test_squared_distance() {
        let euclidean = EuclideanDistance;
        
        let a = vec![0.0f32, 0.0];
        let b = vec![3.0f32, 4.0];
        
        let squared_dist = euclidean.squared_distance(&a, &b);
        assert!((squared_dist - 25.0).abs() < 1e-6);
        
        let dist = euclidean.distance(&a, &b);
        assert!((dist * dist - squared_dist).abs() < 1e-6);
    }
}