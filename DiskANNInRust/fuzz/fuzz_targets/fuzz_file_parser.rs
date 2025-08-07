#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use std::io::Cursor;
use diskann_io::format::{BinaryHeader, read_vectors_f32};

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    // Header data
    num_points: i32,
    num_dimensions: i32,
    // Variable length data following the header
    data: Vec<u8>,
}

fuzz_target!(|input: FuzzInput| {
    // Test binary header parsing with fuzzed data
    let mut cursor = Cursor::new(Vec::new());
    
    // Write fuzzed header
    let header = BinaryHeader {
        num_points: input.num_points,
        num_dimensions: input.num_dimensions,
    };
    
    // Try to write header - this should not panic
    if header.write_to(&mut cursor).is_ok() {
        // Add fuzzed data
        cursor.get_mut().extend_from_slice(&input.data);
        cursor.set_position(0);
        
        // Try to read back header
        if let Ok(parsed_header) = BinaryHeader::read_from(&mut cursor) {
            // Validate should not panic
            let _ = parsed_header.validate();
            
            // Try to read vector data if header seems reasonable
            if parsed_header.num_points > 0 && parsed_header.num_points < 10000 &&
               parsed_header.num_dimensions > 0 && parsed_header.num_dimensions < 1000 {
                cursor.set_position(0); // Start from beginning for full read
                let _ = read_vectors_f32(&mut cursor);
            }
        }
    }
});