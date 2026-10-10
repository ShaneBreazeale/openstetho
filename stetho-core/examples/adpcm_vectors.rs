//! Print decoder test vectors for the Python ADPCM port.
//!
//! `model/openstetho_model/channel.py` reimplements `AdpcmDecoder` so the
//! trainer can push audio through the BLE codec. `model/tests/test_channel.py`
//! checks that port against this output, sample for sample.
//!
//! Output: the vector count on the first line, then one `nibble sample` pair
//! per line.
//!
//!   cargo run -q -p stetho-core --example adpcm_vectors \
//!       > model/tests/fixtures/adpcm_rust.txt

use stetho_core::AdpcmDecoder;

fn vectors() -> Vec<u8> {
    let mut v = Vec::new();
    // Saturate high, then low, so both i16 clamps are exercised.
    v.resize(200, 0x7);
    v.resize(400, 0xF);
    // One of each code.
    v.extend(0..16u8);
    // Pseudo-random tail (Numerical Recipes LCG, top nibble).
    let mut state: u32 = 1;
    while v.len() < 3000 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        v.push((state >> 28) as u8);
    }
    v
}

fn main() {
    let codes = vectors();
    let mut decoder = AdpcmDecoder::new();
    println!("{}", codes.len());
    for code in codes {
        println!("{code} {}", decoder.decode_nibble(code));
    }
}
