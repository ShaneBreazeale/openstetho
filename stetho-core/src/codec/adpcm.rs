//! IMA-style ADPCM decoder for compatible E4-class devices.
//!
//! State carries predictor + step-index across packets. Codec-reset packets
//! (detected by `is_codec_reset_response`) clear state and resync the decoder.
//!
//! Validation status: golden-test harness pending — see `tests/adpcm_golden.rs`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AdpcmError {
    #[error("packet too short: {len} bytes")]
    Truncated { len: usize },
}

/// IMA-ADPCM standard step-size table (89 entries).
const STEP_TABLE: [i32; 89] = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66,
    73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449,
    494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272,
    2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493,
    10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
];

/// IMA-ADPCM index-adjust table (per 4-bit nibble).
const INDEX_TABLE: [i32; 16] = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8];

/// Stateful nibble-by-nibble decoder.
///
/// Packet framing can vary by device family; the per-nibble
/// [`Self::decode_nibble`] entry point is the authoritative API.
pub struct AdpcmDecoder {
    predictor: i32,
    step_index: i32,
}

impl Default for AdpcmDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl AdpcmDecoder {
    pub fn new() -> Self {
        Self {
            predictor: 0,
            step_index: 0,
        }
    }

    /// Drop predictor + step-index back to zero. Called when a codec-reset
    /// packet is observed on the BLE link.
    pub fn reset(&mut self) {
        self.predictor = 0;
        self.step_index = 0;
    }

    /// Decode one 4-bit ADPCM nibble into a 16-bit PCM sample.
    pub fn decode_nibble(&mut self, nibble: u8) -> i16 {
        let n = (nibble & 0x0F) as i32;
        let step = STEP_TABLE[self.step_index.clamp(0, 88) as usize];

        let mut diff = step >> 3;
        if n & 4 != 0 {
            diff += step;
        }
        if n & 2 != 0 {
            diff += step >> 1;
        }
        if n & 1 != 0 {
            diff += step >> 2;
        }
        if n & 8 != 0 {
            self.predictor -= diff;
        } else {
            self.predictor += diff;
        }
        self.predictor = self.predictor.clamp(i16::MIN as i32, i16::MAX as i32);

        self.step_index = (self.step_index + INDEX_TABLE[n as usize]).clamp(0, 88);
        self.predictor as i16
    }

    /// Decode a single 17-byte ADPCM sub-packet (matches
    /// `Core2ADPCMManager.getPacketLength() = 17`):
    ///
    ///   * byte 0 — header:
    ///       - bit 7 (`0x80`) set ⇒ codec-reset marker; decoder state
    ///         is zeroed before decoding this sub-packet
    ///       - bit 6 (`0x40`) set ⇒ codec-reset request (echoed back to device)
    ///       - bits 0..3       ⇒ 4-bit sequence number
    ///   * bytes 1..17 — 16 bytes (32 nibbles, low-nibble first) of ADPCM.
    pub fn decode_subpacket(&mut self, sub: &[u8], out: &mut Vec<i16>) {
        debug_assert_eq!(sub.len(), SUBPACKET_LEN);
        if sub.is_empty() {
            return;
        }
        if sub[0] & PACKET_HEADER_RESET_BIT != 0 {
            self.reset();
        }
        for &b in &sub[1..] {
            out.push(self.decode_nibble(b & 0x0F));
            out.push(self.decode_nibble(b >> 4));
        }
    }

    /// Decode a full BLE notification payload by splitting it into
    /// 17-byte sub-packets and feeding each through [`Self::decode_subpacket`].
    /// A 238-byte Core 2 notification produces 14 sub-packets ⇒ 448 samples.
    pub fn decode_ble_notification(&mut self, bytes: &[u8], out: &mut Vec<i16>) {
        out.reserve((bytes.len() / SUBPACKET_LEN) * SAMPLES_PER_SUBPACKET);
        for sub in bytes.chunks_exact(SUBPACKET_LEN) {
            self.decode_subpacket(sub, out);
        }
    }

    /// Legacy alias kept for the existing CLI / tests. Treats `bytes` as a
    /// full BLE notification of stacked 17-byte sub-packets.
    pub fn decode_packet(&mut self, bytes: &[u8], out: &mut Vec<i16>) {
        self.decode_ble_notification(bytes, out);
    }

    /// Extract the 4-bit sequence number from a sub-packet header byte.
    pub fn seq_from_header(header: u8) -> u8 {
        header & 0x0F
    }

    /// Returns true if `header` carries the codec-reset / stream-init flag.
    pub fn is_codec_reset_packet(header: u8) -> bool {
        header & PACKET_HEADER_RESET_BIT != 0
    }

    /// Returns true if `header` carries the codec-reset-request flag (the
    /// device asks the host to echo a reset back via `createCodecResetPacket`).
    pub fn is_codec_reset_request(header: u8) -> bool {
        header & PACKET_HEADER_RESET_REQUEST_BIT != 0
    }
}

/// Bit 7 of byte 0 ⇒ codec reset / stream-init marker.
pub const PACKET_HEADER_RESET_BIT: u8 = 0x80;
/// Bit 6 of byte 0 ⇒ codec reset request from device.
pub const PACKET_HEADER_RESET_REQUEST_BIT: u8 = 0x40;
/// Core 2 sub-packet length, in bytes.
pub const SUBPACKET_LEN: usize = 17;
/// PCM samples produced per 17-byte sub-packet (16 data bytes × 2 nibbles).
pub const SAMPLES_PER_SUBPACKET: usize = 32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_zeroes_state() {
        let mut d = AdpcmDecoder::new();
        d.decode_nibble(0b0101);
        d.decode_nibble(0b1010);
        d.reset();
        assert_eq!(d.predictor, 0);
        assert_eq!(d.step_index, 0);
    }

    #[test]
    fn zero_nibble_keeps_predictor_small() {
        let mut d = AdpcmDecoder::new();
        for _ in 0..16 {
            d.decode_nibble(0);
        }
        assert!(d.predictor.abs() < 100);
    }

    #[test]
    fn reset_bit_triggers_reset_before_decode() {
        let mut d = AdpcmDecoder::new();
        // Prime non-zero state.
        d.decode_nibble(7);
        d.decode_nibble(6);
        let before_reset = d.predictor;
        // 17-byte sub-packet: header 0x80 + 16 zero data bytes.
        let mut sub = [0u8; SUBPACKET_LEN];
        sub[0] = 0x80;
        let mut out = Vec::new();
        d.decode_subpacket(&sub, &mut out);
        assert_eq!(out.len(), SAMPLES_PER_SUBPACKET);
        assert_ne!(before_reset, 0);
        assert!(d.predictor.abs() < 10);
    }

    #[test]
    fn header_helpers() {
        assert!(AdpcmDecoder::is_codec_reset_packet(0x80));
        assert!(!AdpcmDecoder::is_codec_reset_packet(0x0e));
        assert_eq!(AdpcmDecoder::seq_from_header(0x80), 0);
        assert_eq!(AdpcmDecoder::seq_from_header(0x0e), 14);
        assert_eq!(AdpcmDecoder::seq_from_header(0x8c), 12);
    }
}
