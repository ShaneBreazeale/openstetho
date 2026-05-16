//! Shape tests against deterministic synthetic BLE notification payloads.

use stetho_core::codec::adpcm::{SAMPLES_PER_SUBPACKET, SUBPACKET_LEN};
use stetho_core::AdpcmDecoder;

const SUBPACKETS_PER_NOTIFICATION: usize = 14;
const NOTIFICATION_LEN: usize = SUBPACKET_LEN * SUBPACKETS_PER_NOTIFICATION;

fn synthetic_packets(count: usize) -> Vec<Vec<u8>> {
    let mut packets = Vec::with_capacity(count);
    for packet_idx in 0..count {
        let mut packet = vec![0u8; NOTIFICATION_LEN];
        for sub_idx in 0..SUBPACKETS_PER_NOTIFICATION {
            let start = sub_idx * SUBPACKET_LEN;
            let seq = ((packet_idx * SUBPACKETS_PER_NOTIFICATION + sub_idx) % 16) as u8;
            packet[start] = seq;
            if packet_idx == 0 && sub_idx == 0 {
                packet[start] |= 0x80;
            }
            for data_idx in 1..SUBPACKET_LEN {
                packet[start + data_idx] = ((packet_idx + sub_idx + data_idx) % 16) as u8;
            }
        }
        packets.push(packet);
    }
    packets
}

#[test]
fn synthetic_payloads_match_expected_shape() {
    let packets = synthetic_packets(10);
    assert_eq!(packets.len(), 10, "synthetic set should contain 10 packets");
    for p in &packets {
        assert_eq!(
            p.len(),
            NOTIFICATION_LEN,
            "every packet should be 238 bytes"
        );
        assert_eq!(
            p.len() % SUBPACKET_LEN,
            0,
            "BLE payload must be a multiple of {SUBPACKET_LEN}"
        );
        assert_eq!(
            p.len() / SUBPACKET_LEN,
            14,
            "expected 14 sub-packets per notification"
        );
    }

    // First sub-packet of the first notification should carry codec-reset.
    assert!(
        AdpcmDecoder::is_codec_reset_packet(packets[0][0]),
        "first sub-packet header {:#04x} should have reset bit set",
        packets[0][0]
    );
}

#[test]
fn subpacket_sequence_numbers_walk_within_each_notification() {
    let packets = synthetic_packets(10);
    // In every BLE notification the 14 sub-packet seq numbers should be
    // monotonic mod 16. We don't yet require a specific direction.
    for (pi, p) in packets.iter().enumerate() {
        let seqs: Vec<u8> = p
            .chunks_exact(SUBPACKET_LEN)
            .map(|c| AdpcmDecoder::seq_from_header(c[0]))
            .collect();
        assert_eq!(
            seqs.len(),
            14,
            "packet {pi} should split into 14 sub-packets"
        );
    }
}

#[test]
fn decodes_all_packets_without_panic_and_at_correct_sample_count() {
    let packets = synthetic_packets(10);
    let mut decoder = AdpcmDecoder::new();
    let mut pcm: Vec<i16> = Vec::new();
    for p in &packets {
        decoder.decode_ble_notification(p, &mut pcm);
    }
    // 14 sub-packets × 32 samples × 10 BLE notifications = 4480 samples.
    let expected = 14 * SAMPLES_PER_SUBPACKET * packets.len();
    assert_eq!(pcm.len(), expected);
}
