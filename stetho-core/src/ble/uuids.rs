//! Bluetooth GATT UUIDs observed from owned hardware during interoperability
//! testing.
//!
//! The relevant advertising names are used by both the original E4-class
//! devices (legacy 0x5bf6e5* service) and newer Core 2 / Littmann CORE
//! family devices. A real device often exposes both for back-compat.

use uuid::Uuid;

// === Standard Bluetooth SIG services ===

pub const BATTERY_SERVICE: Uuid = Uuid::from_u128(0x0000180f_0000_1000_8000_00805f9b34fb);
pub const BATTERY_LEVEL_CHAR: Uuid = Uuid::from_u128(0x00002a19_0000_1000_8000_00805f9b34fb);
pub const DEVICE_INFORMATION_SERVICE: Uuid =
    Uuid::from_u128(0x0000180a_0000_1000_8000_00805f9b34fb);

// === Legacy E4-class service ===

pub const EKO_LEGACY_SERVICE: Uuid = Uuid::from_u128(0x5bf6e500_9999_11e3_a116_0002a5d5c51b);
/// Legacy audio characteristic on original E4. NOTIFY + WRITE_WITHOUT_RESPONSE.
pub const EKO_LEGACY_AUDIO_CHAR: Uuid = Uuid::from_u128(0xba9c5360_9999_11e3_966f_0002a5d5c51b);

// === Core 2 / Littmann CORE data service ===

pub const CORE2_DATA_SERVICE: Uuid = Uuid::from_u128(0xf1de0ef3_6e8f_4fa6_b538_5bd318bdbccb);
/// ADPCM-compressed audio stream (notify). Primary audio path on Core 2.
pub const CORE2_ADPCM_CHAR: Uuid = Uuid::from_u128(0xc320d257_d7be_46ac_9a37_7a4edfa84bce);
/// Uncompressed PCM stream (notify). Used for higher-fidelity recording
/// paths and/or ECG depending on device model.
pub const CORE2_PCM_CHAR: Uuid = Uuid::from_u128(0xc2148e84_cb1f_4a05_9ed0_832a1e9fb336);
pub const CORE2_VOLUME_CHAR: Uuid = Uuid::from_u128(0x34696772_1597_429d_a2e3_5c036f9f39de);
pub const CORE2_HEARTBEAT_TYPE_CHAR: Uuid = Uuid::from_u128(0x611ca734_3c3d_4ff7_b908_3587e127db41);
pub const CORE2_RECORDING_STATE_CHAR: Uuid =
    Uuid::from_u128(0xcd54fb7b_61a4_40a4_a34a_e6cfeae11aa6);
pub const CORE2_CUSTOM_NAME_CHAR: Uuid = Uuid::from_u128(0x2af120d7_4d40_4ff1_96c6_d803455a3959);
pub const CORE2_RECORDING_DURATION_CHAR: Uuid =
    Uuid::from_u128(0x8ced7de2_6d15_4e8c_932d_c2be048146da);
pub const CORE2_PAIRING_FAILURE_CHAR: Uuid =
    Uuid::from_u128(0x40b90c02_9306_4dcf_94bd_4cc71515026a);
pub const CORE2_FILTER_SETTING_CHAR: Uuid = Uuid::from_u128(0x75f0a9da_183d_4ce6_bc9b_334812d40a1e);
pub const CORE2_ANC_ENABLE_CHAR: Uuid = Uuid::from_u128(0xe6ea3564_d144_4dd3_a884_c9aaa3bfcc19);

// === Core 2 OTA/DFU ===

pub const CORE2_DFU_START_SERVICE: Uuid = Uuid::from_u128(0xc2d4f30f_e149_43f5_b1b5_b31e7c2ef5d4);
pub const CORE2_DFU_START_CHARGER_CONNECTED: Uuid =
    Uuid::from_u128(0x580b41ec_243f_42d6_a922_8cd6def5f941);
pub const CORE2_DFU_START_CHAR: Uuid = Uuid::from_u128(0x31ddcab1_2788_4af0_b019_9307cebfaf53);
pub const CORE2_BOOTLOADER_SERVICE: Uuid = Uuid::from_u128(0x00060000_f8ce_11e4_abf4_0002a5d5c51b);
pub const CORE2_BOOTLOADER_CHAR: Uuid = Uuid::from_u128(0x00060001_f8ce_11e4_abf4_0002a5d5c51b);
