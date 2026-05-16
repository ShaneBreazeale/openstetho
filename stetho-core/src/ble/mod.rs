pub mod uuids;

use anyhow::{anyhow, Context, Result};
use btleplug::api::{
    BDAddr, Central, CentralEvent, Manager as _, Peripheral as _, ScanFilter, WriteType,
};
use btleplug::platform::{Adapter, Manager, Peripheral, PeripheralId};
use futures::StreamExt;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Lowercase substrings that identify compatible devices by advertised
/// local name.
pub const EKO_NAME_SUBSTRINGS: &[&str] = &[
    "eko core",
    "eko duo",
    "eko core2",
    "eko core2 dfu",
    // Loose match for the 3M Littmann CORE skinning.
    "littmann core",
    "core digital",
];

#[derive(Clone, Debug)]
pub struct DiscoveredDevice {
    pub id: PeripheralId,
    pub address: BDAddr,
    pub name: String,
    pub rssi: Option<i16>,
}

pub async fn manager() -> Result<Manager> {
    Manager::new().await.context("init btleplug Manager")
}

pub async fn default_adapter(mgr: &Manager) -> Result<Adapter> {
    let adapters = mgr.adapters().await.context("list BLE adapters")?;
    adapters
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no BLE adapter found"))
}

/// Scan for `duration`, returning every compatible device discovered.
pub async fn scan(adapter: &Adapter, duration: Duration) -> Result<Vec<DiscoveredDevice>> {
    let mut events = adapter.events().await?;
    adapter
        .start_scan(ScanFilter::default())
        .await
        .context("start_scan")?;
    info!("BLE scan running for {:?}", duration);

    let deadline = tokio::time::Instant::now() + duration;
    let mut hits: Vec<DiscoveredDevice> = Vec::new();
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        match tokio::time::timeout(remaining, events.next()).await {
            Ok(Some(evt)) => {
                if let Some(d) = consider(adapter, evt).await? {
                    if !hits.iter().any(|x| x.id == d.id) {
                        info!("found {} ({}) rssi={:?}", d.name, d.address, d.rssi);
                        hits.push(d);
                    }
                }
            }
            Ok(None) | Err(_) => break,
        }
    }

    adapter.stop_scan().await.ok();
    Ok(hits)
}

async fn consider(adapter: &Adapter, evt: CentralEvent) -> Result<Option<DiscoveredDevice>> {
    let id = match evt {
        CentralEvent::DeviceDiscovered(id) | CentralEvent::DeviceUpdated(id) => id,
        _ => return Ok(None),
    };
    let p = adapter.peripheral(&id).await?;
    let props = match p.properties().await? {
        Some(p) => p,
        None => return Ok(None),
    };
    let name = props.local_name.unwrap_or_default();
    if name.is_empty() {
        return Ok(None);
    }
    let lname = name.to_lowercase();
    if !EKO_NAME_SUBSTRINGS.iter().any(|s| lname.contains(s)) {
        debug!("ignored non-compatible advertisement: {name}");
        return Ok(None);
    }
    Ok(Some(DiscoveredDevice {
        id,
        address: props.address,
        name,
        rssi: props.rssi,
    }))
}

/// Locate a peripheral by Bluetooth address, then by PeripheralId string
/// (macOS uses opaque UUIDs), then by name substring. First match wins.
pub async fn resolve_peripheral(adapter: &Adapter, target: &str) -> Result<Peripheral> {
    let needle = target.to_lowercase();
    for p in adapter.peripherals().await? {
        let addr_match = p.address().to_string().eq_ignore_ascii_case(target);
        let id_str = format!("{:?}", p.id());
        let id_match = id_str.to_lowercase().contains(&needle);
        let name_match = match p.properties().await? {
            Some(props) => props
                .local_name
                .map(|n| n.to_lowercase().contains(&needle))
                .unwrap_or(false),
            None => false,
        };
        if addr_match || id_match || name_match {
            return Ok(p);
        }
    }
    Err(anyhow!("no peripheral matched target {target:?}"))
}

pub async fn connect_and_discover(peripheral: &Peripheral) -> Result<()> {
    if !peripheral.is_connected().await? {
        peripheral.connect().await.context("connect")?;
        sleep(Duration::from_millis(200)).await;
    }
    peripheral
        .discover_services()
        .await
        .context("discover_services")?;
    Ok(())
}

/// Print services + characteristics + descriptors to tracing log.
pub async fn dump_gatt(peripheral: &Peripheral) -> Result<()> {
    for service in peripheral.services() {
        info!("service {} (primary={})", service.uuid, service.primary);
        for ch in service.characteristics {
            info!("  char {} props={:?}", ch.uuid, ch.properties);
            for d in ch.descriptors {
                info!("    desc {}", d.uuid);
            }
        }
    }
    Ok(())
}

/// Read the standard Battery Level characteristic if present.
pub async fn read_battery(peripheral: &Peripheral) -> Result<Option<u8>> {
    for service in peripheral.services() {
        if service.uuid != uuids::BATTERY_SERVICE {
            continue;
        }
        for ch in service.characteristics {
            if ch.uuid == uuids::BATTERY_LEVEL_CHAR {
                let v = peripheral.read(&ch).await.context("read battery level")?;
                return Ok(v.first().copied());
            }
        }
    }
    Ok(None)
}

/// Subscribe to notifications on a single characteristic by UUID.
pub async fn subscribe(peripheral: &Peripheral, char_uuid: Uuid) -> Result<()> {
    for service in peripheral.services() {
        for ch in service.characteristics {
            if ch.uuid == char_uuid {
                peripheral.subscribe(&ch).await.context("subscribe")?;
                return Ok(());
            }
        }
    }
    warn!("characteristic {char_uuid} not present on device");
    Err(anyhow!("characteristic {char_uuid} not found"))
}

/// Subscribe to the Core 2 ADPCM audio stream (primary audio characteristic
/// on the Core 2 / Littmann CORE).
pub async fn subscribe_core2_adpcm(peripheral: &Peripheral) -> Result<()> {
    subscribe(peripheral, uuids::CORE2_ADPCM_CHAR).await
}

/// Subscribe to the legacy E4 audio characteristic.
pub async fn subscribe_legacy_audio(peripheral: &Peripheral) -> Result<()> {
    subscribe(peripheral, uuids::EKO_LEGACY_AUDIO_CHAR).await
}

pub async fn write_request(
    peripheral: &Peripheral,
    char_uuid: Uuid,
    payload: &[u8],
    with_response: bool,
) -> Result<()> {
    for service in peripheral.services() {
        for ch in service.characteristics {
            if ch.uuid == char_uuid {
                let kind = if with_response {
                    WriteType::WithResponse
                } else {
                    WriteType::WithoutResponse
                };
                peripheral
                    .write(&ch, payload, kind)
                    .await
                    .context("write")?;
                return Ok(());
            }
        }
    }
    Err(anyhow!("characteristic {char_uuid} not found"))
}
