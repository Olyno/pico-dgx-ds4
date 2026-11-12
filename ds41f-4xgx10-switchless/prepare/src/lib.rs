//! Cut the DeepSeek-V4.1-Flash checkpoint into per-rank `.dsw` files and Engram
//! shards without ever holding two complete copies on one disk.
//!
//! The formats are the engine's own (`engine/weights/dsw.h`, `engine/engram/engram.h`):
//! nothing here invents a new layout, and every tensor byte is the checkpoint's own.
//! What this crate adds is orchestration: a source reader that can stream ranges
//! instead of storing the checkpoint, a dependency-aware scheduler, a durable
//! journal, bounded output allocation, and reclamation of source blocks once their
//! last consumer is verified.

/// UTC, ISO 8601, for the `.dsw` header's `created` field. No dependency: a
/// civil-from-days conversion of the Unix epoch.
pub fn now_iso8601() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    // Howard Hinnant's days_from_civil, inverted.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

pub mod checkpoint;
pub mod convert;
pub mod dsw;
pub mod engram;
pub mod journal;
pub mod plan;
pub mod reshard;
pub mod safetensors;
pub mod source;
pub mod space;
pub mod tile;
pub mod vision;
