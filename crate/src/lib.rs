// ============================================================
// lib.rs — ferrum-photo WASM Core
//
// RAW image parsing via rawloader.
// Returns raw sensor data + metadata for GPU-side demosaicing.
// ============================================================

use std::io::Cursor;
use wasm_bindgen::prelude::*;

/// Decode a RAW image file and return raw sensor data + metadata.
///
/// Returns a flat byte array with the following layout:
///   Bytes 0..4:   width (u32 LE)
///   Bytes 4..8:   height (u32 LE)
///   Bytes 8..12:  CFA pattern (u32 LE, encoded as 4 color indices packed)
///   Bytes 12..20: black levels (4 × u16 LE)
///   Bytes 20..28: white levels (4 × u16 LE)
///   Bytes 28..44: white balance coefficients (4 × f32 LE, RGBE order)
///   Bytes 44..:   raw sensor data (u16 LE per pixel)
///
/// The GPU will handle demosaicing, normalization, and color processing.
#[wasm_bindgen]
pub fn decode_raw(data: &[u8]) -> Result<Vec<u8>, JsValue> {
    let mut cursor = Cursor::new(data);

    let raw_image = rawloader::decode(&mut cursor)
        .map_err(|e| JsValue::from_str(&format!("RAW decode error: {}", e)))?;

    let width = raw_image.width;
    let height = raw_image.height;

    // Encode CFA pattern as u32: each byte is a color index (0=R,1=G,2=B)
    // Pattern is 2x2: [TL, TR, BL, BR]
    let cfa_pattern: u32 = (raw_image.cfa.color_at(0, 0) as u32)
        | ((raw_image.cfa.color_at(1, 0) as u32) << 8)
        | ((raw_image.cfa.color_at(0, 1) as u32) << 16)
        | ((raw_image.cfa.color_at(1, 1) as u32) << 24);

    // Get raw pixel data as u16
    let raw_u16: &[u16] = match &raw_image.data {
        rawloader::RawImageData::Integer(d) => d,
        rawloader::RawImageData::Float(_) => {
            return Err(JsValue::from_str("Float RAW data not yet supported"));
        }
    };

    // WB coefficients from camera metadata (RGBE order)
    let wb = raw_image.wb_coeffs;

    // Header: 44 bytes (was 28, added 16 bytes for WB)
    // Data: width * height * 2 bytes (u16 per pixel)
    let header_size = 44;
    let data_size = raw_u16.len() * 2;
    let mut result = Vec::with_capacity(header_size + data_size);

    // Write header
    result.extend_from_slice(&(width as u32).to_le_bytes());
    result.extend_from_slice(&(height as u32).to_le_bytes());
    result.extend_from_slice(&cfa_pattern.to_le_bytes());
    for &b in &raw_image.blacklevels {
        result.extend_from_slice(&b.to_le_bytes());
    }
    for &w in &raw_image.whitelevels {
        result.extend_from_slice(&w.to_le_bytes());
    }
    // WB coefficients (4 × f32 LE)
    for &c in &wb {
        result.extend_from_slice(&c.to_le_bytes());
    }

    // Write raw u16 data as little-endian bytes
    // Safety: u16 → [u8; 2] is always valid LE on WASM (little-endian)
    let raw_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(raw_u16.as_ptr() as *const u8, raw_u16.len() * 2) };
    result.extend_from_slice(raw_bytes);

    Ok(result)
}
