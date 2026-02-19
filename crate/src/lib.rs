// ============================================================
// lib.rs — ferrum-photo WASM Core
//
// RAW image decoding via rawloader + basic Bayer demosaic.
// Exports functions callable from JavaScript via wasm-bindgen.
// ============================================================

use std::io::Cursor;
use wasm_bindgen::prelude::*;

/// Decode a RAW image file from bytes and return linear RGB u8 data.
///
/// Returns a flat array: [width(u32 LE), height(u32 LE), R, G, B, A, R, G, B, A, ...]
/// The first 8 bytes encode width and height as little-endian u32.
#[wasm_bindgen]
pub fn decode_raw(data: &[u8]) -> Result<Vec<u8>, JsValue> {
    let mut cursor = Cursor::new(data);

    let raw_image = rawloader::decode(&mut cursor)
        .map_err(|e| JsValue::from_str(&format!("RAW decode error: {}", e)))?;

    let width = raw_image.width;
    let height = raw_image.height;

    // Get the raw pixel data
    let raw_data = match &raw_image.data {
        rawloader::RawImageData::Integer(d) => d.clone(),
        rawloader::RawImageData::Float(d) => {
            // Convert f32 data to u16 range
            d.iter()
                .map(|&v| (v.clamp(0.0, 1.0) * 65535.0) as u16)
                .collect()
        }
    };

    // Simple bilinear Bayer demosaic
    let rgb = demosaic_bayer(
        &raw_data,
        width,
        height,
        &raw_image.cfa,
        raw_image.blacklevels,
        raw_image.whitelevels,
    );

    // Pack result: [width_le_u32, height_le_u32, RGBA...]
    let mut result = Vec::with_capacity(8 + rgb.len());
    result.extend_from_slice(&(width as u32).to_le_bytes());
    result.extend_from_slice(&(height as u32).to_le_bytes());
    result.extend_from_slice(&rgb);

    Ok(result)
}

/// Basic Bayer CFA demosaic (bilinear interpolation)
fn demosaic_bayer(
    raw: &[u16],
    width: usize,
    height: usize,
    cfa: &rawloader::CFA,
    blacks: [u16; 4],
    whites: [u16; 4],
) -> Vec<u8> {
    let mut rgba = vec![0u8; width * height * 4];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let color = cfa.color_at(x, y);

            // Normalize raw value
            let black = blacks[color] as f64;
            let white = whites[color] as f64;
            let range = (white - black).max(1.0);
            let val = ((raw[idx] as f64 - black) / range).clamp(0.0, 1.0);

            // Simple nearest-neighbor for MVP (assign to the correct channel)
            let (r, g, b) = match color {
                0 => (
                    val,
                    get_green_at(raw, x, y, width, height, cfa, &blacks, &whites),
                    get_blue_at(raw, x, y, width, height, cfa, &blacks, &whites),
                ),
                1 => (
                    get_red_at(raw, x, y, width, height, cfa, &blacks, &whites),
                    val,
                    get_blue_at(raw, x, y, width, height, cfa, &blacks, &whites),
                ),
                2 => (
                    get_red_at(raw, x, y, width, height, cfa, &blacks, &whites),
                    get_green_at(raw, x, y, width, height, cfa, &blacks, &whites),
                    val,
                ),
                _ => (val, val, val),
            };

            // Apply simple sRGB gamma
            let out_idx = idx * 4;
            rgba[out_idx] = (linear_to_srgb(r) * 255.0) as u8;
            rgba[out_idx + 1] = (linear_to_srgb(g) * 255.0) as u8;
            rgba[out_idx + 2] = (linear_to_srgb(b) * 255.0) as u8;
            rgba[out_idx + 3] = 255;
        }
    }

    rgba
}

/// Get average of neighboring pixels with matching color
fn get_color_at(
    raw: &[u16],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    cfa: &rawloader::CFA,
    blacks: &[u16; 4],
    whites: &[u16; 4],
    target_color: usize,
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0;

    let x_start = if x > 0 { x - 1 } else { x };
    let x_end = if x + 1 < width { x + 1 } else { x };
    let y_start = if y > 0 { y - 1 } else { y };
    let y_end = if y + 1 < height { y + 1 } else { y };

    for ny in y_start..=y_end {
        for nx in x_start..=x_end {
            if cfa.color_at(nx, ny) == target_color {
                let black = blacks[target_color] as f64;
                let white = whites[target_color] as f64;
                let range = (white - black).max(1.0);
                let val = ((raw[ny * width + nx] as f64 - black) / range).clamp(0.0, 1.0);
                sum += val;
                count += 1;
            }
        }
    }

    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

fn get_red_at(
    raw: &[u16],
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    cfa: &rawloader::CFA,
    b: &[u16; 4],
    wh: &[u16; 4],
) -> f64 {
    get_color_at(raw, x, y, w, h, cfa, b, wh, 0)
}

fn get_green_at(
    raw: &[u16],
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    cfa: &rawloader::CFA,
    b: &[u16; 4],
    wh: &[u16; 4],
) -> f64 {
    get_color_at(raw, x, y, w, h, cfa, b, wh, 1)
}

fn get_blue_at(
    raw: &[u16],
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    cfa: &rawloader::CFA,
    b: &[u16; 4],
    wh: &[u16; 4],
) -> f64 {
    get_color_at(raw, x, y, w, h, cfa, b, wh, 2)
}

/// Linear to sRGB gamma conversion
fn linear_to_srgb(c: f64) -> f64 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}
