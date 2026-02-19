// ============================================================
// shaders.wgsl — ferrum-photo WebGPU Shaders
//
// GPU-accelerated pipeline:
//   1. demosaic_bayer: MHC (Malvar-He-Cutler) demosaicing
//   2. develop: WB, exposure, contrast, HL/SH, vibrance,
//              saturation, sharpening, vignette, sRGB gamma
//   3. vs/fs_fullscreen: render to canvas
// ============================================================

// ─── Common Uniforms ─────────────────────────────────────────

struct RawParams {
  width: u32,
  height: u32,
  cfa_pattern: u32,
  _pad: u32,
  blacks: vec4f,
  whites: vec4f,
}

struct DevelopParams {
  exposure: f32,
  contrast: f32,
  highlights: f32,
  shadows: f32,
  wb_r: f32,
  wb_g: f32,
  wb_b: f32,
  vibrance: f32,
  saturation: f32,
  sharpening: f32,
  vignette: f32,
  _pad: f32,
}

// ─── Demosaic Compute Shader (MHC Algorithm) ─────────────────
// Malvar-He-Cutler demosaicing using 5×5 directional kernels.
// Provides superior edge reconstruction vs bilinear.

@group(0) @binding(0) var<storage, read> raw_data: array<u32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> raw_params: RawParams;

fn read_raw(idx: u32) -> f32 {
  let word_idx = idx / 2u;
  let word = raw_data[word_idx];
  if ((idx & 1u) == 0u) {
    return f32(word & 0xFFFFu);
  } else {
    return f32(word >> 16u);
  }
}

fn cfa_color(x: u32, y: u32) -> u32 {
  let px = x % 2u;
  let py = y % 2u;
  let shift = (py * 2u + px) * 8u;
  return (raw_params.cfa_pattern >> shift) & 0xFFu;
}

fn normalize(val: f32, color: u32) -> f32 {
  let black = raw_params.blacks[color];
  let white = raw_params.whites[color];
  let range = max(white - black, 1.0);
  return clamp((val - black) / range, 0.0, 1.0);
}

// Sample raw value at (x,y), clamped to image bounds, and normalize
fn sn(x: i32, y: i32) -> f32 {
  let cx = clamp(x, 0, i32(raw_params.width) - 1);
  let cy = clamp(y, 0, i32(raw_params.height) - 1);
  let idx = u32(cy) * raw_params.width + u32(cx);
  let color = cfa_color(u32(cx), u32(cy));
  return normalize(read_raw(idx), color);
}

@compute @workgroup_size(8, 8)
fn demosaic_bayer(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= raw_params.width || gid.y >= raw_params.height) {
    return;
  }

  let x = i32(gid.x);
  let y = i32(gid.y);
  let color = cfa_color(gid.x, gid.y);
  let C = sn(x, y);  // center pixel

  // Load the 5×5 neighborhood (only needed values for MHC)
  // Cardinal neighbors
  let N1 = sn(x, y - 1);
  let S1 = sn(x, y + 1);
  let W1 = sn(x - 1, y);
  let E1 = sn(x + 1, y);
  // Diagonal neighbors
  let NW = sn(x - 1, y - 1);
  let NE = sn(x + 1, y - 1);
  let SW = sn(x - 1, y + 1);
  let SE = sn(x + 1, y + 1);
  // 2-away cardinal
  let N2 = sn(x, y - 2);
  let S2 = sn(x, y + 2);
  let W2 = sn(x - 2, y);
  let E2 = sn(x + 2, y);

  var r: f32;
  var g: f32;
  var b: f32;

  // MHC kernels (Malvar, He, Cutler 2004)
  // The key insight: use the center pixel's color channel as a
  // gradient guide to interpolate missing channels.

  switch (color) {
    case 0u: {
      // ── Red pixel: have R, interpolate G and B ──
      r = C;
      // G at R: gradient-corrected average of 4 cardinal G neighbors
      g = 0.25 * (N1 + S1 + W1 + E1)
        + 0.125 * (2.0 * C - N2 - S2 - W2 - E2);
      // B at R: gradient-corrected average of 4 diagonal B neighbors
      b = 0.25 * (NW + NE + SW + SE)
        + 0.1875 * (2.0 * C - N2 - S2 - W2 - E2);
    }
    case 2u: {
      // ── Blue pixel: have B, interpolate G and R ──
      b = C;
      // G at B: same kernel as G at R
      g = 0.25 * (N1 + S1 + W1 + E1)
        + 0.125 * (2.0 * C - N2 - S2 - W2 - E2);
      // R at B: same kernel as B at R
      r = 0.25 * (NW + NE + SW + SE)
        + 0.1875 * (2.0 * C - N2 - S2 - W2 - E2);
    }
    case 1u: {
      // ── Green pixel: have G, interpolate R and B ──
      g = C;
      // Determine if this green is on a red row or blue row
      let is_red_row = (cfa_color(gid.x, gid.y - 1u) == 0u) ||
                       (cfa_color(gid.x, gid.y + 1u) == 0u);

      if (is_red_row) {
        // Red is above/below, Blue is left/right
        r = 0.5 * (N1 + S1)
          + 0.25 * (2.0 * C - N2 - S2);
        b = 0.5 * (W1 + E1)
          + 0.25 * (2.0 * C - W2 - E2);
      } else {
        // Blue is above/below, Red is left/right
        b = 0.5 * (N1 + S1)
          + 0.25 * (2.0 * C - N2 - S2);
        r = 0.5 * (W1 + E1)
          + 0.25 * (2.0 * C - W2 - E2);
      }
    }
    default: {
      r = C; g = C; b = C;
    }
  }

  // Clamp to [0,1] after interpolation (MHC can produce negatives)
  r = clamp(r, 0.0, 1.0);
  g = clamp(g, 0.0, 1.0);
  b = clamp(b, 0.0, 1.0);

  textureStore(output_tex, vec2i(gid.xy), vec4f(r, g, b, 1.0));
}

// ─── Develop Compute Shader ──────────────────────────────────

@group(0) @binding(0) var dev_input: texture_2d<f32>;
@group(0) @binding(1) var dev_output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> develop: DevelopParams;

// sRGB gamma from linear
fn linear_to_srgb(c: f32) -> f32 {
  if (c <= 0.0031308) {
    return c * 12.92;
  }
  return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

// Highlights / shadows tone mapping
fn tone_adjust(val: f32, highlights: f32, shadows: f32) -> f32 {
  var v = val;
  if (shadows != 0.0) {
    let shadow_mask = 1.0 - smoothstep(0.0, 0.5, v);
    v = v + shadows * shadow_mask * 0.3;
  }
  if (highlights != 0.0) {
    let highlight_mask = smoothstep(0.5, 1.0, v);
    v = v + highlights * highlight_mask * 0.3;
  }
  return clamp(v, 0.0, 1.0);
}

// RGB → HSL
fn rgb_to_hsl(rgb: vec3f) -> vec3f {
  let cmax = max(rgb.r, max(rgb.g, rgb.b));
  let cmin = min(rgb.r, min(rgb.g, rgb.b));
  let delta = cmax - cmin;
  let l = (cmax + cmin) * 0.5;

  if (delta < 0.00001) {
    return vec3f(0.0, 0.0, l);
  }

  let s = select(delta / (2.0 - cmax - cmin), delta / (cmax + cmin), l < 0.5);

  var h: f32;
  if (cmax == rgb.r) {
    h = (rgb.g - rgb.b) / delta;
    if (h < 0.0) { h += 6.0; }
  } else if (cmax == rgb.g) {
    h = (rgb.b - rgb.r) / delta + 2.0;
  } else {
    h = (rgb.r - rgb.g) / delta + 4.0;
  }
  h /= 6.0;

  return vec3f(h, s, l);
}

fn hue_to_rgb(p: f32, q: f32, t_in: f32) -> f32 {
  var t = t_in;
  if (t < 0.0) { t += 1.0; }
  if (t > 1.0) { t -= 1.0; }
  if (t < 1.0 / 6.0) { return p + (q - p) * 6.0 * t; }
  if (t < 0.5) { return q; }
  if (t < 2.0 / 3.0) { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
  return p;
}

// HSL → RGB
fn hsl_to_rgb(hsl: vec3f) -> vec3f {
  if (hsl.y < 0.00001) {
    return vec3f(hsl.z, hsl.z, hsl.z);
  }
  let q = select(hsl.z + hsl.y - hsl.z * hsl.y, hsl.z * (1.0 + hsl.y), hsl.z < 0.5);
  let p = 2.0 * hsl.z - q;
  return vec3f(
    hue_to_rgb(p, q, hsl.x + 1.0 / 3.0),
    hue_to_rgb(p, q, hsl.x),
    hue_to_rgb(p, q, hsl.x - 1.0 / 3.0),
  );
}

@compute @workgroup_size(8, 8)
fn develop_image(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(dev_input);
  if (gid.x >= dims.x || gid.y >= dims.y) {
    return;
  }

  let coords = vec2i(gid.xy);
  let pixel = textureLoad(dev_input, coords, 0);

  // 1. White balance
  var r = pixel.r * develop.wb_r;
  var g = pixel.g * develop.wb_g;
  var b = pixel.b * develop.wb_b;

  // 2. Exposure
  let ev_factor = pow(2.0, develop.exposure);
  r *= ev_factor;
  g *= ev_factor;
  b *= ev_factor;

  // 3. Contrast (S-curve)
  if (develop.contrast != 0.0) {
    let c = develop.contrast * 0.5 + 1.0;
    r = clamp((r - 0.5) * c + 0.5, 0.0, 1.0);
    g = clamp((g - 0.5) * c + 0.5, 0.0, 1.0);
    b = clamp((b - 0.5) * c + 0.5, 0.0, 1.0);
  } else {
    r = clamp(r, 0.0, 1.0);
    g = clamp(g, 0.0, 1.0);
    b = clamp(b, 0.0, 1.0);
  }

  // 4. Highlights / Shadows
  r = tone_adjust(r, develop.highlights, develop.shadows);
  g = tone_adjust(g, develop.highlights, develop.shadows);
  b = tone_adjust(b, develop.highlights, develop.shadows);

  // 5. Vibrance & Saturation (in HSL space)
  if (develop.vibrance != 0.0 || develop.saturation != 0.0) {
    var hsl = rgb_to_hsl(vec3f(r, g, b));

    // Vibrance: boost less-saturated colors more (perceptual)
    if (develop.vibrance != 0.0) {
      let boost = develop.vibrance * (1.0 - hsl.y) * 0.5;
      hsl.y = clamp(hsl.y + boost, 0.0, 1.0);
    }

    // Saturation: uniform adjustment
    if (develop.saturation != 0.0) {
      hsl.y = clamp(hsl.y * (1.0 + develop.saturation), 0.0, 1.0);
    }

    let rgb_new = hsl_to_rgb(hsl);
    r = rgb_new.r;
    g = rgb_new.g;
    b = rgb_new.b;
  }

  // 6. Sharpening (3×3 Laplacian unsharp mask)
  if (develop.sharpening != 0.0) {
    // Sample neighbors from dev_input (pre-develop, so we sharpen structure)
    let pN = textureLoad(dev_input, coords + vec2i(0, -1), 0);
    let pS = textureLoad(dev_input, coords + vec2i(0, 1), 0);
    let pW = textureLoad(dev_input, coords + vec2i(-1, 0), 0);
    let pE = textureLoad(dev_input, coords + vec2i(1, 0), 0);

    // Luminance-based sharpening to avoid color fringing
    let lum_c = 0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b;
    let lum_n = 0.299 * pN.r + 0.587 * pN.g + 0.114 * pN.b;
    let lum_s = 0.299 * pS.r + 0.587 * pS.g + 0.114 * pS.b;
    let lum_w = 0.299 * pW.r + 0.587 * pW.g + 0.114 * pW.b;
    let lum_e = 0.299 * pE.r + 0.587 * pE.g + 0.114 * pE.b;

    let edge = 4.0 * lum_c - lum_n - lum_s - lum_w - lum_e;
    let sharp_amount = develop.sharpening * 1.5;
    r = clamp(r + edge * sharp_amount, 0.0, 1.0);
    g = clamp(g + edge * sharp_amount, 0.0, 1.0);
    b = clamp(b + edge * sharp_amount, 0.0, 1.0);
  }

  // 7. Vignette
  if (develop.vignette != 0.0) {
    let uv = vec2f(f32(gid.x) / f32(dims.x), f32(gid.y) / f32(dims.y));
    let center = vec2f(0.5, 0.5);
    let dist = length(uv - center) * 1.414; // normalize to [0,1] at corners
    let vig = 1.0 - develop.vignette * dist * dist;
    r *= vig;
    g *= vig;
    b *= vig;
    r = clamp(r, 0.0, 1.0);
    g = clamp(g, 0.0, 1.0);
    b = clamp(b, 0.0, 1.0);
  }

  // 8. sRGB gamma
  r = linear_to_srgb(r);
  g = linear_to_srgb(g);
  b = linear_to_srgb(b);

  textureStore(dev_output, coords, vec4f(r, g, b, 1.0));
}

// ─── Fullscreen Quad Vertex Shader ───────────────────────────

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VertexOutput {
  var out: VertexOutput;
  let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
  let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
  out.position = vec4f(x, y, 0.0, 1.0);
  out.uv = vec2f(
    (x + 1.0) * 0.5,
    1.0 - (y + 1.0) * 0.5
  );
  return out;
}

// ─── Fullscreen Quad Fragment Shader ─────────────────────────

@group(0) @binding(0) var render_tex: texture_2d<f32>;
@group(0) @binding(1) var render_sampler: sampler;

@fragment
fn fs_fullscreen(@location(0) uv: vec2f) -> @location(0) vec4f {
  return textureSample(render_tex, render_sampler, uv);
}
