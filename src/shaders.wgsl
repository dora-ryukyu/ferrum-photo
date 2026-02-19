// ============================================================
// shaders.wgsl — ferrum-photo WebGPU Shaders
//
// Compute shader for exposure adjustment + fullscreen quad
// rendering pipeline for displaying the processed image.
// ============================================================

// ─── Exposure Compute Shader ─────────────────────────────────

struct Params {
  exposure: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
}

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn adjust_exposure(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(input_tex);
  if (gid.x >= dims.x || gid.y >= dims.y) {
    return;
  }

  let coords = vec2i(gid.xy);
  let pixel = textureLoad(input_tex, coords, 0);

  // Exposure: multiply by 2^exposure (photographic stops)
  let factor = pow(2.0, params.exposure);
  let adjusted = vec4f(
    clamp(pixel.r * factor, 0.0, 1.0),
    clamp(pixel.g * factor, 0.0, 1.0),
    clamp(pixel.b * factor, 0.0, 1.0),
    pixel.a
  );

  textureStore(output_tex, coords, adjusted);
}

// ─── Fullscreen Quad Vertex Shader ───────────────────────────

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

// Generates a fullscreen triangle (3 vertices, no vertex buffer needed)
@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VertexOutput {
  var out: VertexOutput;
  // Creates a triangle that covers the entire screen:
  //   vertex 0: (-1, -1) uv (0, 1)
  //   vertex 1: ( 3, -1) uv (2, 1)
  //   vertex 2: (-1,  3) uv (0, -1)
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
