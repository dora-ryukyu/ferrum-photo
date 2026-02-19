// ============================================================
// main.ts — ferrum-photo WebGPU Entry Point
//
// Initializes WebGPU, loads images via WASM decode or browser
// APIs, runs exposure compute shader, renders to canvas.
// ============================================================

import "./style.css";
import shaderSource from "./shaders.wgsl?raw";
import { decode_raw } from "./wasm/ferrum_photo_core.js";

console.log("✓ WASM module loaded");

// RAW file extensions supported by rawloader
const RAW_EXTENSIONS = new Set([
  "cr2", "cr3", "nef", "nrw", "arw", "srf", "sr2",
  "dng", "raf", "orf", "rw2", "pef", "srw", "x3f",
  "mrw", "mef", "erf", "kdc", "dcr", "dcs", "crw",
]);

// ─── Types ───────────────────────────────────────────────────

interface GPUState {
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  // Compute pipeline
  computePipeline: GPUComputePipeline;
  computeBindGroupLayout: GPUBindGroupLayout;
  // Render pipeline
  renderPipeline: GPURenderPipeline;
  renderBindGroupLayout: GPUBindGroupLayout;
  sampler: GPUSampler;
  // Uniform
  paramsBuffer: GPUBuffer;
}

interface ImageState {
  inputTexture: GPUTexture;
  outputTexture: GPUTexture;
  computeBindGroup: GPUBindGroup;
  renderBindGroup: GPUBindGroup;
  width: number;
  height: number;
}

// ─── Globals ─────────────────────────────────────────────────

let gpu: GPUState | null = null;
let img: ImageState | null = null;
let exposure = 0.0;
let animFrameId = 0;

// ─── DOM ─────────────────────────────────────────────────────

const canvas = document.getElementById("viewport") as HTMLCanvasElement;
const placeholder = document.getElementById("viewport-placeholder")!;
const fileInput = document.getElementById("file-input") as HTMLInputElement;
const exposureSlider = document.getElementById("exposure-slider") as HTMLInputElement;
const exposureValue = document.getElementById("exposure-value")!;
const viewportContainer = document.getElementById("viewport-container")!;
const viewportStatus = document.getElementById("viewport-status")!;
const imageInfo = document.getElementById("image-info")!;
const badgeWebGPU = document.getElementById("badge-webgpu")!;
const badgeCOI = document.getElementById("badge-coi")!;
const badgeWasm = document.getElementById("badge-wasm")!;

// ─── Entry ───────────────────────────────────────────────────

async function main() {
  // WASM loaded (top-level import succeeded)
  badgeWasm.classList.add("active");

  // Check cross-origin isolation
  if (crossOriginIsolated) {
    badgeCOI.classList.add("active");
  }

  // Init WebGPU
  try {
    gpu = await initWebGPU();
    badgeWebGPU.classList.add("active");
    console.log("✓ WebGPU initialized");
  } catch (e) {
    badgeWebGPU.classList.add("error");
    console.error("WebGPU init failed:", e);
    showError(
      "WebGPU is not supported in this browser. Please use Chrome 113+ or Edge 113+."
    );
    return;
  }

  // Event listeners
  fileInput.addEventListener("change", handleFileInput);
  exposureSlider.addEventListener("input", handleExposureChange);

  // Drag & drop
  viewportContainer.addEventListener("dragover", (e) => {
    e.preventDefault();
    viewportContainer.classList.add("drag-over");
  });
  viewportContainer.addEventListener("dragleave", () => {
    viewportContainer.classList.remove("drag-over");
  });
  viewportContainer.addEventListener("drop", (e) => {
    e.preventDefault();
    viewportContainer.classList.remove("drag-over");
    if (e.dataTransfer?.files.length) {
      loadFile(e.dataTransfer.files[0]);
    }
  });

  // Double-click to reset exposure
  exposureSlider.addEventListener("dblclick", () => {
    exposureSlider.value = "0";
    handleExposureChange();
  });

  // Show UI (prevents white flash from CSS loading)
  document.getElementById("app")!.classList.add("ready");
}

// ─── WebGPU Initialization ───────────────────────────────────

async function initWebGPU(): Promise<GPUState> {
  if (!navigator.gpu) {
    throw new Error("WebGPU not supported");
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) throw new Error("No GPU adapter found");

  // Request the adapter's maximum texture dimension to handle large RAW files
  const maxDim = adapter.limits.maxTextureDimension2D;
  console.log(`Adapter maxTextureDimension2D: ${maxDim}`);

  const device = await adapter.requestDevice({
    requiredFeatures: [],
    requiredLimits: {
      maxTextureDimension2D: maxDim,
    },
  });

  const context = canvas.getContext("webgpu");
  if (!context) throw new Error("Failed to get WebGPU context");

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "premultiplied" });

  // Create shader module
  const shaderModule = device.createShaderModule({ code: shaderSource });

  // ── Compute pipeline ──
  const computeBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "float" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: "write-only", format: "rgba8unorm" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
    ],
  });

  const computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [computeBindGroupLayout],
    }),
    compute: {
      module: shaderModule,
      entryPoint: "adjust_exposure",
    },
  });

  // ── Render pipeline ──
  const renderBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: "float" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: {},
      },
    ],
  });

  const renderPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [renderBindGroupLayout],
    }),
    vertex: {
      module: shaderModule,
      entryPoint: "vs_fullscreen",
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_fullscreen",
      targets: [{ format }],
    },
    primitive: {
      topology: "triangle-list",
    },
  });

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  // Params uniform buffer (16 bytes = vec4f alignment)
  const paramsBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  return {
    device,
    context,
    format,
    computePipeline,
    computeBindGroupLayout,
    renderPipeline,
    renderBindGroupLayout,
    sampler,
    paramsBuffer,
  };
}

// ─── File Loading ────────────────────────────────────────────

function handleFileInput() {
  if (fileInput.files?.length) {
    loadFile(fileInput.files[0]);
  }
}

async function loadFile(file: File) {
  if (!gpu) return;

  showLoading(`Decoding ${file.name}…`);

  try {
    const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
    const isRaw = RAW_EXTENSIONS.has(ext);

    if (isRaw) {
      // RAW path: decode via WASM → upload raw RGBA bytes
      console.log(`RAW file detected (.${ext}), decoding via WASM…`);
      const arrayBuf = await file.arrayBuffer();
      const result = decode_raw(new Uint8Array(arrayBuf));

      // First 8 bytes: width (u32 LE) + height (u32 LE)
      const headerBytes = new Uint8Array(result.buffer.slice(result.byteOffset, result.byteOffset + 8));
      const view = new DataView(headerBytes.buffer);
      const width = view.getUint32(0, true);
      const height = view.getUint32(4, true);
      const rgbaData = new Uint8Array(result.buffer.slice(result.byteOffset + 8));

      console.log(`RAW decoded: ${width}×${height}`);
      uploadRGBAToGPU(rgbaData, width, height);
    } else {
      // Standard image path: use browser decoding
      const blob = new Blob([file], { type: file.type || "image/jpeg" });
      const bitmap = await createImageBitmap(blob);
      uploadBitmapToGPU(bitmap);
      bitmap.close();
    }

    placeholder.classList.add("hidden");
    canvas.classList.add("visible");
    viewportStatus.classList.remove("hidden");
    imageInfo.textContent = `${img!.width}×${img!.height} • ${file.name}`;

    render();
    hideLoading();
  } catch (e) {
    hideLoading();
    console.error("Failed to load:", e);
    showError(`Failed to load ${file.name}: ${(e as Error).message}`);
  }
}

// ─── GPU Upload ──────────────────────────────────────────────

/** Upload an ImageBitmap (standard images: JPEG, PNG, etc.) */
function uploadBitmapToGPU(bitmap: ImageBitmap) {
  if (!gpu) return;

  const { device } = gpu;
  const width = bitmap.width;
  const height = bitmap.height;

  setupCanvas(width, height);

  const inputTexture = createInputTexture(device, width, height);
  device.queue.copyExternalImageToTexture(
    { source: bitmap },
    { texture: inputTexture },
    { width, height }
  );

  createBindGroups(inputTexture, width, height);
}

/** Upload raw RGBA u8 data (RAW files decoded via WASM) */
function uploadRGBAToGPU(rgba: Uint8Array, width: number, height: number) {
  if (!gpu) return;

  const { device } = gpu;

  setupCanvas(width, height);

  const inputTexture = createInputTexture(device, width, height);

  // writeTexture needs rows aligned to 256 bytes
  const bytesPerRow = width * 4;
  const alignedBytesPerRow = Math.ceil(bytesPerRow / 256) * 256;

  if (bytesPerRow === alignedBytesPerRow) {
    // No padding needed — upload directly
    device.queue.writeTexture(
      { texture: inputTexture },
      rgba.buffer as ArrayBuffer,
      { offset: rgba.byteOffset, bytesPerRow, rowsPerImage: height },
      { width, height }
    );
  } else {
    // Pad each row to meet 256-byte alignment
    const padded = new Uint8Array(alignedBytesPerRow * height);
    for (let y = 0; y < height; y++) {
      padded.set(
        rgba.subarray(y * bytesPerRow, y * bytesPerRow + bytesPerRow),
        y * alignedBytesPerRow
      );
    }
    device.queue.writeTexture(
      { texture: inputTexture },
      padded.buffer as ArrayBuffer,
      { bytesPerRow: alignedBytesPerRow, rowsPerImage: height },
      { width, height }
    );
  }

  createBindGroups(inputTexture, width, height);
}

// ─── GPU Helpers ─────────────────────────────────────────────

function setupCanvas(width: number, height: number) {
  if (!gpu) return;
  canvas.width = width;
  canvas.height = height;
  gpu.context.configure({
    device: gpu.device,
    format: gpu.format,
    alphaMode: "premultiplied",
  });
  // Destroy previous textures
  img?.inputTexture.destroy();
  img?.outputTexture.destroy();
}

function createInputTexture(device: GPUDevice, width: number, height: number): GPUTexture {
  return device.createTexture({
    size: { width, height },
    format: "rgba8unorm",
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.RENDER_ATTACHMENT,
  });
}

function createBindGroups(inputTexture: GPUTexture, width: number, height: number) {
  if (!gpu) return;

  const { device } = gpu;

  const outputTexture = device.createTexture({
    size: { width, height },
    format: "rgba8unorm",
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING,
  });

  const computeBindGroup = device.createBindGroup({
    layout: gpu.computeBindGroupLayout,
    entries: [
      { binding: 0, resource: inputTexture.createView() },
      { binding: 1, resource: outputTexture.createView() },
      { binding: 2, resource: { buffer: gpu.paramsBuffer } },
    ],
  });

  const renderBindGroup = device.createBindGroup({
    layout: gpu.renderBindGroupLayout,
    entries: [
      { binding: 0, resource: outputTexture.createView() },
      { binding: 1, resource: gpu.sampler },
    ],
  });

  img = {
    inputTexture,
    outputTexture,
    computeBindGroup,
    renderBindGroup,
    width,
    height,
  };
}

// ─── Render ──────────────────────────────────────────────────

function render() {
  if (!gpu || !img) return;
  cancelAnimationFrame(animFrameId);
  animFrameId = requestAnimationFrame(doRender);
}

function doRender() {
  if (!gpu || !img) return;

  const { device, context } = gpu;

  // Update params
  const paramsData = new Float32Array([exposure, 0, 0, 0]);
  device.queue.writeBuffer(gpu.paramsBuffer, 0, paramsData);

  const commandEncoder = device.createCommandEncoder();

  // ── Compute pass: exposure adjustment ──
  const computePass = commandEncoder.beginComputePass();
  computePass.setPipeline(gpu.computePipeline);
  computePass.setBindGroup(0, img.computeBindGroup);
  computePass.dispatchWorkgroups(
    Math.ceil(img.width / 8),
    Math.ceil(img.height / 8)
  );
  computePass.end();

  // ── Render pass: display output ──
  const textureView = context.getCurrentTexture().createView();
  const renderPass = commandEncoder.beginRenderPass({
    colorAttachments: [
      {
        view: textureView,
        clearValue: { r: 0.04, g: 0.04, b: 0.06, a: 1 },
        loadOp: "clear" as GPULoadOp,
        storeOp: "store" as GPUStoreOp,
      },
    ],
  });
  renderPass.setPipeline(gpu.renderPipeline);
  renderPass.setBindGroup(0, img.renderBindGroup);
  renderPass.draw(3); // fullscreen triangle
  renderPass.end();

  device.queue.submit([commandEncoder.finish()]);
}

// ─── Exposure Control ────────────────────────────────────────

function handleExposureChange() {
  exposure = parseFloat(exposureSlider.value);
  const sign = exposure >= 0 ? "+" : "";
  exposureValue.textContent = `${sign}${exposure.toFixed(2)}`;
  render();
}

// ─── UI Helpers ──────────────────────────────────────────────

function showLoading(text: string) {
  let overlay = document.getElementById("loading-overlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.id = "loading-overlay";
    overlay.innerHTML = `
      <div class="spinner"></div>
      <span class="loading-text">${text}</span>
    `;
    viewportContainer.appendChild(overlay);
  } else {
    overlay.querySelector(".loading-text")!.textContent = text;
    overlay.classList.remove("hidden");
  }
}

function hideLoading() {
  const overlay = document.getElementById("loading-overlay");
  if (overlay) overlay.classList.add("hidden");
}

function showError(message: string) {
  const el = document.createElement("div");
  el.style.cssText = `
    position:fixed; bottom:20px; left:50%; transform:translateX(-50%);
    background:rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.3);
    color:#fca5a5; padding:12px 20px; border-radius:10px; font-size:13px;
    backdrop-filter:blur(8px); z-index:100; max-width:500px; text-align:center;
  `;
  el.textContent = message;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 8000);
}

// ─── Start ───────────────────────────────────────────────────

main();
