// ============================================================
// main.ts — ferrum-photo WebGPU Entry Point
//
// Pipeline: RAW file → WASM parse → GPU demosaic → develop → render
// Phase 4: MHC demosaic, vibrance/sat/sharpen/vignette,
//          File System Access API, presets, histogram fix
// ============================================================

import "./style.css";
import shaderSource from "./shaders.wgsl?raw";
import { decode_raw } from "./wasm/ferrum_photo_core.js";

console.log("✓ WASM module loaded");

const RAW_EXTENSIONS = new Set([
  "cr2", "cr3", "nef", "nrw", "arw", "srf", "sr2",
  "dng", "raf", "orf", "rw2", "pef", "srw", "x3f",
  "mrw", "mef", "erf", "kdc", "dcr", "dcs", "crw",
]);

const PRESET_VERSION = 1;

// ─── Types ───────────────────────────────────────────────────

interface GPUState {
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  shaderModule: GPUShaderModule;
  demosaicPipeline: GPUComputePipeline;
  demosaicBindGroupLayout: GPUBindGroupLayout;
  developPipeline: GPUComputePipeline;
  developBindGroupLayout: GPUBindGroupLayout;
  renderPipeline: GPURenderPipeline;
  renderBindGroupLayout: GPUBindGroupLayout;
  sampler: GPUSampler;
  rawParamsBuffer: GPUBuffer;
  developParamsBuffer: GPUBuffer;
}

interface ImageState {
  width: number;
  height: number;
  isRaw: boolean;
  rawBuffer: GPUBuffer | null;
  demosaicTexture: GPUTexture | null;
  demosaicBindGroup: GPUBindGroup | null;
  inputTexture: GPUTexture | null;
  developOutputTexture: GPUTexture;
  developBindGroup: GPUBindGroup;
  renderBindGroup: GPUBindGroup;
}

interface DevelopSettings {
  exposure: number;
  contrast: number;
  highlights: number;
  shadows: number;
  wb_r: number;
  wb_g: number;
  wb_b: number;
  vibrance: number;
  saturation: number;
  sharpening: number;
  vignette: number;
}

interface Preset {
  version: number;
  name: string;
  settings: DevelopSettings;
}

// ─── Defaults ────────────────────────────────────────────────

const DEFAULT_SETTINGS: DevelopSettings = {
  exposure: 0, contrast: 0, highlights: 0, shadows: 0,
  wb_r: 1, wb_g: 1, wb_b: 1,
  vibrance: 0, saturation: 0, sharpening: 0, vignette: 0,
};

// ─── Globals ─────────────────────────────────────────────────

let gpu: GPUState | null = null;
let img: ImageState | null = null;
let develop: DevelopSettings = { ...DEFAULT_SETTINGS };
let animFrameId = 0;

// Zoom & Pan
let zoomLevel = 0; // 0 = fit
let panX = 0;
let panY = 0;
let isPanning = false;
let panStartX = 0;
let panStartY = 0;

// Before/After
let showBefore = false;

// ─── DOM ─────────────────────────────────────────────────────

const canvas = document.getElementById("viewport") as HTMLCanvasElement;
const placeholder = document.getElementById("viewport-placeholder")!;
const fileInput = document.getElementById("file-input") as HTMLInputElement;
const viewportContainer = document.getElementById("viewport-container")!;
const viewportStatus = document.getElementById("viewport-status")!;
const imageInfo = document.getElementById("image-info")!;
const zoomLevelEl = document.getElementById("zoom-level")!;
const badgeWebGPU = document.getElementById("badge-webgpu")!;
const badgeCOI = document.getElementById("badge-coi")!;
const badgeWasm = document.getElementById("badge-wasm")!;
const histogramCanvas = document.getElementById("histogram") as HTMLCanvasElement;
const histogramCtx = histogramCanvas.getContext("2d")!;

// Sliders
const sliders = {
  exposure: document.getElementById("exposure-slider") as HTMLInputElement,
  contrast: document.getElementById("contrast-slider") as HTMLInputElement,
  highlights: document.getElementById("highlights-slider") as HTMLInputElement,
  shadows: document.getElementById("shadows-slider") as HTMLInputElement,
  wb: document.getElementById("wb-slider") as HTMLInputElement,
  vibrance: document.getElementById("vibrance-slider") as HTMLInputElement,
  saturation: document.getElementById("saturation-slider") as HTMLInputElement,
  sharpening: document.getElementById("sharpening-slider") as HTMLInputElement,
  vignette: document.getElementById("vignette-slider") as HTMLInputElement,
};

const valueEls = {
  exposure: document.getElementById("exposure-value")!,
  contrast: document.getElementById("contrast-value")!,
  highlights: document.getElementById("highlights-value")!,
  shadows: document.getElementById("shadows-value")!,
  wb: document.getElementById("wb-value")!,
  vibrance: document.getElementById("vibrance-value")!,
  saturation: document.getElementById("saturation-value")!,
  sharpening: document.getElementById("sharpening-value")!,
  vignette: document.getElementById("vignette-value")!,
};

// Buttons
const btnOpen = document.getElementById("btn-open")!;
const btnExport = document.getElementById("btn-export") as HTMLButtonElement;
const btnReset = document.getElementById("btn-reset")!;
const btnBeforeAfter = document.getElementById("btn-before-after")!;
const btnPresetSave = document.getElementById("btn-preset-save")!;
const btnPresetLoad = document.getElementById("btn-preset-load")!;

// ─── Entry ───────────────────────────────────────────────────

async function main() {
  badgeWasm.classList.add("active");
  if (crossOriginIsolated) badgeCOI.classList.add("active");

  try {
    gpu = await initWebGPU();
    badgeWebGPU.classList.add("active");
    console.log("✓ WebGPU initialized");
  } catch (e) {
    badgeWebGPU.classList.add("error");
    console.error("WebGPU init failed:", e);
    showError("WebGPU is not supported in this browser. Please use Chrome 113+ or Edge 113+.");
    document.getElementById("app")!.classList.add("ready");
    return;
  }

  // File input
  fileInput.addEventListener("change", () => {
    if (fileInput.files?.length) loadFile(fileInput.files[0]);
  });
  btnOpen.addEventListener("click", openFile);

  // All develop sliders
  for (const s of Object.values(sliders)) {
    s?.addEventListener("input", updateDevelop);
  }

  // Double-click slider to reset
  const sliderDefaults: Record<string, string> = {
    exposure: "0", contrast: "0", highlights: "0", shadows: "0",
    wb: "6500", vibrance: "0", saturation: "0", sharpening: "0", vignette: "0",
  };
  for (const [key, el] of Object.entries(sliders)) {
    el?.addEventListener("dblclick", () => {
      el.value = sliderDefaults[key];
      updateDevelop();
    });
  }

  // Action buttons
  btnReset.addEventListener("click", resetAll);
  btnBeforeAfter.addEventListener("click", toggleBeforeAfter);
  btnExport.addEventListener("click", showExportModal);
  btnPresetSave.addEventListener("click", savePreset);
  btnPresetLoad.addEventListener("click", loadPreset);

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
    if (e.dataTransfer?.files.length) loadFile(e.dataTransfer.files[0]);
  });

  // Zoom & Pan
  viewportContainer.addEventListener("wheel", onWheel, { passive: false });
  viewportContainer.addEventListener("mousedown", onPanStart);
  window.addEventListener("mousemove", onPanMove);
  window.addEventListener("mouseup", onPanEnd);
  viewportContainer.addEventListener("dblclick", (e) => {
    if ((e.target as HTMLElement).closest("#viewport-placeholder")) return;
    fitZoom();
  });

  // Keyboard shortcuts
  window.addEventListener("keydown", onKeyDown);

  document.getElementById("app")!.classList.add("ready");
  drawHistogram(null);
}

// ─── WebGPU Initialization ───────────────────────────────────

async function initWebGPU(): Promise<GPUState> {
  if (!navigator.gpu) throw new Error("WebGPU not supported");

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No GPU adapter found");

  const maxDim = adapter.limits.maxTextureDimension2D;
  const maxBufSize = adapter.limits.maxStorageBufferBindingSize;
  const maxBuffer = adapter.limits.maxBufferSize;
  console.log(`Adapter limits: maxTex=${maxDim}, maxBuf=${maxBufSize}, maxBufferSize=${maxBuffer}`);

  const device = await adapter.requestDevice({
    requiredLimits: {
      maxTextureDimension2D: maxDim,
      maxStorageBufferBindingSize: maxBufSize,
      maxBufferSize: maxBuffer,
    },
  });

  const context = canvas.getContext("webgpu");
  if (!context) throw new Error("Failed to get WebGPU context");

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "premultiplied" });

  const shaderModule = device.createShaderModule({ code: shaderSource });

  // ── Demosaic pipeline ──
  const demosaicBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const demosaicPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [demosaicBindGroupLayout] }),
    compute: { module: shaderModule, entryPoint: "demosaic_bayer" },
  });

  // ── Develop pipeline ──
  const developBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba8unorm" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const developPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [developBindGroupLayout] }),
    compute: { module: shaderModule, entryPoint: "develop_image" },
  });

  // ── Render pipeline ──
  const renderBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
    ],
  });
  const renderPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayout] }),
    vertex: { module: shaderModule, entryPoint: "vs_fullscreen" },
    fragment: { module: shaderModule, entryPoint: "fs_fullscreen", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

  const rawParamsBuffer = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // DevelopParams: 12 floats = 48 bytes
  const developParamsBuffer = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  return {
    device, context, format, shaderModule,
    demosaicPipeline, demosaicBindGroupLayout,
    developPipeline, developBindGroupLayout,
    renderPipeline, renderBindGroupLayout,
    sampler, rawParamsBuffer, developParamsBuffer,
  };
}

// ─── File Loading ────────────────────────────────────────────

async function openFile() {
  // Use File System Access API if available
  if ("showOpenFilePicker" in window) {
    try {
      const [handle] = await (window as any).showOpenFilePicker({
        types: [
          {
            description: "Images & RAW files",
            accept: {
              "image/*": [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"],
              "image/x-raw": [".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw"],
            },
          },
        ],
      });
      const file = await handle.getFile();
      loadFile(file);
    } catch {
      // User cancelled
    }
  } else {
    fileInput.click();
  }
}

async function loadFile(file: File) {
  if (!gpu) return;

  showLoading(`Decoding ${file.name}…`);
  const t0 = performance.now();

  try {
    const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
    const isRaw = RAW_EXTENSIONS.has(ext);

    if (isRaw) {
      console.log(`RAW file detected (.${ext}), parsing via WASM…`);
      const arrayBuf = await file.arrayBuffer();
      const result = decode_raw(new Uint8Array(arrayBuf));

      const headerBuf = new ArrayBuffer(28);
      new Uint8Array(headerBuf).set(result.slice(0, 28));
      const hdr = new DataView(headerBuf);
      const width = hdr.getUint32(0, true);
      const height = hdr.getUint32(4, true);
      const cfaPattern = hdr.getUint32(8, true);
      const blacks = [hdr.getUint16(12, true), hdr.getUint16(14, true), hdr.getUint16(16, true), hdr.getUint16(18, true)];
      const whites = [hdr.getUint16(20, true), hdr.getUint16(22, true), hdr.getUint16(24, true), hdr.getUint16(26, true)];

      const rawBytes = result.slice(28);
      const parseTime = performance.now() - t0;
      console.log(`RAW parsed: ${width}×${height} in ${parseTime.toFixed(0)}ms`);

      uploadRawToGPU(rawBytes, width, height, cfaPattern, blacks, whites);
    } else {
      const blob = new Blob([file], { type: file.type || "image/jpeg" });
      const bitmap = await createImageBitmap(blob);
      uploadBitmapToGPU(bitmap);
      bitmap.close();
    }

    const totalTime = performance.now() - t0;
    console.log(`Total load time: ${totalTime.toFixed(0)}ms`);

    placeholder.classList.add("hidden");
    canvas.classList.add("visible");
    viewportStatus.classList.remove("hidden");
    imageInfo.textContent = `${img!.width}×${img!.height} • ${file.name}`;
    btnExport.disabled = false;

    fitZoom();
    render();
    hideLoading();
  } catch (e) {
    hideLoading();
    console.error("Failed to load:", e);
    showError(`Failed to load ${file.name}: ${(e as Error).message}`);
  }
}

// ─── GPU Upload: Standard Images ─────────────────────────────

function uploadBitmapToGPU(bitmap: ImageBitmap) {
  if (!gpu) return;
  const { device } = gpu;
  const width = bitmap.width;
  const height = bitmap.height;

  cleanupImageState();
  setupCanvas(width, height);

  const inputTexture = device.createTexture({
    size: { width, height },
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });

  device.queue.copyExternalImageToTexture(
    { source: bitmap },
    { texture: inputTexture },
    { width, height },
  );

  const developOutputTexture = device.createTexture({
    size: { width, height },
    format: "rgba8unorm",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });

  const developBindGroup = device.createBindGroup({
    layout: gpu.developBindGroupLayout,
    entries: [
      { binding: 0, resource: inputTexture.createView() },
      { binding: 1, resource: developOutputTexture.createView() },
      { binding: 2, resource: { buffer: gpu.developParamsBuffer } },
    ],
  });

  const renderBindGroup = device.createBindGroup({
    layout: gpu.renderBindGroupLayout,
    entries: [
      { binding: 0, resource: developOutputTexture.createView() },
      { binding: 1, resource: gpu.sampler },
    ],
  });

  img = {
    width, height, isRaw: false,
    rawBuffer: null, demosaicTexture: null, demosaicBindGroup: null,
    inputTexture,
    developOutputTexture, developBindGroup, renderBindGroup,
  };
}

// ─── GPU Upload: RAW Data ────────────────────────────────────

function uploadRawToGPU(
  rawBytes: Uint8Array,
  width: number, height: number,
  cfaPattern: number,
  blacks: number[], whites: number[],
) {
  if (!gpu) return;
  const { device } = gpu;

  cleanupImageState();
  setupCanvas(width, height);

  const rawBuffer = device.createBuffer({
    size: Math.ceil(rawBytes.byteLength / 4) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(rawBuffer.getMappedRange()).set(rawBytes);
  rawBuffer.unmap();

  const rawParamsData = new ArrayBuffer(48);
  const rawParamsView = new DataView(rawParamsData);
  rawParamsView.setUint32(0, width, true);
  rawParamsView.setUint32(4, height, true);
  rawParamsView.setUint32(8, cfaPattern, true);
  rawParamsView.setUint32(12, 0, true);
  rawParamsView.setFloat32(16, blacks[0], true);
  rawParamsView.setFloat32(20, blacks[1], true);
  rawParamsView.setFloat32(24, blacks[2], true);
  rawParamsView.setFloat32(28, blacks[3], true);
  rawParamsView.setFloat32(32, whites[0], true);
  rawParamsView.setFloat32(36, whites[1], true);
  rawParamsView.setFloat32(40, whites[2], true);
  rawParamsView.setFloat32(44, whites[3], true);

  device.queue.writeBuffer(gpu.rawParamsBuffer, 0, rawParamsData);

  const demosaicTexture = device.createTexture({
    size: { width, height },
    format: "rgba16float",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });

  const demosaicBindGroup = device.createBindGroup({
    layout: gpu.demosaicBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: rawBuffer } },
      { binding: 1, resource: demosaicTexture.createView() },
      { binding: 2, resource: { buffer: gpu.rawParamsBuffer } },
    ],
  });

  const developOutputTexture = device.createTexture({
    size: { width, height },
    format: "rgba8unorm",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });

  const developBindGroup = device.createBindGroup({
    layout: gpu.developBindGroupLayout,
    entries: [
      { binding: 0, resource: demosaicTexture.createView() },
      { binding: 1, resource: developOutputTexture.createView() },
      { binding: 2, resource: { buffer: gpu.developParamsBuffer } },
    ],
  });

  const renderBindGroup = device.createBindGroup({
    layout: gpu.renderBindGroupLayout,
    entries: [
      { binding: 0, resource: developOutputTexture.createView() },
      { binding: 1, resource: gpu.sampler },
    ],
  });

  img = {
    width, height, isRaw: true,
    rawBuffer, demosaicTexture, demosaicBindGroup,
    inputTexture: null,
    developOutputTexture, developBindGroup, renderBindGroup,
  };

  runDemosaic();
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
}

function cleanupImageState() {
  if (img) {
    img.rawBuffer?.destroy();
    img.demosaicTexture?.destroy();
    img.inputTexture?.destroy();
    img.developOutputTexture.destroy();
    img = null;
  }
}

function runDemosaic() {
  if (!gpu || !img || !img.demosaicBindGroup) return;

  const encoder = gpu.device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(gpu.demosaicPipeline);
  pass.setBindGroup(0, img.demosaicBindGroup);
  pass.dispatchWorkgroups(
    Math.ceil(img.width / 8),
    Math.ceil(img.height / 8),
  );
  pass.end();
  gpu.device.queue.submit([encoder.finish()]);
  console.log("✓ GPU demosaic (MHC) complete");
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

  // Build develop params (12 floats = 48 bytes)
  const s = showBefore ? DEFAULT_SETTINGS : develop;
  const devBuf = new Float32Array([
    s.exposure, s.contrast, s.highlights, s.shadows,
    s.wb_r, s.wb_g, s.wb_b, s.vibrance,
    s.saturation, s.sharpening, s.vignette, 0, // pad
  ]);
  device.queue.writeBuffer(gpu.developParamsBuffer, 0, devBuf);

  const encoder = device.createCommandEncoder();

  // Develop compute pass
  const devPass = encoder.beginComputePass();
  devPass.setPipeline(gpu.developPipeline);
  devPass.setBindGroup(0, img.developBindGroup);
  devPass.dispatchWorkgroups(
    Math.ceil(img.width / 8),
    Math.ceil(img.height / 8),
  );
  devPass.end();

  // Render pass
  const textureView = context.getCurrentTexture().createView();
  const renderPass = encoder.beginRenderPass({
    colorAttachments: [{
      view: textureView,
      clearValue: { r: 0.04, g: 0.04, b: 0.06, a: 1 },
      loadOp: "clear" as GPULoadOp,
      storeOp: "store" as GPUStoreOp,
    }],
  });
  renderPass.setPipeline(gpu.renderPipeline);
  renderPass.setBindGroup(0, img.renderBindGroup);
  renderPass.draw(3);
  renderPass.end();

  device.queue.submit([encoder.finish()]);

  // Schedule histogram after GPU work completes
  scheduleHistogramUpdate();
}

// ─── Develop Controls ────────────────────────────────────────

function updateDevelop() {
  develop.exposure = parseFloat(sliders.exposure?.value ?? "0");
  develop.contrast = parseFloat(sliders.contrast?.value ?? "0");
  develop.highlights = parseFloat(sliders.highlights?.value ?? "0");
  develop.shadows = parseFloat(sliders.shadows?.value ?? "0");
  develop.vibrance = parseFloat(sliders.vibrance?.value ?? "0");
  develop.saturation = parseFloat(sliders.saturation?.value ?? "0");
  develop.sharpening = parseFloat(sliders.sharpening?.value ?? "0");
  develop.vignette = parseFloat(sliders.vignette?.value ?? "0");

  const kelvin = parseFloat(sliders.wb?.value ?? "6500");
  const wb = kelvinToRGB(kelvin);
  develop.wb_r = wb.r;
  develop.wb_g = wb.g;
  develop.wb_b = wb.b;

  // Update value displays
  const fmtSigned = (v: number) => (v >= 0 ? "+" : "") + v.toFixed(2);
  valueEls.exposure.textContent = fmtSigned(develop.exposure);
  valueEls.contrast.textContent = fmtSigned(develop.contrast);
  valueEls.highlights.textContent = fmtSigned(develop.highlights);
  valueEls.shadows.textContent = fmtSigned(develop.shadows);
  valueEls.vibrance.textContent = fmtSigned(develop.vibrance);
  valueEls.saturation.textContent = fmtSigned(develop.saturation);
  valueEls.sharpening.textContent = parseFloat(sliders.sharpening?.value ?? "0").toFixed(2);
  valueEls.vignette.textContent = fmtSigned(develop.vignette);
  valueEls.wb.textContent = `${kelvin.toFixed(0)}K`;

  render();
}

function resetAll() {
  sliders.exposure.value = "0";
  sliders.contrast.value = "0";
  sliders.highlights.value = "0";
  sliders.shadows.value = "0";
  sliders.wb.value = "6500";
  sliders.vibrance.value = "0";
  sliders.saturation.value = "0";
  sliders.sharpening.value = "0";
  sliders.vignette.value = "0";
  updateDevelop();
}

function toggleBeforeAfter() {
  if (!img) return;
  showBefore = !showBefore;
  btnBeforeAfter.classList.toggle("active", showBefore);

  let label = viewportContainer.querySelector(".before-after-label") as HTMLElement;
  if (showBefore) {
    if (!label) {
      label = document.createElement("div");
      label.className = "before-after-label";
      viewportContainer.appendChild(label);
    }
    label.textContent = "BEFORE";
    label.style.display = "";
  } else {
    if (label) label.style.display = "none";
  }
  render();
}

// ─── Zoom & Pan ──────────────────────────────────────────────

function getContainerSize() {
  return { cw: viewportContainer.clientWidth, ch: viewportContainer.clientHeight };
}

function getFitScale(): number {
  if (!img) return 1;
  const { cw, ch } = getContainerSize();
  return Math.min(cw / img.width, ch / img.height, 1);
}

function fitZoom() {
  zoomLevel = 0;
  panX = 0;
  panY = 0;
  applyTransform();
  updateZoomDisplay();
}

function setZoom(scale: number, cx?: number, cy?: number) {
  if (!img) return;
  const fit = getFitScale();
  const minScale = fit;
  const maxScale = Math.max(4, fit * 8);
  const newScale = Math.max(minScale, Math.min(maxScale, scale));

  if (Math.abs(newScale - fit) < 0.01) { fitZoom(); return; }

  const oldScale = zoomLevel === 0 ? fit : zoomLevel;
  if (cx !== undefined && cy !== undefined) {
    const { cw, ch } = getContainerSize();
    const viewCx = cx - cw / 2;
    const viewCy = cy - ch / 2;
    panX = viewCx - (viewCx - panX) * (newScale / oldScale);
    panY = viewCy - (viewCy - panY) * (newScale / oldScale);
  }

  zoomLevel = newScale;
  clampPan();
  applyTransform();
  updateZoomDisplay();
}

function clampPan() {
  if (!img) return;
  const scale = zoomLevel === 0 ? getFitScale() : zoomLevel;
  const { cw, ch } = getContainerSize();
  const imgW = img.width * scale;
  const imgH = img.height * scale;
  const maxPanX = Math.max(0, (imgW - cw) / 2);
  const maxPanY = Math.max(0, (imgH - ch) / 2);
  panX = Math.max(-maxPanX, Math.min(maxPanX, panX));
  panY = Math.max(-maxPanY, Math.min(maxPanY, panY));
}

function applyTransform() {
  if (!img) return;
  const scale = zoomLevel === 0 ? getFitScale() : zoomLevel;
  canvas.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
  canvas.style.maxWidth = "none";
  canvas.style.maxHeight = "none";
  canvas.style.width = `${img.width}px`;
  canvas.style.height = `${img.height}px`;
  const canPan = zoomLevel !== 0;
  viewportContainer.classList.toggle("panning", canPan && !isPanning);
  viewportContainer.classList.toggle("dragging", isPanning);
}

function updateZoomDisplay() {
  zoomLevelEl.textContent = zoomLevel === 0 ? "Fit" : `${Math.round(zoomLevel * 100)}%`;
}

function onWheel(e: WheelEvent) {
  if (!img) return;
  e.preventDefault();
  const rect = viewportContainer.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  const currentScale = zoomLevel === 0 ? getFitScale() : zoomLevel;
  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  setZoom(currentScale * factor, cx, cy);
}

function onPanStart(e: MouseEvent) {
  if (!img || zoomLevel === 0 || e.button !== 0) return;
  isPanning = true;
  panStartX = e.clientX - panX;
  panStartY = e.clientY - panY;
  viewportContainer.classList.remove("panning");
  viewportContainer.classList.add("dragging");
}

function onPanMove(e: MouseEvent) {
  if (!isPanning) return;
  panX = e.clientX - panStartX;
  panY = e.clientY - panStartY;
  clampPan();
  applyTransform();
}

function onPanEnd() {
  if (!isPanning) return;
  isPanning = false;
  viewportContainer.classList.remove("dragging");
  viewportContainer.classList.toggle("panning", zoomLevel !== 0);
}

// ─── Histogram ───────────────────────────────────────────────

let histogramPending = false;

function scheduleHistogramUpdate() {
  if (histogramPending) return;
  histogramPending = true;
  // Wait for GPU to finish, then read pixels
  if (gpu) {
    gpu.device.queue.onSubmittedWorkDone().then(() => {
      histogramPending = false;
      readAndDrawHistogram();
    });
  }
}

async function readAndDrawHistogram() {
  if (!gpu || !img) return;

  try {
    // createImageBitmap on a WebGPU canvas after onSubmittedWorkDone
    const bitmap = await createImageBitmap(canvas);
    const sw = Math.min(512, img.width);
    const sh = Math.round(sw * (img.height / img.width));
    const offscreen = new OffscreenCanvas(sw, sh);
    const ctx2d = offscreen.getContext("2d")!;
    ctx2d.drawImage(bitmap, 0, 0, sw, sh);
    bitmap.close();

    const imageData = ctx2d.getImageData(0, 0, sw, sh);
    drawHistogram(imageData.data);
  } catch {
    drawHistogram(null);
  }
}

function drawHistogram(data: Uint8ClampedArray | null) {
  const w = histogramCanvas.width;
  const h = histogramCanvas.height;
  histogramCtx.clearRect(0, 0, w, h);

  if (!data) {
    histogramCtx.fillStyle = "rgba(255,255,255,0.03)";
    histogramCtx.fillRect(0, 0, w, h);
    return;
  }

  const bins = 256;
  const rHist = new Uint32Array(bins);
  const gHist = new Uint32Array(bins);
  const bHist = new Uint32Array(bins);
  const lHist = new Uint32Array(bins);

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    rHist[r]++;
    gHist[g]++;
    bHist[b]++;
    lHist[Math.round(0.299 * r + 0.587 * g + 0.114 * b)]++;
  }

  let maxVal = 0;
  for (let i = 1; i < bins - 1; i++) {
    maxVal = Math.max(maxVal, rHist[i], gHist[i], bHist[i]);
  }
  if (maxVal === 0) maxVal = 1;

  const drawChannel = (hist: Uint32Array, color: string) => {
    histogramCtx.beginPath();
    histogramCtx.moveTo(0, h);
    for (let i = 0; i < bins; i++) {
      const x = (i / (bins - 1)) * w;
      const y = h - (Math.log1p(hist[i]) / Math.log1p(maxVal)) * h;
      histogramCtx.lineTo(x, y);
    }
    histogramCtx.lineTo(w, h);
    histogramCtx.closePath();
    histogramCtx.fillStyle = color;
    histogramCtx.fill();
  };

  histogramCtx.globalCompositeOperation = "screen";
  drawChannel(lHist, "rgba(180, 180, 180, 0.25)");
  drawChannel(rHist, "rgba(255, 80, 80, 0.35)");
  drawChannel(gHist, "rgba(80, 255, 80, 0.35)");
  drawChannel(bHist, "rgba(80, 80, 255, 0.35)");
  histogramCtx.globalCompositeOperation = "source-over";
}

// ─── Export ──────────────────────────────────────────────────

function showExportModal() {
  if (!img) return;

  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  backdrop.innerHTML = `
    <div class="modal">
      <h3>Export Image</h3>
      <div class="modal-row">
        <label>Format</label>
        <select id="export-format">
          <option value="jpeg" selected>JPEG</option>
          <option value="png">PNG</option>
        </select>
      </div>
      <div class="modal-row" id="quality-row">
        <label>Quality</label>
        <div style="display:flex;align-items:center;gap:8px;">
          <input type="range" id="export-quality" min="10" max="100" step="5" value="92" class="slider" style="width:100px" />
          <span id="export-quality-value" style="font-family:var(--font-mono);font-size:12px;color:var(--accent-orange);">92%</span>
        </div>
      </div>
      <div class="modal-row">
        <label>Size</label>
        <span style="font-family:var(--font-mono);font-size:12px;color:var(--text-secondary);">${img.width}×${img.height}</span>
      </div>
      <div class="modal-actions">
        <button class="btn-modal btn-modal-cancel" id="export-cancel">Cancel</button>
        <button class="btn-modal btn-modal-primary" id="export-confirm">Save</button>
      </div>
    </div>
  `;

  document.body.appendChild(backdrop);

  const formatSelect = backdrop.querySelector("#export-format") as HTMLSelectElement;
  const qualitySlider = backdrop.querySelector("#export-quality") as HTMLInputElement;
  const qualityValue = backdrop.querySelector("#export-quality-value")!;
  const qualityRow = backdrop.querySelector("#quality-row") as HTMLElement;

  qualitySlider.addEventListener("input", () => {
    qualityValue.textContent = `${qualitySlider.value}%`;
  });
  formatSelect.addEventListener("change", () => {
    qualityRow.style.display = formatSelect.value === "png" ? "none" : "";
  });

  backdrop.querySelector("#export-cancel")!.addEventListener("click", () => backdrop.remove());
  backdrop.addEventListener("click", (e) => { if (e.target === backdrop) backdrop.remove(); });

  backdrop.querySelector("#export-confirm")!.addEventListener("click", async () => {
    const format = formatSelect.value as "jpeg" | "png";
    const quality = parseInt(qualitySlider.value) / 100;
    backdrop.remove();
    await doExport(format, quality);
  });
}

async function doExport(format: "jpeg" | "png", quality: number) {
  if (!img || !gpu) return;
  showLoading("Exporting…");

  try {
    // Wait for GPU to finish
    await gpu.device.queue.onSubmittedWorkDone();

    const bitmap = await createImageBitmap(canvas);
    const offscreen = new OffscreenCanvas(img.width, img.height);
    const ctx2d = offscreen.getContext("2d")!;
    ctx2d.drawImage(bitmap, 0, 0, img.width, img.height);
    bitmap.close();

    const mimeType = format === "png" ? "image/png" : "image/jpeg";
    const blob = await offscreen.convertToBlob({ type: mimeType, quality: format === "jpeg" ? quality : undefined });

    // Try File System Access API for saving
    if ("showSaveFilePicker" in window) {
      try {
        const ext = format === "png" ? "png" : "jpg";
        const handle = await (window as any).showSaveFilePicker({
          suggestedName: `ferrum-photo-export.${ext}`,
          types: [{
            description: format.toUpperCase(),
            accept: { [mimeType]: [`.${ext}`] },
          }],
        });
        const writable = await handle.createWritable();
        await writable.write(blob);
        await writable.close();
        hideLoading();
        return;
      } catch {
        // User cancelled save picker, fall through to download
      }
    }

    // Fallback: download via link
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ferrum-photo-export.${format === "png" ? "png" : "jpg"}`;
    a.click();
    URL.revokeObjectURL(url);
    hideLoading();
  } catch (e) {
    hideLoading();
    showError(`Export failed: ${(e as Error).message}`);
  }
}

// ─── Presets ─────────────────────────────────────────────────

function savePreset() {
  const preset: Preset = {
    version: PRESET_VERSION,
    name: "Custom Preset",
    settings: { ...develop },
  };

  const json = JSON.stringify(preset, null, 2);
  const blob = new Blob([json], { type: "application/json" });

  if ("showSaveFilePicker" in window) {
    (window as any).showSaveFilePicker({
      suggestedName: "ferrum-preset.json",
      types: [{ description: "Ferrum Preset", accept: { "application/json": [".json"] } }],
    }).then(async (handle: any) => {
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
    }).catch(() => { /* cancelled */ });
  } else {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "ferrum-preset.json";
    a.click();
    URL.revokeObjectURL(url);
  }
}

function loadPreset() {
  if ("showOpenFilePicker" in window) {
    (window as any).showOpenFilePicker({
      types: [{ description: "Ferrum Preset", accept: { "application/json": [".json"] } }],
    }).then(async ([handle]: any[]) => {
      const file = await handle.getFile();
      applyPresetFile(file);
    }).catch(() => { /* cancelled */ });
  } else {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.addEventListener("change", () => {
      if (input.files?.length) applyPresetFile(input.files[0]);
    });
    input.click();
  }
}

async function applyPresetFile(file: File) {
  try {
    const text = await file.text();
    const preset = JSON.parse(text) as Preset;

    if (!preset.settings) {
      showError("Invalid preset file");
      return;
    }

    // Apply settings to sliders
    const s = preset.settings;
    sliders.exposure.value = String(s.exposure ?? 0);
    sliders.contrast.value = String(s.contrast ?? 0);
    sliders.highlights.value = String(s.highlights ?? 0);
    sliders.shadows.value = String(s.shadows ?? 0);
    sliders.vibrance.value = String(s.vibrance ?? 0);
    sliders.saturation.value = String(s.saturation ?? 0);
    sliders.sharpening.value = String(s.sharpening ?? 0);
    sliders.vignette.value = String(s.vignette ?? 0);

    // Reverse kelvin from wb_r/wb_b (approximate: use 6500 if neutral)
    if (s.wb_r === 1 && s.wb_g === 1 && s.wb_b === 1) {
      sliders.wb.value = "6500";
    }
    // Otherwise keep current WB slider — preset stores computed multipliers

    updateDevelop();
    console.log(`Preset loaded: ${preset.name}`);
  } catch {
    showError("Failed to load preset");
  }
}

// ─── Keyboard Shortcuts ──────────────────────────────────────

function onKeyDown(e: KeyboardEvent) {
  if ((e.target as HTMLElement).tagName === "INPUT" || (e.target as HTMLElement).tagName === "SELECT") return;

  if (e.key === "0") { fitZoom(); }
  else if (e.key === "1") { setZoom(1); }
  else if (e.key === "2") { setZoom(2); }
  else if (e.key === "\\") { toggleBeforeAfter(); }
  else if ((e.key === "r" || e.key === "R") && !e.ctrlKey && !e.metaKey) { resetAll(); }
  else if ((e.ctrlKey || e.metaKey) && e.key === "s") { e.preventDefault(); if (img) showExportModal(); }
  else if ((e.ctrlKey || e.metaKey) && e.key === "o") { e.preventDefault(); openFile(); }
}

// ─── Color Temperature ───────────────────────────────────────

function kelvinToRGB(kelvin: number): { r: number; g: number; b: number } {
  const temp = kelvin / 100;
  let r: number, g: number, b: number;

  if (temp <= 66) { r = 1.0; } else { r = 1.292936 * Math.pow(temp - 60, -0.1332047592); }
  if (temp <= 66) { g = 0.39008 * Math.log(temp) - 0.63184; } else { g = 1.129891 * Math.pow(temp - 60, -0.0755148492); }
  if (temp >= 66) { b = 1.0; } else if (temp <= 19) { b = 0.0; } else { b = 0.54321 * Math.log(temp - 10) - 1.19625; }

  const ref6500 = kelvinToRGBRaw(6500);
  return {
    r: Math.max(0, r) / ref6500.r,
    g: Math.max(0, g) / ref6500.g,
    b: Math.max(0, b) / ref6500.b,
  };
}

function kelvinToRGBRaw(kelvin: number): { r: number; g: number; b: number } {
  const temp = kelvin / 100;
  const r = temp <= 66 ? 1.0 : 1.292936 * Math.pow(temp - 60, -0.1332047592);
  const g = temp <= 66 ? 0.39008 * Math.log(temp) - 0.63184 : 1.129891 * Math.pow(temp - 60, -0.0755148492);
  const b = temp >= 66 ? 1.0 : temp <= 19 ? 0.0 : 0.54321 * Math.log(temp - 10) - 1.19625;
  return { r: Math.max(0, r), g: Math.max(0, g), b: Math.max(0, b) };
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
