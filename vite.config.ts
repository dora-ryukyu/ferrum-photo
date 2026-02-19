import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [
    wasm(),
    topLevelAwait(),
    tailwindcss(),
  ],

  build: {
    target: "esnext",
  },

  // Ensure .wasm files are served with correct MIME type
  assetsInclude: ["**/*.wgsl"],

  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
