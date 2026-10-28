import { defineConfig } from "vite";

export default defineConfig({
  build: {
    lib: {
      entry: "src/index.ts",
      name: "PosturaPose",
      fileName: () => `index.js`,
      formats: ["es"],
    },
    rollupOptions: {
      // Load ort from CDN; its wasm resolves next to it. Keep version in sync with package.json.
      external: ["onnxruntime-web/wasm"],
      output: {
        paths: {
          "onnxruntime-web/wasm":
            "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.wasm.min.mjs",
        },
      },
    },
    outDir: "build",
    emptyOutDir: true,
    minify: true,
  },
});