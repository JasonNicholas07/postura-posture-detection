import { defineConfig } from "vite";

export default defineConfig({
  build: {
    lib: {
      entry: "src/index.ts",
      name: "PosturaPose",
      fileName: () => `index.js`,
      formats: ["es"],
    },
    outDir: "build",
    emptyOutDir: true,
    minify: true,
  },
});