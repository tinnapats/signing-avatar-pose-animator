import { cp, mkdir } from 'node:fs/promises';
import { fileURLToPath, URL } from 'node:url';
import { defineConfig } from 'vite';

const projectRoot = fileURLToPath(new URL('../pose-animator/', import.meta.url));
const outputDir = fileURLToPath(new URL('../pose-animator-dist/', import.meta.url));

export default defineConfig({
  root: projectRoot,
  base: './',
  publicDir: false,
  resolve: { alias: { paper: fileURLToPath(new URL('../pose-animator/paper-global.js', import.meta.url)) } },
  build: {
    outDir: outputDir,
    emptyOutDir: true,
    rollupOptions: { input: fileURLToPath(new URL('../pose-animator/dataset_player.html', import.meta.url)) },
  },
  plugins: [{
    name: 'copy-signing-avatar-runtime-assets',
    async closeBundle() {
      await mkdir(outputDir, { recursive: true });
      await cp(
        fileURLToPath(new URL('../pose-animator/resources/illustration/', import.meta.url)),
        fileURLToPath(new URL('../pose-animator-dist/resources/illustration/', import.meta.url)),
        { recursive: true },
      );      await cp(
        fileURLToPath(new URL('../pose-animator/vendor/', import.meta.url)),
        fileURLToPath(new URL('../pose-animator-dist/vendor/', import.meta.url)),
        { recursive: true },
      );
    },
  }],
});