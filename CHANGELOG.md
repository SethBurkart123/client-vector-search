# client-vector-search

## 1.3.0

### Minor Changes

- WebGPU support: switched underlying transformers dependency from `@xenova/transformers` v2 to `@huggingface/transformers` v3, added optional `device` parameter (`'webgpu' | 'wasm' | 'cpu' | 'auto'`) to `initializeModel` and `getEmbedding`, with auto-detect that prefers WebGPU when an adapter is available and falls back to the runtime default otherwise. Roughly 20x faster on long inputs (~150 tokens) on Apple Silicon vs WASM.

## 0.2.0

### Minor Changes

- support for experimental hnsw that runs on node and browser with json and binary serialization opitons

### Patch Changes

- f09bc2f: updates the docs and dynamic import for @xenova/transformers
- 46e07d6: creates a proper embedding index
- 13bddbb: adds in-memory index creation and brute force knn search
