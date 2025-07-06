# CytoVerse <img src="web/public/cytoverse.svg" width="32" height="32" alt="Description">

Search, display and label cells in foundation model embedding space

NOTE: Under heavy development and refactoring, not ready for reproducible installs...

[https://cells-test.gi.ucsc.edu/cytoverse](https://cells-test.gi.ucsc.edu/cytoverse)

![Alt text](screenshot.png?raw=true 'CytoVerse Screenshot')

## Overview

CytoVerse presents a fully browser-based single-cell RNA-seq analysis platform that enables real-time cell type annotation through foundation model embeddings. The system leverages SCimilarity for client-side cell embedding, parametric UMAP for 2D visualization against comprehensive training datasets, and Inverted File with Product Quantization (IVFPQ) for approximate nearest neighbor search to generate cell type labels. Built on WebAssembly and ONNX for performance, the platform scales to millions of training samples while supporting unlimited streaming analysis of h5ad files via h5wasm. This architecture enables distributed collaborative discovery, allowing consortium researchers to identify overlapping or complementary assays through shared embedding space exploration—a critical capability for the emerging perturbseq-driven discovery paradigm.

## Status

- Training dataset of 1M points plotted using WebGL.
- Run streams the selected h5ad, embeds client side using scimiliarity, maps using a parametric umap trained on the training dataset and plots as larger red dots.
- CPU embedding and mapping functions - GPU in progress

Next:

- Explore quantizing the encodings using PQ and sharding via IVF towards including the entire scimilarity training dataset (23.4M cells)

## Install

Download and unpack the scimilarity model and dataset (~30GB) into data/scimilarity/model_v1.1.

Install python dependencies and create a virtual env:

```
uv venv
source .venv/bin/activate
uv sync
```

Select a stratified subset of 1M cells from the scimilarity tiledb, export their embeddings, train a parametric umap model, export it as onnx and save the mappings and labels in compressed arrow files for the web app to render:

```
make scimilarity-export umap-train umap-map umap-export
```

Verify that web/public/models/scimilarity is populated

```
❯ ls web/public/models/scimilarity
embedder.genes   metadata.json    tissue.arrow
embedder.onnx    prediction.arrow x.arrow
umap.onnx      study.arrow      y.arrow
```

Install web app dependencies and run local dev server:

```
cd web
npm install
npm run dev
```

## Terminology

Vector IDs: Generic id for a vector which in our case is a sample id used to get back to the original sample and/or related annotations/labels.

Partition: Subset of vectors or samples. There could be many 'clusters' in a partition. Partition is used in the IVF implementations. A sample can only be in a single partition (but it 'could' be in multiple clusters or neighborhoods)

Clusters: Group of vectors/samples that somehow go together. Might be from multiple partitions. Similar to a neighborhood in fuziness.

## References

SCimilarity [paper](https://doi.org/10.1038/s41586-024-08411-y), [repo](https://github.com/Genentech/scimilarity), [model and dataset](https://zenodo.org/records/10685499)

SIMS Web [repo](https://github.com/braingeneers/sims-web) and [demo](https://cells-test.gi.ucsc.edu/sims/)

[Product Quantization for Similarity Search](https://towardsdatascience.com/product-quantization-for-similarity-search-2f1f67c5fddd/)

[IVFPQ + HNSW for Billion-scale Similarity Search](https://towardsdatascience.com/ivfpq-hnsw-for-billion-scale-similarity-search-89ff2f89d90e/)

[Wikipedia search-by-vibes through millions of pages offline](https://www.leebutterman.com/2023/06/01/offline-realtime-embedding-search.html) and associated [Product Quantization (PQ) in Javascript](https://github.com/lsb)

[TorchPQ](https://github.com/DeMoriarty/TorchPQ)
