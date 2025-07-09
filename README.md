# CytoVerse <img src="web/public/cytoverse.svg" width="32" height="32" alt="Description">

Search, display and label cells in foundation model embedding space

[https://cells-test.gi.ucsc.edu/cytoverse](https://cells-test.gi.ucsc.edu/cytoverse)

![Alt text](screenshot.png?raw=true 'CytoVerse Screenshot')

## Overview

CytoVerse is a browser-based platform for single-cell RNA-seq analysis, designed for cell annotation using foundation model embeddings. It operates entirely in the browser, streaming h5ad files from local storage without uploading data or requiring server computation. Key features include:

- **Embedding and Visualization**: Uses SCimilarity for cell embeddings and parametric UMAP for 2D visualization, leveraging large training datasets for accurate cell annotation.
- **Efficient Search**: Employs Inverted File with Product Quantization (IVFPQ) for fast approximate nearest neighbor searches across over 20 million samples.
- **Performance**: Built on WebAssembly and ONNX, enabling high-speed processing directly in the browser.
- **Scalability**: Supports unlimited streaming analysis of h5ad files via h5wasm, ensuring flexibility for large datasets.
- **Collaboration**: Facilitates distributed discovery by allowing researchers to explore shared embedding spaces, identifying overlapping or complementary assays, particularly for perturbseq-driven research.

This architecture ensures privacy, scalability, and collaborative potential without server dependency.

## Install

Download and unpack the scimilarity [model and dataset](https://zenodo.org/records/10685499) (~30GB) into data/scimilarity/model_v1.1.

Install python dependencies and create a virtual env:

```
uv venv
source .venv/bin/activate
uv sync
```

Export SCimilarity embeddings and labels, train a parametric umap model on a stratified subset of cells, train IVFPQ, populate partitions and export models to ONNX:

```
make scimilarity pumap ivfpq
```

Verify that web/public/models/scimilarity is populated (~700MB Total):

```
web/public/models/scimilarity
├── embedding
│   ├── embedding.onnx
│   ├── genes.txt
│   ├── model.onnx
│   └── preprocessing.onnx
├── ivf
│   ├── centroids.arrow
│   ├── metadata.json
│   └── model.pkl
├── ivfpq
│   ├── centroids.arrow
│   ├── metadata.json
│   └── partitions
│       ├── partition_0000.arrow
│       ├── partition_0001.arrow
│       ├── ....
│       ├── partition_0062.arrow
│       └── partition_0063.arrow
├── pq
│   ├── codebooks.bin
│   ├── metadata.json
│   ├── model.onnx
│   └── model.pkl
└── pumap
    ├── author_label.arrow
    ├── metadata.json
    ├── model.onnx
    ├── prediction.arrow
    ├── study.arrow
    ├── tissue.arrow
    ├── x.arrow
    └── y.arrow
```

Install web app dependencies and run local dev server:

```
cd web
npm install
npm run dev
```

## References

SCimilarity [paper](https://doi.org/10.1038/s41586-024-08411-y), [repo](https://github.com/Genentech/scimilarity), [model and dataset](https://zenodo.org/records/10685499)

SIMS Web [repo](https://github.com/braingeneers/sims-web) and [demo](https://cells-test.gi.ucsc.edu/sims/)

[Product Quantization for Similarity Search](https://towardsdatascience.com/product-quantization-for-similarity-search-2f1f67c5fddd/)

[IVFPQ + HNSW for Billion-scale Similarity Search](https://towardsdatascience.com/ivfpq-hnsw-for-billion-scale-similarity-search-89ff2f89d90e/)

[Wikipedia search-by-vibes through millions of pages offline](https://www.leebutterman.com/2023/06/01/offline-realtime-embedding-search.html) and associated [Product Quantization (PQ) in Javascript](https://github.com/lsb)

[TorchPQ](https://github.com/DeMoriarty/TorchPQ)
