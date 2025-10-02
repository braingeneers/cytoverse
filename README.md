# CytoVerse <img src="public/cytoverse.svg" width="32" height="32" alt="Cytoverse">

Search, display and label cells in foundation model embedding space

[https://cells-test.gi.ucsc.edu/cytoverse](https://cells-test.gi.ucsc.edu/cytoverse)

![Alt text](screenshot.png?raw=true 'CytoVerse Screenshot')

## Overview

CytoVerse is a browser-based platform for single-cell RNA-seq analysis, designed for cell annotation using foundation model embeddings. It operates entirely in the browser, streaming h5ad files from local storage without uploading data or requiring server computation. Key features include:

- **Embedding and Visualization**: Uses SCimilarity for cell embeddings and parametric UMAP for 2D visualization, leveraging large training datasets for accurate cell annotation.
- **Efficient Search**: Employs Inverted File with Product Quantization (IVFPQ) for fast approximate nearest neighbor searches across over 20 million samples.
- **Performance**: Built on WebAssembly and ONNX, enabling high-speed processing directly in the browser.
- **Scalability**: Supports unlimited streaming analysis of h5ad files via h5wasm, ensuring flexibility for large datasets.
- **Customization**: Save your labeled cells as a user reference.
- **Collaboration**: Facilitates distributed discovery by allowing researchers to explore shared embedding spaces, identifying overlapping or complementary assays, particularly for perturbseq-driven research.

This architecture ensures privacy, scalability, and collaborative potential without server dependency.

## Data and Compute Flow

![Alt text](notebooks/figures/architecture.svg?raw=true 'CytoVerse Data Flow')

## Install

Create a ./data/ folder and download and unpack the scimilarity [model and dataset](https://zenodo.org/records/10685499) (~30GB) into data/models/scimilarity/model_v1.1 and the tutorial [h5ad file](https://zenodo.org/records/13685881) into ./data/GSE136831_subsample.h5ad

Install python dependencies and create a virtual env:

```
uv venv
source .venv/bin/activate
uv sync

npm install
npx playwright install
```

Export SCimilarity embeddings and labels, train a parametric umap model on a stratified subset of cells, train IVFPQ, populate partitions and export models to ONNX:

```
make scimilarity
```

Verify that public/models/scimilarity is populated (~1.2G Total):

```
> tree public/models/scimilarity
public/models/scimilarity
├── embedding
│   ├── embedding.onnx
│   ├── genes.txt
│   ├── model.onnx
│   └── preprocessing.onnx
├── ivfpq
│   ├── ivf_centroids.bin
│   ├── ivf_coarse.onnx
│   ├── ivf_forward_dynamic.onnx
│   ├── ivf_forward.onnx
│   ├── ivf_metadata.json
│   ├── partitions
│   │   ├── partition_0000.bin
│   │   ├── partition_0001.bin
│   │   ....
│   │   ├── partition_4833.bin
│   │   └── partition_4834.bin
│   ├── pq_codebooks.bin
│   ├── pq_distance.onnx
│   ├── pq_encode.onnx
│   └── pq_metadata.json
└── pumap
    ├── metadata.json
    ├── model.onnx
    ├── prediction.bin
    ├── study.bin
    ├── tissue.bin
    ├── x.bin
    └── y.bin
```

Run tests:

```
make test
```

Install web app dependencies and run local dev server:

```
npm run dev
```

Open http://localhost:5173/ in a browser

Generate a distribution including all models under public/models in dist/:

```
npm run build
```

## Creating References

Given an h5ad file you can generate a reference from it via:

```
export model_id="new_model_id"
python scripts/h5ad_to_embeddings.py \
    <path to your h5ad file> \
	data/models/scimilarity/model_v1.1 \
	data/references/$model_id
	--labels <first label in obs> \
	--labels <second label in obs>
    ...

make ivfpq-train pumap
```

## Concordance and Performance

See [the analysis notebook and figures](notebooks/analysis.ipynb) for a detailed comparison with running on a server with the underlying foundation model.

## References

SCimilarity [paper](https://doi.org/10.1038/s41586-024-08411-y), [repo](https://github.com/Genentech/scimilarity), [model and dataset](https://zenodo.org/records/10685499)

SIMS Web [repo](https://github.com/braingeneers/sims-web) and [demo](https://cells-test.gi.ucsc.edu/sims/)

[Vector similarity search methods](https://weaxsey.org/en/articles/2023-10-30/)

[Product Quantization for Similarity Search](https://towardsdatascience.com/product-quantization-for-similarity-search-2f1f67c5fddd/)

[IVFPQ + HNSW for Billion-scale Similarity Search](https://towardsdatascience.com/ivfpq-hnsw-for-billion-scale-similarity-search-89ff2f89d90e/)

[Wikipedia search-by-vibes through millions of pages offline](https://www.leebutterman.com/2023/06/01/offline-realtime-embedding-search.html) and associated [Product Quantization (PQ) in Javascript](https://github.com/lsb)

[TorchPQ](https://github.com/DeMoriarty/TorchPQ)
