# CytoVerse <img src="public/cytoverse.svg" width="32" height="32" alt="Cytoverse">

Label, share and display cells using foundation models client side within the browser.

[![DOI](https://img.shields.io/badge/DOI-10.1101/2025.01.15.123456-blue)](https://doi.org/10.64898/2026.01.29.702554)

[https://cells-test.gi.ucsc.edu/cytoverse](https://cells-test.gi.ucsc.edu/cytoverse)

![Alt text](screenshot.png?raw=true 'CytoVerse Screenshot')

## Overview

CytoVerse is a browser-based platform for single-cell RNA-seq analysis, designed for cell annotation using foundation model embeddings. It operates entirely in the browser, streaming h5ad files from local storage without uploading data or requiring server computation. Key features include:

- **Embedding and Visualization**: Uses SCimilarity for cell embeddings and parametric UMAP for 2D visualization, leveraging large training datasets for accurate cell annotation.
- **Efficient Search**: Employs Inverted File with Product Quantization (IVFPQ) for fast approximate nearest neighbor searches across over 20 million samples.
- **Performance**: Built on WebAssembly and ONNX, enabling high-speed processing directly in the browser.
- **Scalability**: Supports unlimited streaming analysis of h5ad files via h5wasm, ensuring flexibility for large datasets.
- **Customization**: Save your labeled cells as a user reference and share with other researchers.
- **Collaboration**: Facilitates distributed discovery by allowing researchers to explore shared embedding spaces, identifying overlapping or complementary assays, particularly for perturbseq-driven research.

This architecture ensures privacy, scalability, and collaborative potential without server dependency.

## Data and Compute Flow

![Alt text](notebooks/figures/architecture_horizontal_full.svg?raw=true 'CytoVerse Data Flow')

## Develop

Install python dependencies and create a virtual env:

```
uv venv
source .venv/bin/activate
uv sync

npm install
npx playwright install
```

Create a ./data/ folder and download and unpack the scimilarity [model and dataset](https://zenodo.org/records/10685499) (~30GB) into data/models/scimilarity/model_v1.1.

Then to create a small reference for development and end to end testing purposes:

```
make scimilarity-subset
```

Run the web app:

```
npm run dev
```

And then open http://localhost:5173/ in a browser.

Generate a distribution including all models under public/models in dist/:

```
npm run build
```

## Full SCimilarity Reference

To export all 23.3 million SCimilarity embeddings and labels, train a parametric
umap model on a stratified subset of cells, train IVFPQ, populate partitions and
export models to ONNX:

```
make scimilarity
```

NOTE: This will take several hours on a laptop

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

Run unit and e2e including browser tests (assumes full SCimilarity reference build)

```
make test
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

## Citing CytoVerse

If you use **CytoVerse** for analysis or exploration, consider citing our publication -

```bibtex
@article {Currie2026.01.29.702554,
    author = {Currie, Robert and Gonzalez Ferrer, Jesus and Mostajo-Radji, Mohammed A and Haussler, David},
    title = {CytoVerse: Single-Cell AI Foundation Models in the Browser},
    elocation-id = {2026.01.29.702554},
    year = {2026},
    doi = {10.64898/2026.01.29.702554},
    publisher = {Cold Spring Harbor Laboratory},
    url = {https://www.biorxiv.org/content/early/2026/01/30/2026.01.29.702554},
    eprint = {https://www.biorxiv.org/content/early/2026/01/30/2026.01.29.702554.full.pdf},
    journal = {bioRxiv}
}
```

## References

SCimilarity [paper](https://doi.org/10.1038/s41586-024-08411-y), [repo](https://github.com/Genentech/scimilarity), [model and dataset](https://zenodo.org/records/10685499)

SIMS Web [repo](https://github.com/braingeneers/sims-web) and [demo](https://cells-test.gi.ucsc.edu/sims/)

[Vector similarity search methods](https://weaxsey.org/en/articles/2023-10-30/)

[Product Quantization for Similarity Search](https://towardsdatascience.com/product-quantization-for-similarity-search-2f1f67c5fddd/)

[IVFPQ + HNSW for Billion-scale Similarity Search](https://towardsdatascience.com/ivfpq-hnsw-for-billion-scale-similarity-search-89ff2f89d90e/)

[Wikipedia search-by-vibes through millions of pages offline](https://www.leebutterman.com/2023/06/01/offline-realtime-embedding-search.html) and associated [Product Quantization (PQ) in Javascript](https://github.com/lsb)

[TorchPQ](https://github.com/DeMoriarty/TorchPQ)
