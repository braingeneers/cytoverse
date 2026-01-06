#!/usr/bin/env python3

import logging
import pickle
import typer
from pathlib import Path
import numpy as np
from typing import Dict

import anndata
import scanpy as sc
import torch
import onnx
import onnxruntime as ort
from transformers import BertForMaskedLM

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Export Geneformer model to ONNX format",
    add_completion=False,
)


class GeneformerPreprocessingWrapper(torch.nn.Module):
    """
    Standalone preprocessing: median norm + rank encoding + tokenization.

    Implements Geneformer preprocessing pipeline:
    1. Divide by gene medians (corpus normalization)
    2. Rank genes using TopK operation
    3. Map gene indices to token IDs via embedding layer
    4. Prepend CLS token if V2 model

    Args:
        gene_medians: Corpus median expression values [n_genes]
        ensembl_to_token: Mapping from gene index to token ID
        model_input_size: 2048 (V1) or 4096 (V2)
        use_cls_token: False (V1) or True (V2)
    """

    def __init__(
        self,
        gene_medians: torch.Tensor,
        ensembl_to_token: torch.Tensor,
        model_input_size: int,
        use_cls_token: bool,
        cls_token_id: int = 0,
    ):
        super().__init__()
        self.register_buffer("gene_medians", gene_medians)
        self.register_buffer("ensembl_to_token", ensembl_to_token)
        self.model_input_size = model_input_size
        self.use_cls_token = use_cls_token
        self.cls_token_id = cls_token_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with preprocessing.

        Args:
            x: Raw counts [batch_size, n_genes]

        Returns:
            Token IDs [batch_size, model_input_size]
        """
        x = x.to(torch.float32)
        x = torch.nan_to_num(x, nan=0.0)

        # Step 1: Median normalization
        # Replace zero medians with 1 to avoid division by zero
        safe_medians = torch.where(
            self.gene_medians == 0,
            torch.tensor(1.0, dtype=torch.float32),
            self.gene_medians,
        )
        x_norm = x / safe_medians

        # Step 2: Rank genes using TopK
        # TopK returns (values, indices) in descending order
        k = (
            self.model_input_size
            if not self.use_cls_token
            else self.model_input_size - 1
        )
        _, top_indices = torch.topk(x_norm, k=k, dim=1, largest=True, sorted=True)

        # Step 3: Map gene indices to token IDs
        # Use gather to select token IDs based on ranked gene indices
        batch_size = x.shape[0]
        tokens = torch.gather(
            self.ensembl_to_token.unsqueeze(0).expand(batch_size, -1),
            1,
            top_indices,
        )

        # Step 4: Prepend CLS token if V2
        if self.use_cls_token:
            cls_tokens = torch.full(
                (batch_size, 1), self.cls_token_id, dtype=tokens.dtype
            )
            tokens = torch.cat([cls_tokens, tokens], dim=1)

        return tokens.to(torch.int64)


class GeneformerEmbeddingWrapper(torch.nn.Module):
    """
    Extract embeddings from BERT hidden states.

    Args:
        bert_model: BertForMaskedLM model
        layer_to_extract: -1 (2nd to last) or 0 (last)
        use_cls_token: False (V1) or True (V2)
    """

    def __init__(
        self,
        bert_model: BertForMaskedLM,
        layer_to_extract: int,
        use_cls_token: bool,
    ):
        super().__init__()
        self.bert = bert_model.bert
        self.layer_to_extract = layer_to_extract
        self.use_cls_token = use_cls_token

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract embeddings.

        Args:
            token_ids: Token IDs [batch_size, seq_len]

        Returns:
            Cell embeddings [batch_size, hidden_size]
        """
        # Create attention mask (all ones since we have no padding in preprocessing)
        attention_mask = torch.ones_like(token_ids)

        # Run through BERT
        outputs = self.bert(
            input_ids=token_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Extract hidden states from specified layer
        hidden_states = outputs.hidden_states[self.layer_to_extract]

        # Extract embeddings
        if self.use_cls_token:
            # V2: Use CLS token (position 0)
            embeddings = hidden_states[:, 0, :]
        else:
            # V1: Mean pooling over all tokens
            embeddings = torch.mean(hidden_states, dim=1)

        return embeddings


class GeneformerCombinedWrapper(torch.nn.Module):
    """Combined preprocessing + embedding."""

    def __init__(
        self,
        preprocessing: GeneformerPreprocessingWrapper,
        embedding: GeneformerEmbeddingWrapper,
    ):
        super().__init__()
        self.preprocessing = preprocessing
        self.embedding = embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with preprocessing followed by embedding.

        Args:
            x: Raw counts [batch_size, n_genes]

        Returns:
            Cell embeddings [batch_size, hidden_size]
        """
        token_ids = self.preprocessing(x)
        embeddings = self.embedding(token_ids)
        return embeddings


def load_gene_medians(geneformer_path: Path, model_version: str) -> Dict[str, float]:
    """
    Load gene median dictionary from Geneformer package.

    Args:
        geneformer_path: Path to geneformer package directory
        model_version: "v1" or "v2"

    Returns:
        Dictionary mapping Ensembl IDs to median expression values
    """
    if model_version == "v1":
        median_file = (
            geneformer_path
            / "geneformer"
            / "gene_dictionaries_30m"
            / "gene_median_dictionary_gc30M.pkl"
        )
    else:
        median_file = (
            geneformer_path / "geneformer" / "gene_median_dictionary_gc104M.pkl"
        )

    logger.info("Loading gene medians from %s", median_file)
    with open(median_file, "rb") as f:
        gene_medians = pickle.load(f)

    logger.info("Loaded %d gene medians", len(gene_medians))
    return gene_medians


def load_token_dictionary(geneformer_path: Path, model_version: str) -> Dict[str, int]:
    """
    Load token dictionary from Geneformer package.

    Args:
        geneformer_path: Path to geneformer package directory
        model_version: "v1" or "v2"

    Returns:
        Dictionary mapping Ensembl IDs to token IDs
    """
    if model_version == "v1":
        token_file = (
            geneformer_path
            / "geneformer"
            / "gene_dictionaries_30m"
            / "token_dictionary_gc30M.pkl"
        )
    else:
        token_file = geneformer_path / "geneformer" / "token_dictionary_gc104M.pkl"

    logger.info("Loading token dictionary from %s", token_file)

    with open(token_file, "rb") as f:
        token_dict = pickle.load(f)

    logger.info("Loaded %d tokens", len(token_dict))
    return token_dict


@app.command()
def model(
    model_path: Path = typer.Argument(
        exists=True,
        help="Path to Geneformer model directory (e.g., ~/geneformer/Geneformer-V1-10M/)",
    ),
    output_path: Path = typer.Argument(
        help="Path to save ONNX models (e.g., public/models/geneformer-v1-10m/embedding/)",
    ),
    geneformer_pkg_path: Path = typer.Option(
        "data/models/geneformer",
        help="Path to geneformer package directory",
    ),
    model_version: str = typer.Option(
        "v1",
        help="Model version: 'v1' or 'v2'",
    ),
    validation_h5ad: Path = typer.Option(
        "fixtures/GSE136831_subsample_100.h5ad",
        help="Path to h5ad file for validation",
    ),
) -> None:
    """
    Export Geneformer to ONNX format.

    Exports 3 ONNX models:
    1. preprocessing.onnx - Median normalization + rank encoding + tokenization
    2. embedding.onnx - BERT embedding extraction
    3. model.onnx - Combined preprocessing + embedding

    Also exports:
    - genes.txt - Simple list of gene symbols (one per line, for inflation mapping)
    - genes.json - Full metadata (JSON per line: ensembl_id, symbol, token)
    - config.json - Model configuration metadata
    """
    logger.info("Loading Geneformer model from %s", model_path)
    bert_model = BertForMaskedLM.from_pretrained(model_path)
    bert_model.eval()

    # Get model configuration
    use_cls_token = model_version == "v2"
    model_input_size = 4096 if model_version == "v2" else 2048
    embedding_layer = -1
    embedding_dim = bert_model.config.hidden_size

    logger.info("Model configuration:")
    logger.info("  Version: %s", model_version)
    logger.info("  Input size: %d", model_input_size)
    logger.info("  Use CLS token: %s", use_cls_token)
    logger.info("  Embedding dimension: %d", embedding_dim)

    # Load gene medians and token dictionary
    gene_medians = load_gene_medians(geneformer_pkg_path, model_version)
    token_dict = load_token_dictionary(geneformer_pkg_path, model_version)

    # Create gene ordering based on token dictionary
    # Sort by token ID to get consistent gene ordering
    sorted_genes = sorted(
        [(eid, tid) for eid, tid in token_dict.items() if eid.startswith("ENSG")],
        key=lambda x: x[1],
    )
    ensembl_ids = [eid for eid, _ in sorted_genes]

    # Create tensor for gene medians (aligned with ensembl_ids)
    gene_medians_tensor = torch.tensor(
        [gene_medians.get(eid, 1.0) for eid in ensembl_ids], dtype=torch.float32
    )

    # Create tensor for ensembl_to_token mapping (index → token ID)
    ensembl_to_token_tensor = torch.tensor(
        [token_dict[eid] for eid in ensembl_ids], dtype=torch.int64
    )

    cls_token_id = token_dict.get("<cls>", 0)
    pad_token_id = token_dict.get("<pad>", 0)

    logger.info("Gene vocabulary:")
    logger.info("  Total genes: %d", len(ensembl_ids))
    logger.info("  CLS token ID: %d", cls_token_id)
    logger.info("  PAD token ID: %d", pad_token_id)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Export preprocessing module
    logger.info("\n1. Exporting preprocessing module...")
    preprocessing_model = GeneformerPreprocessingWrapper(
        gene_medians=gene_medians_tensor,
        ensembl_to_token=ensembl_to_token_tensor,
        model_input_size=model_input_size,
        use_cls_token=use_cls_token,
        cls_token_id=cls_token_id,
    )
    preprocessing_path = output_path / "preprocessing.onnx"

    torch.onnx.export(
        preprocessing_model,
        torch.zeros(1, len(ensembl_ids)),
        preprocessing_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["token_ids"],
        dynamic_axes={"input": {0: "batch_size"}, "token_ids": {0: "batch_size"}},
        verbose=False,
        dynamo=False,  # Use legacy TorchScript exporter (PyTorch 2.9+)
    )
    logger.info("  Preprocessing ONNX saved to %s", preprocessing_path)

    # Step 2: Export embedding module
    logger.info("\n2. Exporting embedding module...")
    embedding_model = GeneformerEmbeddingWrapper(
        bert_model=bert_model,
        layer_to_extract=embedding_layer,
        use_cls_token=use_cls_token,
    )
    embedding_path = output_path / "embedding.onnx"

    torch.onnx.export(
        embedding_model,
        torch.zeros(1, model_input_size, dtype=torch.int64),
        embedding_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["token_ids"],
        output_names=["embedding"],
        dynamic_axes={"token_ids": {0: "batch_size"}, "embedding": {0: "batch_size"}},
        verbose=False,
        dynamo=False,  # Use legacy TorchScript exporter (PyTorch 2.9+)
    )
    logger.info("  Embedding ONNX saved to %s", embedding_path)

    # Step 3: Export combined model
    logger.info("\n3. Exporting combined model...")
    combined_model = GeneformerCombinedWrapper(preprocessing_model, embedding_model)
    combined_model.eval()
    combined_path = output_path / "model.onnx"

    # Use opset 17 for maximum compatibility with onnxruntime-web
    torch.onnx.export(
        combined_model,
        torch.zeros(1, len(ensembl_ids)),
        combined_path,
        export_params=True,
        opset_version=17,  # Use opset 17 for compatibility with onnxruntime-web 1.23
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        verbose=False,
        dynamo=False,  # Use legacy TorchScript exporter (PyTorch 2.9+)
    )
    logger.info("  Combined ONNX saved to %s", combined_path)

    # Validate all models
    logger.info("\n4. Validating ONNX models...")
    for model_file, name in [
        (preprocessing_path, "Preprocessing"),
        (embedding_path, "Embedding"),
        (combined_path, "Combined"),
    ]:
        onnx_model = onnx.load(str(model_file))
        onnx.checker.check_model(onnx_model)
        logger.info("  ✅ %s ONNX model validation passed", name)

    # Query biomart for Ensembl ID to gene symbol mapping
    logger.info("\n5. Querying biomart for gene symbols...")
    import requests
    import json

    ensembl_to_symbol: dict[str, str] = {}
    batch_size = 500

    for i in range(0, len(ensembl_ids), batch_size):
        batch = ensembl_ids[i : i + batch_size]
        query = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query  virtualSchemaName = "default" formatter = "TSV" header = "0" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" >
    <Dataset name = "hsapiens_gene_ensembl" interface = "default" >
        <Filter name = "ensembl_gene_id" value = "{','.join(batch)}"/>
        <Attribute name = "ensembl_gene_id" />
        <Attribute name = "external_gene_name" />
    </Dataset>
</Query>"""

        try:
            response = requests.post(
                "http://www.ensembl.org/biomart/martservice",
                data={"query": query},
                timeout=60,
            )

            if response.status_code == 200:
                for line in response.text.strip().split("\n"):
                    if line:
                        parts = line.split("\t")
                        if len(parts) == 2:
                            ensembl_id, symbol = parts
                            if symbol:
                                ensembl_to_symbol[ensembl_id] = symbol

            logger.info(
                "  Processed %d/%d Ensembl IDs",
                min(i + batch_size, len(ensembl_ids)),
                len(ensembl_ids),
            )
        except Exception as e:
            logger.warning("  Failed to query biomart batch: %s", e)

    logger.info("  Mapped %d/%d Ensembl IDs to gene symbols", len(ensembl_to_symbol), len(ensembl_ids))

    # Export genes.txt (simple list of gene symbols, one per line)
    # Use gene symbol if available, otherwise fall back to Ensembl ID
    logger.info("\n6. Exporting genes.txt...")
    genes_txt_path = output_path / "genes.txt"

    gene_symbols = [ensembl_to_symbol.get(eid, eid) for eid in ensembl_ids]
    with open(genes_txt_path, "w") as f:
        f.write("\n".join(gene_symbols))

    logger.info("  Genes list saved to %s", genes_txt_path)

    # Export genes.json with full metadata
    logger.info("\n7. Exporting genes.json...")
    genes_json_path = output_path / "genes.json"

    # Write genes.json (one JSON object per line)
    with open(genes_json_path, "w") as f:
        for eid in ensembl_ids:
            gene_info = {
                "ensembl_id": eid,
                "symbol": ensembl_to_symbol.get(eid, ""),
                "token": int(token_dict[eid]),
            }
            f.write(json.dumps(gene_info) + "\n")

    logger.info("  Genes metadata saved to %s", genes_json_path)

    # Export config.json
    logger.info("\n8. Exporting config.json...")
    config_path = output_path / "config.json"

    config = {
        "model_type": "geneformer",
        "version": model_version,
        "model_input_size": model_input_size,
        "use_cls_token": use_cls_token,
        "embedding_layer": embedding_layer,
        "embedding_dim": embedding_dim,
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info("  Config saved to %s", config_path)

    logger.info("\n🎉 Model export complete!")
    logger.info("  - Preprocessing: %s", preprocessing_path)
    logger.info("  - Embedding: %s", embedding_path)
    logger.info("  - Combined: %s", combined_path)
    logger.info("  - Genes (list): %s", genes_txt_path)
    logger.info("  - Genes (metadata): %s", genes_json_path)
    logger.info("  - Config: %s", config_path)


if __name__ == "__main__":
    app()
