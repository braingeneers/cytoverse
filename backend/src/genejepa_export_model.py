#!/usr/bin/env python3

import logging
import json
import typer
from pathlib import Path
from typing import Dict

import torch
import onnx

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Export GeneJEPA model to ONNX format",
    add_completion=False,
)


class GeneJEPAPreprocessingWrapper(torch.nn.Module):
    """
    GeneJEPA preprocessing: log1p normalization + standardization.

    Implements GeneJEPA preprocessing pipeline:
    1. Apply log1p transform to raw counts
    2. Standardize using global mean/std from training corpus
    3. Return ragged tensor representation (indices, values, offsets)

    Args:
        global_mean: Mean of log1p(counts) from training corpus
        global_std: Std of log1p(counts) from training corpus
    """

    def __init__(
        self,
        global_mean: float,
        global_std: float,
    ):
        super().__init__()
        self.register_buffer(
            "global_mean", torch.tensor(global_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "global_std", torch.tensor(global_std, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with preprocessing.

        Args:
            x: Raw counts [batch_size, n_genes]

        Returns:
            Normalized values [batch_size, n_genes]
        """
        x = x.to(torch.float32)
        x = torch.nan_to_num(x, nan=0.0)

        # Step 1: Log1p transform
        x_log = torch.log1p(x)

        # Step 2: Standardize using global statistics
        x_norm = (x_log - self.global_mean) / (self.global_std + 1e-6)

        return x_norm


class GeneJEPATokenizerWrapper(torch.nn.Module):
    """
    GeneJEPA tokenizer: gene identity + Fourier-encoded expression values.

    Creates token embeddings by combining:
    1. Gene identity embedding (learned lookup table)
    2. Fourier-encoded expression values (sin/cos features)

    Args:
        tokenizer: scRNATokenizer module from trained model
    """

    def __init__(self, tokenizer: torch.nn.Module):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to create token embeddings.

        Args:
            indices: Gene token IDs [total_genes]
            values: Normalized expression values [total_genes]

        Returns:
            Token embeddings [total_genes, d]
        """
        return self.tokenizer(indices, values)


class GeneJEPAEncoderWrapper(torch.nn.Module):
    """
    GeneJEPA encoder: Perceiver-style cross-attention to fixed latents.

    Simplified wrapper that just forwards to the encoder for ONNX export.
    The encoder's pad_sequence operation will be handled by PyTorch's ONNX exporter
    through symbolic registration or we accept warnings for now.

    Args:
        encoder: GenePerceiverEncoder module from trained model (teacher or student)
    """

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(
        self, indices: torch.Tensor, values: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass to extract embeddings.

        Args:
            indices: Gene token IDs, concatenated [total_genes]
            values: Normalized expression values, concatenated [total_genes]
            offsets: Prefix sums marking cell boundaries [batch_size + 1]

        Returns:
            Cell embeddings [batch_size, hidden_size]
        """
        return self.encoder(indices, values, offsets)


class GeneJEPACombinedWrapper(torch.nn.Module):
    """Combined preprocessing + tokenization + encoding."""

    def __init__(
        self,
        preprocessing: GeneJEPAPreprocessingWrapper,
        encoder: GeneJEPAEncoderWrapper,
        gene_vocab_size: int,
    ):
        super().__init__()
        self.preprocessing = preprocessing
        self.encoder = encoder
        self.gene_vocab_size = gene_vocab_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with full pipeline.

        Args:
            x: Raw counts [batch_size, n_genes]

        Returns:
            Cell embeddings [batch_size, hidden_size]
        """
        batch_size = x.shape[0]

        # Step 1: Preprocessing
        x_norm = self.preprocessing(x)

        # Step 2: Create ragged tensor representation
        # For each cell, identify non-zero genes and their normalized values
        indices_list = []
        values_list = []
        offsets = [0]

        for i in range(batch_size):
            # Find non-zero entries (genes present in this cell)
            nonzero_mask = x[i] > 0
            gene_indices = torch.nonzero(nonzero_mask, as_tuple=True)[0]

            # Get normalized values for these genes
            gene_values = x_norm[i, gene_indices]

            indices_list.append(gene_indices)
            values_list.append(gene_values)
            offsets.append(offsets[-1] + len(gene_indices))

        # Concatenate into ragged format
        indices = (
            torch.cat(indices_list)
            if indices_list
            else torch.empty(0, dtype=torch.int64)
        )
        values = (
            torch.cat(values_list)
            if values_list
            else torch.empty(0, dtype=torch.float32)
        )
        offsets_tensor = torch.tensor(offsets, dtype=torch.int64)

        # Step 3: Encode
        embeddings = self.encoder(indices, values, offsets_tensor)

        return embeddings


def load_gene_metadata(metadata_path: Path) -> Dict[str, int]:
    """
    Load gene metadata from local parquet file.

    Args:
        metadata_path: Path to gene_metadata.parquet file

    Returns:
        Dictionary mapping gene symbols to token IDs
    """
    logger.info("Loading gene metadata from %s", metadata_path)

    import pandas as pd

    df = pd.read_parquet(metadata_path)

    # Create mapping from gene symbol to token ID
    gene_map = {row["gene_symbol"]: row["token_id"] for _, row in df.iterrows()}

    logger.info("Loaded %d gene mappings", len(gene_map))
    return gene_map


def load_global_stats(stats_path: Path) -> tuple[float, float]:
    """
    Load global normalization statistics from local JSON file.

    Args:
        stats_path: Path to global_stats.json file

    Returns:
        Tuple of (global_mean, global_std)
    """
    logger.info("Loading global stats from %s", stats_path)

    with open(stats_path, "r") as f:
        stats = json.load(f)
    global_mean = stats["mean"]
    global_std = stats["std"]

    logger.info("Loaded global stats: mean=%.4f, std=%.4f", global_mean, global_std)
    return global_mean, global_std


@app.command()
def model(
    output_path: Path = typer.Argument(
        default=Path("public/models/genejepa/embedding/"),
        help="Path to save ONNX models (e.g., public/models/genejepa/embedding/)",
    ),
    checkpoint_path: Path = typer.Option(
        "data/models/GeneJEPA/checkpoints/genejepa-epoch=49.ckpt",
        help="Path to GeneJEPA checkpoint file",
    ),
    metadata_path: Path = typer.Option(
        "data/models/GeneJEPA/checkpoints/gene_metadata.parquet",
        help="Path to gene metadata file",
    ),
    stats_path: Path = typer.Option(
        "data/models/GeneJEPA/checkpoints/global_stats.json",
        help="Path to global stats file",
    ),
    genejepa_repo_path: Path = typer.Option(
        "data/models/GeneJEPA",
        help="Path to GeneJEPA repository directory",
    ),
    use_teacher: bool = typer.Option(
        True,
        help="Use teacher encoder (EMA) instead of student encoder",
    ),
) -> None:
    """
    Export GeneJEPA to ONNX format.

    Exports 3 ONNX models:
    1. preprocessing.onnx - Log1p + standardization
    2. embedding.onnx - Tokenizer + Perceiver encoder
    3. model.onnx - Combined preprocessing + embedding

    Also exports:
    - genes.txt - Simple list of gene symbols (one per line)
    - genes.json - Full metadata (JSON per line: symbol, token_id)
    - config.json - Model configuration metadata
    """
    # Add GeneJEPA to Python path
    import sys

    sys.path.insert(0, str(genejepa_repo_path))

    from genejepa.models import GenePerceiverJEPA
    from genejepa.configs import ModelConfig, TrainingConfig, DataConfig, ExperimentConfig

    # Load gene metadata first to get vocab size
    gene_map = load_gene_metadata(metadata_path)

    # Load global statistics
    global_mean, global_std = load_global_stats(stats_path)

    logger.info("Loading GeneJEPA checkpoint from %s", checkpoint_path)

    # Register classes in __main__ for unpickling checkpoint
    # The checkpoint was saved from the training script with __main__ references
    import __main__
    __main__.ModelConfig = ModelConfig
    __main__.TrainingConfig = TrainingConfig
    __main__.DataConfig = DataConfig
    __main__.ExperimentConfig = ExperimentConfig

    # Load checkpoint directly
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )

    # Extract hyperparameters and state dict
    if "hyper_parameters" in checkpoint:
        hparams = checkpoint["hyper_parameters"]
        model_config = hparams["model_config"]
    else:
        logger.error("Checkpoint missing hyper_parameters")
        raise ValueError("Invalid checkpoint format")

    # Create model from config
    gene_vocab_size = model_config.gene_vocab_size
    model_obj = GenePerceiverJEPA(model_config)

    # Load state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Remove 'model.' prefix from keys if present
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_state_dict[key[6:]] = value
            else:
                new_state_dict[key] = value
        model_obj.load_state_dict(new_state_dict, strict=False)
    else:
        logger.error("Checkpoint missing state_dict")
        raise ValueError("Invalid checkpoint format")

    model_obj.eval()

    # Verify vocab size matches
    if len(gene_map) != gene_vocab_size:
        logger.warning(
            "Gene map size %d doesn't match checkpoint vocab size %d",
            len(gene_map),
            gene_vocab_size
        )

    # Get model configuration
    embedding_dim = model_config.d
    gene_vocab_size = model_config.gene_vocab_size

    logger.info("Model configuration:")
    logger.info("  Embedding dimension: %d", embedding_dim)
    logger.info("  Gene vocabulary size: %d", gene_vocab_size)
    logger.info("  Latents: %d", model_config.latents_L)
    logger.info("  Transformer blocks: %d", model_config.blocks_D)
    logger.info("  Attention heads: %d", model_config.heads_h)

    # Create gene ordering (sorted by token ID)
    sorted_genes = sorted(gene_map.items(), key=lambda x: x[1])
    gene_symbols = [symbol for symbol, _ in sorted_genes]
    token_ids = [token_id for _, token_id in sorted_genes]

    assert (
        len(gene_symbols) == gene_vocab_size
    ), f"Gene map size {len(gene_symbols)} != vocab size {gene_vocab_size}"

    logger.info("Gene vocabulary:")
    logger.info("  Total genes: %d", len(gene_symbols))

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Select encoder (teacher or student)
    if use_teacher:
        logger.info("Using teacher encoder (EMA)")
        encoder = model_obj.teacher_encoder.ema_model
    else:
        logger.info("Using student encoder")
        encoder = model_obj.student_encoder

    encoder.eval()

    # Step 1: Export preprocessing module
    logger.info("\n1. Exporting preprocessing module...")
    preprocessing_model = GeneJEPAPreprocessingWrapper(
        global_mean=global_mean,
        global_std=global_std,
    )
    preprocessing_path = output_path / "preprocessing.onnx"

    torch.onnx.export(
        preprocessing_model,
        torch.rand(1, gene_vocab_size),
        preprocessing_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["normalized"],
        dynamic_axes={"input": {0: "batch_size"}, "normalized": {0: "batch_size"}},
        verbose=False,
    )
    logger.info("  Preprocessing ONNX saved to %s", preprocessing_path)

    # Step 2: Export encoder module (includes tokenizer)
    logger.info("\n2. Exporting encoder module...")
    encoder_model = GeneJEPAEncoderWrapper(encoder)
    encoder_path = output_path / "embedding.onnx"

    # Create example ragged inputs
    example_indices = torch.tensor([10, 42, 7, 3, 9, 1, 2], dtype=torch.int64)
    example_values = torch.tensor(
        [0.1, 0.5, 0.3, 2.1, -0.2, 0.0, 1.1], dtype=torch.float32
    )
    example_offsets = torch.tensor([0, 5, 7], dtype=torch.int64)  # 2 samples

    # Use dynamo-based export to handle pad_sequence
    logger.info("  Using dynamo-based ONNX export...")
    torch.onnx.export(
        encoder_model,
        (example_indices, example_values, example_offsets),
        encoder_path,
        export_params=True,
        dynamo=True,
        input_names=["indices", "values", "offsets"],
        output_names=["embedding"],
    )
    logger.info("  Encoder ONNX saved to %s", encoder_path)

    # Step 3: Export combined model
    logger.info("\n3. Exporting combined model...")
    combined_model = GeneJEPACombinedWrapper(
        preprocessing_model,
        encoder_model,
        gene_vocab_size,
    )
    combined_model.eval()
    combined_path = output_path / "model.onnx"

    torch.onnx.export(
        combined_model,
        torch.rand(1, gene_vocab_size),
        combined_path,
        export_params=True,
        dynamo=True,
        input_names=["input"],
        output_names=["cell_embedding"],
    )
    logger.info("  Combined ONNX saved to %s", combined_path)

    # Validate all models
    logger.info("\n4. Validating ONNX models...")
    for model_file, name in [
        (preprocessing_path, "Preprocessing"),
        (encoder_path, "Encoder"),
        (combined_path, "Combined"),
    ]:
        onnx_model = onnx.load(str(model_file))
        onnx.checker.check_model(onnx_model)
        logger.info("  ✅ %s ONNX model validation passed", name)

    # Export genes.txt (simple list of gene symbols, one per line)
    logger.info("\n5. Exporting genes.txt...")
    genes_txt_path = output_path / "genes.txt"

    with open(genes_txt_path, "w") as f:
        f.write("\n".join(gene_symbols))

    logger.info("  Genes list saved to %s", genes_txt_path)

    # Export genes.json with full metadata
    logger.info("\n6. Exporting genes.json...")
    genes_json_path = output_path / "genes.json"

    # Write genes.json (one JSON object per line)
    with open(genes_json_path, "w") as f:
        for symbol, token_id in zip(gene_symbols, token_ids):
            gene_info = {
                "symbol": symbol,
                "token_id": int(token_id),
            }
            f.write(json.dumps(gene_info) + "\n")

    logger.info("  Genes metadata saved to %s", genes_json_path)

    # Export config.json
    logger.info("\n7. Exporting config.json...")
    config_path = output_path / "config.json"

    config = {
        "model_type": "genejepa",
        "embedding_dim": embedding_dim,
        "gene_vocab_size": gene_vocab_size,
        "latents": model_config.latents_L,
        "blocks": model_config.blocks_D,
        "heads": model_config.heads_h,
        "global_mean": global_mean,
        "global_std": global_std,
        "use_teacher": use_teacher,
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info("  Config saved to %s", config_path)

    logger.info("\n🎉 Model export complete!")
    logger.info("  - Preprocessing: %s", preprocessing_path)
    logger.info("  - Encoder: %s", encoder_path)
    logger.info("  - Combined: %s", combined_path)
    logger.info("  - Genes (list): %s", genes_txt_path)
    logger.info("  - Genes (metadata): %s", genes_json_path)
    logger.info("  - Config: %s", config_path)


if __name__ == "__main__":
    app()
