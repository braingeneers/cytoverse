import click
from pathlib import Path
import numpy as np

import torch
import onnx
import onnxruntime as ort
from scimilarity import CellEmbedding


@click.command()
@click.option(
    "--model_path",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    default="data/scimilarity/model_v1.1",
    help="Path to encoder checkpoint file",
)
@click.option(
    "--output_path",
    type=click.Path(exists=False, file_okay=True, path_type=Path),
    default="web/public/models/scimilarity.onnx",
    help="Path to save the ONNX model",
)
def main(model_path: Path, output_path: Path) -> None:
    """
    Convert a scimilarity encoder checkpoint to ONNX format.
    """
    print(f"Loading model from {model_path}...")
    ce = CellEmbedding(model_path)

    print("Converting to ONNX format...")
    torch.onnx.export(
        ce.model,
        torch.zeros(1, ce.n_genes),
        output_path,
        export_params=True,
        opset_version=14,  # Use version 14 for better transformer support
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        verbose=False,  # Reduce verbosity
    )

    print(f"ONNX model saved to {output_path}")

    print("Validating ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX model validation passed")

    print("Checking concordance with scimilarity model...")

    # Create ONNX runtime session
    ort_session = ort.InferenceSession(str(output_path))

    # Generate random test data (batch of 5 samples)
    batch_size = 5
    np.random.seed(42)  # For reproducible results
    test_input = np.random.lognormal(
        mean=0, sigma=1, size=(batch_size, ce.n_genes)
    ).astype(np.float32)

    # Run through original PyTorch model
    ce.model.eval()
    with torch.no_grad():
        torch_input = torch.from_numpy(test_input)
        torch_output = ce.model(torch_input).numpy()

    # Run through ONNX model
    onnx_output = ort_session.run(None, {"input": test_input})[0]

    # Compare outputs
    max_diff = np.max(np.abs(torch_output - onnx_output))
    mean_diff = np.mean(np.abs(torch_output - onnx_output))

    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")

    if max_diff < 1e-5:
        print("  ✅ ✓ Models are concordant (differences < 1e-5)")
    elif max_diff < 1e-3:
        print("  ⚠️ Models have small differences (< 1e-3)")
    else:
        print(f"  ❌ Models have significant differences (max diff: {max_diff:.2e})")

    print("Exporting genes...")

    # REMIND: Switch to saving as .gz and sort out issues with vite server mime...
    # genes_path = (output_path.parent / output_path.stem).with_suffix(".genes.gz")
    # with gzip.open(genes_path, "wb") as f:
    #     f.write("\n".join(map(str, ce.gene_order)).encode('utf-8'))

    genes_path = (output_path.parent / output_path.stem).with_suffix(".genes")
    with open(genes_path, "w") as f:
        f.write("\n".join(map(str, ce.gene_order)))

    print("🎉 Export complete!")


if __name__ == "__main__":
    main()
