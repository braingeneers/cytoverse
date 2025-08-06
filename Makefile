# Default ivfpq parameters
export model_id=scimilarity

test:
	python -m pytest ivfpq/python/tests
	python -m pytest tests/unit
	npx vitest run ivfpq

test-no-capture:
	python -m pytest ivfpq/python/tests --capture=no --log-cli-level=DEBUG

# Populate model artifacts for the selected model to web/public/models/<model_id>
notice:
	@echo "⚙️ Populating $(model_id)"
	@echo "- Embeddings & metadata to data/references/$(model_id)"
	@echo "- Web Artificats to web/public/models/$(model_id)"

scimilarity: model_id=scimilarity
scimilarity: notice scimilarity-embeddings scimilarity-model ivfpq-train pumap

# brain: model_id=brain
# brain: notice brain-embeddings scimilarity-model pumap ivfpq

# Embeddings (Must be first as model uses these to validate the export)
scimilarity-embeddings:
	python scripts/scimilarity_to_embeddings.py \
	data/models/scimilarity/model_v1.1/cellsearch \
	data/references/$(model_id)/ \
	--labels prediction \
	--labels tissue \
	--labels author_label \
	--labels study \
	--validate

# brain-embeddings:
# 	python scripts/ingest_h5ad_as_reference.py \
# 	data/brain.h5ad \
# 	data/models/scimilarity/model_v1.1 \
# 	data/brain \
# 	--labels CellType \
# 	--labels tissue_type

# Embedding Model
scimilarity-model:
	python scripts/scimilarity_export_model.py \
	data/models/scimilarity/model_v1.1 \
	public/models/$(model_id)/embedding


# IVFPQ tune
ivfpq-tune:
	python scripts/ivfpq_tune.py \
        data/references/scimilarity/embeddings.parquet \
        data/references/scimilarity/labels.parquet \
        --num-samples 16 \
        --max-concurrent-trials 8

# IVF
ivfpq-train:
	python scripts/ivfpq_train.py train \
	data/references/$(model_id)/embeddings.npy \
	public/models/$(model_id) \
	--n-partitions 512 \
	--max-vectors-for-training 100000 \
	--n-iterations 30

ivfpq-test:
	python scripts/ivfpq_train.py test \
	public/models/$(model_id) \
	data/references/$(model_id)/embeddings.npy

# PUMAP
pumap-train:
	python scripts/pumap_train.py train \
	data/references/$(model_id)/embeddings.npy \
	data/references/$(model_id)/labels.parquet \
	public/models/$(model_id)/pumap \
	--stratify-label prediction \
	--num-vectors 250000

pumap-map:
	python scripts/pumap_train.py map \
	public/models/$(model_id)/pumap/model.onnx \
	data/references/$(model_id)/embeddings.npy \
	data/references/$(model_id)/ \
	--export-png \
	--num-vectors 1000000

pumap-export:
	python scripts/pumap_train.py export \
	data/references/$(model_id)/ \
	public/models/$(model_id)/pumap/

pumap: pumap-train pumap-map pumap-export



# Testing
update-validations:
	python scripts/label.py \
	      ivfpq/python/tests/fixtures/GSE136831_subsample_100.h5ad \
	      tests/e2e/GSE136831_subsample_100.labels.csv
	python scripts/label.py \
	      public/sample.h5ad \
	      tests/e2e/sample.labels.csv