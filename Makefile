# Default ivfpq parameters
export model_id=scimilarity
export pq_m=16
export n_partitions=256

test:
	python -m pytest tests/ -v

# Populate model artifacts for the selected model to web/public/models/<model_id>
notice:
	@echo "⚙️ Populating $(model_id)"
	@echo "- Embeddings & metadata to data/$(model_id)"
	@echo "- Web Artificats to web/public/models/$(model_id)"
	@echo "- PQ M=$(pq_m)"
	@echo "- N Partitions=$(n_partitions)"

scimilarity: model_id=scimilarity
scimilarity: notice scimilarity-embeddings embedding-model pumap ivfpq

brain: model_id=brain
brain: notice brain-embeddings embedding-model pumap ivfpq

# Embedding Model
embedding-model:
	python src/cytoverse/scripts/scimilarity_export_model.py \
	~/data/scimilarity/model_v1.1 \
	web/public/models/$(model_id)/embedding


# Embeddings
scimilarity-embeddings:
	python src/cytoverse/scripts/scimilarity_to_embeddings.py \
	~/data/scimilarity/model_v1.1/cellsearch \
	data/$(model_id)/ \
	--labels prediction \
	--labels tissue \
	--labels author_label \
	--labels study \
	--validate

brain-embeddings:
	python src/cytoverse/scripts/ingest_h5ad_as_reference.py \
	data/brain.h5ad \
	data/scimilarity/model_v1.1 \
	data/brain \
	--labels CellType \
	--labels tissue_type


# PUMAP
pumap-train:
	python src/cytoverse/scripts/pumap_train.py train \
	data/$(model_id)/vectors.npy \
	data/$(model_id)/labels.parquet \
	web/public/models/$(model_id)/pumap \
	--stratify-label prediction \
	--num-vectors 250000

pumap-map:
	python src/cytoverse/scripts/pumap_train.py map \
	web/public/models/$(model_id)/pumap/model.onnx \
	data/$(model_id)/vectors.npy \
	data/$(model_id)/ \
	--export-png \
	--num-vectors 1000000

pumap-export:
	python src/cytoverse/scripts/pumap_train.py export \
	data/$(model_id)/ \
	web/public/models/$(model_id)/pumap/

pumap: pumap-train pumap-map pumap-export

# PQ
pq-train:
	python src/cytoverse/scripts/ivfpq_train.py pq-train \
	data/$(model_id)/vectors.npy \
	web/public/models/$(model_id)/pq/ \
	--m $(pq_m) \
	--k $(n_partitions) \
	--max-vectors 100000 \
	--n-iterations 30
	
# IVF
ivf-train:
	python src/cytoverse/scripts/ivfpq_train.py ivf-train \
	data/$(model_id)/vectors.npy \
	web/public/models/$(model_id)/ivf/ \
	--n-partitions 256 \
	--max-vectors 100000 \
	--n-iterations 30

# IVFPQ
ivfpq-export:
	python src/cytoverse/scripts/ivfpq_train.py ivfpq-export \
	web/public/models/$(model_id)/ \
	data/$(model_id)/vectors.npy \
	--max-vectors 100000 \
	--test-performance

ivfpq: pq-train ivf-train ivfpq-export


# Testing
update-validations:
	python src/cytoverse/scripts/label.py \
	      tests/GSE136831_subsample_100.h5ad \
	      web/tests/GSE136831_subsample_100.labels.csv
	python src/cytoverse/scripts/label.py \
	      web/public/sample.h5ad \
	      web/tests/sample.labels.csv

# Ray
rclone-run:
	@echo "Starting local s3 server on port 9000..."
	AWS_PROFILE=local-s3 rclone \
		serve s3 local-s3:~/data \
		--addr :9000 \
		--vfs-cache-mode off \
		-v
	@echo "Local s3 server is running. You can access it at http://localhost:9000"

# Generate embeddings from local h5ad file
embed:
	python -m cytoverse.scripts.tune embed data/input.h5ad data/embeddings.parquet
