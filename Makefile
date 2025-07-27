# Default to scimilarity model and reference dataaset
export model_id=scimilarity

test:
	python -m pytest tests/ -v

# Populate model artifacts for the selected model to web/public/models/<model_id>
notice:
	@echo "⚙️ Populating $(model_id)"
	@echo "- Embeddings & metadata to data/$(model_id)"
	@echo "- Web Artificats to web/public/models/$(model_id)"

brain: model_id=brain
brain: notice brain-embeddings embedding-model pumap ivfpq

scimilarity: model_id=scimilarity
scimilarity: notice scimilarity-embeddings embedding-model pumap ivfpq

# Embeddings
scimilarity-embeddings:
	python src/cytoverse/scripts/scimilarity_export_embeddings.py \
	data/scimilarity/model_v1.1/cellsearch \
	data/scimilarity/ \
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

# Embedding Model
embedding-model:
	python src/cytoverse/scripts/scimilarity_export_model.py \
	data/scimilarity/model_v1.1 \
	web/public/models/$(model_id)/embedding

# PUMAP
pumap-train:
	python src/cytoverse/scripts/pumap_train.py train \
	data/$(model_id)/vectors.npy \
	data/$(model_id)/labels.parquet \
	web/public/models/$(model_id)/pumap \
	--stratify-label CellType \
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
	--m 16 \
	--k 256 \
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
	      tests/GSE136831_subsample_10.h5ad \
	      tests/GSE136831_subsample_10.labels.csv
