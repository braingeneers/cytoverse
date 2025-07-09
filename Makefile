test:
	python -m pytest tests/ -v

all: scimilarity pumap ivfpq

reset:
	rm data/scimilarity/*.*
	rm -rf web/public/models/scimilarity

# SCimilarity
scimilarity-model:
	python src/cytoverse/scripts/scimilarity_export_model.py \
	data/scimilarity/model_v1.1 \
	web/public/models/scimilarity/embedding

scimilarity-embeddings:
	python src/cytoverse/scripts/scimilarity_export_embeddings.py \
	data/scimilarity/model_v1.1/cellsearch \
	data/scimilarity/ \
	--labels prediction \
	--labels tissue \
	--labels author_label \
	--labels study \
	--validate
	parquet-tools inspect data/scimilarity/labels.parquet

scimilarity: scimilarity-model scimilarity-embeddings

# PUMAP
pumap-train:
	python src/cytoverse/scripts/pumap_train.py train \
	data/scimilarity/vectors.npy \
	data/scimilarity/labels.parquet \
	web/public/models/scimilarity/pumap \
	--num-vectors 250000

pumap-map:
	python src/cytoverse/scripts/pumap_train.py map \
	web/public/models/scimilarity/pumap/model.onnx \
	data/scimilarity/vectors.npy \
	data/scimilarity/ \
	--export-png \
	--num-vectors 1000000

pumap-export:
	python src/cytoverse/scripts/pumap_train.py export \
	data/scimilarity/ \
	web/public/models/scimilarity/pumap/

pumap: pumap-train pumap-map pumap-export

# PQ
pq-train:
	python src/cytoverse/scripts/ivfpq_train.py pq-train \
	data/scimilarity/vectors.npy \
	data/scimilarity/labels.parquet \
	web/public/models/scimilarity/pq/ \
	--m 16 \
	--k 256 \
	--max-vectors 10000 \
	--n-iterations 30
	
# IVF
ivf-train:
	python src/cytoverse/scripts/ivfpq_train.py ivf-train \
	data/scimilarity/vectors.npy \
	data/scimilarity/labels.parquet \
	web/public/models/scimilarity/ivf/ \
	--n-partitions 64 \
	--max-vectors 10000 \
	--n-iterations 30

# IVFPQ
ivfpq-export:
	python src/cytoverse/scripts/ivfpq_train.py ivfpq-export \
	web/public/models/scimilarity \
	data/scimilarity/vectors.npy \
	data/scimilarity/labels.parquet \
	--test-performance

ivfpq: pq-train ivf-train ivfpq-export