test:
	python -m pytest tests/ -v

all: scimilarity pumap ivfpq

reset:
	rm data/scimilarity/*.*
	rm -rf web/public/models/scimilarity

model:
	python src/cytoverse/scripts/scimilarity_export_model.py \
	data/scimilarity/model_v1.1 \
	web/public/models/$(model_id)/embedding

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

# Brain
pull-brain-embeddings:
	rsync -avh \
	rcurrie@hgwdev.gi.ucsc.edu:/scratch/rcurrie/cytoverse/brain/ \
	data/brain/

brain-embeddings:
	python src/cytoverse/scripts/ingest_h5ad_as_reference.py \
	data/integrated_data_10k.h5ad \
	data/scimilarity/model_v1.1 \
	data/brain \
	--labels CellType \
	--labels tissue_type \
	--max-cells 10

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
