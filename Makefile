test:
	python -m pytest tests/ -v

reset:
	rm -rf web/public/models/scimilarity
	rm data/scimilarity/*

# SCimilarity
scimilarity-model:
	python src/cytoverse/scripts/scimilarity_export.py model \
	data/scimilarity/model_v1.1 \
	web/public/models/scimilarity/embedding

scimilarity-embeddings:
	python src/cytoverse/scripts/scimilarity_export.py embeddings \
	data/scimilarity/model_v1.1/cellsearch \
	data/scimilarity/ \
	--labels tissue \
	--labels prediction \
	--labels study \
	--validate \
	--stratify \
	--num-embeddings 5000000
	parquet-tools inspect data/scimilarity/labels.parquet

scimilarity: scimilarity-model scimilarity-embeddings

# PUMAP
pumap-train:
	python src/cytoverse/scripts/pumap_train.py train \
        data/scimilarity/vectors.npy \
        web/public/models/scimilarity/pumap \
        --num-vectors 250000

pumap-map:
	python src/cytoverse/scripts/pumap_train.py map \
        web/public/models/scimilarity/pumap/model.onnx \
        data/scimilarity/vectors.npy \
        data/scimilarity/ \
		--export-png \

pumap-export:
	python src/cytoverse/scripts/pumap_train.py export \
        data/scimilarity/ \
        web/public/models/scimilarity/pumap/

pumap: pumap-train pumap-map pumap-export

# PQ
pq-train:
	python src/cytoverse/scripts/ivfpq_train.py pq-train \
		data/scimilarity/vectors.npy \
		data/scimilarity/vector_ids.npy \
		web/public/models/scimilarity/pq/ \
		--m 16 \
		--k 256 \
		--max-vectors 10000 \
		--n-iterations 30
	
# IVF
ivf-train:
	python src/cytoverse/scripts/ivfpq_train.py ivf-train \
		data/scimilarity/vectors.npy \
		data/scimilarity/vector_ids.npy \
		web/public/models/scimilarity/ivf/ \
		--n-partitions 64 \
		--max-vectors 10000 \
		--n-iterations 30

# IVFPQ
ivfpq-export:
	python src/cytoverse/scripts/ivfpq_train.py ivfpq-export \
		web/public/models/scimilarity \
		data/scimilarity/vectors.npy \
		data/scimilarity/vector_ids.npy \
		--max-vectors 10000 \
		--test-performance

ivfpq: ivf-train pq-train ivfpq-export