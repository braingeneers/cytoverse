test:
	python -m pytest tests/ -v

reset:
	rm -rf web/public/models/scimilarity
	rm data/scimilarity/*

# SCimilarity
scimilarity-model:
	python src/cellspace/scripts/scimilarity_export.py model \
	data/scimilarity/model_v1.1 \
	web/public/models/scimilarity/embedding

scimilarity-embeddings:
	python src/cellspace/scripts/scimilarity_export.py embeddings \
	data/scimilarity/model_v1.1/cellsearch \
	data/scimilarity/ \
	--labels tissue \
	--labels prediction \
	--labels study \
	--validate \
	--stratify \
	--num-embeddings 1000000
	parquet-tools inspect data/scimilarity/labels.parquet

scimilarity: scimilarity-model scimilarity-embeddings

# PUMAP
pumap-train:
	python src/cellspace/scripts/pumap_train.py train \
        data/scimilarity/vectors.npy \
        web/public/models/scimilarity/pumap \
        --num-vectors 100000

pumap-map:
	python src/cellspace/scripts/pumap_train.py map \
        web/public/models/scimilarity/pumap/model.onnx \
        data/scimilarity/vectors.npy \
        data/scimilarity/ \
		--export-png \

pumap-export:
	python src/cellspace/scripts/pumap_train.py export \
        data/scimilarity/ \
        web/public/models/scimilarity/pumap/

pumap: pumap-train pumap-map pumap-export

# PQ
pq-train:
	python src/cellspace/scripts/ivfpq_train.py pq-train \
		data/scimilarity/vectors.npy \
		web/public/models/scimilarity/pq/ \
		--m 16 \
		--k 256 \
		--max-vectors 10000 \
		--n-iterations 30
	
# IVF
ivf-train:
	python src/cellspace/scripts/ivfpq_train.py ivf-train \
		data/scimilarity/vectors.npy \
		data/scimilarity/vector_ids.npy \
		web/public/models/scimilarity/ivf/ \
		--n-partitions 64 \
		--max-vectors 10000 \
		--n-iterations 30

# ivf-train-full:
# 	python src/cellspace/scripts/ivfpq_train.py train-ivf \
# 		data/scimilarity/vectors.npy \
# 		web/public/models/scimilarity/ivf/ \
# 		--n-clusters 256 \
# 		--n-iterations 50

# IVFPQ
ivfpq-train:
	python src/cellspace/scripts/ivfpq_train.py train-ivfpq \
		data/scimilarity/vectors.npy \
		web/public/models/scimilarity/ivfpq/ \
		--m 16 \
		--k 256 \
		--n-clusters 64 \
		--max-vectors 10000 \
		--pq-iterations 30 \
		--ivf-iterations 30

ivfpq-train-full:
	python src/cellspace/scripts/ivfpq_train.py train-ivfpq \
		data/scimilarity/vectors.npy \
		web/public/models/scimilarity/ivfpq/ \
		--m 64 \
		--k 256 \
		--n-clusters 256 \
		--pq-iterations 50 \
		--ivf-iterations 50

ivfpq-test:
	python src/cellspace/scripts/ivfpq_train.py test-trained-models \
		web/public/models/scimilarity/ivfpq/ \
		--vectors-path data/scimilarity/vectors.npy \
		--n-test-vectors 1000

# Complete IVFPQ (Arrow format only)
ivfpq-complete-train:
	python src/cellspace/scripts/ivfpq_train.py train-complete-ivfpq \
		data/scimilarity/vectors.npy \
		web/public/models/scimilarity/ivfpq/ \
		--m 16 \
		--k 256 \
		--n-clusters 64 \
		--max-vectors 10000 \
		--pq-iterations 30 \
		--ivf-iterations 30

ivfpq-complete-train-full:
	python src/cellspace/scripts/ivfpq_train.py train-complete-ivfpq \
		data/scimilarity/vectors.npy \
		web/public/models/scimilarity/ivfpq/ \
		--m 16 \
		--k 256 \
		--n-clusters 256 \
		--pq-iterations 50 \
		--ivf-iterations 50

# Export browser assets from trained models (legacy compatibility)
# Export browser assets from trained models (legacy compatibility)
ivfpq-export-browser:
	@echo "Note: Browser assets are now exported directly during training"
	@echo "Use 'make ivfpq-complete-train' to generate Arrow format assets"

# Test Arrow export functionality
test-arrow-export:
	python tests/test_arrow_export.py

# Validate browser assets
validate-browser-assets:
	python scripts/validate_browser_assets.py web/public/models/scimilarity/ivfpq/