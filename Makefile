test:
	python -m pytest tests/ -v

# SCimilarity
scimilarity-export-model:
	python src/cellspace/scripts/scimilarity_export.py model \
	data/scimilarity/model_v1.1 \
	web/public/models/scimilarity

scimilarity-export-embeddings:
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

mapper-train:
	python src/cellspace/scripts/mapper.py train \
        data/scimilarity/embeddings.npy \
        web/public/models/scimilarity/ \
        --num-embeddings 100000

mapper-map:
	python src/cellspace/scripts/mapper.py map \
        web/public/models/scimilarity/mapper.onnx \
        data/scimilarity/embeddings.npy \
        data/scimilarity/ \
		--export-png \

mapper-export:
	python src/cellspace/scripts/mapper.py export \
        data/scimilarity/ \
        web/public/models/scimilarity/ 

# PQ
pq-train:
	python src/cellspace/scripts/pq_train.py train \
		data/scimilarity/embeddings.npy \
		web/public/models/scimilarity/pq/ \
		--m 16 \
		--k 256 \
		--max-vectors 10000 \
		--n-iterations 30

pq-train-full:
	python src/cellspace/scripts/pq_train.py train \
		data/scimilarity/embeddings.npy \
		web/public/models/scimilarity/pq/ \
		--m 16 \
		--k 256 \
		--n-iterations 50

pq-train-test:
	python src/cellspace/scripts/pq_train.py test \
		web/public/models/scimilarity/pq/pq_model.pkl \
		data/scimilarity/embeddings.npy \
		--n-test 1000

# IVF
ivf-train:
	python src/cellspace/scripts/ivfpq_train.py train-ivf \
		data/scimilarity/embeddings.npy \
		web/public/models/scimilarity/ivf/ \
		--n-clusters 64 \
		--max-vectors 10000 \
		--n-iterations 30

ivf-train-full:
	python src/cellspace/scripts/ivfpq_train.py train-ivf \
		data/scimilarity/embeddings.npy \
		web/public/models/scimilarity/ivf/ \
		--n-clusters 256 \
		--n-iterations 50

# IVFPQ
ivfpq-train:
	python src/cellspace/scripts/ivfpq_train.py train-ivfpq \
		data/scimilarity/embeddings.npy \
		web/public/models/scimilarity/ivfpq/ \
		--m 16 \
		--k 256 \
		--n-clusters 64 \
		--max-vectors 10000 \
		--pq-iterations 30 \
		--ivf-iterations 30

ivfpq-train-full:
	python src/cellspace/scripts/ivfpq_train.py train-ivfpq \
		data/scimilarity/embeddings.npy \
		web/public/models/scimilarity/ivfpq/ \
		--m 64 \
		--k 256 \
		--n-clusters 256 \
		--pq-iterations 50 \
		--ivf-iterations 50

ivfpq-test:
	python src/cellspace/scripts/ivfpq_train.py test-trained-models \
		web/public/models/scimilarity/ivfpq/ \
		--embeddings-path data/scimilarity/embeddings.npy \
		--n-test-vectors 1000