# Default ivfpq parameters
model_id ?=scimilarity

test:
	python -m pytest backend/tests/unit
	python -m pytest backend/tests/e2e
	npx vitest run frontend/tests/unit --silent
	npx playwright test --config frontend/playwright.config.ts

test-no-capture:
	cd backend && python -m pytest tests --capture=no --log-cli-level=DEBUG

benchmark:
	cd backend && python -m pytest tests/unit/test_performance.py --capture=no --log-cli-level=DEBUG


# Frontend
build:
	npm run build
	echo "Removing dev from models list"
	sed -i '' '/^dev$$/d' ./dist/models/models.txt
	rm -rf dist/models/dev

deploy:
	echo "Updating https://cells-test.gi.ucsc.edu/cytoverse/..."
	rsync -avh --delete dist/ \
		rcurrie@hgwdev.gi.ucsc.edu:/usr/local/apache/htdocs-cells/cytoverse/


# Data ingestion
inspect-h5ad:
	python backend/src/h5ad_to_embeddings.py inspect \
		fixtures/GSE136831_subsample_100.h5ad	

	python backend/src/h5ad_to_embeddings.py inspect \
		~/data/h5ad/adata_metaatlas_final_raw.h5ad

	python backend/src/h5ad_to_embeddings.py inspect \
		~/data/h5ad/allen-celltypes+human-cortex+various-cortical-areas.h5ad

# Populate model artifacts for the selected model to public/models/<model_id>
notice:
	@echo "⚙️ Populating $(model_id)"
	@echo "- Embeddings & metadata to data/references/$(model_id)"
	@echo "- Web Artificats to public/models/$(model_id)"

update-models-list:
	ls -d public/models/*/ | while read -r dir; do basename "$$dir"; done > public/models/models.txt

scimilarity: model_id=scimilarity
scimilarity: stratify_label=prediction
scimilarity: notice scimilarity-model \
	scimilarity-embeddings ivfpq-train pumap update-models-list

sspsygene: model_id=sspsygene
sspsygene: stratify_label=Type.v1
sspsygene: notice scimilarity-model \
	sspsygene-embeddings ivfpq-train pumap update-models-list

# Embedding Model (only one for now)
scimilarity-model:
	cd backend && python src/scimilarity_export_model.py \
	../data/models/scimilarity/model_v1.1 \
	../public/models/$(model_id)/embedding

# Embeddings (Must be first as model uses these to validate the export)
scimilarity-embeddings:
	cd backend && python src/scimilarity_to_embeddings.py embeddings \
	../data/models/scimilarity/model_v1.1/cellsearch \
	../data/references/$(model_id)/ \
	--labels prediction \
	--labels tissue \
	--labels study \
	--validate

scimilarity-aligned-embeddings:
	cd backend && python src/scimilarity_to_embeddings.py annotation-aligned \
	../data/models/scimilarity/model_v1.1 \
	../data/references/$(model_id)

sspsygene-embeddings:
	cd backend && python src/h5ad_to_embeddings.py \
	~/data/h5ad/adata_metaatlas_final_raw.h5ad \
	../data/models/scimilarity/model_v1.1 \
	../data/references/sspsygene \
	--labels Dataset \
	--labels Gestational_week \
	--labels Class \
	--labels Type.v1 \
	--labels Subtype.v1


# sspsygene embeddings from geneformer model
sspsygene-geneformer-model:
	python backend/src/geneformer_export_model.py \
	data/models/Geneformer/Geneformer-V1-10M \
	public/models/sspsygene-geneformer/embedding

sspsygene-geneformer-embeddings:
	python backend/src/h5ad_to_embeddings.py \
	~/data/h5ad/adata_metaatlas_final_raw.h5ad \
	public/models/sspsygene-geneformer/embedding/model.onnx \
	public/models/sspsygene-geneformer/embedding/genes.txt \
	data/references/sspsygene-geneformer \
	--max-cells 10_000 \
	--batch-size 32 \
	--labels Dataset \
	--labels Gestational_week \
	--labels Class \
	--labels Type.v1 \
	--labels Subtype.v1

# sspsygene embeddings from geneformer model
arc-stack-model:
	python backend/src/arc_stack_export_model.py \
	data/models/arc-stack/Stack-Large \
	public/models/arc-stack/embedding

sspsygene-arc-stack-embeddings:
	python backend/src/h5ad_to_embeddings.py \
	~/data/h5ad/adata_metaatlas_final_raw.h5ad \
	public/models/arc-stack/embedding/model.onnx \
	public/models/arc-stack/embedding/genes.txt \
	data/references/arc-stack \
	--max-cells 10_000 \
	--batch-size 4 \
	--labels Type.v1 \
	--labels Subtype.v1

# IVF
ivfpq-train:
	python backend/src/ivfpq_train.py train \
	data/references/$(model_id)/embeddings.npy \
	public/models/$(model_id)/ivfpq \
	--max-vectors-for-training 100_000 \
	--pq-m 32 \
	--pq-k 256 \
	--sample-training-vectors \
	--test

# PUMAP
pumap-train:
	python backend/src/pumap_train.py train \
	data/references/$(model_id)/embeddings.npy \
	data/references/$(model_id)/labels.parquet \
	public/models/$(model_id)/pumap \
	--stratify-label $(stratify_label) \
	--num-vectors 250_000

pumap-map:
	python backend/src/pumap_train.py map \
	public/models/$(model_id)/pumap/model.onnx \
	data/references/$(model_id)/embeddings.npy \
	data/references/$(model_id)/ \
	--export-png \
	--num-vectors 1_000_000

pumap-export:
	python backend/src/pumap_train.py export \
	data/references/$(model_id)/ \
	public/models/$(model_id)/pumap/

pumap: pumap-train pumap-map pumap-export

# Testing
update-validations:
	python backend/src/label.py \
	      fixtures/GSE136831_subsample_100.h5ad \
	      fixtures/GSE136831_subsample_100.labels.csv
	python backend/src/label.py \
	      public/sample.h5ad \
	      fixtures/sample.labels.csv
