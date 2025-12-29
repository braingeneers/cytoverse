# Default ivfpq parameters
export model_id=scimilarity

test:
	cd frontend && npx vitest run tests/unit
	cd backend && python -m pytest tests/unit
	cd backend && python -m pytest tests/e2e
	cd frontend && npx playwright test tests/e2e/run.spec.ts

build:
	cd frontend && npm run build
	echo "Removing dev from models list"
	sed -i '' '/^dev$$/d' ./dist/models/models.txt
	rm -rf dist/models/dev

deploy:
	echo "Updating https://cells-test.gi.ucsc.edu/cytoverse/..."
	rsync -avh --delete dist/ \
		rcurrie@hgwdev.gi.ucsc.edu:/usr/local/apache/htdocs-cells/cytoverse/

benchmark:
	cd backend && python -m pytest tests/unit/test_performance.py --capture=no --log-cli-level=DEBUG

test-no-capture:
	cd backend && python -m pytest tests --capture=no --log-cli-level=DEBUG

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

# IVF
ivfpq-train:
	cd backend && python src/ivfpq_train.py train \
	../data/references/$(model_id)/embeddings.npy \
	../public/models/$(model_id)/ivfpq \
	--max-vectors-for-training 2_400_000 \
	--pq-m 32 \
	--pq-k 256 \
	--sample-training-vectors \
	--test

# PUMAP
pumap-train:
	cd backend && python src/pumap_train.py train \
	../data/references/$(model_id)/embeddings.npy \
	../data/references/$(model_id)/labels.parquet \
	../public/models/$(model_id)/pumap \
	--stratify-label $(stratify_label) \
	--num-vectors 250000

pumap-map:
	cd backend && python src/pumap_train.py map \
	../public/models/$(model_id)/pumap/model.onnx \
	../data/references/$(model_id)/embeddings.npy \
	../data/references/$(model_id)/ \
	--export-png \
	--num-vectors 1_000_000

pumap-export:
	cd backend && python src/pumap_train.py export \
	../data/references/$(model_id)/ \
	../public/models/$(model_id)/pumap/

pumap: pumap-train pumap-map pumap-export

# Testing
update-validations:
	cd backend && python src/label.py \
	      ../frontend/tests/fixtures/GSE136831_subsample_100.h5ad \
	      ../frontend/tests/fixtures/GSE136831_subsample_100.labels.csv
	cd backend && python src/label.py \
	      ../public/sample.h5ad \
	      ../frontend/tests/fixtures/sample.labels.csv
