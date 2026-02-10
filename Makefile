# Default
model_id ?= "scimilarity"

# Populate model artifacts for the selected model to public/models/<model_id>
notice:
	@echo "⚙️ Populating $(model_id)"
	@echo "- Embeddings & metadata to data/references/$(model_id)"
	@echo "- Web Artificats to web/public/models/$(model_id)"

update-models-list:
	ls -d public/models/*/ | while read -r dir; do basename "$$dir"; done > public/models/models.txt

# Model
scimilarity-model:
	python scripts/scimilarity_export_model.py \
	data/models/scimilarity/model_v1.1 \
	public/models/$(model_id)/embedding

# Embeddings
scimilarity-embeddings:
	python scripts/scimilarity_to_embeddings.py  embeddings \
	data/models/scimilarity/model_v1.1/cellsearch \
	data/references/$(model_id)/ \
	--labels prediction \
	--labels tissue \
	--labels study \
	--validate

scimilarity-aligned-embeddings:
	python scripts/scimilarity_to_embeddings.py annotation-aligned \
	data/models/scimilarity/model_v1.1 \
	data/references/$(model_id)

scimilarity-subset-embeddings:
	python scripts/scimilarity_to_embeddings.py  embeddings \
	data/models/scimilarity/model_v1.1/cellsearch \
	data/references/$(model_id)/ \
	--num-embeddings 10_000 \
	--labels prediction \
	--labels study \
	--validate

sspsygene-embeddings:
	python scripts/h5ad_to_embeddings.py \
	~/data/h5ad/adata_metaatlas_final_raw.h5ad \
	data/models/scimilarity/model_v1.1 \
	data/references/sspsygene \
	--labels Dataset \
	--labels Gestational_week \
	--labels Class \
	--labels Type.v1 \
	--labels Subtype.v1

# IVF
ivfpq-train:
	python scripts/ivfpq_train.py train \
	data/references/$(model_id)/embeddings.npy \
	public/models/$(model_id)/ivfpq \
	--max-vectors-for-training 2_400_000 \
	--pq-m 32 \
	--pq-k 256 \
	--sample-training-vectors \
	--test

# PUMAP
pumap-train:
	python scripts/pumap_train.py train \
	data/references/$(model_id)/embeddings.npy \
	data/references/$(model_id)/labels.parquet \
	public/models/$(model_id)/pumap \
	--stratify-label $(stratify_label) \
	--num-vectors 250000

pumap-map:
	python scripts/pumap_train.py map \
	public/models/$(model_id)/pumap/model.onnx \
	data/references/$(model_id)/embeddings.npy \
	data/references/$(model_id)/ \
	--export-png \
	--num-vectors 1_000_000

pumap-export:
	python scripts/pumap_train.py export \
	data/references/$(model_id)/ \
	public/models/$(model_id)/pumap/

pumap: pumap-train pumap-map pumap-export

# Specific models and references
scimilarity-subset: model_id=scimilarity-subset
scimilarity-subset: stratify_label=prediction
scimilarity-subset: notice scimilarity-model \
	scimilarity-subset-embeddings ivfpq-train pumap update-models-list

scimilarity: model_id=scimilarity
scimilarity: stratify_label=prediction
scimilarity: notice scimilarity-model \
	scimilarity-embeddings ivfpq-train pumap update-models-list

sspsygene: model_id=sspsygene
sspsygene: stratify_label=Type.v1
sspsygene: notice scimilarity-model \
	sspsygene-embeddings ivfpq-train pumap update-models-list

# Testing
update-validations:
	python scripts/label.py \
	      ivfpq/python/tests/fixtures/GSE136831_subsample_100.h5ad \
	      tests/e2e/GSE136831_subsample_100.labels.csv
	python scripts/label.py \
	      public/sample.h5ad \
	      tests/e2e/sample.labels.csv

test:
	npx vitest run tests/unit
	python -m pytest tests/unit
	python -m pytest tests/e2e
	npx playwright test tests/e2e/run.spec.ts

build:
	npm run build
	echo "Removing scimilarity-subset from models list"
	sed -i '' '/^scimilarity-subset$$/d' ./dist/models/models.txt
	rm -rf dist/models/scimilarity-subset

deploy-dry:
	rsync -a --delete --dry-run --out-format="%n" \
		dist/ \
		rcurrie@hgwdev.gi.ucsc.edu:/usr/local/apache/htdocs-cells/cytoverse/

deploy:
	echo "Updating https://cells-test.gi.ucsc.edu/cytoverse/..."
	rsync -avh --delete dist/ \
		rcurrie@hgwdev.gi.ucsc.edu:/usr/local/apache/htdocs-cells/cytoverse/

benchmark:
	python -m pytest tests/unit/test_performance.py --capture=no --log-cli-level=DEBUG

test-no-capture:
	python -m pytest ivfpq/python/tests --capture=no --log-cli-level=DEBUG