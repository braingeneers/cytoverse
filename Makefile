scimilarity-export:
	python src/cellspace/scripts/scimilarity_export_cells.py \
	data/scimilarity/model_v1.1/cellsearch \
	data/scimilarity/ \
	--labels tissue \
	--labels prediction \
	--labels study \
	--num-cells 1000000

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
        --num-embeddings 1000000

mapper-export:
	python src/cellspace/scripts/mapper.py export \
        data/scimilarity/ \
        web/public/models/scimilarity/ 

inspect:
	parquet-tools inspect web/public/models/scimilarity/mappings.parquet
	parquet-tools inspect web/public/models/scimilarity/mappings.parquet