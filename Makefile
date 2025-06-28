subset:
	python src/cellspace/scripts/scimilarity_export_cells.py \
	data/scimilarity/model_v1.1/cellsearch \
	web/public/models/scimilarity \
	--labels tissue \
	--labels prediction \
	--labels study \
	--num-cells 100000

train:
	python src/cellspace/scripts/mapper.py train \
        web/public/models/scimilarity/embeddings.npy \
        web/public/models/scimilarity/ \
        --num-embeddings 100000

map:
	python src/cellspace/scripts/mapper.py map \
        web/public/models/scimilarity/mapper.onnx \
        web/public/models/scimilarity/embeddings.npy \
        web/public/models/scimilarity/ \
        --num-embeddings 100000