# Store the labeler results in a browser innodb with csv export

- In start create a new 'results' innodb. If one already exists clear it and create a new empty database
- Store the current categoryLabels in a table
- As labeled batches come in store them in a testResults table with the following row structure:
  (vectorID, categoryID, x, y)
- Use the idb package for database access and manipulations
- Add a new function, 'exportResultsToCSV' that generates a csv file in memory to be downloaded
- Add a download icon button left of the share button that when pressed calls exportResultsToCSV

# Ray Data/Train/Tune
Create a new python script, tune.py, that utilizes the Ray AIR ecosystem (data, train, tune) to do the following:

Data: Generate a dataset that streams batches from an h5ad file using AnnData and runs them through SCimilarity ce.get_embeddings() yielding batches of embeddings. Support dividing the input into train/validate/test splits stratified by the obs "CellType" label with a 60%/20%/20% ratio. See ingest_h5ad_as_reference for some examples of how this is done, but develop a clean pattern using Ray DataSets and no files saved to disk.

Train: Train a PQ Embedding with hyperpameters pq_m (number of subspaces) and pq_k (number of centroids per subspace). Build and IVF with a hyper parameter of the number of partitions (num_partitions, default to 256). See ivfpq_train.py for examples of how this is done. Only use the train portion of the dataset.  

Tune: The Data and Train portions should generate a populated (in memory) IVFPQ index. Create an objective function that using the validation vectors finds nearest neighbors with two more hyper parameters: number of partitions (n_probe) to look for and top k (probe_top_k) to rank. It should find the k nearest neighbors accross the partitions using asymetric distance (see pq.py). It should then use SCimilarity (see label.py) to find the same number of nearest neighbors and calculate the % overlap between this and the IVFPQ found vectors. WE ultimately want to optimize for higher overlap but while keeping overall memory storage and search time down.

Look through all three steps and carefully plan so that design decisions in the first step antiticipate needs in later steps. Follow canonical Ray structure and keep things as minimal and simple as possible. Goal should be running simple approach that we can validate and enhance. Make assumptions where it can make things simpler (and not them in a comment)

# Tune based on label

Rewrite src/cytoverse/scripts/tune.py taking the following approach to using Ray Tune:
- Add stratify_by parameter, default to "prediction"
- Stratify generate x_train/x_test embeddings stratifying on stratify_by (no more val required)
- Generate y_train/y_test to go with the embeddings. These should be the coordsponding index id of the stratify_by column in the labels.parquet
- Update the open and parsing of the parquet file to use backed so that we only need to read from disk the first max_embeddings to populate train/test

Then rewrite objective to:
- Train pq and ivf given parameters on the train set
- For each of the test samples use pq.compute_asymetric_distances to find the 50 closest train embeddings to the test sample, convert to label indexes and compute the consenus label index
- Update the overlap calculation as the % of test samples where the consensus label index from pq is the same as the actual test label index from the ground truth (ie the respective index in y_test)

The actual parquet files will be: ./data/scimilarity/embeddings.parquet and ./data/scimilarity/labels.parquet
