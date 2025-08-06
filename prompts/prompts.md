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

# PyTorch pq.py

We have a kmeans pytorch implementation in kmeans.py with tests in test_kmeans.py and kmeans.test.tx. It is implimented using pytorch and exports onnx models.

Generate a new file, src/pq.py, that implements ProductQuantization. It should have the following pytorch models:

Encode: given an embeddings tensor and a code book it should output the pq encodings of the vectors.

Decode: given a tensor of pq encoded embeddings and codebook it should decode them back into embeddings

Distance: given a query tensor (a pq encoded embedding), a code book, and a reference tensor (a tensor of pq encoded embeddings), and an integer k, return the index of the closest k reference tensors to the query tensor sorted by distance. Calculate the asymetric distance between the query and each of the reference tensors using the codebook to create a distance table for the query. The resulting distances are not exposed and do not need to be any exact distance i.e. euclidean distance itself, it could be squared distance so we don't need to calculate a square root. We'll only be using the index of the top k for further processing.

Also include a utility function that given a tensor of training vectors, PQ parameters m and k, uses the onnx models from kmeans.py to generate the PQ codebook that can be used above. This codebook should be binary as it will be used later in a web application. See web/src/embedding.ts for an example use of an existing implimentation that we're replacing. 

Generate a new tests/test_pq_onnx.py file that tests all of this ala pytest. Assume defaults of 128d embedding vectors, m = 8 and k = 256 for the codebook training and k = 50 for the topk distance calculation.

# Cleanup PQ and support python and onnx kmeans

We currently have two implementations if PQ:
- src/pq.py utilizes src/cytoverse/kmeans.py for training, which is onnx based and has onnx optimized classes for encoding, decoding and importantly distance which we will use in the web application
- src/cytoverse/pq.py is an older version which uses python for kmeans

Related to this src/cytoverse/ivf.py uses the onnx kmeans.py, but as it needs to deal with the full embedding width (128d) vs. the sub-spaces of pq (16d) it takes a very long time.

Finally there is some redundancy between ivfpq.py and ivf.py and the two pq.py

With that as context:
- With the python kmeans implementation in src/cytoverse/pq.py as an example, enhance kmeans.py to support computing using python OR onnx.
- Replace src/cytoverse/pq.py with src/pq.py functionality i.e. the ONNX encode, decode and distance models which are used in the web application.
- Elliminate src/cytoverse/ivfpq.py moving its functionality into ivf.py which then depends on pq.py and kmeans.py
- Update ivfpq_train to use this now consolidated kmeans.py, pq.py and ivf.py. It should have a new command line option, onnx-kmeans, that defaults to false that determines which kmeans will be used. 
- Fold any relevant tests from test_ivfpq.py into test_ivf.py
- Make sure updates to web/src/pq.ts and web/src/ivf.ts from all of this, but there should not be any as all we're changing is the way kmeans is used to generate artifacts that both of these files use.

At the end of this we should have a complete set of packages that:
- Enable training pq on a set of embeddings and emitting a codebook, encode, decode and distance onnx models for use in the browser from typescript
- Enable generating an ivf index leveraging a trained pq model also emitting browser artifacts

# Rewrite IVFPQ to use Residual Vectors

Let's completely rewrite ivf.py and pq.py to use residual vectors:

ivf train:
- Take a tensor of N vectors, run kmeans to generate tensor of k centroids.
- Allocate the residual of each vector to the closest centroid (where residual = vector minus centroid).
- Train a pq model on the residual vectors and export a pq codebook as well as encode and distance onnx models.
- Encode the residual vectors for each partition with the pq codebook and export k partition parquet files with pq residual vectors and the original vector index number (int32).

pq 
- The distance onnx model should take in a query residual (full-precision) and each pq encode residual in a partion and output a sorted list of (vector index, distance). 
- See the current pq.py implementation and note usage of gather elements in onnx to optimize this operation and simplify the emitted distance onnx model.

ivf search:
- Take in a query vector and centroids and find the nearest n_probe partitions (all in full original precision i.e. exact presision)
- For each partition using the pq distance onnx model find the top k closest pq encode residual vectors to the query residual.
- Merge the n_probe distance lists and yield the top k distances and associated original vector index for each

pq/ivf typescript:

Update ivfpq/typescript/src ivf.ts and pq.ts to use the artifacts from all of this python code - as they do currently but potentially any changes to support the residual flow etc...

These final vector index's will be used to lookup annotations associated with the original vectors as well as triage the distances further.

Additional Notes
- Use scikit-learn optimized kmeans for all of this (NOT kmeans.py in this repo) and include tqdm based progress.
- All data artifacts (centroids, pq codebook, partitions) should raw binary .bin files so they can be efficiently pulled over http and loaded as Float32Array (for centroids/codebook), UInt8Array for PQ codes etc...
- Update/replace test_pq.py and test_ivf.py to use the new residual flow.
- Update ivfpq/typescript/tests/ vitest based unit tests to validate the above
