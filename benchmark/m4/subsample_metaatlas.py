#!/usr/bin/env python
# Deterministic without-replacement subsample of the 599k metaatlas (the SCimilarity reference
# source) into a query h5ad of N cells. Uses anndata backed mode so the 10GB file isn't fully loaded.
#   subsample_metaatlas.py N [out.h5ad]
import sys, anndata as ad, numpy as np
SRC = "/Users/rcurrie/data/h5ad/adata_metaatlas_final_raw.h5ad"
N = int(sys.argv[1])
OUT = sys.argv[2] if len(sys.argv) > 2 else f"/Users/rcurrie/data/cytoverse.main/metaatlas_cells_{N}.h5ad"
a = ad.read_h5ad(SRC, backed="r")
n = a.n_obs
idx = np.sort(np.random.default_rng(42).choice(n, size=N, replace=False))
sub = a[idx].to_memory()
# ensure float32 X, drop heavy obsm/uns to keep the query lean (only X + var genes matter)
import scipy.sparse as sp
if sp.issparse(sub.X):
    sub.X = sub.X.astype(np.float32)
sub.obsm.clear(); sub.varm.clear(); sub.uns.clear(); sub.obsp.clear(); sub.varp.clear()
sub.write(OUT)
print(f"wrote {OUT}: {sub.n_obs} cells x {sub.n_vars} genes, X dtype {sub.X.dtype}")
