from hpdex.de import parallel_differential_expression
import scanpy as sc

print("Loading data...")
adata = sc.read_h5ad("/Volumes/Wzq/Datasets/scperturb/AdamsonWeissman2016_GSM2406681_10X010.h5ad")
sc.pp.normalize_total(adata, target_sum=1e4)

print("Running differential expression analysis...")
result = parallel_differential_expression(
    adata,
    groupby_key="perturbation",
    reference=None,
    groups=None,
    threads=1000,
    clip_value=2000
)

result.to_csv("result.csv", index=False)
print(result.head())
print("Result saved to result.csv")