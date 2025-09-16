from hpdex.de import parallel_differential_expression
import scanpy as sc

print("Loading data...")
adata = sc.read_h5ad("/ssdwork/zhoujingbo/Datasets/vcc_data/adata_Training.h5ad")
sc.pp.normalize_total(adata, target_sum=1e4)

result = parallel_differential_expression(
    adata,
    groupby_key="target_gene",
    reference="non-targeting",
    groups=None,
    threads=1000,
    clip_value=2000,
    show_progress=True
)

result.to_csv("result.csv", index=False)
print(result.head())
print("Result saved to result.csv")