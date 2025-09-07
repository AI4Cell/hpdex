<div align="center">

<!-- Optional logo -->

<!-- <img src=".github/assets/hpdex-logo.svg" width="96" alt="hpdex logo" /> -->

<h1>hpdex</h1>

<p><em>High‑performance differential expression analysis for single‑cell data</em></p>

<p>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License" /></a>
  <img src="https://img.shields.io/badge/status-experimental-purple" alt="Status: experimental" />
</p>

<p>
  <a href="#-overview">Overview</a> ·
  <a href="#-installation">Installation</a> ·
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-api-reference">API</a> ·
  <a href="#-statistical-kernels">Kernels</a> ·
  <a href="#-testing">Testing</a> ·
  <a href="#-faq">FAQ</a> ·
  <a href="#-license">License</a>
</p>

</div>

---

## 🔎 Overview

**hpdex** provides efficient differential expression (DE) analysis for single‑cell data using **multiprocessing** and an optimized **Mann–Whitney U** implementation. It aims to be *statistically consistent* with `scipy.stats.mannwhitneyu` while scaling to large datasets.

<table>
  <tr>
    <td>⚡️ <strong>Fast</strong><br/><small>Batch processing & shared memory minimize copies.</small></td>
    <td>🧠 <strong>Accurate</strong><br/><small>Tie‑aware U statistics; normal approximation for large <i>n</i>.</small></td>
    <td>🧰 <strong>Versatile</strong><br/><small>Float & histogram kernels auto‑selected by data type.</small></td>
  </tr>
  <tr>
    <td>🧵 <strong>Parallel</strong><br/><small>Simple <code>num_workers</code> control.</small></td>
    <td>💾 <strong>Memory‑savvy</strong><br/><small>Reuses pre‑sorted references across comparisons.</small></td>
    <td>📊 <strong>Streaming</strong><br/><small>Handles datasets larger than RAM via chunking.</small></td>
  </tr>
</table>

---

## ⚙️ Installation

### Quick Install (coming soon)

```bash
pip install hpdex
```

### uv (recommended)

```bash
git clone https://github.com/AI4Cell/hpdex.git
cd hpdex
uv sync
```

### pip (from source)

```bash
git clone https://github.com/AI4Cell/hpdex.git
cd hpdex
pip install -e .
```

<details>
<summary><strong>Requirements</strong></summary>

* Python ≥ 3.10
* <code>numpy</code>, <code>scipy</code>, <code>numba</code>, <code>pandas</code>, <code>anndata</code>

</details>

---

## 🚀 Quick Start

```python
import anndata as ad
from hpdex import parallel_differential_expression

# Load your data
adata = ad.read_h5ad("data.h5ad")

# Run differential expression analysis
results = parallel_differential_expression(
    adata,
    groupby_key="perturbation",
    reference="control",
    num_workers=4,
)

# Save results
results.to_csv("de_results.csv", index=False)
```

**Output schema** (DataFrame columns):

| column             | description                              |
| ------------------ | ---------------------------------------- |
| `target`           | target group name                        |
| `feature`          | gene / feature id                        |
| `p_value`          | (two‑sided) p‑value from Mann–Whitney U  |
| `fold_change`      | mean(target) / mean(reference)           |
| `log2_fold_change` | `log2(fold_change)`                      |
| `fdr`              | BH‑adjusted p‑value (Benjamini–Hochberg) |

---

## 📚 API Reference

### `parallel_differential_expression`

Main entry for DE analysis.

```python
parallel_differential_expression(
    adata: ad.AnnData,
    groupby_key: str,
    reference: str,
    groups: Optional[List[str]] = None,
    metric: str = "wilcoxon",
    tie_correction: bool = True,
    continuity_correction: bool = True,
    use_asymptotic: Optional[bool] = None,
    min_samples: int = 2,
    max_bins: int = 100_000,
    prefer_hist_if_int: bool = False,
    num_workers: int = 1,
    clip_value: float = 20.0,
) -> pd.DataFrame
```

**Parameters**

* `adata` — `AnnData` object containing expression matrix & metadata
* `groupby_key` — column in `adata.obs` for grouping
* `reference` — reference group name (e.g., "control")
* `groups` — optional subset of target groups (auto if `None`)
* `metric` — currently `"wilcoxon"` (Mann–Whitney U)
* `tie_correction` — whether to apply tie correction
* `continuity_correction` — whether to apply continuity correction
* `use_asymptotic` — whether to use asymptotic approximation
* `min_samples` — minimum number of samples per group
* `max_bins` — maximum number of bins for histogram algorithm
* `prefer_hist_if_int` — prefer histogram algorithm for integer data
* `num_workers` — number of worker processes
* `clip_value` — clip fold change if infinite or NaN

> 💡 **Tips**
>
> * For UMI counts, set `prefer_hist_if_int=True` to favor the histogram kernel.
> * Very large samples may produce extremely small `p_value`s due to underflow; rely on `fdr` for decisions.

**Returns** — `pd.DataFrame` (see *Output schema* above)

---

## 🧪 Statistical Kernels

hpdex implements two complementary kernels and auto‑selects by data type.

### Float Kernel

* **Use**: continuous expression (e.g., log‑counts)
* **Alg**: merge‑rank computation for U; Numba JIT; vectorized batches
* **Mem**: `O(n)` working memory for sorted arrays

### Histogram Kernel

* **Use**: integer/discrete counts (e.g., UMI)
* **Alg**: bucketized rank computation; reduces sorting overhead
* **Mem**: `O(bins)` working memory, typically ≪ data size

**Common features**

* Proper **tie handling** and variance correction
* **Asymptotic normal** approximation for large samples
* **Batching** across gene × group pairs
* **Reference re‑use** to save sorting cost

> The kernels aim to match `scipy.stats.mannwhitneyu` numerically under equivalent settings.

---

## 🧷 Testing

See **[test/README.md](test/README.md)** for full docs.

**Quick test**

```bash
cd test
python test.py config_quick.yml
```

**Full suite**

```bash
python test.py config.yml
```

---

## ❓ FAQ

<details>
<summary><strong>Does hpdex correct for multiple testing?</strong></summary>
Yes. The returned <code>fdr</code> column applies Benjamini–Hochberg (BH) adjustment to the raw <code>p_value</code>s.
</details>

<details>
<summary><strong>Why do I see extremely small <code>p_value</code>s (close to 0)?</strong></summary>
For very large samples and strong effects, underflow can make values effectively <code>0.0</code> in float precision. This is expected; rely on <code>fdr</code> for decision making.
</details>

<details>
<summary><strong>When should I prefer the Histogram kernel?</strong></summary>
When the data are integer UMI counts with limited range. It avoids full sorting per target and is usually faster and more memory‑efficient.
</details>

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

<div align="center">
  <sub>Built for large‑scale single‑cell perturbation analysis.</sub>
</div>
