<div align="center">

<h1>hpdex</h1>

<p><em>High-performance differential expression analysis for single-cell data</em></p>

<p>
  <a href="https://pypi.org/project/hpdex/"><img src="https://img.shields.io/pypi/v/hpdex.svg?color=blue" alt="PyPI version" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License" /></a>
  <img src="https://img.shields.io/badge/status-stable-brightgreen" alt="Status: stable" />
  <img src="https://img.shields.io/badge/backend-C%2B%2B%20%2B%20multithreaded-1f6feb" alt="C++ backend" />
</p>

<p>
  <a href="#-overview">Overview</a> ·
  <a href="#-installation">Installation</a> ·
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-api-reference">API</a> ·
  <a href="#-statistical-kernels">Kernels</a> ·
  <a href="#-benchmarks">Benchmarks</a> ·
  <a href="#-faq">FAQ</a> ·
  <a href="#-license">License</a>
</p>

</div>

---

## 🔎 Overview

**hpdex** now ships a **C++ backend** with careful memory layout and **true multithreading**, delivering large speedups for Mann–Whitney U–based DE on single-cell matrices:

* ⚡ **Sparse CSC**: in DE scenarios (gene × group pairs), the **pure Mann–Whitney U kernel** is often **hundreds of times faster** than SciPy when multi-threaded.
* 🧩 **End-to-end DE** (`parallel_differential_expression`): with scheduling and reuse, typical pipelines are **2–3 orders of magnitude faster** than pdex-style Python loops.
* 📦 **Dense**: even single-thread is commonly **\~3× faster** than SciPy on dense inputs; multi-thread scales further.
* 📈 **Statistical parity**: tie-aware U, continuity correction, and asymptotic p-values match `scipy.stats.mannwhitneyu` under equivalent settings (within float tolerance).

Key capabilities:

* 🧵 C++ multithreading, per-thread workspaces, low allocator pressure
* 🧮 Float & histogram kernels (auto-selected by data type)
* 🧠 Tie & continuity corrections; exact or asymptotic p-values in backend
* 🧱 Robust sparse support (CSC highly recommended)
* 🧰 Drop-in friendly design & clean Python API

---

## ⚙️ Installation

### From PyPI

```bash
pip install hpdex
```

### From source

```bash
git clone https://github.com/AI4Cell/hpdex.git
cd hpdex
pip install -e .
# or: uv sync
```

<details>
<summary><strong>Requirements</strong></summary>

* Python ≥ 3.10
* <code>numpy</code>, <code>scipy</code>, <code>pandas</code>, <code>anndata</code>, <code>tqdm</code>
* Building from source requires a C++17 compiler; OpenMP recommended for threading

</details>

---

## 🚀 Quick Start

```python
import anndata as ad
from hpdex import parallel_differential_expression

adata = ad.read_h5ad("data.h5ad")

df = parallel_differential_expression(
    adata,
    groupby_key="perturbation",
    reference="control",
    threads=8,                 # C++ multithreading
    tie_correction=True,
    use_continuity=True,       # continuity correction for U -> Z
    show_progress=True,
)

df.to_csv("de_results.csv", index=False)
```

**Output schema** (DataFrame):

| column             | description                           |
| ------------------ | ------------------------------------- |
| `target`           | target group name                     |
| `feature`          | gene / feature id                     |
| `p_value`          | two-sided p-value from Mann–Whitney U |
| `u_statistic`      | U₁ statistic (reference vs target)    |
| `fold_change`      | mean(target) / mean(reference)        |
| `log2_fold_change` | `log2(fold_change)`                   |
| `fdr`              | BH-adjusted p-value                   |

---

## 📚 API Reference

### `parallel_differential_expression`

```python
parallel_differential_expression(
    adata: ad.AnnData,
    groupby_key: str,
    reference: str | None,          # None -> treated as "non-targeting" baseline
    groups: list[str] | None = None,
    metric: str = "wilcoxon",       # currently only "wilcoxon" (Mann–Whitney U)
    tie_correction: bool = True,
    use_continuity: bool = True,
    min_samples: int = 2,
    threads: int = -1,              # -1 -> use all logical cores
    clip_value: float = 20.0,
    show_progress: bool = True,
) -> pd.DataFrame
```

**Notes**

* `threads` controls the **C++ backend** concurrency.
* `reference=None` creates a “non-targeting” baseline for missing labels.
* Internally the matrix is converted to **CSC** (`scipy.sparse.csc_matrix`) for best performance.
* The backend currently defaults to **zero-handling = "min"** (zeros at head), suitable for UMI counts.

### Low-level backend (advanced)

```python
from hpdex.backend import mannwhitneyu, group_mean

U1, P = mannwhitneyu(
    matrix_csc,           # scipy.sparse.csc_matrix or ndarray (dense handled internally)
    group_id_int32,       # shape: (n_cells,), 0=reference, 1..G-1=targets, -1=ignored
    n_targets: int,
    ref_sorted=False,     # set True if your ref slice is already sorted
    tar_sorted=False,     # set True if target slices are sorted
    use_continuity=True,
    tie_correction=True,
    zero_handling="min",  # "none" | "min" | "max" | "mix"
    threads=-1,
    show_progress=False,
)

means = group_mean(
    matrix_csc,
    group_id_int32,
    G,                    # total groups (reference + targets)
    include_zeros=True,
    threads=-1,
)  # returns array shape: (G, n_genes)
```

---

## 🧪 Statistical Kernels

**Float kernel**

* Merge-rank U with tie tracking; contiguous slices; vector-friendly
* Memory: `O(n)` per slice; scales well with threads

**Histogram kernel**

* For integer/UMI counts with limited range
* Bucketized ranks reduce sorting; memory `O(bins)` (≪ n)

**Common**

* Exact/asymptotic p-values in backend
* Tie & continuity corrections
* Batch-wise scheduling across (gene × group)

---

## 📈 Benchmarks

Typical observations (indicative; varies by CPU, threads, sparsity):

| Scenario                                 | Matrix   | Speedup vs SciPy |
| ---------------------------------------- | -------- | ---------------- |
| **Dense** single-thread U per gene       | dense    | \~3×             |
| **Sparse** multi-thread U (DE setting)   | CSC, 64T | 100×–300×        |
| **End-to-end DE** vs pdex-style baseline | CSC, 64T | 10²–10³×         |

**Why fast?** Thread-local workspaces (no per-column allocations), cache-friendly slices, zero-aware rank merge, and reduced Python overhead.

---

## 🧷 Testing

```bash
cd test && cat bench_mwu.txt | xargs python bench_mwu.py
```

---

## ❓ FAQ

<details>
<summary><strong>Do you correct for multiple testing?</strong></summary>
Yes. The output <code>fdr</code> applies Benjamini–Hochberg (BH).
</details>

<details>
<summary><strong>Why are p-values sometimes ~0?</strong></summary>
Very large samples and strong effects can underflow in float. This is expected; prefer <code>fdr</code> for decisions.
</details>

<details>
<summary><strong>How are zeros handled?</strong></summary>
Backend supports <code>"none"</code>, <code>"min"</code>, <code>"max"</code>, <code>"mix"</code>. The high-level API currently uses <code>"min"</code> by default, which is suitable for UMI counts.
</details>

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
  <sub>Built for large-scale single-cell perturbation analysis, now powered by a C++ backend.</sub>
</div>