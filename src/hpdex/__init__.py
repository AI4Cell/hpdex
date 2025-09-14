from .backen import parallel_differential_expression
from .stream import \
	parallel_differential_expression as parallel_differential_expression_stream

# Operator exports
from .operator import (
	wilcoxon_dense,
	wilcoxon_hist_dense,
	wilcoxon_all_csc,
	wilcoxon_sorted_ptr,
	wilcoxon_ref_sorted_csc,
)

__version__ = "0.2.0"

__all__ = [
	"parallel_differential_expression",
	"parallel_differential_expression_stream",
	"wilcoxon_dense",
	"wilcoxon_hist_dense",
	"wilcoxon_all_csc",
	"wilcoxon_sorted_ptr",
	"wilcoxon_ref_sorted_csc",
]

de_analysis = parallel_differential_expression
streaming_de_analysis = parallel_differential_expression_stream
