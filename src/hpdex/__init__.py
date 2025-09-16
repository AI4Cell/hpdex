from .de import parallel_differential_expression
from .backend import parallel_differential_expression_numba as pden
from .stream import \
	parallel_differential_expression as parallel_differential_expression_stream

pde = parallel_differential_expression
 
__version__ = "1.0.0"

__all__ = [
	"parallel_differential_expression",
	"parallel_differential_expression_stream",
	"pden",
	"pde",
]