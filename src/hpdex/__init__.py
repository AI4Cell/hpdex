from .backen import parallel_differential_expression
from .stream import \
    parallel_differential_expression as parallel_differential_expression_stream

__version__ = "0.2.0"

__all__ = [
    "parallel_differential_expression",
    "parallel_differential_expression_stream"
]

de_analysis = parallel_differential_expression
streaming_de_analysis = parallel_differential_expression_stream
