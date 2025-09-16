import os, sys

_pkg_dir = os.path.dirname(__file__)
_bin_dir = os.path.join(_pkg_dir, "bin")

if _bin_dir not in sys.path:
    sys.path.insert(0, _bin_dir)

# 惰性导入 C++ 扩展模块
_kernel = None

def __getattr__(name):
    global _kernel
    if name == 'kernel':
        if _kernel is None:
            try:
                import kernel as _kernel_module
                _kernel = _kernel_module
            except ImportError as e:
                raise ImportError(f"Failed to import kernel module: {e}")
        return _kernel
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")