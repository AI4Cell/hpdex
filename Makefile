# ==============================================
# HPDEx C++ 项目 Makefile
# 自动管理 CMake 构建和 LibTorch 环境
# ==============================================

# 颜色定义
RED    := \033[0;31m
GREEN  := \033[0;32m
YELLOW := \033[0;33m
BLUE   := \033[0;34m
PURPLE := \033[0;35m
CYAN   := \033[0;36m
RESET  := \033[0m

# 基础配置
PROJECT_NAME := hpdex_cpp
BUILD_TYPE   ?= Release
BUILD_DIR    := build
JOBS         ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# 选项
USE_OPENMP        ?= ON
USE_HIGHWAY       ?= ON
FAST_NORM         ?= ON

# Python 相关
PYTHON := python3
VENV_DIR := venv

# LibTorch 相关
LIBTORCH_VERSION := 2.5.1
LIBTORCH_SEARCH_PATHS := $(wildcard $(HOME)/local/libtorch_*_*/libtorch)
ifdef LIBTORCH_ROOT
	CMAKE_PREFIX_PATH := $(LIBTORCH_ROOT)
else ifneq ($(LIBTORCH_SEARCH_PATHS),)
	# 使用最新版本
	CMAKE_PREFIX_PATH := $(lastword $(sort $(LIBTORCH_SEARCH_PATHS)))
	export LIBTORCH_ROOT := $(CMAKE_PREFIX_PATH)
endif

# CMake 参数
CMAKE_FLAGS := \
	-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	-DUSE_CUDA=$(USE_CUDA) \
	-DUSE_MKL=$(USE_MKL) \
	-DUSE_OPENMP=$(USE_OPENMP) \
	-DUSE_HIGHWAY=$(USE_HIGHWAY) \
	-DBUILD_TESTS=$(BUILD_TESTS) \
	-DVERBOSE_CONFIG=$(VERBOSE_CONFIG) \
	-DUSE_SYSTEM_TORCH=$(USE_SYSTEM_TORCH)

ifdef CMAKE_PREFIX_PATH
	CMAKE_FLAGS += -DCMAKE_PREFIX_PATH="$(CMAKE_PREFIX_PATH)"
endif

# 默认目标
.PHONY: all
all: build

# ==============================================
# 主要构建目标
# ==============================================

.PHONY: help
help:
	@echo "$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	@echo "$(CYAN)       HPDEx C++ 项目 Makefile$(RESET)"
	@echo "$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	@echo ""
	@echo "$(GREEN)主要目标:$(RESET)"
	@echo "  $(YELLOW)make build$(RESET)        - 构建项目（默认）"
	@echo "  $(YELLOW)make clean$(RESET)        - 清理构建目录"
	@echo "  $(YELLOW)make install$(RESET)      - 安装到 Python 环境"
	@echo "  $(YELLOW)make test$(RESET)         - 运行测试"
	@echo "  $(YELLOW)make rebuild$(RESET)      - 清理并重新构建"
	@echo ""
	@echo "$(GREEN)环境设置:$(RESET)"
	@echo "  $(YELLOW)make setup$(RESET)        - 完整环境设置"
	@echo "  $(YELLOW)make install-deps$(RESET) - 安装依赖"
	@echo "  $(YELLOW)make install-torch$(RESET)- 安装 LibTorch"
	@echo "  $(YELLOW)make venv$(RESET)         - 创建虚拟环境"
	@echo ""
	@echo "$(GREEN)构建变体:$(RESET)"
	@echo "  $(YELLOW)make debug$(RESET)        - Debug 构建"
	@echo "  $(YELLOW)make release$(RESET)      - Release 构建（默认）"
	@echo "  $(YELLOW)make cuda$(RESET)         - 启用 CUDA 构建"
	@echo "  $(YELLOW)make simd$(RESET)         - 启用 SIMD 优化构建"
	@echo "  $(YELLOW)make no-simd$(RESET)      - 禁用 SIMD 优化构建"
	@echo ""
	@echo "$(GREEN)开发工具:$(RESET)"
	@echo "  $(YELLOW)make format$(RESET)       - 格式化代码"
	@echo "  $(YELLOW)make check$(RESET)        - 静态检查"
	@echo "  $(YELLOW)make dev$(RESET)          - 开发模式安装"
	@echo ""
	@echo "$(GREEN)配置选项:$(RESET)"
	@echo "  BUILD_TYPE=$(BUILD_TYPE) (Debug/Release/RelWithDebInfo)"
	@echo "  USE_CUDA=$(USE_CUDA)"
	@echo "  USE_MKL=$(USE_MKL)"
	@echo "  USE_OPENMP=$(USE_OPENMP)"
	@echo "  USE_HIGHWAY=$(USE_HIGHWAY)"
	@echo "  BUILD_TESTS=$(BUILD_TESTS)"
	@echo "  JOBS=$(JOBS)"
	@echo ""
	@echo "$(GREEN)当前 LibTorch:$(RESET)"
	@if [ -n "$(CMAKE_PREFIX_PATH)" ]; then \
		echo "  $(BLUE)$(CMAKE_PREFIX_PATH)$(RESET)"; \
	else \
		echo "  $(RED)未找到 (运行 'make install-torch')$(RESET)"; \
	fi
	@echo "$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"

.PHONY: configure
configure:
	@echo "$(BLUE)[配置]$(RESET) 生成构建文件..."
	@if [ -z "$(CMAKE_PREFIX_PATH)" ]; then \
		echo "$(YELLOW)[警告]$(RESET) 未设置 LIBTORCH_ROOT，尝试自动检测..."; \
	fi
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. $(CMAKE_FLAGS)

.PHONY: build
build: check-torch configure
	@echo "$(BLUE)[构建]$(RESET) 编译项目 ($(BUILD_TYPE) 模式, $(JOBS) 线程)..."
	@cmake --build $(BUILD_DIR) -j $(JOBS)
	@echo "$(GREEN)[完成]$(RESET) 构建成功！"

.PHONY: clean
clean:
	@echo "$(BLUE)[清理]$(RESET) 删除构建目录..."
	@rm -rf $(BUILD_DIR)
	@rm -rf __pycache__ .pytest_cache
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)[完成]$(RESET) 清理完成"

.PHONY: rebuild
rebuild: clean build

.PHONY: install
install: build
	@echo "$(BLUE)[安装]$(RESET) 安装到 Python 环境..."
	@cd $(BUILD_DIR) && cmake --install . --config $(BUILD_TYPE)
	@echo "$(GREEN)[完成]$(RESET) 安装成功"

# ==============================================
# 构建变体
# ==============================================

.PHONY: debug
debug:
	@$(MAKE) BUILD_TYPE=Debug build

.PHONY: release
release:
	@$(MAKE) BUILD_TYPE=Release build

.PHONY: relwithdebinfo
relwithdebinfo:
	@$(MAKE) BUILD_TYPE=RelWithDebInfo build

.PHONY: cuda
cuda:
	@$(MAKE) USE_CUDA=ON build

.PHONY: cuda-debug
cuda-debug:
	@$(MAKE) USE_CUDA=ON BUILD_TYPE=Debug build

.PHONY: simd
simd:
	@$(MAKE) USE_HIGHWAY=ON USE_OPENMP=ON build

.PHONY: no-simd
no-simd:
	@$(MAKE) USE_HIGHWAY=OFF USE_OPENMP=OFF build

# ==============================================
# 环境设置
# ==============================================

.PHONY: setup
setup: install-deps install-torch configure
	@echo "$(GREEN)[完成]$(RESET) 环境设置完成！"
	@echo "$(CYAN)下一步：运行 'make build' 构建项目$(RESET)"

.PHONY: install-deps
install-deps:
	@echo "$(BLUE)[依赖]$(RESET) 安装系统依赖..."
	@if [ "$$(uname)" = "Darwin" ]; then \
		if command -v brew >/dev/null 2>&1; then \
			brew list eigen >/dev/null 2>&1 || brew install eigen; \
			brew list pybind11 >/dev/null 2>&1 || brew install pybind11; \
			brew list cmake >/dev/null 2>&1 || brew install cmake; \
		else \
			echo "$(RED)[错误]$(RESET) 未找到 Homebrew，请先安装"; \
			exit 1; \
		fi \
	elif [ "$$(uname)" = "Linux" ]; then \
		if command -v apt-get >/dev/null 2>&1; then \
			sudo apt-get update && sudo apt-get install -y \
				build-essential cmake libeigen3-dev pybind11-dev; \
		elif command -v yum >/dev/null 2>&1; then \
			sudo yum install -y gcc-c++ cmake eigen3-devel python3-pybind11; \
		elif command -v pacman >/dev/null 2>&1; then \
			sudo pacman -S --needed base-devel cmake eigen pybind11; \
		fi \
	fi
	@echo "$(BLUE)[依赖]$(RESET) 安装 Python 依赖..."
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install torch pybind11 numpy pytest

.PHONY: install-torch
install-torch:
	@echo "$(BLUE)[LibTorch]$(RESET) 安装 LibTorch $(LIBTORCH_VERSION)..."
	@if [ -f install_libtorch.sh ]; then \
		bash install_libtorch.sh; \
	else \
		echo "$(RED)[错误]$(RESET) 未找到 install_libtorch.sh"; \
		exit 1; \
	fi
	@echo "$(GREEN)[完成]$(RESET) LibTorch 安装完成"
	@echo "$(YELLOW)[提示]$(RESET) 运行以下命令加载环境："
	@echo "  source ~/local/libtorch_*/env_libtorch.sh"

.PHONY: check-torch
check-torch:
	@if [ -z "$(CMAKE_PREFIX_PATH)" ]; then \
		echo "$(RED)[错误]$(RESET) 未找到 LibTorch"; \
		echo "$(YELLOW)[提示]$(RESET) 请运行以下命令之一："; \
		echo "  1. make install-torch  # 安装 LibTorch"; \
		echo "  2. source ~/local/libtorch_*/env_libtorch.sh  # 加载已安装的"; \
		echo "  3. export LIBTORCH_ROOT=/path/to/libtorch  # 手动设置"; \
		exit 1; \
	fi

# ==============================================
# 虚拟环境
# ==============================================

.PHONY: venv
venv:
	@echo "$(BLUE)[环境]$(RESET) 创建 Python 虚拟环境..."
	@$(PYTHON) -m venv $(VENV_DIR)
	@echo "$(GREEN)[完成]$(RESET) 虚拟环境创建成功"
	@echo "$(YELLOW)[提示]$(RESET) 激活环境: source $(VENV_DIR)/bin/activate"

.PHONY: venv-install
venv-install: venv
	@echo "$(BLUE)[环境]$(RESET) 在虚拟环境中安装依赖..."
	@. $(VENV_DIR)/bin/activate && $(PYTHON) -m pip install --upgrade pip
	@. $(VENV_DIR)/bin/activate && $(PYTHON) -m pip install torch pybind11 numpy pytest
	@. $(VENV_DIR)/bin/activate && $(MAKE) build install

# ==============================================
# 测试
# ==============================================

.PHONY: test
test:
	@echo "$(BLUE)[测试]$(RESET) 运行单元测试..."
	@if [ "$(BUILD_TESTS)" = "OFF" ]; then \
		echo "$(YELLOW)[提示]$(RESET) 重新构建并启用测试..."; \
		$(MAKE) BUILD_TESTS=ON build; \
	fi
	@cd $(BUILD_DIR) && ctest --output-on-failure -j $(JOBS)

.PHONY: test-python
test-python:
	@echo "$(BLUE)[测试]$(RESET) 运行 Python 测试..."
	@$(PYTHON) -m pytest tests/ -v

.PHONY: test-import
test-import:
	@echo "$(BLUE)[测试]$(RESET) 测试模块导入..."
	@$(PYTHON) -c "import $(PROJECT_NAME); print('✓ 模块导入成功')"

# ==============================================
# 开发工具
# ==============================================

.PHONY: dev
dev:
	@echo "$(BLUE)[开发]$(RESET) 开发模式安装..."
	@$(PYTHON) -m pip install -e .
	@echo "$(GREEN)[完成]$(RESET) 开发模式安装成功"

.PHONY: format
format:
	@echo "$(BLUE)[格式化]$(RESET) 格式化 C++ 代码..."
	@if command -v clang-format >/dev/null 2>&1; then \
		find src -name "*.cpp" -o -name "*.hpp" -o -name "*.h" | \
			xargs clang-format -i -style=file; \
		echo "$(GREEN)[完成]$(RESET) 代码格式化完成"; \
	else \
		echo "$(YELLOW)[警告]$(RESET) 未找到 clang-format"; \
	fi

.PHONY: check
check:
	@echo "$(BLUE)[检查]$(RESET) 运行静态分析..."
	@if command -v cppcheck >/dev/null 2>&1; then \
		cppcheck --enable=all --suppress=missingIncludeSystem \
			--inline-suppr --quiet --error-exitcode=1 \
			src/ 2>&1 | tee cppcheck.log; \
	else \
		echo "$(YELLOW)[警告]$(RESET) 未找到 cppcheck"; \
	fi
	@if command -v clang-tidy >/dev/null 2>&1 && [ -f $(BUILD_DIR)/compile_commands.json ]; then \
		clang-tidy src/**/*.cpp -p $(BUILD_DIR); \
	fi

.PHONY: bench
bench: build
	@echo "$(BLUE)[基准]$(RESET) 运行性能基准测试..."
	@if [ -f benchmarks/run_benchmarks.py ]; then \
		$(PYTHON) benchmarks/run_benchmarks.py; \
	else \
		echo "$(YELLOW)[警告]$(RESET) 未找到基准测试脚本"; \
	fi

# ==============================================
# 信息显示
# ==============================================

.PHONY: info
info:
	@echo "$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	@echo "$(CYAN)系统信息:$(RESET)"
	@echo "  OS: $$(uname -s) $$(uname -r)"
	@echo "  架构: $$(uname -m)"
	@echo "  CPU 核心: $(JOBS)"
	@echo ""
	@echo "$(CYAN)编译器:$(RESET)"
	@if command -v gcc >/dev/null 2>&1; then \
		echo "  GCC: $$(gcc --version | head -1)"; \
	fi
	@if command -v clang >/dev/null 2>&1; then \
		echo "  Clang: $$(clang --version | head -1)"; \
	fi
	@echo "  CMake: $$(cmake --version | head -1)"
	@echo ""
	@echo "$(CYAN)Python:$(RESET)"
	@echo "  版本: $$($(PYTHON) --version)"
	@echo "  路径: $$(which $(PYTHON))"
	@echo ""
	@echo "$(CYAN)LibTorch:$(RESET)"
	@if [ -n "$(CMAKE_PREFIX_PATH)" ]; then \
		echo "  路径: $(CMAKE_PREFIX_PATH)"; \
		if [ -f "$(CMAKE_PREFIX_PATH)/build-version" ]; then \
			echo "  版本: $$(cat $(CMAKE_PREFIX_PATH)/build-version)"; \
		fi \
	else \
		echo "  $(RED)未设置$(RESET)"; \
	fi
	@echo "$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"

.PHONY: torch-info
torch-info:
	@echo "$(CYAN)LibTorch 信息:$(RESET)"
	@$(PYTHON) -c "import torch; \
		print(f'PyTorch 版本: {torch.__version__}'); \
		print(f'CUDA 可用: {torch.cuda.is_available()}'); \
		print(f'CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); \
		print(f'cuDNN 版本: {torch.backends.cudnn.version() if torch.cuda.is_available() else \"N/A\"}')"

# ==============================================
# CI/CD 目标
# ==============================================

.PHONY: ci
ci: clean
	@echo "$(BLUE)[CI]$(RESET) 运行 CI 构建..."
	@$(MAKE) BUILD_TESTS=ON build
	@$(MAKE) test
	@$(MAKE) check

.PHONY: package
package: build
	@echo "$(BLUE)[打包]$(RESET) 创建发布包..."
	@cd $(BUILD_DIR) && cpack -G TGZ
	@echo "$(GREEN)[完成]$(RESET) 打包完成"

# 防止删除中间文件
.PRECIOUS: $(BUILD_DIR)/CMakeCache.txt

# 设置默认目标
.DEFAULT_GOAL := help
