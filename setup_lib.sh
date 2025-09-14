#!/usr/bin/env bash
set -euo pipefail

# ============================================
# 配置
# ============================================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="${ROOT_DIR}/Lib"
mkdir -p "${LIB_DIR}"

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

echo "===> 系统: ${OS}, 架构: ${ARCH}"
echo "===> 安装目录: ${LIB_DIR}"

# ============================================
# 1. Eigen
# ============================================
install_eigen() {
    echo "===> 安装 Eigen"
    local eigen_ver="3.4.0"
    local url="https://gitlab.com/libeigen/eigen/-/archive/${eigen_ver}/eigen-${eigen_ver}.tar.gz"
    local out="${LIB_DIR}/eigen-${eigen_ver}"
    if [ ! -d "${out}" ]; then
        curl -L "$url" | tar xz -C "${LIB_DIR}"
    fi
    echo "Eigen 已安装到 ${out}"
}

# ============================================
# 2. LibTorch (智能检测 CUDA 并下载对应版本)
# ============================================
install_libtorch() {
    echo "===> 安装 LibTorch"
    local out="${LIB_DIR}/libtorch"
    
    # 检查是否已经存在 LibTorch
    if [ -d "${out}" ]; then
        echo "LibTorch 已存在于 ${out}"
        return 0
    fi
    
    # 检测 CUDA 可用性
    local cuda_available=false
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            cuda_available=true
            echo "✅ 检测到 CUDA 支持"
        fi
    fi
    
    # 方法1: 检查本地是否有 LibTorch 安装
    local local_torch_paths=(
        "/Users/wzq/local/libtorch_2.5.1_cpu/libtorch"
        "/Users/wzq/local/libtorch_2.5.1_cuda/libtorch"
        "${HOME}/local/libtorch_2.5.1_cpu/libtorch"
        "${HOME}/local/libtorch_2.5.1_cuda/libtorch"
        "/usr/local/libtorch"
        "/opt/libtorch"
    )
    
    for local_path in "${local_torch_paths[@]}"; do
        if [ -d "${local_path}" ]; then
            echo "发现本地 LibTorch 安装: ${local_path}"
            ln -sf "${local_path}" "${out}"
            echo "LibTorch 已链接到 ${out}"
            return 0
        fi
    done
    
    # 方法2: 使用 conda 安装 (如果可用)
    if command -v conda &> /dev/null; then
        echo "使用 conda 安装 LibTorch..."
        local conda_package="pytorch-cpu"
        if [ "$cuda_available" = true ] && [[ "${OS}" != "darwin" ]]; then
            conda_package="pytorch"
            echo "安装 CUDA 版本的 PyTorch"
        else
            echo "安装 CPU 版本的 PyTorch"
        fi
        
        if conda install -c pytorch "$conda_package" -y; then
            # 找到 conda 安装的 LibTorch 路径
            local conda_torch_path=$(conda info --envs | grep -E '^\*' | awk '{print $NF}')/lib/python*/site-packages/torch
            if [ -d "${conda_torch_path}" ]; then
                # 创建 libtorch 目录结构
                mkdir -p "${out}"
                ln -sf "${conda_torch_path}/lib" "${out}/lib"
                ln -sf "${conda_torch_path}/include" "${out}/include"
                ln -sf "${conda_torch_path}/share" "${out}/share"
                echo "LibTorch 已通过 conda 安装到 ${out}"
                return 0
            fi
        fi
    fi
    
    # 方法3: 使用 pip 安装 (如果可用)
    if command -v pip &> /dev/null; then
        echo "使用 pip 安装 LibTorch..."
        local pip_index="https://download.pytorch.org/whl/cpu"
        if [ "$cuda_available" = true ] && [[ "${OS}" != "darwin" ]]; then
            pip_index="https://download.pytorch.org/whl/cu121"
            echo "安装 CUDA 版本的 PyTorch"
        else
            echo "安装 CPU 版本的 PyTorch"
        fi
        
        if pip install torch torchvision torchaudio --index-url "$pip_index"; then
            # 找到 pip 安装的 LibTorch 路径
            local pip_torch_path=$(python -c "import torch; print(torch.__path__[0])" 2>/dev/null)
            if [ -d "${pip_torch_path}" ]; then
                # 创建 libtorch 目录结构
                mkdir -p "${out}"
                ln -sf "${pip_torch_path}/lib" "${out}/lib"
                ln -sf "${pip_torch_path}/include" "${out}/include"
                ln -sf "${pip_torch_path}/share" "${out}/share"
                echo "LibTorch 已通过 pip 安装到 ${out}"
                return 0
            fi
        fi
    fi
    
    # 方法4: 手动下载 LibTorch
    echo "尝试手动下载 LibTorch..."
    local torch_ver="2.5.0"
    local url=""
    
    if [[ "${OS}" == "darwin" ]]; then
        echo "⚠️ macOS 不支持 CUDA，下载 CPU 版本"
        url="https://download.pytorch.org/libtorch/cpu/libtorch-macos-${torch_ver}.zip"
    elif [[ "${OS}" == "linux" ]]; then
        if [ "$cuda_available" = true ]; then
            echo "🚀 下载 CUDA 12.1 版本"
            url="https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-${torch_ver}%2Bcu121.zip"
        else
            echo "💻 下载 CPU 版本"
            url="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${torch_ver}.zip"
        fi
    elif [[ "${OS}" =~ (mingw|msys|cygwin) ]]; then
        if [ "$cuda_available" = true ]; then
            echo "🚀 下载 CUDA 12.1 版本"
            url="https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-${torch_ver}%2Bcu121.zip"
        else
            echo "💻 下载 CPU 版本"
            url="https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${torch_ver}.zip"
        fi
    else
        echo "不支持的系统: ${OS}"
        exit 1
    fi
    
    tmp_zip="${LIB_DIR}/libtorch.zip"
    echo "下载: $url"
    
    if curl -L --fail --silent --show-error "$url" -o "$tmp_zip"; then
        if file "$tmp_zip" | grep -q "Zip archive"; then
            unzip -q "$tmp_zip" -d "${LIB_DIR}"
            rm "$tmp_zip"
            echo "LibTorch 已下载并安装到 ${out}"
            return 0
        fi
    fi
    
    echo "❌ 所有安装方法都失败了"
    echo "请手动安装 LibTorch:"
    echo "1. 访问 https://pytorch.org/get-started/locally/"
    if [ "$cuda_available" = true ]; then
        echo "2. 选择 CUDA 版本和您的操作系统"
    else
        echo "2. 选择 CPU 版本和您的操作系统"
    fi
    echo "3. 按照官方说明安装"
    echo "4. 或者将 LibTorch 解压到 ${out}"
    exit 1
}

# ============================================
# 3. Highway
# ============================================
install_highway() {
    echo "===> 安装 Highway"
    local repo="https://github.com/google/highway.git"
    local out="${LIB_DIR}/highway"
    if [ ! -d "${out}" ]; then
        git clone --depth 1 "$repo" "$out"
    else
        (cd "$out" && git pull)
    fi
    echo "Highway 已安装到 ${out}"
}

# ============================================
# 4. OpenMP (macOS / Linux)
# ============================================
install_openmp() {
    echo "===> 检查 OpenMP"
    if [[ "${OS}" == "darwin" ]]; then
        if command -v brew &>/dev/null; then
            brew install libomp
        else
            echo "请先安装 Homebrew: https://brew.sh/"
            exit 1
        fi
    elif [[ "${OS}" == "linux" ]]; then
        if command -v apt &>/dev/null; then
            sudo apt-get update
            sudo apt-get install -y libomp-dev
        elif command -v yum &>/dev/null; then
            sudo yum install -y libomp-devel
        fi
    else
        echo "Windows 下 OpenMP 由 MSVC 自带，无需安装"
    fi
}

# ============================================
# 5. pybind11 (uv 管理)
# ============================================
install_pybind11() {
    echo "===> 安装 pybind11 (通过 uv)"
    uv add pybind11
}

# ============================================
# 执行安装
# ============================================
install_eigen
install_libtorch
install_highway
install_openmp
install_pybind11

echo ""
echo "===> 所有依赖安装完成！"
echo "你可以在 CMakeLists.txt 中添加："
echo "  set(CMAKE_PREFIX_PATH \"${LIB_DIR}/libtorch\")"
echo "  include_directories(${LIB_DIR}/eigen-3.4.0)"
echo "  include_directories(${LIB_DIR}/highway)"
