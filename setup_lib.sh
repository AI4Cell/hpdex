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
# 1. Highway
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
# 2. OpenMP (macOS / Linux)
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
# 3. pybind11 (uv 管理)
# ============================================
install_pybind11() {
    echo "===> 安装 pybind11 (通过 uv)"
    uv add pybind11
}

# ============================================
# 执行安装
# ============================================
install_highway
install_openmp
install_pybind11

echo ""
echo "===> 所有依赖安装完成！"
echo "你可以在 CMakeLists.txt 中添加："
echo "  include_directories(${LIB_DIR}/highway)"
