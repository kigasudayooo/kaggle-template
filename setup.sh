#!/bin/bash
#
# Kaggle Competition Project Setup Script
# ========================================
# 使用方法:
#   ./setup.sh                    # 現在のディレクトリでセットアップ
#   ./setup.sh /path/to/project   # 指定ディレクトリでセットアップ
#
# パッケージ管理: pyproject.toml で一元管理
# インストール: uv sync --all-extras
#

set -e

# =============================================================================
# カラー出力
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# =============================================================================
# デフォルト設定
# =============================================================================
DEFAULT_PYTHON_VERSION="3.10"
DEFAULT_PROJECT_NAME="kaggle-competition"

# =============================================================================
# 引数処理
# =============================================================================
PROJECT_DIR="${1:-.}"

if [ "$PROJECT_DIR" != "." ]; then
    mkdir -p "$PROJECT_DIR"
    cd "$PROJECT_DIR"
    info "プロジェクトディレクトリ: $PROJECT_DIR"
fi

# =============================================================================
# 設定読み込み
# =============================================================================
PYTHON_VERSION="$DEFAULT_PYTHON_VERSION"
PROJECT_NAME="$DEFAULT_PROJECT_NAME"

if [ -f "setup_config.json" ]; then
    info "setup_config.json を読み込み中..."

    if command -v jq &> /dev/null; then
        PYTHON_VERSION=$(jq -r '.python_version // "3.10"' setup_config.json)
        PROJECT_NAME=$(jq -r '.project.name // "kaggle-competition"' setup_config.json)
    elif command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c "import json; print(json.load(open('setup_config.json')).get('python_version', '3.10'))")
        PROJECT_NAME=$(python3 -c "import json; print(json.load(open('setup_config.json')).get('project', {}).get('name', 'kaggle-competition'))")
    fi

    success "設定読み込み完了: Python $PYTHON_VERSION, Project: $PROJECT_NAME"
else
    warn "setup_config.json が見つかりません。デフォルト設定を使用します。"
fi

# =============================================================================
# uv インストール確認・インストール
# =============================================================================
install_uv() {
    if command -v uv &> /dev/null; then
        success "uv は既にインストールされています: $(uv --version)"
        return
    fi

    info "uv をインストール中..."

    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi

    export PATH="$HOME/.local/bin:$PATH"
    export PATH="$HOME/.cargo/bin:$PATH"

    if command -v uv &> /dev/null; then
        success "uv インストール完了: $(uv --version)"
    else
        error "uv のインストールに失敗しました"
    fi
}

# =============================================================================
# Task (go-task) インストール確認
# =============================================================================
check_task() {
    if command -v task &> /dev/null; then
        success "Task は既にインストールされています: $(task --version)"
        return
    fi

    warn "Task (go-task) がインストールされていません。"
    echo "  macOS:  brew install go-task"
    echo "  Linux:  sh -c \"\$(curl --location https://taskfile.dev/install.sh)\" -- -d -b ~/.local/bin"
}

# =============================================================================
# ディレクトリ構造作成
# =============================================================================
create_directories() {
    info "ディレクトリ構造を作成中..."

    local dirs=("data" "notebooks" "src" "models" "experiments" "tests" "configs")

    # setup_config.jsonからカスタムディレクトリを読み込み
    if [ -f "setup_config.json" ] && command -v jq &> /dev/null; then
        while IFS= read -r dir; do
            dirs+=("$dir")
        done < <(jq -r '.directories[]? // empty' setup_config.json)
    fi

    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        touch "$dir/.gitkeep" 2>/dev/null || true
    done

    success "ディレクトリ構造作成完了"
}

# =============================================================================
# Git初期化
# =============================================================================
init_git() {
    if [ -d ".git" ]; then
        info "Git リポジトリは既に初期化されています"
        return
    fi

    info "Git リポジトリを初期化中..."
    git init
    success "Git 初期化完了"
}

# =============================================================================
# src/__init__.py作成
# =============================================================================
create_init_files() {
    info "__init__.py を作成中..."
    if [ ! -f "src/__init__.py" ]; then
        touch "src/__init__.py"
    fi
    success "__init__.py 作成完了"
}

# =============================================================================
# Python環境セットアップ
# =============================================================================
setup_python_env() {
    info "Python $PYTHON_VERSION 環境をセットアップ中..."

    uv python install "$PYTHON_VERSION" 2>/dev/null || true
    uv python pin "$PYTHON_VERSION"

    success "Python 環境セットアップ完了"
}

# =============================================================================
# 依存パッケージインストール (pyproject.tomlから)
# =============================================================================
install_dependencies() {
    info "依存パッケージをインストール中 (uv sync)..."

    if [ ! -f "pyproject.toml" ]; then
        error "pyproject.toml が見つかりません"
    fi

    uv sync --all-extras

    success "依存パッケージインストール完了"
}

# =============================================================================
# pre-commit設定
# =============================================================================
setup_precommit() {
    if [ ! -f ".pre-commit-config.yaml" ]; then
        warn ".pre-commit-config.yaml が見つかりません"
        return
    fi

    info "pre-commit をセットアップ中..."
    uv run pre-commit install 2>/dev/null || warn "pre-commit install に失敗"
    success "pre-commit セットアップ完了"
}

# =============================================================================
# メイン処理
# =============================================================================
main() {
    echo ""
    echo "========================================"
    echo "  Kaggle Project Setup Script"
    echo "========================================"
    echo ""

    install_uv
    check_task
    create_directories
    init_git
    create_init_files
    setup_python_env
    install_dependencies
    setup_precommit

    echo ""
    echo "========================================"
    success "セットアップ完了!"
    echo "========================================"
    echo ""
    echo "次のステップ:"
    echo "  task          # 利用可能なタスクを表示"
    echo "  task jupyter  # Jupyter Lab を起動"
    echo ""
}

main
