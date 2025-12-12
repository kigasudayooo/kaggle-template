# Kaggle Competition Template

## セットアップ

```bash
./setup.sh
```

または手動で:

```bash
uv sync --all-extras
uv run pre-commit install
```

## コマンド

```bash
task lint       # Lintチェック
task lint:fix   # Lint自動修正
task test       # テスト実行
task check      # 全チェック
task jupyter    # Jupyter起動
```

## ディレクトリ構成

```
├── configs/        # 設定ファイル
├── data/           # データ
├── notebooks/      # Jupyter notebooks
├── src/            # ソースコード
├── models/         # 訓練済みモデル
├── experiments/    # 実験出力
└── tests/          # テスト
```
