# Kaggle Competition - Universal Guide

**Language**: 日本語で返答してください (Always respond in Japanese)

このドキュメントは、KaggleコンペティションプロジェクトのためのClaude Code向けコアガイドラインです。

---

## 📋 基本方針

このプロジェクトは**Kaggleコンペティション用テンプレート**です：

- **Python 3.10**（Kaggle環境互換）
- **uvパッケージ管理**（pip禁止）
- **Trackio実験管理**（ローカルファースト）
- **TODO.md中心のワークフロー**
- **2025年ベストプラクティス準拠**

---

## ⚠️ 絶対的な禁止事項

### 1. pipの使用禁止

```bash
❌ pip install package-name
❌ uv pip install package-name
❌ python -m pip install package-name

✅ uv add package-name
✅ uv add --dev dev-package
```

**理由**: uvは10-100倍高速で依存関係解決が確実

### 2. Kaggle提出でのalbumentations使用禁止

```python
❌ import albumentations as A  # Kaggle環境で動作しない

✅ from torchvision import transforms  # 唯一の信頼できるオプション
```

**理由**: Kaggleノートブックではalbumentationsが利用不可

---

## 📝 TODO管理の基本原則

**絶対ルール**: すべての作業前にtodo.mdを更新すること

### 正しいワークフロー

```
1. todo.mdを開く
2. これから行う作業をtodo.mdに追加/更新
3. Commit: "docs: Add [task] to TODO"
4. 作業を開始する
5. 作業完了後、即座にtodo.mdで完了マーク
6. Commit: "docs: Mark [task] as completed"
```

### TODO.md更新が必要な作業トリガー

1. ファイル作成・編集
2. 実験・訓練
3. デバッグ・問題解決
4. データ処理・分析
5. 推論・提出
6. リファクタリング・整理

---

## 📁 プロジェクト構造（概要）

```
project/
├── data/              # データファイル（gitignored）
├── notebooks/         # Jupyter notebooks
├── src/
│   ├── data/         # データセット
│   ├── models/       # モデル定義
│   ├── features/     # 特徴量エンジニアリング
│   ├── utils/        # ユーティリティ
│   └── scripts/      # 訓練・推論スクリプト
├── configs/          # YAML設定
├── models/           # 訓練済みモデル（gitignored）
├── experiments/      # 実験出力（gitignored）
├── todo.md           # TODO管理（重要）
└── pyproject.toml    # uv設定
```

---

## 🎯 詳細情報の参照先

詳細なガイドラインは以下に分散配置されています：

### Rules（path-specific、起動時読み込み）

- `~/.claude/rules/kaggle-constraints.md` - Kaggle特有の制約
- `~/.claude/rules/coding-standards.md` - コーディング規約
- `~/.claude/rules/uv-package-management.md` - uvルール

### Slash Commands（ユーザー呼び出し）

- `/setup` - 環境構築手順
- `/git:commit` - Git運用ルール
- `/kaggle:submit` - Kaggle提出チェックリスト
- `/train` - 訓練開始手順
- `/trackio` - Trackio使用方法

### Skills（AI自動判断）

- `experiment-tracking` - 実験管理（Trackio統合）
- `performance-optimization` - パフォーマンス最適化
- `kaggle-workflow` - Kaggle特有のワークフロー

### Subagents（独立コンテキスト）

- `debug-assistant` - デバッグ・トラブルシューティング
- `architecture-validator` - アーキテクチャ検証

---

## 🚀 Quick Start

### 新規プロジェクトセットアップ

```bash
# 環境構築の詳細ガイド
/setup
```

### モデル訓練開始

```bash
# 訓練前の確認事項と手順
/train
```

### Kaggle提出準備

```bash
# 提出前チェックリスト
/kaggle:submit
```

### Git コミット

```bash
# Git運用ルールとコミットフォーマット
/git:commit
```

---

## 💡 ベストプラクティス

1. **TODO.md優先**: すべての作業前に更新
2. **Trackio必須**: すべての訓練をトラッキング
3. **uvのみ使用**: pipは絶対禁止
4. **Kaggle制約遵守**: albumentations禁止、torchvision使用
5. **小さくコミット**: 意味のある単位で頻繁にコミット

---

## 🔗 重要なリンク

- [TODO管理テンプレート](todo.md)
- [Taskfile（タスクランナー）](Taskfile.yml)
- [プロジェクトREADME](README.md)

---

## 📚 さらに詳しく

- 環境構築: `/setup`
- 実験管理: `experiment-tracking` skill（自動発動）
- パフォーマンス最適化: `performance-optimization` skill（自動発動）
- デバッグ: `debug-assistant` subagent（自動発動）

---

**Last Updated**: 2025-12-17
**Version**: 2.0 (Context-Optimized)

このガイドは2025年のClaude Codeベストプラクティスに準拠し、コンテキスト効率を最大化しています。
