#!/usr/bin/env bash
# ============================================================
# Kaggle コンペ初期セットアップスクリプト
#
# Usage:
#   ./setup-competition.sh <competition-slug>
#
# Example:
#   ./setup-competition.sh m5-forecasting-accuracy
#   ./setup-competition.sh house-prices-advanced-regression-techniques
#
# 事前準備:
#   1. ~/.kaggle/access_token に KGAT_ トークンを配置
#   2. pip install kaggle (v2.0+)
#   詳細は SETUP.md を参照
# ============================================================

set -euo pipefail

# --- 引数チェック ---
if [ $# -lt 1 ]; then
    echo "Usage: $0 <competition-slug>"
    echo "Example: $0 m5-forecasting-accuracy"
    exit 1
fi

COMPETITION="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMP_DIR="${SCRIPT_DIR}/${COMPETITION}"

echo "=== Kaggle Competition Setup ==="
echo "Competition: ${COMPETITION}"
echo "Directory:   ${COMP_DIR}"
echo ""

# --- Step 1: 認証チェック ---
echo "[1/5] 認証チェック..."
if ! kaggle competitions list -s "${COMPETITION}" > /dev/null 2>&1; then
    echo "ERROR: Kaggle API 認証に失敗しました"
    echo ""
    echo "以下を確認してください:"
    echo "  1. ~/.kaggle/access_token に有効な KGAT_ トークンがあること"
    echo "  2. トークンが期限切れでないこと"
    echo "  詳細は SETUP.md を参照"
    exit 1
fi
echo "  OK: 認証成功"

# --- Step 2: コンペ存在確認 ---
echo "[2/5] コンペティション確認..."
COMP_INFO=$(kaggle competitions list -s "${COMPETITION}" 2>&1)
if ! echo "${COMP_INFO}" | grep -q "${COMPETITION}"; then
    echo "ERROR: コンペティション '${COMPETITION}' が見つかりません"
    echo "スラッグを確認してください (URLの末尾部分)"
    exit 1
fi
echo "  OK: コンペティション確認済み"
echo "${COMP_INFO}" | head -3

# --- Step 3: ディレクトリ作成 ---
echo ""
echo "[3/5] ディレクトリ作成..."
mkdir -p "${COMP_DIR}"
echo "  OK: ${COMP_DIR}"

# --- Step 4: データダウンロード ---
echo "[4/5] データダウンロード..."
cd "${COMP_DIR}"
kaggle competitions download -c "${COMPETITION}"

# zipファイルを展開して削除
ZIP_FILE="${COMPETITION}.zip"
if [ -f "${ZIP_FILE}" ]; then
    echo "  展開中..."
    unzip -o "${ZIP_FILE}"
    rm "${ZIP_FILE}"
fi
echo "  OK: ダウンロード完了"
echo ""
echo "  ファイル一覧:"
ls -lh "${COMP_DIR}"/ | grep -v "^total"

# --- Step 5: .gitignore 更新 ---
echo ""
echo "[5/5] .gitignore 更新..."
GITIGNORE="${SCRIPT_DIR}/.gitignore"
IGNORE_PATTERN="${COMPETITION}/*.csv"
if grep -qF "${IGNORE_PATTERN}" "${GITIGNORE}" 2>/dev/null; then
    echo "  OK: パターン既に登録済み"
else
    echo "${IGNORE_PATTERN}" >> "${GITIGNORE}"
    echo "  OK: '${IGNORE_PATTERN}' を追加"
fi

# --- 完了 ---
echo ""
echo "=== セットアップ完了 ==="
echo ""
echo "次のステップ:"
echo "  1. cd ${COMP_DIR}"
echo "  2. EDA ノートブックを作成"
echo "  3. kaggle competitions submit -c ${COMPETITION} -f submission.csv -m 'first submission'"
