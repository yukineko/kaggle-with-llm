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
echo "[1/7] 認証チェック..."
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
echo "[2/7] コンペティション確認..."
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
echo "[3/7] ディレクトリ作成..."
mkdir -p "${COMP_DIR}/figures"
echo "  OK: ${COMP_DIR}"
echo "  OK: ${COMP_DIR}/figures"

# --- Step 4: データダウンロード ---
echo "[4/7] データダウンロード..."
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
echo "[5/7] .gitignore 更新..."
GITIGNORE="${SCRIPT_DIR}/.gitignore"
IGNORE_PATTERN="${COMPETITION}/*.csv"
if grep -qF "${IGNORE_PATTERN}" "${GITIGNORE}" 2>/dev/null; then
    echo "  OK: パターン既に登録済み"
else
    echo "${IGNORE_PATTERN}" >> "${GITIGNORE}"
    echo "  OK: '${IGNORE_PATTERN}' を追加"
fi

# --- Step 6: DATA_FILES リストを自動検出 ---
echo ""
echo "[6/7] CSVファイルリスト検出..."
CSV_LIST=$(python3 -c "
import glob, os, sys
files = sorted(os.path.basename(f) for f in glob.glob('${COMP_DIR}/*.csv'))
if files:
    print(', '.join(repr(f) for f in files))
    sys.stderr.write('  検出: ' + ', '.join(files) + '\n')
else:
    print(\"'TODO.csv'\")
    sys.stderr.write('  WARN: CSVファイルが見つかりません\n')
" 2>&1 >/tmp/_csv_list.txt; cat /tmp/_csv_list.txt >&2; python3 -c "
import glob, os
files = sorted(os.path.basename(f) for f in glob.glob('${COMP_DIR}/*.csv'))
print(', '.join(repr(f) for f in files) if files else \"'TODO.csv'\")
")

# --- Step 7: eda.ipynb 生成 ---
echo "[7/7] eda.ipynb 生成..."
NOTEBOOK="${COMP_DIR}/eda.ipynb"
if [ -f "${NOTEBOOK}" ]; then
    echo "  SKIP: eda.ipynb が既に存在します"
else
    COMPETITION="${COMPETITION}" CSV_LIST="${CSV_LIST}" COMP_DIR="${COMP_DIR}" NOTEBOOK="${NOTEBOOK}" \
    python3 - <<'PYEOF'
import json, os

competition = os.environ['COMPETITION']
csv_list    = os.environ['CSV_LIST']
local_path  = os.environ['COMP_DIR']
notebook    = os.environ['NOTEBOOK']

setup_code = "\n".join([
    "# ============================================================",
    "# SETUP CELL — 環境・認証・データ確認・読み込み",
    "# ============================================================",
    "import sys, os",
    "import pandas as pd",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "import matplotlib.dates as mdates",
    "import seaborn as sns",
    "import warnings",
    "warnings.filterwarnings('ignore')",
    "",
    "plt.rcParams['figure.figsize'] = (16, 6)",
    "plt.rcParams['font.size'] = 12",
    "sns.set_style('whitegrid')",
    "",
    "# ============================================================",
    "# [CONFIG] 手動設定 (自動検出する場合は None のまま)",
    "# ============================================================",
    f"USER_DATA_DIR = None   # 例: '/content/drive/MyDrive/{competition}'",
    "",
    "# ============================================================",
    "# [1] 環境検出",
    "# ============================================================",
    "from pathlib import Path",
    "",
    "IS_COLAB = 'google.colab' in sys.modules or 'COLAB_GPU' in os.environ",
    "print(f\"[1] Environment : {'Google Colab' if IS_COLAB else 'Local'}\")",
    "",
    f"COMPETITION = '{competition}'",
    f"DATA_FILES  = [{csv_list}]",
    "",
    "def has_all_files(d: Path) -> bool:",
    "    return d.exists() and all((d / f).exists() for f in DATA_FILES)",
    "",
    "# ============================================================",
    "# [2] DATA_DIR の決定",
    "# ============================================================",
    "DATA_DIR = None",
    "",
    "if USER_DATA_DIR is not None:",
    "    DATA_DIR = Path(USER_DATA_DIR)",
    "",
    "elif IS_COLAB:",
    "    # 2a. /content/ をチェック (通信なし)",
    "    for c in [Path(f'/content/{COMPETITION}'), Path('/content/data')]:",
    "        if has_all_files(c):",
    "            DATA_DIR = c",
    "            break",
    "",
    "    # 2b. Drive をマウントして確認 (通信1回)",
    "    if DATA_DIR is None:",
    "        print('[2] Mounting Google Drive...')",
    "        from google.colab import drive",
    "        drive.mount('/content/drive', force_remount=False)",
    "        for c in [",
    "            Path(f'/content/drive/MyDrive/kaggle/{COMPETITION}'),",
    "            Path(f'/content/drive/MyDrive/{COMPETITION}'),",
    "        ]:",
    "            if has_all_files(c):",
    "                DATA_DIR = c",
    "                break",
    "",
    "    # 2c. Drive にも見つからない → Kaggle API でダウンロード",
    "    if DATA_DIR is None:",
    "        DATA_DIR = Path(f'/content/{COMPETITION}')",
    "        DATA_DIR.mkdir(parents=True, exist_ok=True)",
    "        _token_path = Path('/content/drive/MyDrive/.kaggle/access_token')",
    "        if not _token_path.exists():",
    "            raise FileNotFoundError('Google Drive の マイドライブ/.kaggle/access_token が見つかりません')",
    "        _token = _token_path.read_text().strip()",
    "        print(f\"[2] access_token : {_token[:8]}... (from Drive)\")",
    "        import requests, zipfile",
    "        _url = f'https://www.kaggle.com/api/v1/competitions/data/download-all/{COMPETITION}'",
    "        print('[3] Downloading from Kaggle API...')",
    "        with requests.get(_url, headers={'Authorization': f'Bearer {_token}'},",
    "                          stream=True, allow_redirects=True) as _r:",
    "            if _r.status_code == 401:",
    "                raise RuntimeError('401 Unauthorized — access_token が無効です')",
    "            if _r.status_code == 403:",
    "                raise RuntimeError('403 Forbidden — コンペのルール同意が必要です')",
    "            _r.raise_for_status()",
    "            _zip = DATA_DIR / f'{COMPETITION}.zip'",
    "            _total = int(_r.headers.get('content-length', 0))",
    "            _done = 0",
    "            with open(_zip, 'wb') as _f:",
    "                for _chunk in _r.iter_content(1024 * 1024):",
    "                    _f.write(_chunk)",
    "                    _done += len(_chunk)",
    "                    if _total:",
    "                        print(f'\\r    {_done/1e6:.0f} / {_total/1e6:.0f} MB', end='')",
    "        print()",
    "        with zipfile.ZipFile(_zip) as _z:",
    "            _z.extractall(DATA_DIR)",
    "        _zip.unlink()",
    "        del _token",
    "",
    "else:",
    "    # ローカル: 通信なし、データは手動管理",
    "    for c in [",
    f"        Path('{local_path}'),",
    "        Path('.'),",
    "    ]:",
    "        if has_all_files(c):",
    "            DATA_DIR = c",
    "            break",
    "",
    "# ============================================================",
    "# [3] ファイル確認",
    "# ============================================================",
    "assert DATA_DIR is not None, 'DATA_DIR が未設定です'",
    "assert has_all_files(DATA_DIR), \\",
    "    f\"ファイルが揃っていません: {[f for f in DATA_FILES if not (DATA_DIR/f).exists()]}\"",
    "",
    "print(f'[2] DATA_DIR    : {DATA_DIR}')",
    "for f in DATA_FILES:",
    "    size = (DATA_DIR / f).stat().st_size / 1e6",
    "    print(f'    {f:<40} {size:6.1f} MB')",
    "",
    "# ============================================================",
    "# [4] データ読み込み",
    "# ============================================================",
    "FIG_DIR = DATA_DIR / 'figures'",
    "FIG_DIR.mkdir(exist_ok=True)",
    "DATA_DIR = str(DATA_DIR) + '/'",
    "FIG_DIR  = str(FIG_DIR) + '/'",
    "",
    "# TODO: 以下をコンペのデータに合わせて変更する",
    "# df = pd.read_csv(DATA_DIR + DATA_FILES[0])",
    "# print(f'    df : {df.shape}')",
    "print('✓ Ready')",
])

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"}
    },
    "cells": [
        {
            "cell_type": "markdown",
            "id": "title",
            "metadata": {},
            "source": [f"# {competition} — EDA\n\nTODO: コンペの説明を記入"]
        },
        {
            "cell_type": "code",
            "id": "setup",
            "metadata": {},
            "source": [setup_code],
            "outputs": [],
            "execution_count": None
        }
    ]
}

with open(notebook, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("  OK: eda.ipynb を生成しました")
PYEOF
fi

# --- 完了 ---
echo ""
echo "=== セットアップ完了 ==="
echo ""
echo "次のステップ:"
echo "  1. ${COMP_DIR}/eda.ipynb を開く"
echo "  2. DATA_FILES リストを確認・調整する"
echo "  3. [4] データ読み込みコードをコンペに合わせて実装する"
echo "  4. kaggle competitions submit -c ${COMPETITION} -f submission.csv -m 'first submission'"
