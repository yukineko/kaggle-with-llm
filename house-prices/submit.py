#!/usr/bin/env python3
"""
Kaggle 提出自動化スクリプト (Bearer認証)

使い方:
  python submit.py                          # Downloads/ から最新の submission*.csv を自動検出
  python submit.py submission_multiseed.csv # ファイル名指定 (Downloads/ から検索)
  python submit.py /path/to/file.csv       # フルパス指定

動作:
  1. Downloads/ から提出ファイルを検出 → submissions/ にコピー
  2. Kaggle API (Bearer認証) で3ステップアップロード
  3. スコア確定までポーリング (最大5分)
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("requests が必要です: pip install requests")
    sys.exit(1)

# --- 設定 ---
COMPETITION = "house-prices-advanced-regression-techniques"
KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"
DOWNLOADS_DIR = Path.home() / "Downloads"
SCRIPT_DIR = Path(__file__).resolve().parent
SUBMISSIONS_DIR = SCRIPT_DIR / "submissions"
POLL_INTERVAL = 10  # seconds
POLL_TIMEOUT = 300  # 5 minutes


def get_token():
    """kaggle.json から KGAT トークンを取得"""
    if not KAGGLE_JSON.exists():
        print(f"エラー: {KAGGLE_JSON} が見つかりません")
        sys.exit(1)
    with open(KAGGLE_JSON) as f:
        data = json.load(f)
    token = data.get("key", "")
    if not token.startswith("KGAT_"):
        print(f"警告: トークンが KGAT_ で始まりません (Basic認証は非対応)")
    return token


def find_submission(filename=None):
    """提出ファイルを検索: Downloads/ → submissions/ → カレントディレクトリ"""
    search_dirs = [DOWNLOADS_DIR, SUBMISSIONS_DIR, Path.cwd()]

    if filename:
        # フルパス指定
        if os.path.isabs(filename) or os.path.exists(filename):
            p = Path(filename)
            if p.exists():
                return p
        # ファイル名のみ → 各ディレクトリを検索
        for d in search_dirs:
            p = d / filename
            if p.exists():
                return p
        print(f"エラー: {filename} が見つかりません")
        sys.exit(1)

    # 自動検出: Downloads/ の最新 submission*.csv
    candidates = sorted(DOWNLOADS_DIR.glob("submission*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        latest = candidates[0]
        age_min = (time.time() - latest.stat().st_mtime) / 60
        print(f"自動検出: {latest.name} ({age_min:.0f}分前)")
        return latest

    # Downloads になければ submissions/ の最新
    candidates = sorted(SUBMISSIONS_DIR.glob("submission*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]

    print("エラー: 提出ファイルが見つかりません")
    sys.exit(1)


def copy_to_submissions(src):
    """submissions/ にコピー"""
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    dst = SUBMISSIONS_DIR / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(str(src), str(dst))
        print(f"コピー: {src} → {dst}")
    return dst


def upload_submission(token, filepath, description=""):
    """3ステップ Kaggle API アップロード"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    file_size = filepath.stat().st_size

    # Step 1: StartSubmissionUpload
    print("Step 1/3: アップロード開始...", end="", flush=True)
    r = requests.post(
        "https://api.kaggle.com/v1/competitions.CompetitionApiService/StartSubmissionUpload",
        headers=headers,
        json={"competitionName": COMPETITION, "fileName": filepath.name, "contentLength": file_size},
    )
    r.raise_for_status()
    data = r.json()
    blob_token = data["token"]
    upload_url = data["createUrl"]
    print(" OK")

    # Step 2: PUT to GCS
    print("Step 2/3: ファイルアップロード...", end="", flush=True)
    with open(filepath, "rb") as f:
        r = requests.put(upload_url, headers={"Content-Type": "text/csv"}, data=f)
    r.raise_for_status()
    print(" OK")

    # Step 3: CreateSubmission
    print("Step 3/3: 提出作成...", end="", flush=True)
    r = requests.post(
        "https://api.kaggle.com/v1/competitions.CompetitionApiService/CreateSubmission",
        headers=headers,
        json={
            "competitionName": COMPETITION,
            "blobFileTokens": [blob_token],
            "submissionDescription": description,
        },
    )
    r.raise_for_status()
    print(" OK")
    return True


def wait_for_score(token):
    """スコア確定までポーリング"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    print(f"\nスコア待機中 (最大{POLL_TIMEOUT}秒)...", flush=True)

    start = time.time()
    while time.time() - start < POLL_TIMEOUT:
        time.sleep(POLL_INTERVAL)
        r = requests.post(
            "https://api.kaggle.com/v1/competitions.CompetitionApiService/ListSubmissions",
            headers=headers,
            json={"competitionName": COMPETITION, "pageSize": 1},
        )
        r.raise_for_status()
        subs = r.json().get("submissions", [])
        if not subs:
            continue

        latest = subs[0]
        status = latest.get("status", "")
        score = latest.get("publicScore", "")
        elapsed = int(time.time() - start)

        if status == "COMPLETE" and score:
            print(f"\n{'=' * 50}")
            print(f"  Public Score: {score}")
            print(f"  File: {latest.get('fileName', '')}")
            print(f"  Date: {latest.get('date', '')}")
            print(f"  ({elapsed}秒で確定)")
            print(f"{'=' * 50}")
            return score
        elif status == "ERROR":
            print(f"\nエラー: 提出が失敗しました")
            print(json.dumps(latest, indent=2, ensure_ascii=False))
            return None
        else:
            print(f"  ... {elapsed}秒経過 (status: {status})", flush=True)

    print(f"\nタイムアウト ({POLL_TIMEOUT}秒). Kaggle Web で確認してください。")
    return None


def main():
    parser = argparse.ArgumentParser(description="Kaggle House Prices 提出自動化")
    parser.add_argument("file", nargs="?", help="提出ファイル (省略時: Downloads/ の最新を自動検出)")
    parser.add_argument("-d", "--description", default="", help="提出の説明")
    parser.add_argument("--no-wait", action="store_true", help="スコア待機をスキップ")
    args = parser.parse_args()

    token = get_token()
    src = find_submission(args.file)
    print(f"提出ファイル: {src} ({src.stat().st_size:,} bytes)")

    dst = copy_to_submissions(src)
    desc = args.description or src.stem.replace("_", " ")

    upload_submission(token, dst, description=desc)

    if not args.no_wait:
        wait_for_score(token)


if __name__ == "__main__":
    main()
