# Kaggle 環境セットアップガイド

## 1. Kaggle API 認証

### トークンの種類 (Kaggle CLI 2.0+)

Kaggle CLI 2.0 では認証方式が2つある:

| 方式 | ファイル | 形式 | 認証ヘッダー |
|------|----------|------|-------------|
| **Bearer (推奨)** | `~/.kaggle/access_token` | `KGAT_xxxxxxx` | `Authorization: Bearer KGAT_...` |
| Legacy API Key | `~/.kaggle/kaggle.json` | `{"username":"...","key":"..."}` | Basic認証 (username:key) |

### 重要: `KGAT_` トークンは Bearer 認証専用

Kaggle Settings で生成される `KGAT_` 形式のトークンは **Bearer 認証用**。
`kaggle.json` の `key` フィールドに入れると Basic 認証として送信され **401 エラー**になる。

### セットアップ手順

1. https://www.kaggle.com/settings → API → **Create New Token**
2. トークン (`KGAT_xxxx...`) を `~/.kaggle/access_token` に保存:

```bash
echo -n "KGAT_your_token_here" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

3. 認証テスト:

```bash
kaggle competitions list
```

### 環境変数による認証 (代替)

```bash
export KAGGLE_API_TOKEN="KGAT_your_token_here"
```

### トークンの優先順位 (CLI 2.0)

1. OAuth トークン (Kaggle Notebook 内)
2. `KAGGLE_API_TOKEN` 環境変数
3. `~/.kaggle/access_token` ファイル
4. `~/.kaggle/kaggle.json` (Legacy, Basic認証)

### トラブルシューティング

| 症状 | 原因 | 対処 |
|------|------|------|
| `401 Unauthenticated` | トークン期限切れ or 形式不一致 | 新トークン取得 → access_token に保存 |
| `kaggle.json` にKGAT入れて401 | Bearer トークンをBasic認証で送信 | `access_token` ファイルに移動 |
| `403 Forbidden` | コンペルール未同意 | Webでコンペページを開きルールに同意 |

## 2. 新規コンペのセットアップ

### スクリプトで自動化

```bash
./setup-competition.sh <competition-slug>
```

実行内容:
1. 認証チェック
2. コンペ存在確認
3. ディレクトリ作成 (`<slug>/`)
4. データダウンロード & zip展開
5. `.gitignore` に `<slug>/*.csv` 追加

### 手動セットアップ

```bash
mkdir <competition-slug>
cd <competition-slug>
kaggle competitions download -c <competition-slug>
unzip <competition-slug>.zip && rm <competition-slug>.zip
```

## 3. 提出

```bash
kaggle competitions submit -c <competition-slug> -f submission.csv -m "description"
kaggle competitions submissions -c <competition-slug>  # 結果確認
```
