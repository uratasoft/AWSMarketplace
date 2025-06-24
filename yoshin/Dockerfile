# ベースイメージ
FROM python:3.12-slim

# 作業ディレクトリ
WORKDIR /app

# 必要ファイルをコピー
COPY . /app

# 必要なPythonパッケージをインストール（pipアップグレード込み）
RUN pip install --upgrade pip && \
    pip install --no-cache-dir flask numpy pandas lightgbm

# Flaskアプリの起動
CMD ["python", "predict.py"]
