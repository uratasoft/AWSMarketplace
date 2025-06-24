import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import pickle

# データ読み込み
df = pd.read_csv("credit_data_cleaned.csv")

# 列名の設定（もし既に設定済みならこの行はコメントアウト可）
df.columns = [f"A{i}" for i in range(1, 17)]

# ラベルと説明変数に分ける
X = df.drop("A16", axis=1)
y = df["A16"].apply(lambda x: 1 if x == "+" else 0)

# ラベルエンコーディング
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデル学習
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# モデル保存
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ モデルを保存しました: model.pkl")
