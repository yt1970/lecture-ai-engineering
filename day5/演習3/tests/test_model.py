import os
import pytest
import pandas as pd
import numpy as np
import tempfile
import pickle
import time
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
# MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
# model_filename = f"{name}_model.pkl"
# MODEL_PATH = os.path.join(MODEL_DIR, model_filename)


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target

        # 必要なカラムのみ選択
        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


@pytest.fixture
def preprocessor():
    """前処理パイプラインを定義"""
    # 数値カラムと文字列カラムを定義
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    # 数値特徴量の前処理（欠損値補完と標準化）
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # カテゴリカル特徴量の前処理（欠損値補完とOne-hotエンコーディング）
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 前処理をまとめる
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


@pytest.fixture(scope="session", autouse=True)
def setup_mlflow():
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))  # 任意のローカルURI
    mlflow.set_experiment("Titanic_Model_Tests")
    # グローバル run を明示的に開始
    with mlflow.start_run(run_name="Titanic_Model_Test_Session") as run:
        yield  # テストがすべてこの run の中で実行される


# モデル一覧
@pytest.fixture(
    params=[
        ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("LogisticRegression", LogisticRegression(max_iter=1000)),
        ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric="logloss")),
    ]
)
def model_and_name(request):
    return request.param  # (name, model) tuple


@pytest.fixture
def train_model(sample_data, preprocessor, model_and_name):
    # 複数のモデルをパラメータとして渡す
    name, model_algo = model_and_name
    """モデルの学習とテストデータの準備"""
    # データの分割とラベル変換
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデルパイプラインの作成
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model_algo),
        ]
    )

    # モデルの学習
    model.fit(X_train, y_train)

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
    model_filename = f"{name}_model.pkl"
    MODEL_PATH = os.path.join(MODEL_DIR, model_filename)

    # モデルの保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return name, model, X_test, y_test


# def test_model_exists():
#     """モデルファイルが存在するか確認"""
#     if not os.path.exists(MODEL_PATH):
#         pytest.skip("モデルファイルが存在しないためスキップします")
#     assert os.path.exists(MODEL_PATH), "モデルファイルが存在しません"


def test_model_accuracy(train_model):
    """モデルの精度を検証"""
    name, model, X_test, y_test = train_model

    # 予測と精度計算
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} accuracy: {accuracy:.4f}")
    

    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", accuracy)

    # 保存してモデルとして記録
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, f"{name}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(model_path, artifact_path=f"model_{name}")

    # Titanicデータセットでは0.75以上の精度が一般的に良いとされる
    assert accuracy >= 0.75, f"{name}モデルの精度が低すぎます: {accuracy}"


def test_model_inference_time(train_model):
    # 複数のモデルをパラメータとして渡す
    """モデルの推論時間を検証"""
    name, model, X_test, _ = train_model

    # 推論時間の計測
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()

    inference_time = end_time - start_time
    print(f"{name} inference time: {inference_time:.4f} sec")

    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_metric("inference_time", inference_time)

    # 推論時間が1秒未満であることを確認
    assert inference_time < 1.0, f"{name}の推論時間が長すぎます: {inference_time}秒"


def test_model_reproducibility(sample_data, preprocessor, model_and_name):
    # 複数のモデルをパラメータとして渡す。
    name, model_algo = model_and_name
    """モデルの再現性を検証"""

    params = model_algo.get_params()
    if "random_state" in params:
        params["random_state"] = 42
    # # 別インスタンスを2つ作る
    # model1 = type(model_algo)(**model_algo.get_params())
    # model2 = type(model_algo)(**model_algo.get_params())

    # データの分割
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 同じパラメータで２つのモデルを作成
    model1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", type(model_algo)(**model_algo.get_params())),
        ]
    )

    model2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", type(model_algo)(**model_algo.get_params())),
        ]
    )

    # 学習
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    # 同じ予測結果になることを確認
    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test)

    assert np.array_equal(
        predictions1, predictions2
    ), "モデルの予測結果に再現性がありません"
