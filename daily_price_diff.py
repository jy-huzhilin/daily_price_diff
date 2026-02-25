import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from typing import Dict
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class daily_price_diff:
    """
    模型管理功能测试项目
    生成合成数据，训练线性回归模型，通过 MLflow 记录训练过程。
    框架（worker_service.py）已通过环境变量注入 tracking URI、experiment 及父 run 上下文。
    """

    def compute(
        self,
        input_dataframes: Dict[str, pd.DataFrame],
        current_time: datetime,
    ):
        """单次模式入口，直接复用批量训练逻辑。"""
        ts = current_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(current_time, datetime) else str(current_time)
        self.compute_history(input_dataframes, ts, ts, [current_time])
        return {}

    def compute_history(
        self,
        input_dataframes: Dict[str, pd.DataFrame],
        start_time: str,
        end_time: str,
        run_times: list,
    ):
        """批量历史模式入口（无真实数据依赖，内部生成合成数据）"""

        # ── 超参数 ──────────────────────────────────────────────────────────
        params = {
            "n_features":   10,
            "n_samples":    1000,
            "test_size":    0.2,
            "noise_level":  0.5,
            "random_state": 42,
            "start_time":   start_time,
            "end_time":     end_time,
        }

        # ── 从框架注入的环境变量初始化 MLflow ──────────────────────────────
        # worker_service.py 在调用本方法前已设置：
        #   JADE_TRACKING_URI, JADE_EXPERIMENT_NAME, JADE_PARENT_MLFLOW_RUN_ID
        tracking_uri = os.getenv("JADE_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        experiment_name = os.getenv("JADE_EXPERIMENT_NAME")
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        run_name = f"linear_regression_{start_time[:10]}_to_{end_time[:10]}"

        # nested=True：将本 run 挂载到框架已启动的父 run 下
        with mlflow.start_run(run_name=run_name, nested=True):
            # 记录超参数
            mlflow.log_params(params)

            # ── 生成合成数据 ────────────────────────────────────────────────
            rng = np.random.RandomState(params["random_state"])
            X = rng.randn(params["n_samples"], params["n_features"])
            true_coef = rng.randn(params["n_features"])
            y = X @ true_coef + params["noise_level"] * rng.randn(params["n_samples"])

            n_train = int(params["n_samples"] * (1 - params["test_size"]))
            X_train, X_test = X[:n_train], X[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]

            # ── 分 epoch 模拟训练，记录曲线 ─────────────────────────────────
            model = LinearRegression()
            n_epochs = 10
            for epoch in range(n_epochs):
                n = max(1, int(n_train * (epoch + 1) / n_epochs))
                model.fit(X_train[:n], y_train[:n])
                train_mse = mean_squared_error(y_train[:n], model.predict(X_train[:n]))
                test_mse  = mean_squared_error(y_test,      model.predict(X_test))
                train_r2  = r2_score(y_train[:n], model.predict(X_train[:n]))
                test_r2   = r2_score(y_test,      model.predict(X_test))
                mlflow.log_metrics(
                    {"train_mse": train_mse, "test_mse": test_mse,
                     "train_r2":  train_r2,  "test_r2":  test_r2},
                    step=epoch,
                )

            # ── 最终全量训练并记录汇总指标 ───────────────────────────────────
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mlflow.log_metrics({
                "final_mse": mean_squared_error(y_test, y_pred),
                "final_mae": mean_absolute_error(y_test, y_pred),
                "final_r2":  r2_score(y_test, y_pred),
            })

            # ── 记录模型 ────────────────────────────────────────────────────
            mlflow.sklearn.log_model(model, "model")

        return {}, {}
