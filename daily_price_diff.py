import os
import sys
import numpy as np
import pandas as pd
from typing import Dict
from datetime import datetime

# 确保 jade_ml 可以被 import（worker 进程继承了父进程的 sys.path，但以防万一）
_jade_root = '/tech/home/hzl/Repo/legendary_jade'
if _jade_root not in sys.path:
    sys.path.insert(0, _jade_root)

from jade_ml.tracker import JadeTracker


class daily_price_diff:
    """
    模型管理功能测试项目
    生成合成数据，训练线性回归模型，通过 JadeTracker 记录到 MLflow。
    """

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

        # ── user_id：优先环境变量，其次默认 huzl ────────────────────────────
        user_id = os.environ.get("JADE_USER_ID", "huzl")

        # ── JadeTracker ─────────────────────────────────────────────────────
        tracker = JadeTracker()
        run_name = f"linear_regression_{start_time[:10]}_to_{end_time[:10]}"

        with tracker.start_run(
            run_name=run_name,
            user_id=user_id,
            tags={
                "jade.project_name": "daily_price_diff",
                "jade.team":         "cbond",
                "jade.repo.url":     "https://github.com/jy-huzhilin/daily_price_diff.git",
                "jade.repo.branch":  "model-train-test",
                "jade.repo.commit":  _get_git_commit(),
            },
        ):
            # 记录超参数
            tracker.log_params(params)

            # ── 生成合成数据 ────────────────────────────────────────────────
            rng = np.random.RandomState(params["random_state"])
            X = rng.randn(params["n_samples"], params["n_features"])
            true_coef = rng.randn(params["n_features"])
            y = X @ true_coef + params["noise_level"] * rng.randn(params["n_samples"])

            n_train = int(params["n_samples"] * (1 - params["test_size"]))
            X_train, X_test = X[:n_train], X[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]

            # ── 分 epoch 模拟训练，记录曲线 ─────────────────────────────────
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

            model = LinearRegression()
            n_epochs = 10
            for epoch in range(n_epochs):
                n = max(1, int(n_train * (epoch + 1) / n_epochs))
                model.fit(X_train[:n], y_train[:n])
                train_mse = mean_squared_error(y_train[:n], model.predict(X_train[:n]))
                test_mse  = mean_squared_error(y_test,      model.predict(X_test))
                train_r2  = r2_score(y_train[:n], model.predict(X_train[:n]))
                test_r2   = r2_score(y_test,      model.predict(X_test))
                tracker.log_metrics(
                    {"train_mse": train_mse, "test_mse": test_mse,
                     "train_r2":  train_r2,  "test_r2":  test_r2},
                    step=epoch,
                )

            # ── 最终全量训练并记录汇总指标 ───────────────────────────────────
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            tracker.log_metrics({
                "final_mse": mean_squared_error(y_test, y_pred),
                "final_mae": mean_absolute_error(y_test, y_pred),
                "final_r2":  r2_score(y_test, y_pred),
            })

            # ── 记录模型 ────────────────────────────────────────────────────
            tracker.log_model(model, "model", model_type="sklearn")

        return {}, {}


def _get_git_commit() -> str:
    """尝试读取当前 HEAD commit hash，失败时返回占位字符串。"""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"
