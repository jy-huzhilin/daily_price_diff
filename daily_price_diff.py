from Validator import *
from typing import Dict
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn

class daily_price_diff():
    def compute(self, input: Dict[str, pd.DataFrame],  current_time: datetime) -> Dict[str, pd.DataFrame]:
        # 获取输入数据
        df = input['cbond.stock_daily_quotes_non_ror']
        # 去重处理,取create_time最新字段
        df = df.sort_values(by='create_time', ascending=False).drop_duplicates(subset=['date', 'ths_code'], keep='first')

        # 处理数据计算差值
        df_sorted = df.sort_values(by=['ths_code', 'date'])
        df_sorted[['open_diff', 'close_diff', 'high_diff', 'low_diff']] = df_sorted.groupby('ths_code')[['open', 'close', 'high', 'low']].diff()
        df_sorted = df_sorted[df_sorted[['open_diff', 'close_diff', 'high_diff', 'low_diff']].notna().any(axis=1)].reset_index(drop=True)

        # 保留计算结果
        result_open = df_sorted[['ths_code', 'open_diff']]
        result_close = df_sorted[['ths_code', 'close_diff']]
        result_high = df_sorted[['ths_code', 'high_diff']]
        result_low = df_sorted[['ths_code', 'low_diff']]

        # 将计算结果组合成标准格式
        result_open = result_open.rename(columns={'ths_code': 'symbol', 'open_diff': 'value'})
        result_open['time'] = current_time
        result_close = result_close.rename(columns={'ths_code': 'symbol', 'close_diff': 'value'})
        result_close['time'] = current_time
        result_high = result_high.rename(columns={'ths_code': 'symbol', 'high_diff': 'value'})
        result_high['time'] = current_time
        result_low = result_low.rename(columns={'ths_code': 'symbol', 'low_diff': 'value'})
        result_low['time'] = current_time

        X = torch.rand(20, 5)
        y = torch.randint(0, 2, (20,))

        model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 2))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        for _ in range(5):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # 组合结果
        res = {
            'daily_open_diff': result_open,
            'daily_close_diff': result_close,
            'daily_high_diff': result_high,
            'daily_low_diff': result_low,
            'random_model': {current_time: model.state_dict()}
        }
        return res
    