from Validator import *
from typing import Dict
from datetime import datetime
import pandas as pd

class daily_price_diff():
    @validation(validators=[
        DtypeValidator(),
        DuplicateValidator(),
        CountValidator(min_count=5000, max_count=6000)
    ])
    def compute(self, input: Dict[str, pd.DataFrame],  current_time: datetime) -> Dict[str, pd.DataFrame]:
        # 获取输入数据
        df = input['cbond.stock_daily_quotes_non_ror']
        # 去重处理,取create_time最新字段
        df = df.sort_values(by='create_time', ascending=False).drop_duplicates(subset=['date', 'ths_code'], keep='first')

        # 处理数据计算差值
        df_sorted = df.sort_values(by=['ths_code', 'date'])
        df_sorted[['open_diff', 'close_diff', 'high_diff', 'low_diff']] = df_sorted.groupby('ths_code')[['open', 'close', 'high', 'low']].diff()
        df_sorted = df_sorted[df_sorted['date'] == current_time.date()].reset_index(drop=True)

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

        # 组合结果
        res2 = {
            'daily_open_diff': result_open,
            'daily_close_diff': result_close,
            'daily_high_diff': result_high,
            'daily_low_diff': result_low
        }
        return res2
    # 111
    
    @validation()
    def compute_history(self, input: Dict[str, pd.DataFrame], start_time:datetime, end_time:datetime, time_list:list) -> Dict[str, pd.DataFrame]:
        pass