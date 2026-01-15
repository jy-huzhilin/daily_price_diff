from Validator import *
from typing import Dict
from datetime import datetime
import pandas as pd
import torch


class daily_price_diff():
    def run(self, input: Dict[str, pd.DataFrame],  current_time: datetime) -> Dict[str, pd.DataFrame]:
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

        # 组合结果
        res = {
            'daily_open_diff': result_open,
            'daily_close_diff': result_close,
            'daily_high_diff': result_high,
            'daily_low_diff': result_low,
        }
        return res

    def run2(self, input: Dict[str, pd.DataFrame],  current_time: datetime) -> Dict[str, pd.DataFrame]:
        # 获取输入数据
        open_df = input['daily_open_diff']
        close_df = input['daily_close_diff']
        high_df = input['daily_high_diff']
        low_df = input['daily_low_diff']

        # 去重处理,取create_time最新字段
        open_df = open_df.sort_values(by='ct', ascending=False).drop_duplicates(subset=['time', 'symbol'], keep='first')
        close_df = close_df.sort_values(by='ct', ascending=False).drop_duplicates(subset=['time', 'symbol'], keep='first')
        high_df = high_df.sort_values(by='ct', ascending=False).drop_duplicates(subset=['time', 'symbol'], keep='first')
        low_df = low_df.sort_values(by='ct', ascending=False).drop_duplicates(subset=['time', 'symbol'], keep='first')

        # 合并四个因子
        merged_df = open_df[['time', 'symbol', 'value']].rename(columns={'value': 'open'})
        merged_df = merged_df.merge(close_df[['time', 'symbol', 'value']].rename(columns={'value': 'close'}),
                                    on=['time', 'symbol'], how='outer')
        merged_df = merged_df.merge(high_df[['time', 'symbol', 'value']].rename(columns={'value': 'high'}),
                                    on=['time', 'symbol'], how='outer')
        merged_df = merged_df.merge(low_df[['time', 'symbol', 'value']].rename(columns={'value': 'low'}),
                                    on=['time', 'symbol'], how='outer')

        # 计算最大值和最小值
        merged_df['max_value'] = merged_df[['open', 'close', 'high', 'low']].max(axis=1)
        merged_df['min_value'] = merged_df[['open', 'close', 'high', 'low']].min(axis=1)
        merged_df['mean_value'] = merged_df[['open', 'close', 'high', 'low']].mean(axis=1)
        merged_df['median_value'] = merged_df[['open', 'close', 'high', 'low']].median(axis=1)

        # 构造结果 DataFrame
        max_df = merged_df[['time', 'symbol', 'max_value']].rename(columns={'max_value': 'value'})
        min_df = merged_df[['time', 'symbol', 'min_value']].rename(columns={'min_value': 'value'})
        mean_df = merged_df[['time', 'symbol', 'mean_value']].rename(columns={'mean_value': 'value'})
        median_df = merged_df[['time', 'symbol', 'median_value']].rename(columns={'median_value': 'value'})

        # 组合结果
        res = {
            'daily_diff_max': max_df,  
            'daily_diff_min': min_df,
            'daily_diff_mean': mean_df,
            'daily_diff_median': median_df
        }
        return res