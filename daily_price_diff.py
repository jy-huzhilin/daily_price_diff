from Validator import *
from typing import Dict
from datetime import datetime
import pandas as pd
import logging

class daily_price_diff():
    @validation(validators=[
        DtypeValidator(),
        DuplicateValidator(),
        CountValidator(min_count=5000, max_count=6000)
    ])
    def compute(self, input: Dict[str, pd.DataFrame],  current_time: datetime) -> Dict[str, pd.DataFrame]:
        # 获取输入数据
        open_df = input['daily_open_diff']
        close_df = input['daily_close_diff']
        high_df = input['daily_high_diff']
        low_df = input['daily_low_diff']

        logging.info(close_df)
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
    
    @validation()
    def compute_history(self, input: Dict[str, pd.DataFrame], start_time:datetime, end_time:datetime, time_list:list) -> Dict[str, pd.DataFrame]:
        pass