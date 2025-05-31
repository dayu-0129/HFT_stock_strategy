import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def detect_first_limit_up(df, volume_thresh=200000):
    df['new_limit_up'] = (
        (df['ask_price1'] == df['bid_price1']) &
        (df['ask_volume1'] > volume_thresh) &
        (df['current'] < df['ask_price1']) &
        (df['ask_price1'] != df['ask_price1'].shift(1))
    )
    return df
def compute_seal_strength(df, rolling_window=3):
    """
    计算封板强度指标：当前涨停封单量 / 滚动成交量
    """
    df['rolling_volume'] = df['volume'].rolling(rolling_window).sum().replace(0, 1)
    df['seal_strength'] = df['ask_volume1'] / df['rolling_volume']
    return df

def label_zha_ban(df, limit_up_col='ask_price1'):
    """
    标注炸板label
    """
    df = df.copy()
    df['label'] = 0  # 默认不炸板
    df = df.reset_index(drop=True)  # 保证index和行号一致
    limit_up_pos = df.index[df['new_limit_up']].tolist()
    for i in limit_up_pos:
        next_i = i + 1
        if next_i < len(df):
            if (df.iloc[next_i]['ask_price1'] > df.iloc[next_i]['bid_price1']) and \
               (df.iloc[next_i]['current'] < df.iloc[next_i]['ask_price1']):
                df.at[i, 'label'] = 1  # 炸板
    return df

def detect_zha_ban(df, i):
    return (df.loc[i, 'ask_price1'] > df.loc[i, 'bid_price1']) and \
           (df.loc[i, 'current'] < df.loc[i, 'ask_price1'])






