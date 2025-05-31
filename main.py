import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from utility import label_zha_ban
from utility import compute_seal_strength
from utility import detect_first_limit_up
from sklearn.model_selection import train_test_split
from buy_signal import trigger_buy
from sell_signal import trigger_sell
from risk_control import check_exit_conditions
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_parquet('data.parquet')
df["date"]=pd.to_datetime(df["clock"]).dt.date
df['spread'] = df['ask_price1'] - df['bid_price1']
df['ask1_locked'] = df['ask_price1'].diff().abs() < 1e-6  # 最近价格是否锁死
df['order_imbalance'] = (
    df[[f'bid_volume{i}' for i in range(1, 11)]].sum(axis=1) -
    df[[f'ask_volume{i}' for i in range(1, 11)]].sum(axis=1)
) / (
    df[[f'bid_volume{i}' for i in range(1, 11)]].sum(axis=1) +
    df[[f'ask_volume{i}' for i in range(1, 11)]].sum(axis=1)
)
df=compute_seal_strength(df)
df=detect_first_limit_up(df)
df = df.reset_index(drop=True)
train_df, test_df = train_test_split(df, test_size=0.4, random_state=42)
train_df["seal_success_label"] = label_zha_ban(train_df)["label"]

initial_capital = 1_000_000  # 初始资金100万

def train_seal_model(train_df, features, label_col='seal_success_label'):
    """
    训练炸板预测模型
    """
    X_train = train_df[features]
    y_train = train_df[label_col]
    
    X_train = X_train.fillna(0)
    y_train = y_train.fillna(0)
    model = LGBMClassifier(n_estimators=100, max_depth=5, random_state=0)
    model.fit(X_train, y_train)
    return model

# 读取数据
features = [
    'seal_strength', 'ask_volume1', 'bid_volume1',
    'order_imbalance', 'spread', 'ask1_locked'
]

# 训练模型
model = train_seal_model(train_df, features, label_col='seal_success_label')

# 预测test_df的炸板概率
X_test = test_df[features]
X_test = X_test.fillna(0)
test_df['seal_success_pred'] = model.predict(X_test)



def backtest_limit_up(test_df):
    test_df = test_df.copy()
    trades = []
    position = 0
    entry_idx = None
    entry_price = None
    entry_limit_up = None
    entry_date = None
    sell_state = {}
    capital = initial_capital
    used_capital = 0

    for idx in range(len(test_df)):
        row = test_df.iloc[idx]
        # 开仓
        if position == 0 and trigger_buy(test_df, idx):
            used_capital = capital * 0.8
            position = 1
            entry_idx = idx
            entry_price = row['ask_price1']
            entry_limit_up = row['ask_price1']
            entry_date = row['date']
            sell_state = {}  # 初始化分批止盈状态
            continue

        if position == 1:
            # 次日开盘低开止损
            if row['date'] > entry_date:
                if idx == 0 or test_df.iloc[idx - 1]['date'] == entry_date:
                    if row['current'] < entry_limit_up * (1 - 0.03):  # 0.03为低开止损阈值
                        exit_reason = "次日开盘低于涨停价3%止损"
                        exit_price = row['current']
                        pnl = (exit_price - entry_price) / entry_price
                        profit = used_capital * pnl
                        capital  += profit  
                        trades.append({
                            'entry_time': test_df.iloc[entry_idx]['clock'],
                            'exit_time': row['clock'],
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'used_capital': used_capital,
                            'profit': profit,
                            'capital': capital,
                            'reason': exit_reason
                        })
                        position = 0
                        sell_state = None
                        continue

            # 主动止盈（分批止盈）
            if trigger_sell(test_df, idx, entry_idx, entry_price, entry_date, sell_state):
                exit_reason = '主动止盈'
                exit_price = row['current']
                pnl = (exit_price-entry_price)/entry_price
                profit = used_capital * pnl
                capital = capital - used_capital + used_capital + profit  # 实际就是 capital += profit
                trades.append({
                    'entry_time': test_df.iloc[entry_idx]['clock'],
                    'exit_time': row['clock'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'used_capital': used_capital,
                    'profit': profit,
                    'capital': capital,
                    'reason': exit_reason
                })
                position = 0
                sell_state = None
                continue

            # 次日未连板主动止盈
            if row['date'] > entry_date:
                # 判断是否为次日最后一个tick
                next_idx = idx + 1
                is_last_tick_of_next_day = (next_idx == len(test_df)) or (test_df.iloc[next_idx]['date'] != row['date'])
                if is_last_tick_of_next_day:
                    df_next = test_df[test_df['date'] == row['date']]
                    limit_up_hit = (df_next['ask_price1'] == df_next['bid_price1']).any()
                    if not limit_up_hit:
                        exit_reason = "次日未连板，主动止盈"
                        exit_price = row['bid_price1']
                        pnl = (exit_price - entry_price) / entry_price
                        profit = used_capital * pnl
                        capital = capital - used_capital + used_capital + profit  # 实际就是 capital += profit
                        trades.append({
                            'entry_time': test_df.iloc[entry_idx]['clock'],
                            'exit_time': row['clock'],
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'used_capital': used_capital,
                            'profit': profit,
                            'capital': capital,
                            'reason': exit_reason
                        })
                        position = 0
                        sell_state = None
                        continue

            # risk control
            reason, exit_price = check_exit_conditions(
                df_tick=test_df,
                entry_index=entry_idx,
                
                enable_intraday_pullback=True,
            )
            if reason:
                pnl = (exit_price - entry_price) / entry_price
                profit = used_capital * pnl
                capital = capital - used_capital + used_capital + profit  # 实际就是 capital += profit
                trades.append({
                    'entry_time': test_df.iloc[entry_idx]['clock'],
                    'exit_time': row['clock'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'used_capital': used_capital,
                    'profit': profit,
                    'capital': capital,
                    'reason': reason
                })
                position = 0
                sell_state = None
                continue

    # 若最后还持仓，强制平仓
    if position == 1:
        exit_reason = '到期强平'
        exit_price = test_df.iloc[-1]['current']
        pnl = (exit_price - entry_price) / entry_price
        profit = used_capital * pnl
        capital = capital - used_capital + used_capital + profit  # 实际就是 capital += profit
        trades.append({
            'entry_time': test_df.iloc[entry_idx]['clock'],
            'exit_time': row['clock'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'used_capital': used_capital,
            'profit': profit,
            'capital': capital,
            'reason': exit_reason
        })

    return pd.DataFrame(trades)


if __name__ == '__main__':
    result = backtest_limit_up(test_df)
    result.to_csv('backtest_trades.csv', index=False)

    # 统计
    final_capital = result['capital'].iloc[-1] if not result.empty else initial_capital
    total_trades = len(result)
    win_trades = result[result['profit'] > 0]
    win_rate = len(win_trades) / total_trades if total_trades > 0 else 0
    max_profit = result['profit'].max() if not result.empty else 0
    max_loss = result['profit'].min() if not result.empty else 0
    avg_profit = result['profit'].mean() if not result.empty else 0
    sharpe = avg_profit / result['profit'].std() * np.sqrt(total_trades) if result['profit'].std() != 0 and total_trades > 0 else 0
    # 最大回撤
    
    max_drawdown = max_loss

    print('最终资金: {:.2f} 元'.format(final_capital))
    print('总交易次数:', total_trades)
    print('胜率: {:.2%}'.format(win_rate))
    print('最大单笔收益: {:.2%}'.format(max_profit))
    print('最大单笔亏损: {:.2%}'.format(max_loss))
    print('夏普比率: {:.4f}'.format(sharpe))
    print('最大回撤: {:.2f} 元'.format(abs(max_drawdown)))
    # 确保为datetime类型
    result['entry_time'] = pd.to_datetime(result['entry_time'])
    result['exit_time'] = pd.to_datetime(result['exit_time'])

    # 计算每笔持有时间（以小时为例）
    result['hold_time'] = (result['exit_time'] - result['entry_time']).dt.total_seconds() / 3600

    # 平均持有时间（小时）
    avg_hold_time = result['hold_time'].mean()
    print('平均持有时间: {:.2f} 小时'.format(avg_hold_time))

    