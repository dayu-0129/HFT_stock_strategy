# HFT_stock_strategy
# 高频打板策略回测系统 

本项目基于逐笔 tick 级盘口数据（含十档盘口 + 成交价）实现了一个用于日内打板策略的信号识别、风控控制和收益回测系统，结合机器学习封板评分。

---

##  功能简介

-  封板信号识别（盘口挂单识别首次涨停）
-  封板强度指标构建（成交量 vs 封单量）
-  炸板检测与未回封止损机制
-  多种退出逻辑（次日低开、盘中回撤、封板失败等）
-  策略回测并输出交易记录（含收益率、退出理由）
-  LightGBM 模型预测封板成功概率

---

##  项目结构

<pre>
tick/
  main.py              # 回测主执行逻辑
  bactest_trades.csv   # 交易记录
  risk_control.py      # 风控判断模块
  buy_signal.py        # 买入信号触发模块
  sell_signal.py       # 卖出信号触发模块
  data.parquet         # Tick 数据
  utility.py           # 通用函数
</pre>

---

##  核心逻辑简要说明

###  买入信号 (`trigger_buy`)
- ask_price1 == bid_price1 == 涨停价`
- ask_volume1 > 封板量阈值`
- current < ask_price1` 且 `ask_price1` 相比上一tick变化（首次封板）
- seal_success_pred == 0（预测不会炸板）
###  止损风控 (`check_exit_conditions`)
- 次日开盘低开 > 3%
- 盘中最大回撤 > 5%
- 连板失败（次日未继续涨停）

###  主动退出 (`trigger_sell`)
- 分批止盈（5%，8%）

---

##  LightGBM 模型预测封板成功概率
### features：
1. 买一卖一价差（反映价差宽度）
df['spread'] = df['ask_price1'] - df['bid_price1']

 2. 判断卖一价格是否锁死（涨停时常见）
df['ask1_locked'] = df['ask_price1'].diff().abs() < 1e-6

3. 委托单方向不平衡指标（Order Imbalance）
df['order_imbalance'] = (
    df[[f'bid_volume{i}' for i in range(1, 11)]].sum(axis=1) -
    df[[f'ask_volume{i}' for i in range(1, 11)]].sum(axis=1)
) / (
    df[[f'bid_volume{i}' for i in range(1, 11)]].sum(axis=1) +
    df[[f'ask_volume{i}' for i in range(1, 11)]].sum(axis=1)
)
### label：下一个 tick 是否出现炸板行为
train_df["seal_success_label"] = label_zha_ban(train_df)["label"]

 炸板判定函数：卖一价上移且成交价低于卖一价
def detect_zha_ban(df, i):
    return (df.loc[i, 'ask_price1'] > df.loc[i, 'bid_price1']) and \
           (df.loc[i, 'current'] < df.loc[i, 'ask_price1'])

---

## 📊 输出说明

回测将输出 `trades.csv`，含：

| 字段名 | 含义 |
|--------|------|
| `entry_time` | 建仓时间 |
| `exit_time`  | 平仓时间 |
| `entry_price` | 买入价 |
| `exit_price` | 卖出价 |
| `pnl` | 收益率 |
| `reason` | 平仓理由（止损、止盈、未封板等） |

---



## 回测结果
初始资金：1000000 元  
最终资金：153454.57 元  

总交易次数：99  
胜率：38.38%  

最大单笔收益：218623  
最大单笔亏损：-227194

夏普比率：-1.4517  
最大回撤：227194.24 元  

平均持有时间：51.89 小时  
