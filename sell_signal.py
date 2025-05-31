def trigger_sell(df, idx, entry_idx, entry_price, entry_date, sell_state=None):
    """
    分批止盈逻辑：
    - 第一次止盈：当前收益>=5%
    - 第二次止盈：当前收益>=8%
    sell_state: 可选，记录止盈状态（如已止盈次数）
    返回True表示触发止盈，否则False
    """
    current_price = df.iloc[idx]['current']
    pnl = (current_price - entry_price) / entry_price
    # 分批止盈状态管理
    if sell_state is not None:
        # 如果第一次止盈已触发，判断第二次止盈
        if sell_state.get('first_sell', False) and not sell_state.get('second_sell', False):
            if pnl >= 0.08:
                sell_state['second_sell'] = True
                return True
        # 如果还没止盈，判断第一次止盈
        elif not sell_state.get('first_sell', False):
            if pnl >= 0.05:
                sell_state['first_sell'] = True
                return True
        return False
    else:
        # 无状态时，单次止盈逻辑
        if pnl >= 0.05:
            return True
        return False
