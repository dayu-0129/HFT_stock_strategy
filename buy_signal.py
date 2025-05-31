def trigger_buy(df, idx, threshold=200000):
    """
    买入信号逻辑：
    - new_limit_up为True
    - seal_strength >= threshold
    - seal_success_pred == 0（预测不会炸板）
    """
    row = df.iloc[idx]
    return (
        row.get('new_limit_up', False)
        and row.get('seal_strength', 0) >= threshold
        and row.get('seal_success_pred', 1) == 0
    )
