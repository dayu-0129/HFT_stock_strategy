import pandas as pd
def check_exit_conditions(df_tick, entry_index, 
                          enable_intraday_pullback=True,
                          pullback_threshold=0.05,
                          ):
    df_tick['date'] = pd.to_datetime(df_tick['clock']).dt.date
    exit_reason = None
    exit_price = None

   # 回撤止损 
    if enable_intraday_pullback:
        post_entry = df_tick.iloc[entry_index:]
        cummax_price = post_entry['current'].cummax()
        pullback = (cummax_price - post_entry['current']) / cummax_price
        if pullback.max() > pullback_threshold:
            exit_row = post_entry[pullback > pullback_threshold].iloc[0]
            exit_reason = f"持仓期间回撤超过{pullback_threshold*100:.0f}%止损"
            exit_price = exit_row['bid_price1']
            return exit_reason, exit_price

    return None, None  