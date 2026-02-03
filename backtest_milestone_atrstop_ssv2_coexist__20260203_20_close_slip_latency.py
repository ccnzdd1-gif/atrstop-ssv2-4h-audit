import pandas as pd
import numpy as np

DATA_PATH = 'quant_model/BTCUSDT_4h_real_binance__20260203_20.csv'
OUT_TRADES = 'quant_model/final_versions/MILESTONE_20260203_ATRSTOP_V1V3_SENTINEL/atrstop_ssv2_coexist_trades.csv'
FEE_RATE = 0.0004
# Backtest execution realism
SLIPPAGE_PCT = 0.001  # 0.10% slippage per trade side
LATENCY_BARS = 1      # execute on next bar close


def calculate_indicators(df):
    # Core MAs
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['ema200_slope'] = df['ema200'].diff(5)
    df['ema_trend_slope_monthly_neg'] = df['ema200_slope'].rolling(30).apply(lambda x: (x < 0).all(), raw=False).fillna(0)
    df['ema_fast'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=50, adjust=False).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['sma50_slope'] = df['sma50'].diff()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # ADX
    high = df['high']
    low = df['low']
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr = np.maximum(high - low, np.maximum((high - df['close'].shift(1)).abs(), (low - df['close'].shift(1)).abs()))
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr14 + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr14 + 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    df['adx'] = dx.rolling(14).mean().fillna(0)

    # Bollinger
    df['sma20'] = df['close'].rolling(20).mean()
    df['std20'] = df['close'].rolling(20).std()
    df['upper_bb'] = df['sma20'] + (2 * df['std20'])
    df['bb_width_pct'] = (df['upper_bb'] - (df['sma20'] - 2 * df['std20'])) / (df['sma20'] + 1e-9)

    # Volume
    df['vol_sma20'] = df['volume'].rolling(20).mean()

    # ATR + vol ratio
    prev_close = df['close'].shift(1)
    tr2 = np.maximum(df['high'] - df['low'], np.maximum((df['high'] - prev_close).abs(), (df['low'] - prev_close).abs()))
    df['atr'] = tr2.rolling(14).mean()
    df['atr_ma100'] = df['atr'].rolling(100).mean()
    df['vol_ratio'] = (df['atr'] / (df['atr_ma100'] + 1e-9)).fillna(1.0)

    # Accel
    df['close_smooth'] = df['close'].ewm(span=5, adjust=False).mean()
    df['velocity'] = df['close_smooth'].diff()
    df['acceleration'] = df['velocity'].diff()
    df['accel_z'] = (df['acceleration'] - df['acceleration'].rolling(100).mean()) / (df['acceleration'].rolling(100).std() + 1e-9)

    # Ranges
    df['high_10d'] = df['high'].rolling(60).max().shift(1)
    df['low_10d'] = df['low'].rolling(60).min().shift(1)
    df['range_pct_10d'] = (df['high_10d'] - df['low_10d']) / (df['low_10d'] + 1e-9)
    df['2d_ret'] = df['close'].pct_change(12)
    df['range_regime'] = (df['adx'] < 20) & (df['bb_width_pct'] < 0.06) & (df['range_pct_10d'] < 0.15)

    # Long signals
    r_v = df['high'] - df['low']
    b_v = abs(df['close'] - df['open'])
    l_s = np.minimum(df['open'], df['close']) - df['low']
    is_low = df['low'] == df['low'].rolling(14, center=True).min()
    df['sig_hammer'] = (b_v < (r_v / 3)) & (l_s > (2 * b_v)) & (r_v > 0) & (is_low) & (df['rsi'] < 40)
    df['sig_sma50_b'] = (df['sma50_slope'] > 0) & (df['low'] <= df['sma50'] * 1.02) & (df['close_smooth'] > df['sma50']) & (df['close'] > df['open'])
    df['sig_ema200_rec'] = (df['close'].shift(1) < df['ema200'].shift(1)) & (df['close'] > df['ema200']) & (df['volume'] > df['vol_sma20'] * 2.5) & (df['rsi'] < 80)
    df['sig_trend_rec'] = (df['close'].shift(1) < df['sma50'].shift(1)) & (df['close'] > df['sma50']) & (df['close'] > df['ema200'])
    df['sig_panic'] = df['2d_ret'] < -0.10
    df['sig_boll'] = (df['bb_width_pct'].rolling(5).min() < 0.08) & (df['close'] > df['upper_bb']) & (df['volume'] > df['vol_sma20'] * 1.5) & (df['rsi'] < 75)

    # Short signals
    df['sig_short'] = (df['close_smooth'] < df['ema200']) & (df['rsi'] > 60) & (df['close'] < df['open'])

    # Shooting star V2 (wick/body constraints)
    upper = df['high'] - np.maximum(df['open'], df['close'])
    lower = np.minimum(df['open'], df['close']) - df['low']
    df['sig_shooting_star_v2'] = (upper >= 2 * b_v) & (lower <= b_v) & (b_v > 0) & (df['close'] < df['open'])

    return df


def run_backtest(file_path=DATA_PATH):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = calculate_indicators(df)
    df = df[df['timestamp'] >= '2021-02-01'].reset_index(drop=True)

    initial_balance = 10000.0
    balance = initial_balance
    equity = initial_balance

    pos_long = 0.0
    entry_long = 0.0
    pos_short = 0.0
    entry_short = 0.0
    l_meta = {}
    s_meta = {}
    peak_p = 0.0

    equity_curve = []
    trades = []

    def exec_price(price, side, is_entry=True):
        # adverse slippage
        if side == 'LONG':
            return price * (1 + SLIPPAGE_PCT) if is_entry else price * (1 - SLIPPAGE_PCT)
        else:
            return price * (1 - SLIPPAGE_PCT) if is_entry else price * (1 + SLIPPAGE_PCT)

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['close']
        vol_ratio = float(np.clip(row['vol_ratio'], 0.7, 1.6))

        pnl_long = (price - entry_long) * pos_long
        pnl_short = (entry_short - price) * abs(pos_short)
        equity = balance + pnl_long + pnl_short
        equity_curve.append({'time': row['timestamp'], 'equity': equity})

        # LONG EXIT (close-price trigger)
        if pos_long > 0:
            peak_p = max(peak_p, row['close'])
            sl = l_meta['sl']; tp = l_meta['tp']; ts = l_meta.get('ts')
            executed = False; exit_p = price; reason=None
            if price <= sl: executed=True; exit_p=sl; reason='SL'
            elif tp and price >= tp: executed=True; exit_p=tp; reason='TP'
            elif ts and price <= peak_p*(1-ts): executed=True; exit_p=peak_p*(1-ts); reason='TRAIL'
            elif (price-entry_long)/entry_long > 0.03 and row['accel_z'] < -1.5: executed=True; exit_p=price; reason='ACCEL'
            elif row['ema_fast'] < prev['ema_fast'] and row['ema_fast'] < row['ema_slow']: executed=True; exit_p=price; reason='EMA'
            if executed and i + LATENCY_BARS < len(df):
                exec_row = df.iloc[i + LATENCY_BARS]
                exec_p = exec_price(exit_p, 'LONG', is_entry=False)
                fee = (pos_long * exec_p) * FEE_RATE
                balance += (exec_p - entry_long) * pos_long - fee
                trades.append({'entry_time': l_meta['entry_time'], 'exit_time': exec_row['timestamp'], 'side': 'LONG',
                               'entry_price': entry_long, 'exit_price': exec_p, 'reason': reason})
                pos_long = 0; entry_long = 0

        # SHORT EXIT (V2: TP1/TP2, close-price trigger)
        if pos_short < 0:
            sl = s_meta['sl']; tp2 = s_meta['tp2']
            tp1 = s_meta.get('tp1'); tp1_hit = s_meta.get('tp1_hit', False)
            executed = False; exit_p = price; reason=None

            # TP1 partial 50%
            if (not tp1_hit) and tp1 and price <= tp1 and i + LATENCY_BARS < len(df):
                exec_row = df.iloc[i + LATENCY_BARS]
                cover_amt = abs(pos_short) * 0.5
                exec_p = exec_price(tp1, 'SHORT', is_entry=False)
                fee = (cover_amt * exec_p) * FEE_RATE
                balance += (entry_short - exec_p) * cover_amt - fee
                pos_short += cover_amt
                s_meta['tp1_hit'] = True
                s_meta['sl'] = entry_short  # move SL to BE

            if price >= sl: executed=True; exit_p=sl; reason='SL'
            elif tp2 and price <= tp2: executed=True; exit_p=tp2; reason='TP2'
            elif (entry_short - price)/entry_short > 0.03 and row['accel_z'] > 1.5:
                executed=True; exit_p=price; reason='ACCEL'

            if executed and i + LATENCY_BARS < len(df):
                exec_row = df.iloc[i + LATENCY_BARS]
                exec_p = exec_price(exit_p, 'SHORT', is_entry=False)
                fee = (abs(pos_short) * exec_p) * FEE_RATE
                balance += (entry_short - exec_p) * abs(pos_short) - fee
                trades.append({'entry_time': s_meta['entry_time'], 'exit_time': exec_row['timestamp'], 'side': 'SHORT',
                               'entry_price': entry_short, 'exit_price': exec_p, 'reason': reason})
                pos_short = 0; entry_short = 0

        # ENTRIES (long)
        if pos_long == 0:
            trig = False
            sma50_trig = False
            if row['sig_hammer']:
                sl=price*0.925; tp=price*1.10; ts=None; trig=True
            elif row['sig_panic']:
                sl=price*0.912; tp=None; ts=0.15; trig=True
            elif row['sig_ema200_rec']:
                sl=price*0.92; tp=price*1.12; ts=None; trig=True
            elif price >= row['ema200']:
                if (not row.get('range_regime', False)) and row['sig_sma50_b'] and row['ema200_slope']>0 and row['bb_width_pct']>0.04 and row['volume']>row['vol_sma20']:
                    sl=price*0.95; tp=None; ts=0.08; trig=True; sma50_trig=True
                elif row['sig_trend_rec']:
                    sl=price*0.94; tp=price*1.06; ts=None; trig=True
                elif row['sig_boll']:
                    sl=row['sma20']
                    if (price-sl)/price < 0.005: sl=price*0.99
                    tp=None; ts=0.10; trig=True
            if trig and i + LATENCY_BARS < len(df):
                exec_row = df.iloc[i + LATENCY_BARS]
                exec_p = exec_price(price, 'LONG', is_entry=True)
                leverage_long = 2.0 / vol_ratio
                if sma50_trig:
                    leverage_long = min(leverage_long, 1.2)
                notional = equity * leverage_long; fee = notional * FEE_RATE
                pos_long = (notional - fee) / exec_p
                entry_long = exec_p
                l_meta = {'sl': sl, 'tp': tp, 'ts': ts, 'entry_time': exec_row['timestamp']}
                # Original ATRSTOP SL override (fixed 3*ATR)
                l_meta['sl'] = exec_p - 3.0 * row['atr']
                balance -= fee; peak_p = exec_p

        # ENTRIES (short: original + shooting star v2)
        if pos_short == 0 and (not row.get('range_regime', False)):
            allow_short = False
            if row['sig_short'] and row['ema_trend_slope_monthly_neg'] == 1.0 and row['bb_width_pct']>0.06 and row['rsi']>65:
                if (price < row['ema200'] * 0.98 and row['ema200_slope'] < 0):
                    allow_short = True
            if row['sig_shooting_star_v2'] and (row['close'] < row['ema200']) and (row['adx'] > 15) and (row['rsi'] > 55):
                allow_short = True
            if allow_short and i + LATENCY_BARS < len(df):
                exec_row = df.iloc[i + LATENCY_BARS]
                exec_p = exec_price(price, 'SHORT', is_entry=True)
                leverage_short = 1.0 / vol_ratio
                notional = equity * leverage_short; fee = notional * FEE_RATE
                pos_short = -((notional - fee) / exec_p)
                entry_short = exec_p
                s_meta = {
                    'sl': exec_p + 3.0 * row['atr'],
                    'tp1': exec_p - 1.5 * row['atr'],
                    'tp2': exec_p - 3.0 * row['atr'],
                    'tp1_hit': False,
                    'entry_time': exec_row['timestamp']
                }
                balance -= fee

    # yearly return + max DD
    rec = pd.DataFrame(equity_curve)
    rec['year'] = rec['time'].dt.year
    yearly = []
    for y, g in rec.groupby('year'):
        start_eq = g['equity'].iloc[0]
        end_eq = g['equity'].iloc[-1]
        ret = (end_eq - start_eq) / start_eq
        peak = g['equity'].cummax()
        dd = (g['equity'] - peak) / peak
        max_dd = dd.min()
        yearly.append((y, ret, max_dd))

    print("\n--- ATRSTOP Milestone + Shooting Star V2 (Coexist) ---")
    for y, r, d in sorted(yearly):
        print(f"{y}: {r:+.2%} | MaxDD {d:.2%}")

    # total return + overall max DD
    total_return = (rec['equity'].iloc[-1] - initial_balance) / initial_balance
    peak = rec['equity'].cummax()
    dd = (rec['equity'] - peak) / peak
    max_dd = dd.min()
    print(f"TOTAL: {total_return:+.2%} | MaxDD {max_dd:.2%}")

    if trades:
        out = pd.DataFrame(trades)
        out.to_csv(OUT_TRADES, index=False)
        print(f"Trades saved: {OUT_TRADES} ({len(out)} rows)")


if __name__ == '__main__':
    run_backtest(DATA_PATH)
