"""
红利低波ETF (512890) RSI策略回测
策略：RSI(15) < 32 买入，RSI(15) > 77 卖出

注意：512890是累积型ETF，分红已自动再投资体现在价格中，无需额外处理分红
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import time
import requests

try:
    import akshare as ak
    AKSHARE_IMPORT_ERROR = None
except Exception as e:
    ak = None
    AKSHARE_IMPORT_ERROR = e

# ============ 配置参数 ============
ETF_CODE = "512890"
ETF_NAME = "红利低波ETF"
RSI_PERIOD = 15  # 优化后：15日（原14日）
RSI_BUY_THRESHOLD = 32  # 优化后：32（原66）
RSI_SELL_THRESHOLD = 77  # 优化后：77（原81）
INITIAL_CAPITAL = 100000  # 初始资金10万
DATA_FETCH_RETRIES = int(os.environ.get("DATA_FETCH_RETRIES", 3))
DATA_FETCH_TIMEOUT = int(os.environ.get("DATA_FETCH_TIMEOUT", 20))
SINA_DATALEN = int(os.environ.get("SINA_DATALEN", 1800))

DATA_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://quote.eastmoney.com/",
}

# 基准ETF配置
BENCHMARK_ETFS = {
    'hs300': {'code': '510300', 'name': '沪深300ETF'},
    'gold': {'code': '518880', 'name': '黄金ETF'},
    'nasdaq': {'code': '159941', 'name': '纳指ETF'},
    'sp500': {'code': '513500', 'name': '标普500ETF'},
}


def load_previous_result(path):
    """Load previous backtest JSON for benchmark fallback when API fetch fails."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

# ============ RSI计算 ============
def calculate_rsi(prices, period=15):
    """计算RSI指标（使用EMA平滑，更敏感）"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    # 使用EMA而非SMA（更敏感，与优化脚本一致）
    alpha = 1 / period  # EMA平滑因子
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ============ 获取数据 ============
def normalize_etf_history(df, source_name, adjusted=True):
    """统一 ETF 历史行情字段，并做基础清洗。"""
    if df is None or df.empty:
        raise ValueError(f"{source_name} 返回空数据")

    column_map = {
        '日期': 'date',
        '开盘': 'open',
        '收盘': 'close',
        '最高': 'high',
        '最低': 'low',
        '成交量': 'volume',
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    missing_columns = {'date', 'close'} - set(df.columns)
    if missing_columns:
        raise ValueError(f"{source_name} 缺少字段: {', '.join(sorted(missing_columns))}")

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    for column in ['open', 'high', 'low', 'volume']:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')

    df = df.dropna(subset=['date', 'close'])
    df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{source_name} 清洗后无有效数据")

    start_date = df['date'].min().strftime('%Y-%m-%d')
    end_date = df['date'].max().strftime('%Y-%m-%d')
    print(f"{source_name} 获取到 {len(df)} 条数据，从 {start_date} 到 {end_date}")
    df.attrs['source_name'] = source_name
    df.attrs['adjusted'] = adjusted
    return df


def fetch_etf_data_from_akshare(code):
    """使用 AKShare 获取 ETF 前复权日线数据。"""
    if ak is None:
        raise RuntimeError(f"AKShare 导入失败: {AKSHARE_IMPORT_ERROR}")

    df = ak.fund_etf_hist_em(symbol=code, period="daily", adjust="qfq")
    return normalize_etf_history(df, "AKShare")


def get_eastmoney_secids(code):
    """生成东方财富 secid，ETF 512/510 等上海代码通常为 1.x。"""
    preferred_market = "1" if code.startswith(("5", "6", "9")) else "0"
    fallback_market = "0" if preferred_market == "1" else "1"
    return [f"{preferred_market}.{code}", f"{fallback_market}.{code}"]


def fetch_etf_data_from_eastmoney(code):
    """直接调用东方财富 K 线接口，作为 AKShare 被断连时的备用数据源。"""
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    last_error = None
    session = requests.Session()
    session.trust_env = False

    for secid in get_eastmoney_secids(code):
        params = {
            "secid": secid,
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": "101",
            "fqt": "1",  # 前复权
            "beg": "0",
            "end": "20500101",
        }
        try:
            response = session.get(
                url,
                params=params,
                headers=DATA_REQUEST_HEADERS,
                timeout=DATA_FETCH_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            klines = (payload.get("data") or {}).get("klines") or []
            if not klines:
                raise ValueError(f"东方财富 {secid} 无 K 线数据")

            rows = []
            for line in klines:
                parts = line.split(",")
                if len(parts) >= 6:
                    rows.append({
                        "date": parts[0],
                        "open": parts[1],
                        "close": parts[2],
                        "high": parts[3],
                        "low": parts[4],
                        "volume": parts[5],
                    })

            return normalize_etf_history(pd.DataFrame(rows), f"东方财富({secid})")
        except Exception as e:
            last_error = e
            print(f"东方财富 {secid} 获取失败: {e}")

    raise RuntimeError(f"东方财富备用数据源不可用: {last_error}")


def get_sina_symbol(code):
    """生成新浪行情 symbol，上海 ETF 使用 sh 前缀。"""
    market_prefix = "sh" if code.startswith(("5", "6", "9")) else "sz"
    return f"{market_prefix}{code}"


def fetch_etf_data_from_sina(code):
    """调用新浪日线接口，作为行情源被断开时的第二备用数据源。"""
    url = "https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20data=/CN_MarketDataService.getKLineData"
    symbol = get_sina_symbol(code)
    session = requests.Session()
    session.trust_env = False
    datalen_candidates = []
    for datalen in [SINA_DATALEN, 1800, 1500, 1000, 500]:
        if datalen > 0 and datalen not in datalen_candidates:
            datalen_candidates.append(datalen)

    last_error = None
    for datalen in datalen_candidates:
        try:
            response = session.get(
                url,
                params={
                    "symbol": symbol,
                    "scale": "240",
                    "ma": "no",
                    "datalen": str(datalen),
                },
                headers=DATA_REQUEST_HEADERS,
                timeout=DATA_FETCH_TIMEOUT,
            )
            response.raise_for_status()

            text = response.text
            start = text.find("[")
            end = text.rfind("]")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"新浪 datalen={datalen} 返回内容不是有效的 JSONP K 线数据")

            rows = [
                {
                    "date": item.get("day"),
                    "open": item.get("open"),
                    "close": item.get("close"),
                    "high": item.get("high"),
                    "low": item.get("low"),
                    "volume": item.get("volume"),
                }
                for item in json.loads(text[start : end + 1])
            ]
            return normalize_etf_history(pd.DataFrame(rows), f"新浪({symbol}, datalen={datalen})", adjusted=False)
        except Exception as e:
            last_error = e
            print(f"新浪 {symbol} datalen={datalen} 获取失败: {e}")

    raise RuntimeError(f"新浪备用数据源不可用: {last_error}")


def fetch_with_retries(source_name, fetcher):
    """对容易被 GitHub Actions 网络波动影响的数据源做短重试。"""
    last_error = None
    retries = max(1, DATA_FETCH_RETRIES)

    for attempt in range(1, retries + 1):
        try:
            return fetcher()
        except Exception as e:
            last_error = e
            print(f"{source_name} 第 {attempt}/{retries} 次获取失败: {e}")
            if attempt < retries:
                time.sleep(min(2 * attempt, 8))

    print(f"{source_name} 多次获取失败，准备切换数据源: {last_error}")
    return None


def get_etf_data(code):
    """获取ETF日线数据"""
    print(f"正在获取 {code} 历史数据...")

    data_sources = [
        ("AKShare", lambda: fetch_etf_data_from_akshare(code)),
        ("东方财富备用接口", lambda: fetch_etf_data_from_eastmoney(code)),
        ("新浪备用接口", lambda: fetch_etf_data_from_sina(code)),
    ]

    for source_name, fetcher in data_sources:
        df = fetch_with_retries(source_name, fetcher)
        if df is not None:
            return df

    print("获取ETF数据失败: 所有数据源均不可用")
    return None


def cached_strategy_prices(previous_result):
    """从上次回测结果中恢复主 ETF 的前复权收盘价缓存。"""
    strategy_values = ((previous_result or {}).get('daily_values') or {}).get('strategy') or []
    rows = [
        {'date': item.get('date'), 'close': item.get('close')}
        for item in strategy_values
        if item.get('date') and item.get('close') is not None
    ]
    if not rows:
        return None

    try:
        return normalize_etf_history(pd.DataFrame(rows), "缓存前复权历史")
    except Exception as e:
        print(f"读取缓存前复权历史失败: {e}")
        return None


def merge_with_cached_adjusted_history(df, previous_result):
    """新浪历史价未复权；仅用它补充缓存之后的新交易日，保持回测口径稳定。"""
    if df is None or df.attrs.get('adjusted', True):
        return df

    cached_df = cached_strategy_prices(previous_result)
    if cached_df is None or cached_df.empty:
        print("未找到可用的前复权历史缓存，继续使用当前数据源。")
        return df

    latest_cached_date = cached_df['date'].max()
    new_rows = df[df['date'] > latest_cached_date][['date', 'close']].copy()

    if new_rows.empty:
        print(f"当前数据源为未复权历史价，使用缓存前复权历史至 {latest_cached_date.strftime('%Y-%m-%d')}")
        return cached_df

    merged = pd.concat([cached_df[['date', 'close']], new_rows], ignore_index=True)
    merged = merged.sort_values('date').drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)
    merged.attrs['source_name'] = f"{cached_df.attrs.get('source_name', '缓存')} + {df.attrs.get('source_name', '新浪')} 新数据"
    merged.attrs['adjusted'] = True
    print(
        f"使用缓存前复权历史 {len(cached_df)} 条，追加 {len(new_rows)} 条新数据，"
        f"最新日期 {merged['date'].max().strftime('%Y-%m-%d')}"
    )
    return merged


def get_benchmark_data(code, name, index_type="index"):
    """获取基准指数数据"""
    print(f"正在获取基准 {name} 数据...")
    try:
        if index_type == "index":
            # 国内指数
            df = ak.index_zh_a_hist(symbol=code, period="daily", start_date="20131201")
        elif index_type == "us":
            # 美股指数 - 纳指100
            df = ak.index_us_stock_sina(symbol=code)
            
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.rename(columns={
            '日期': 'date',
            '收盘': 'close'
        })
        df = df.sort_values('date').reset_index(drop=True)
        print(f"获取到 {name} {len(df)} 条数据")
        return df[['date', 'close']]
    except Exception as e:
        print(f"获取 {name} 数据失败: {e}")
        return None


# ============ 回测引擎 ============
def run_backtest(df, initial_capital=INITIAL_CAPITAL):
    """
    执行RSI策略回测
    
    注意：512890是累积型ETF，分红已自动再投资体现在前复权价格中
    返回：交易记录、每日净值
    """
    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
    
    # 初始化
    cash = initial_capital
    shares = 0
    position = 0  # 0: 空仓, 1: 持仓
    
    trades = []  # 交易记录
    daily_values = []  # 每日净值
    
    for i, row in df.iterrows():
        date = row['date']
        price = row['close']
        rsi = row['rsi']
        date_str = date.strftime('%Y-%m-%d')
        
        # RSI信号判断
        if pd.notna(rsi):
            if rsi < RSI_BUY_THRESHOLD and position == 0:
                # 买入信号：满仓买入
                shares_to_buy = int(cash / price / 100) * 100  # 整百份
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    cash -= cost
                    shares += shares_to_buy
                    position = 1
                    trades.append({
                        'date': date_str,
                        'action': '买入',
                        'price': price,
                        'shares': shares_to_buy,
                        'amount': cost,
                        'rsi': rsi,
                        'total_shares': shares,
                        'cash': cash
                    })
                    
            elif rsi > RSI_SELL_THRESHOLD and position == 1:
                # 卖出信号：全部卖出
                if shares > 0:
                    sell_shares = int(shares / 100) * 100  # 整百份
                    if sell_shares > 0:
                        revenue = sell_shares * price
                        cash += revenue
                        shares -= sell_shares
                        if shares < 100:
                            # 剩余零头也卖掉
                            cash += shares * price
                            shares = 0
                        position = 0
                        trades.append({
                            'date': date_str,
                            'action': '卖出',
                            'price': price,
                            'shares': sell_shares,
                            'amount': revenue,
                            'rsi': rsi,
                            'total_shares': shares,
                            'cash': cash
                        })
        
        # 计算当日总资产
        total_value = cash + shares * price
        daily_values.append({
            'date': date_str,
            'close': price,
            'rsi': rsi if pd.notna(rsi) else None,
            'cash': cash,
            'shares': shares,
            'total_value': total_value,
            'return': (total_value / initial_capital - 1) * 100
        })
    
    return trades, daily_values


def calculate_buy_and_hold(df, initial_capital=INITIAL_CAPITAL):
    """计算买入持有策略
    
    注意：512890是累积型ETF，分红已体现在前复权价格中
    """
    start_price = df.iloc[0]['close']
    shares = int(initial_capital / start_price / 100) * 100
    remaining_cash = initial_capital - shares * start_price
    
    daily_values = []
    for _, row in df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        price = row['close']
        
        total_value = remaining_cash + shares * price
        daily_values.append({
            'date': date_str,
            'total_value': total_value,
            'return': (total_value / initial_capital - 1) * 100
        })
    
    return daily_values


def calculate_benchmark_return(df, initial_capital=INITIAL_CAPITAL, reference_dates=None):
    """计算基准收益
    
    Args:
        df: 基准数据DataFrame
        initial_capital: 初始资金
        reference_dates: 参考日期列表，用于对齐数据。如果提供，只返回这些日期的数据
    """
    if df is None or len(df) == 0:
        return []
    
    # 创建日期到价格的映射
    date_price_map = {}
    for _, row in df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        date_price_map[date_str] = row['close']
    
    # 如果提供了参考日期，按参考日期对齐
    if reference_dates:
        start_price = None
        daily_values = []
        
        for date_str in reference_dates:
            if date_str in date_price_map:
                price = date_price_map[date_str]
                if start_price is None:
                    start_price = price
                total_value = initial_capital * (price / start_price)
                daily_values.append({
                    'date': date_str,
                    'total_value': total_value,
                    'return': (total_value / initial_capital - 1) * 100
                })
            # 如果日期不存在，使用前一个值（向前填充）
            elif daily_values:
                daily_values.append({
                    'date': date_str,
                    'total_value': daily_values[-1]['total_value'],
                    'return': daily_values[-1]['return']
                })
        
        return daily_values
    
    # 原始逻辑
    start_price = df.iloc[0]['close']
    daily_values = []
    
    for _, row in df.iterrows():
        price = row['close']
        total_value = initial_capital * (price / start_price)
        daily_values.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'total_value': total_value,
            'return': (total_value / initial_capital - 1) * 100
        })
    
    return daily_values


def calculate_statistics(daily_values, trades):
    """计算策略统计指标"""
    if not daily_values:
        return {}
    
    returns = [d['return'] for d in daily_values]
    values = [d['total_value'] for d in daily_values]
    
    # 计算最大回撤
    peak = values[0]
    max_drawdown = 0
    for v in values:
        if v > peak:
            peak = v
        drawdown = (peak - v) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # 计算年化收益（使用自然日天数，而非交易日数）
    trading_days = len(daily_values)
    total_return = returns[-1]
    # 计算起止日期的自然天数
    from datetime import datetime
    start_date = datetime.strptime(daily_values[0]['date'], '%Y-%m-%d')
    end_date = datetime.strptime(daily_values[-1]['date'], '%Y-%m-%d')
    calendar_days = (end_date - start_date).days
    annual_return = ((1 + total_return / 100) ** (365 / calendar_days) - 1) * 100 if calendar_days > 0 else 0
    
    # 交易统计
    buy_trades = [t for t in trades if t['action'] == '买入']
    sell_trades = [t for t in trades if t['action'] == '卖出']
    
    # 计算胜率
    wins = 0
    for i, sell in enumerate(sell_trades):
        if i < len(buy_trades):
            if sell['price'] > buy_trades[i]['price']:
                wins += 1
    win_rate = (wins / len(sell_trades) * 100) if sell_trades else 0
    
    return {
        'total_return': round(total_return, 2),
        'annual_return': round(annual_return, 2),
        'max_drawdown': round(max_drawdown, 2),
        'trade_count': len(buy_trades),
        'win_rate': round(win_rate, 2),
        'start_date': daily_values[0]['date'],
        'end_date': daily_values[-1]['date'],
        'days': trading_days,
        'calendar_days': calendar_days
    }


def calculate_annual_return(total_return_pct, calendar_days):
    """计算复利年化收益率
    
    Args:
        total_return_pct: 总收益率百分比
        calendar_days: 自然日天数（非交易日）
    
    公式: annual_return = (1 + total_return) ^ (365/days) - 1
    """
    if calendar_days <= 0 or total_return_pct is None:
        return None
    return round(((1 + total_return_pct / 100) ** (365 / calendar_days) - 1) * 100, 2)


# ============ 主程序 ============
def main():
    print("=" * 60)
    print("红利低波ETF (512890) RSI策略回测")
    print("=" * 60)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "backtest_result.json")
    previous_result = load_previous_result(output_file)
    previous_stats = previous_result.get('statistics', {}) if previous_result else {}
    previous_daily = previous_result.get('daily_values', {}) if previous_result else {}
    
    # 1. 获取数据
    etf_df = get_etf_data(ETF_CODE)
    if etf_df is None:
        print("无法获取ETF数据，退出")
        return
    etf_df = merge_with_cached_adjusted_history(etf_df, previous_result)
    
    # 2. 统一时间范围
    start_date = etf_df['date'].min()
    end_date = etf_df['date'].max()
    print(f"\n回测区间: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    
    # 3. 获取基准ETF数据
    benchmark_data = {}
    for key, info in BENCHMARK_ETFS.items():
        print(f"正在获取 {info['name']} ({info['code']}) 数据...")
        try:
            df = get_etf_data(info['code'])
            if df is None:
                raise RuntimeError("所有数据源均不可用")
            if not df.attrs.get('adjusted', True):
                raise RuntimeError(f"{df.attrs.get('source_name', '当前数据源')} 为未复权历史价，使用上次基准结果兜底")
            # 筛选到相同时间范围
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            benchmark_data[key] = df[['date', 'close']]
            print(f"  获取到 {len(df)} 条数据")
        except Exception as e:
            print(f"  获取失败: {e}")
            benchmark_data[key] = None
    
    # 4. 执行回测（无需分红处理，累积型ETF分红已体现在价格中）
    print("\n正在执行RSI策略回测...")
    trades, strategy_values = run_backtest(etf_df)
    
    print("正在计算买入持有收益...")
    buyhold_values = calculate_buy_and_hold(etf_df)
    
    print("正在计算基准收益...")
    
    # 获取策略的日期列表，用于对齐所有基准数据
    strategy_dates = [d['date'] for d in strategy_values]
    
    # 计算各基准收益（使用策略日期对齐）
    benchmark_values = {}
    benchmark_returns = {}
    for key, df in benchmark_data.items():
        if df is not None and len(df) > 0:
            values = calculate_benchmark_return(df, reference_dates=strategy_dates)
            benchmark_values[key] = values
            benchmark_returns[key] = round(values[-1]['return'], 2) if values else None
        else:
            fallback_values = previous_daily.get(key, [])
            fallback_return = previous_stats.get(f'{key}_return')
            benchmark_values[key] = fallback_values if fallback_values else []
            benchmark_returns[key] = fallback_return
            if fallback_return is not None:
                print(f"  {BENCHMARK_ETFS[key]['name']} 使用上次结果兜底: {fallback_return:.2f}%")
            else:
                print(f"  {BENCHMARK_ETFS[key]['name']} 无可用兜底数据")
    
    # 5. 计算统计指标
    strategy_stats = calculate_statistics(strategy_values, trades)
    buyhold_stats = calculate_statistics(buyhold_values, [])
    
    # 使用自然日天数计算年化收益率
    calendar_days = strategy_stats.get('calendar_days', strategy_stats['days'])
    
    # 计算各基准的年化收益率
    benchmark_annuals = {}
    for key in benchmark_returns:
        annual = calculate_annual_return(benchmark_returns.get(key), calendar_days)
        if annual is None:
            annual = previous_stats.get(f'{key}_annual')
        benchmark_annuals[key] = annual

    # 记录回测天数供导出使用
    backtest_days = strategy_stats.get('days', len(strategy_values))
    
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    print(f"\n【RSI策略】")
    print(f"  总收益率: {strategy_stats['total_return']:.2f}%")
    print(f"  年化收益: {strategy_stats['annual_return']:.2f}%")
    print(f"  最大回撤: {strategy_stats['max_drawdown']:.2f}%")
    print(f"  交易次数: {strategy_stats['trade_count']} 次")
    print(f"  胜率: {strategy_stats['win_rate']:.2f}%")
    
    print(f"\n【买入持有】")
    print(f"  总收益率: {buyhold_stats['total_return']:.2f}%")
    print(f"  年化收益: {buyhold_stats['annual_return']:.2f}%")
    
    for key, info in BENCHMARK_ETFS.items():
        if benchmark_returns.get(key) is not None:
            print(f"\n【{info['name']}】")
            print(f"  总收益率: {benchmark_returns[key]:.2f}%")
    
    # 6. 导出数据为JSON
    # 准备导出数据
    export_data = {
        'meta': {
            'etf_code': ETF_CODE,
            'etf_name': ETF_NAME,
            'strategy': f'RSI({RSI_PERIOD}) < {RSI_BUY_THRESHOLD} 买入, > {RSI_SELL_THRESHOLD} 卖出',
            'initial_capital': INITIAL_CAPITAL,
            'start_date': strategy_stats['start_date'],
            'end_date': strategy_stats['end_date'],
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'statistics': {
            'strategy': strategy_stats,
            'buyhold': buyhold_stats,
            'hs300_return': benchmark_returns.get('hs300'),
            'hs300_annual': benchmark_annuals.get('hs300'),
            'gold_return': benchmark_returns.get('gold'),
            'gold_annual': benchmark_annuals.get('gold'),
            'nasdaq_return': benchmark_returns.get('nasdaq'),
            'nasdaq_annual': benchmark_annuals.get('nasdaq'),
            'sp500_return': benchmark_returns.get('sp500'),
            'sp500_annual': benchmark_annuals.get('sp500'),
            'backtest_days': backtest_days,
        },
        'trades': trades,
        'daily_values': {
            'strategy': strategy_values,
            'buyhold': buyhold_values,
            'hs300': benchmark_values.get('hs300', []),
            'gold': benchmark_values.get('gold', []),
            'nasdaq': benchmark_values.get('nasdaq', []),
            'sp500': benchmark_values.get('sp500', []),
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n回测结果已保存至: {output_file}")
    
    # 同时复制到docs目录供网页使用
    docs_output = os.path.join(os.path.dirname(output_dir), "docs", "backtest_result.json")
    with open(docs_output, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False)
    print(f"网页数据已保存至: {docs_output}")
    
    return export_data


if __name__ == "__main__":
    main()
