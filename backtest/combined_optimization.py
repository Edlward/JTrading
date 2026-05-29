import copy
import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd


# ============ Configuration ============
ETF_CODE = "512890"
INITIAL_CAPITAL = 100000
ITERATIONS = 3000
RSI_PERIOD = 15
VOL_ANCHOR = 15.0


def get_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    return {
        "script_dir": script_dir,
        "project_dir": project_dir,
        "backtest_result": os.path.join(script_dir, "backtest_result.json"),
        "docs_result": os.path.join(project_dir, "docs", "backtest_result.json"),
        "best_params": os.path.join(script_dir, "best_combined_params.json"),
    }


def load_source_data():
    """Load the existing backtest payload and extract the shared price series."""
    paths = get_paths()

    with open(paths["backtest_result"], "r", encoding="utf-8") as file:
        source_data = json.load(file)

    daily_values = source_data.get("daily_values", {})
    price_series = (
        daily_values.get("strategy_dynamic")
        or daily_values.get("strategy_ideal")
        or daily_values.get("strategy")
        or daily_values.get("buyhold")
    )

    if not price_series:
        raise ValueError("backtest_result.json does not contain a usable price series.")

    df = pd.DataFrame(
        [
            {
                "date": pd.to_datetime(item["date"]),
                "close": float(item["close"]),
            }
            for item in price_series
        ]
    )
    df = df.sort_values("date").reset_index(drop=True)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["rsi_15"] = calculate_rsi_ema(df["close"], RSI_PERIOD)
    return source_data, df, paths


# ============ Indicator Functions ============

def calculate_rsi_ema(series, period):
    """RSI with EMA smoothing."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    alpha = 1 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100)
    return rsi


def calculate_volatility(log_ret, window):
    """Historical volatility (annualized %)."""
    return log_ret.rolling(window=window, min_periods=window).std() * np.sqrt(252) * 100


def normalize_params(params):
    return {
        "rsi_period": int(params["rsi_period"]),
        "rsi_buy_base": int(params["rsi_buy_base"]),
        "rsi_sell_base": int(params["rsi_sell_base"]),
        "vol_window": int(params["vol_window"]),
        "k_vol": float(params["k_vol"]),
        "vol_anchor": float(params.get("vol_anchor", VOL_ANCHOR)),
    }


def round_or_none(value, digits=6):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


# ============ Backtest Engine ============

def run_combined_backtest(df, params, initial_capital=INITIAL_CAPITAL, capture_details=False):
    params = normalize_params(params)

    if params["rsi_period"] == RSI_PERIOD and "rsi_15" in df.columns:
        rsi = df["rsi_15"]
    else:
        rsi = calculate_rsi_ema(df["close"], params["rsi_period"])

    vol = calculate_volatility(df["log_ret"], params["vol_window"])
    buy_threshold = (params["rsi_buy_base"] - params["k_vol"] * (vol - params["vol_anchor"])).clip(20, 50)
    sell_threshold = (params["rsi_sell_base"] + params["k_vol"] * (vol - params["vol_anchor"])).clip(60, 90)

    cash = float(initial_capital)
    shares = 0.0
    position = 0
    trades = [] if capture_details else None
    daily_values = [] if capture_details else None
    last_price = float(df["close"].iloc[-1])

    for row in zip(
        df["date"],
        df["close"],
        rsi,
        vol,
        buy_threshold,
        sell_threshold,
    ):
        trade_date, price, current_rsi, current_vol, current_buy, current_sell = row
        date_str = trade_date.strftime("%Y-%m-%d")
        price = float(price)
        last_price = price

        rsi_value = None if pd.isna(current_rsi) else float(current_rsi)
        vol_value = None if pd.isna(current_vol) else float(current_vol)
        buy_value = None if pd.isna(current_buy) else float(current_buy)
        sell_value = None if pd.isna(current_sell) else float(current_sell)

        if rsi_value is not None and vol_value is not None:
            if position == 0 and rsi_value < buy_value:
                shares_to_buy = cash / price if price > 0 else 0.0
                if shares_to_buy > 0:
                    amount = shares_to_buy * price
                    shares = shares_to_buy
                    cash = 0.0
                    position = 1
                    if capture_details:
                        trades.append(
                            {
                                "date": date_str,
                                "action": "买入",
                                "price": round(price, 6),
                                "shares": round(shares_to_buy, 6),
                                "amount": round(amount, 6),
                                "rsi": round(rsi_value, 4),
                                "volatility": round(vol_value, 4),
                                "buy_threshold": round(buy_value, 4),
                                "sell_threshold": round(sell_value, 4),
                                "reason": (
                                    f"RSI({rsi_value:.1f}) < 动态买入阈值({buy_value:.1f}) | "
                                    f"Vol:{vol_value:.1f}"
                                ),
                            }
                        )
            elif position == 1 and rsi_value > sell_value:
                if shares > 0:
                    amount = shares * price
                    cash += amount
                    if capture_details:
                        trades.append(
                            {
                                "date": date_str,
                                "action": "卖出",
                                "price": round(price, 6),
                                "shares": round(shares, 6),
                                "amount": round(amount, 6),
                                "rsi": round(rsi_value, 4),
                                "volatility": round(vol_value, 4),
                                "buy_threshold": round(buy_value, 4),
                                "sell_threshold": round(sell_value, 4),
                                "reason": (
                                    f"RSI({rsi_value:.1f}) > 动态卖出阈值({sell_value:.1f}) | "
                                    f"Vol:{vol_value:.1f}"
                                ),
                            }
                        )
                    shares = 0.0
                    position = 0

        if capture_details:
            total_value = cash + shares * price
            daily_values.append(
                {
                    "date": date_str,
                    "close": round(price, 6),
                    "rsi": round_or_none(rsi_value, 4),
                    "volatility": round_or_none(vol_value, 4),
                    "buy_threshold": round_or_none(buy_value, 4),
                    "sell_threshold": round_or_none(sell_value, 4),
                    "cash": round(cash, 6),
                    "shares": round(shares, 6),
                    "total_value": round(total_value, 6),
                    "return": round((total_value / initial_capital - 1) * 100, 6),
                }
            )

    final_value = cash + shares * last_price
    total_return = (final_value / initial_capital - 1) * 100

    result = {
        "params": params,
        "final_value": final_value,
        "total_return": total_return,
    }
    if capture_details:
        result["trades"] = trades
        result["daily_values"] = daily_values
    return result


def calculate_statistics(daily_values, trades):
    """Calculate the statistics required by the backtest UI."""
    if not daily_values:
        return {}

    values = [float(item["total_value"]) for item in daily_values]
    total_return = float(daily_values[-1]["return"])

    peak = values[0]
    max_drawdown = 0.0
    for value in values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100 if peak else 0.0
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    start_date = datetime.strptime(daily_values[0]["date"], "%Y-%m-%d")
    end_date = datetime.strptime(daily_values[-1]["date"], "%Y-%m-%d")
    calendar_days = (end_date - start_date).days
    annual_return = ((1 + total_return / 100) ** (365 / calendar_days) - 1) * 100 if calendar_days > 0 else 0.0

    buy_trades = [trade for trade in trades if trade["action"] == "买入"]
    sell_trades = [trade for trade in trades if trade["action"] == "卖出"]
    wins = sum(
        1
        for buy_trade, sell_trade in zip(buy_trades, sell_trades)
        if float(sell_trade["price"]) > float(buy_trade["price"])
    )
    win_rate = (wins / len(sell_trades) * 100) if sell_trades else 0.0

    return {
        "total_return": round(total_return, 2),
        "annual_return": round(annual_return, 2),
        "max_drawdown": round(max_drawdown, 2),
        "trade_count": len(buy_trades),
        "win_rate": round(win_rate, 2),
        "start_date": daily_values[0]["date"],
        "end_date": daily_values[-1]["date"],
        "days": len(daily_values),
        "calendar_days": calendar_days,
    }


def generate_random_params():
    return {
        "rsi_period": RSI_PERIOD,
        "rsi_buy_base": random.randint(25, 45),
        "rsi_sell_base": random.randint(65, 85),
        "vol_window": random.randint(10, 60),
        "k_vol": random.uniform(-0.5, 1.0),
        "vol_anchor": VOL_ANCHOR,
    }


def build_export_payload(source_data, best_result, best_stats, base_result, base_stats):
    export_data = copy.deepcopy(source_data)

    export_data.setdefault("meta", {})
    export_data.setdefault("statistics", {})
    export_data.setdefault("daily_values", {})

    export_data["meta"]["etf_code"] = ETF_CODE
    export_data["meta"]["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    export_data["meta"]["dynamic_params"] = best_result["params"]
    export_data["meta"]["dynamic_strategy"] = (
        "动态 RSI 阈值: "
        f"RSI({best_result['params']['rsi_period']}), "
        f"买入基准 {best_result['params']['rsi_buy_base']}, "
        f"卖出基准 {best_result['params']['rsi_sell_base']}, "
        f"波动率窗口 {best_result['params']['vol_window']}, "
        f"k={best_result['params']['k_vol']:.6f}"
    )
    export_data["meta"]["dynamic_optimization"] = {
        "iterations": ITERATIONS,
        "baseline_params": base_result["params"],
        "baseline_total_return": round(base_result["total_return"], 2),
        "best_total_return": round(best_result["total_return"], 2),
        "improvement": round(best_result["total_return"] - base_result["total_return"], 2),
    }

    export_data["statistics"]["strategy_dynamic"] = best_stats
    export_data["statistics"]["strategy_dynamic_baseline"] = base_stats
    export_data["statistics"]["strategy_dynamic_improvement"] = round(
        best_stats["total_return"] - base_stats["total_return"], 2
    )
    export_data["statistics"]["strategy_ideal"] = base_stats
    export_data["statistics"]["backtest_days"] = best_stats["days"]

    export_data["trades_dynamic"] = best_result["trades"]
    export_data["trades_ideal"] = base_result["trades"]
    export_data["daily_values"]["strategy_dynamic"] = best_result["daily_values"]
    export_data["daily_values"]["strategy_ideal"] = base_result["daily_values"]

    return export_data


def write_output_files(paths, export_data, best_params):
    with open(paths["best_params"], "w", encoding="utf-8") as file:
        json.dump(best_params, file, ensure_ascii=False, indent=2)

    with open(paths["backtest_result"], "w", encoding="utf-8") as file:
        json.dump(export_data, file, ensure_ascii=False, indent=2)

    with open(paths["docs_result"], "w", encoding="utf-8") as file:
        json.dump(export_data, file, ensure_ascii=False)


def main():
    print(f"Loading data and optimizing RSI + volatility ({ITERATIONS} iterations)...")
    source_data, df, paths = load_source_data()
    initial_capital = float(source_data.get("meta", {}).get("initial_capital", INITIAL_CAPITAL))

    base_params = {
        "rsi_period": RSI_PERIOD,
        "rsi_buy_base": 32,
        "rsi_sell_base": 77,
        "vol_window": 20,
        "k_vol": 0.0,
        "vol_anchor": VOL_ANCHOR,
    }

    base_result = run_combined_backtest(df, base_params, initial_capital=initial_capital, capture_details=True)
    base_stats = calculate_statistics(base_result["daily_values"], base_result["trades"])
    print(
        "Baseline RSI(15) 32/77: "
        f"{base_stats['total_return']:.2f}% "
        f"(annual {base_stats['annual_return']:.2f}%, drawdown {base_stats['max_drawdown']:.2f}%)"
    )

    best_return = float("-inf")
    best_params = None

    for index in range(ITERATIONS):
        params = generate_random_params()
        result = run_combined_backtest(df, params, initial_capital=initial_capital, capture_details=False)

        if result["total_return"] > best_return:
            best_return = result["total_return"]
            best_params = result["params"]
            print(
                f"New Best [{index}]: {best_return:.2f}% | "
                f"Params: {json.dumps(best_params, ensure_ascii=False)}"
            )

    best_result = run_combined_backtest(df, best_params, initial_capital=initial_capital, capture_details=True)
    best_stats = calculate_statistics(best_result["daily_values"], best_result["trades"])
    export_data = build_export_payload(source_data, best_result, best_stats, base_result, base_stats)
    write_output_files(paths, export_data, best_result["params"])

    print("\nOptimization complete.")
    print(
        f"Optimized dynamic return: {best_stats['total_return']:.2f}% "
        f"(annual {best_stats['annual_return']:.2f}%, drawdown {best_stats['max_drawdown']:.2f}%)"
    )
    print(f"Improvement vs fixed RSI(15) 32/77: {best_stats['total_return'] - base_stats['total_return']:.2f}%")
    print("Best parameters:")
    print(json.dumps(best_result["params"], ensure_ascii=False, indent=2))
    print(f"\nSaved params: {paths['best_params']}")
    print(f"Updated backtest JSON: {paths['backtest_result']}")
    print(f"Updated docs JSON: {paths['docs_result']}")


if __name__ == "__main__":
    main()
