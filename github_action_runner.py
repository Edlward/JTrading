import os
import json
import re
import smtplib
import time
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np

# ==========================================
# 配置读取 (优先从环境变量读取)
# ==========================================
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.126.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 465))
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")
# Server酱 Key
SERVERCHAN_KEY = os.environ.get("SERVERCHAN_KEY")
# Gist 订阅者列表配置
GIST_SUBSCRIBERS_URL = os.environ.get("GIST_SUBSCRIBERS_URL")
GIST_TOKEN = os.environ.get("GIST_TOKEN") or os.environ.get("GIST_TOKEN_READ") or os.environ.get("GIST_TOKEN_WRITE")
GIST_ID = os.environ.get("GIST_ID")
GIST_FILENAME = os.environ.get("GIST_FILENAME") or "subscribers.txt"
EMAIL_BATCH_SIZE = 10
EMAIL_BATCH_DELAY_SECONDS = 2

# ==========================================
# 最优策略参数配置 (来自回测优化结果)
# RSI(15) EMA 32/77 - 联结基金模式（理论最优）
# 总收益268.02%, 年化20.90%
# 注：联结基金可小数份额申购，ETF需100份整手交易
# ==========================================
ETF_CODE = "512890"  # 红利低波ETF
ETF_NAME = "红利低波ETF"
RSI_PERIOD = 15  # RSI周期（使用EMA平滑）
RSI_BUY_THRESHOLD = int(os.environ.get("RSI_BUY_THRESHOLD", 32))  # 买入阈值
RSI_SELL_THRESHOLD = int(os.environ.get("RSI_SELL_THRESHOLD", 77))  # 卖出阈值
DATA_FETCH_RETRIES = int(os.environ.get("DATA_FETCH_RETRIES", 3))
DATA_FETCH_TIMEOUT = int(os.environ.get("DATA_FETCH_TIMEOUT", 20))

BEST_PARAMS_PATH = os.path.join("backtest", "best_combined_params.json")
BACKTEST_RESULT_PATH = os.path.join("backtest", "backtest_result.json")

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

EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def load_json_file(file_path):
    """安全读取 JSON 文件。"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"读取 {file_path} 失败: {e}")
        return None


def load_backtest_summary():
    """读取回测统计摘要，避免前端展示过期硬编码收益。"""
    data = load_json_file(BACKTEST_RESULT_PATH)
    stats = (data or {}).get("statistics", {})
    strategy_ideal = stats.get("strategy_ideal") or stats.get("strategy") or {}
    strategy_dynamic = stats.get("strategy_dynamic") or {}
    return {
        "classic_total": strategy_ideal.get("total_return"),
        "classic_annual": strategy_ideal.get("annual_return"),
        "dynamic_total": strategy_dynamic.get("total_return"),
        "dynamic_annual": strategy_dynamic.get("annual_return"),
    }


def load_dynamic_params():
    """读取动态策略最优参数；若缺失则回退到保守默认值。"""
    params = load_json_file(BEST_PARAMS_PATH) or {}
    return {
        "rsi_period": int(params.get("rsi_period", RSI_PERIOD)),
        "rsi_buy_base": float(params.get("rsi_buy_base", 34)),
        "rsi_sell_base": float(params.get("rsi_sell_base", 72)),
        "vol_window": int(params.get("vol_window", 20)),
        "k_vol": float(params.get("k_vol", 0.0)),
        "vol_anchor": float(params.get("vol_anchor", 15.0)),
    }


def calculate_volatility_annualized(close_series, window):
    """按回测口径计算年化波动率(%)。"""
    if close_series is None or len(close_series) < window + 1:
        return None

    log_ret = np.log(close_series / close_series.shift(1))
    vol = log_ret.rolling(window=window).std() * np.sqrt(252) * 100
    latest = vol.iloc[-1]
    if pd.isna(latest):
        return None
    return float(latest)


def compute_dynamic_signal(rsi_value, close_series, params):
    """基于 RSI + 波动率参数计算当日动态阈值与信号。"""
    vol = calculate_volatility_annualized(close_series, params["vol_window"])
    if vol is None or rsi_value is None:
        return None

    adjustment = params["k_vol"] * (vol - params["vol_anchor"])
    buy_threshold = params["rsi_buy_base"] - adjustment
    sell_threshold = params["rsi_sell_base"] + adjustment

    # 与回测前端保持一致的阈值边界
    buy_threshold = min(50.0, max(20.0, buy_threshold))
    sell_threshold = min(90.0, max(60.0, sell_threshold))

    if rsi_value < buy_threshold:
        signal = "买入"
        signal_color = "#22c55e"
    elif rsi_value > sell_threshold:
        signal = "卖出"
        signal_color = "#ef4444"
    else:
        signal = "持有"
        signal_color = "#3b82f6"

    return {
        "volatility": round(vol, 2),
        "buy_threshold": round(buy_threshold, 2),
        "sell_threshold": round(sell_threshold, 2),
        "signal": signal,
        "signal_color": signal_color,
    }

def parse_subscriber_emails(content, include_pending=False):
    """解析订阅者邮箱，支持换行、逗号/分号分隔和 [pending]/[confirmed] 标记。"""
    emails = []
    seen = set()

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        status_match = re.match(r"^\[(pending|confirmed)\]\s*(.+)$", line, re.IGNORECASE)
        if status_match:
            status = status_match.group(1).lower()
            line = status_match.group(2).strip()
            if status == "pending" and not include_pending:
                continue

        for candidate in re.split(r"[,;，；\s]+", line):
            email = candidate.strip()
            if not email:
                continue
            if not EMAIL_PATTERN.match(email):
                print(f"跳过无效订阅邮箱: {email}")
                continue

            key = email.lower()
            if key in seen:
                continue

            seen.add(key)
            emails.append(email)

    return emails


def fetch_gist_subscriber_content():
    """优先从 raw URL 读取 Gist；未配置 raw URL 时回退到 Gist API。"""
    if not GIST_TOKEN:
        return None

    headers_raw = {
        'Authorization': f'token {GIST_TOKEN}',
        'Accept': 'application/vnd.github.v3.raw'
    }
    headers_json = {
        'Authorization': f'token {GIST_TOKEN}',
        'Accept': 'application/vnd.github.v3+json'
    }

    if GIST_SUBSCRIBERS_URL:
        try:
            response = requests.get(GIST_SUBSCRIBERS_URL, headers=headers_raw, timeout=10)
            if response.status_code == 200:
                return response.text
            print(f"从 Gist raw URL 获取邮箱失败: HTTP {response.status_code}")
        except Exception as e:
            print(f"从 Gist raw URL 获取邮箱出错: {e}")

    if GIST_ID:
        try:
            url = f"https://api.github.com/gists/{GIST_ID}"
            response = requests.get(url, headers=headers_json, timeout=10)
            if response.status_code == 200:
                gist = response.json()
                file_info = gist.get("files", {}).get(GIST_FILENAME)
                if not file_info:
                    print(f"Gist 中找不到订阅文件: {GIST_FILENAME}")
                    return None
                return file_info.get("content", "")
            print(f"从 Gist API 获取邮箱失败: HTTP {response.status_code}")
        except Exception as e:
            print(f"从 Gist API 获取邮箱出错: {e}")

    return None


def fetch_subscriber_emails():
    """
    从私有 Gist 获取订阅者邮箱列表。
    如果 Gist 配置不存在或读取失败，则回退到环境变量 SUBSCRIBER_EMAILS。
    """
    content = fetch_gist_subscriber_content()
    if content is not None:
        emails = parse_subscriber_emails(content)
        print(f"从 Gist 获取到 {len(emails)} 个已确认订阅者邮箱")
        return emails

    fallback_emails = os.environ.get("SUBSCRIBER_EMAILS", "")
    if fallback_emails:
        emails = parse_subscriber_emails(fallback_emails)
        print(f"使用环境变量 SUBSCRIBER_EMAILS，共 {len(emails)} 个订阅者")
        return emails

    return []

def calculate_rsi_ema(prices, period):
    """
    计算RSI指标（使用EMA平滑，更敏感）
    与回测代码保持一致
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    # 使用EMA而非SMA（更敏感）
    alpha = 1 / period  # EMA平滑因子
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def normalize_etf_history(df, source_name, days):
    """统一不同数据源的字段，并做基础清洗。"""
    if df is None or df.empty:
        raise ValueError(f"{source_name} 返回空数据")

    column_map = {}
    if "日期" in df.columns:
        column_map["日期"] = "date"
    if "收盘" in df.columns:
        column_map["收盘"] = "close"
    df = df.rename(columns=column_map)

    missing_columns = {"date", "close"} - set(df.columns)
    if missing_columns:
        raise ValueError(f"{source_name} 缺少字段: {', '.join(sorted(missing_columns))}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"{source_name} 清洗后无有效数据")

    df = df.tail(days).reset_index(drop=True)
    start_date = df["date"].min().strftime("%Y-%m-%d")
    end_date = df["date"].max().strftime("%Y-%m-%d")
    print(f"{source_name} 获取到 {len(df)} 条数据，从 {start_date} 到 {end_date}")
    return df


def fetch_etf_data_from_akshare(code, days):
    """使用 AKShare 获取 ETF 前复权日线数据。"""
    import akshare as ak

    df = ak.fund_etf_hist_em(symbol=code, period="daily", adjust="qfq")
    return normalize_etf_history(df, "AKShare", days)


def get_eastmoney_secids(code):
    """生成东方财富 secid，ETF 512/510 等上海代码通常为 1.x。"""
    preferred_market = "1" if code.startswith(("5", "6", "9")) else "0"
    fallback_market = "0" if preferred_market == "1" else "1"
    return [f"{preferred_market}.{code}", f"{fallback_market}.{code}"]


def fetch_etf_data_from_eastmoney(code, days):
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
                if len(parts) >= 3:
                    rows.append({"date": parts[0], "close": parts[2]})

            return normalize_etf_history(pd.DataFrame(rows), f"东方财富({secid})", days)
        except Exception as e:
            last_error = e
            print(f"东方财富 {secid} 获取失败: {e}")

    raise RuntimeError(f"东方财富备用数据源不可用: {last_error}")


def get_sina_symbol(code):
    """生成新浪行情 symbol，上海 ETF 使用 sh 前缀。"""
    market_prefix = "sh" if code.startswith(("5", "6", "9")) else "sz"
    return f"{market_prefix}{code}"


def fetch_etf_data_from_sina(code, days):
    """调用新浪日线接口，作为行情源被断开时的第二备用数据源。"""
    url = "https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20data=/CN_MarketDataService.getKLineData"
    symbol = get_sina_symbol(code)
    session = requests.Session()
    session.trust_env = False

    response = session.get(
        url,
        params={
            "symbol": symbol,
            "scale": "240",
            "ma": "no",
            "datalen": str(max(days, RSI_PERIOD + 5)),
        },
        headers=DATA_REQUEST_HEADERS,
        timeout=DATA_FETCH_TIMEOUT,
    )
    response.raise_for_status()

    text = response.text
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("新浪返回内容不是有效的 JSONP K 线数据")

    rows = [
        {"date": item.get("day"), "close": item.get("close")}
        for item in json.loads(text[start : end + 1])
    ]
    return normalize_etf_history(pd.DataFrame(rows), f"新浪({symbol})", days)


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


def fetch_etf_data(code, days=60):
    """
    获取ETF历史数据。
    优先使用 AKShare；若 GitHub Actions 网络被对端断开，则回退到东方财富直连接口。
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始获取 {code} 数据...")

    data_sources = [
        ("AKShare", lambda: fetch_etf_data_from_akshare(code, days)),
        ("东方财富备用接口", lambda: fetch_etf_data_from_eastmoney(code, days)),
        ("新浪备用接口", lambda: fetch_etf_data_from_sina(code, days)),
    ]

    for source_name, fetcher in data_sources:
        df = fetch_with_retries(source_name, fetcher)
        if df is not None:
            return df

    print("获取ETF数据失败: 所有数据源均不可用")
    return None


def fetch_rsi_and_price():
    """
    获取 RSI 和 价格数据
    使用自己计算的 RSI(15) EMA，与回测策略保持一致
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始获取数据...")
    
    # 获取ETF历史数据
    df = fetch_etf_data(ETF_CODE, days=60)  # 获取60天数据，确保RSI计算准确
    
    if df is None or len(df) < RSI_PERIOD + 5:
        print("无法获取足够的历史数据")
        return None, None, None, None
    
    # 计算RSI(15) EMA
    df['rsi'] = calculate_rsi_ema(df['close'], RSI_PERIOD)
    
    # 获取最新的RSI和价格
    latest = df.iloc[-1]
    rsi_value = latest['rsi']
    latest_price = latest['close']
    latest_date = latest['date'].strftime('%Y-%m-%d')
    
    if pd.notna(rsi_value):
        print(f"获取到 RSI({RSI_PERIOD}) EMA: {rsi_value:.2f}")
        print(f"最新价格: {latest_price:.4f}")
        print(f"数据日期: {latest_date}")
    else:
        print("RSI计算失败，数据不足")
        return None, None, None, None
    
    return rsi_value, latest_price, latest_date, df

def build_alert_message(to_header, subject, content):
    """构建 RSI 提醒邮件。"""
    html_content = f"""
    <html>
    <body style="font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background-color: #f4f6f9; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 10px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
            <h2 style="color: #2c3e50; margin-top: 0; border-bottom: 2px solid #3498db; padding-bottom: 10px;">{subject}</h2>
            
            <div style="font-size: 16px; line-height: 1.6; color: #34495e; margin: 20px 0;">
                {content.replace(chr(10), '<br>')}
            </div>
            
            <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ecf0f1; font-size: 12px; color: #95a5a6; text-align: center;">
                <p>此邮件由 GitHub Actions 自动发送，请勿直接回复。</p>
                <p>
                    如果您不想继续接收此类邮件，可以 
                    <a href="mailto:{SENDER_EMAIL}?subject=取消订阅 RSI 监控&body=请将我的邮箱从订阅列表中移除" style="color: #e74c3c; text-decoration: none; font-weight: bold;">点击此处取消订阅</a>
                </p>
            </div>
        </div>
    </body>
    </html>
    """

    message = MIMEText(html_content, 'html', 'utf-8')
    # 只编码显示名，邮箱地址保持 RFC5322 可解析的 addr-spec。
    message['From'] = formataddr(("RSI 监控助手", SENDER_EMAIL), charset='utf-8')
    message['To'] = to_header
    message['Subject'] = Header(subject, 'utf-8')
    return message


def send_email(to_email, subject, content):
    """发送单个收件人的提醒邮件，保留给临时测试或手动调用。"""
    return send_email_batch([to_email], subject, content)


def send_email_batch(to_emails, subject, content):
    """通过 BCC 方式向一批收件人发送提醒邮件，收件人彼此不可见。"""
    recipients = [email.strip() for email in to_emails if email and email.strip()]
    if not recipients:
        return 0

    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("未配置发件人邮箱或密码，跳过发送邮件。")
        return 0

    message = build_alert_message("undisclosed-recipients:;", subject, content)

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            refused = server.sendmail(SENDER_EMAIL, recipients, message.as_string())

        refused_emails = set(refused.keys()) if refused else set()
        sent_count = len([email for email in recipients if email not in refused_emails])
        if refused_emails:
            print(f"本批部分邮箱被 SMTP 拒绝: {', '.join(refused_emails)}")
        print(f"邮件批量发送完成: {sent_count}/{len(recipients)}，收件人互相不可见")
        return sent_count
    except Exception as e:
        print(f"邮件批量发送失败 ({', '.join(recipients)}): {e}")
        return 0


def send_email_batches(subscribers, subject, content):
    """按固定大小分批发送提醒邮件。"""
    if not subscribers:
        return 0

    total_sent = 0
    total_batches = (len(subscribers) + EMAIL_BATCH_SIZE - 1) // EMAIL_BATCH_SIZE
    for batch_start in range(0, len(subscribers), EMAIL_BATCH_SIZE):
        batch = subscribers[batch_start:batch_start + EMAIL_BATCH_SIZE]
        batch_no = batch_start // EMAIL_BATCH_SIZE + 1
        print(f"发送第 {batch_no}/{total_batches} 批邮件，共 {len(batch)} 个收件人")
        total_sent += send_email_batch(batch, subject, content)
        if batch_start + EMAIL_BATCH_SIZE < len(subscribers) and EMAIL_BATCH_DELAY_SECONDS > 0:
            time.sleep(EMAIL_BATCH_DELAY_SECONDS)

    return total_sent

def send_wechat(title, content):
    """
    微信通知 (Server酱)
    """
    if not SERVERCHAN_KEY:
        print("未配置 SERVERCHAN_KEY，跳过微信通知。")
        return
    
    data = {'title': title,
            'desp': content,
            'channel': 9}
    msg_url = f"https://sctapi.ftqq.com/{SERVERCHAN_KEY}.send"

    try:
        response = requests.post(msg_url, data=data)
        print(f"微信通知发送结果: {response.text}")
    except Exception as e:
        print(f"微信通知发送失败: {e}")

def main():
    rsi, price, latest_date, market_df = fetch_rsi_and_price()
    
    if rsi is None:
        print("未能获取有效 RSI 数据，程序结束。")
        raise SystemExit(1)

    print(f"当前状态: RSI={rsi}, 价格={price}")

    subject = ""
    content = ""

    if rsi < RSI_BUY_THRESHOLD:
        subject = f"【买入提醒】{ETF_NAME} RSI低于{RSI_BUY_THRESHOLD}"
        content = f"""当前{ETF_NAME} ({ETF_CODE}) 的 RSI({RSI_PERIOD}) EMA 为 {rsi:.2f}，已低于 {RSI_BUY_THRESHOLD}，建议关注买入机会。

📊 策略参数:
- RSI周期: {RSI_PERIOD}日 (EMA平滑)
- 买入阈值: RSI < {RSI_BUY_THRESHOLD}
- 卖出阈值: RSI > {RSI_SELL_THRESHOLD}

💰 回测表现:
- 总收益: 268.02%
- 年化收益: 20.90%

当前价格: {price}"""
    elif rsi > RSI_SELL_THRESHOLD:
        subject = f"【卖出提醒】{ETF_NAME} RSI高于{RSI_SELL_THRESHOLD}"
        content = f"""当前{ETF_NAME} ({ETF_CODE}) 的 RSI({RSI_PERIOD}) EMA 为 {rsi:.2f}，已高于 {RSI_SELL_THRESHOLD}，建议关注卖出风险。

📊 策略参数:
- RSI周期: {RSI_PERIOD}日 (EMA平滑)
- 买入阈值: RSI < {RSI_BUY_THRESHOLD}
- 卖出阈值: RSI > {RSI_SELL_THRESHOLD}

💰 回测表现:
- 总收益: 268.02%
- 年化收益: 20.90%

当前价格: {price}"""
    else:
        print(f"RSI 在正常范围内 ({RSI_BUY_THRESHOLD}-{RSI_SELL_THRESHOLD})，无需发送提醒。")

    if subject:
        print(f"触发条件，准备发送邮件: {subject}")
        subscribers = fetch_subscriber_emails()
        if not subscribers:
            print("没有配置订阅者邮箱，无法发送。")
        
        send_email_batches(subscribers, subject, content)
            
        # 发送微信通知
        send_wechat(subject, content)

    # ==========================================
    # 生成静态数据文件 (供 GitHub Pages 使用)
    # ==========================================
    docs_dir = "docs"
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
    
    # GitHub Actions 运行在 UTC 时区，需要转换为北京时间 (UTC+8)
    beijing_time = datetime.utcnow() + timedelta(hours=8)
    
    # 计算买卖信号状态
    if rsi < RSI_BUY_THRESHOLD:
        signal = "买入"
        signal_color = "#22c55e"  # 绿色
    elif rsi > RSI_SELL_THRESHOLD:
        signal = "卖出"
        signal_color = "#ef4444"  # 红色
    else:
        signal = "持有"
        signal_color = "#3b82f6"  # 蓝色
    
    backtest_summary = load_backtest_summary()
    dynamic_params = load_dynamic_params()
    dynamic_signal = compute_dynamic_signal(rsi, market_df["close"] if market_df is not None else None, dynamic_params)

    data = {
        "etf_code": ETF_CODE,
        "etf_name": ETF_NAME,
        "rsi": round(rsi, 2),
        "rsi_period": RSI_PERIOD,
        "market_date": latest_date,
        "price": round(price, 4) if price else None,
        "buy_threshold": RSI_BUY_THRESHOLD,
        "sell_threshold": RSI_SELL_THRESHOLD,
        "signal": signal,
        "signal_color": signal_color,
        "strategy": f"RSI({RSI_PERIOD}) EMA {RSI_BUY_THRESHOLD}/{RSI_SELL_THRESHOLD}",
        "backtest_return": f"{backtest_summary['classic_total']:.2f}%" if backtest_summary["classic_total"] is not None else "--",
        "backtest_annual": f"{backtest_summary['classic_annual']:.2f}%" if backtest_summary["classic_annual"] is not None else "--",
        "timestamp": beijing_time.strftime("%Y-%m-%d %H:%M:%S") + " (北京时间)"
    }
    
    with open(os.path.join(docs_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"静态数据已保存至 {docs_dir}/data.json")

    dynamic_data = {
        "etf_code": ETF_CODE,
        "etf_name": ETF_NAME,
        "market_date": latest_date,
        "price": round(price, 4) if price else None,
        "rsi": round(rsi, 2),
        "rsi_period": dynamic_params["rsi_period"],
        "rsi_buy_base": dynamic_params["rsi_buy_base"],
        "rsi_sell_base": dynamic_params["rsi_sell_base"],
        "vol_window": dynamic_params["vol_window"],
        "k_vol": dynamic_params["k_vol"],
        "vol_anchor": dynamic_params["vol_anchor"],
        "volatility": dynamic_signal["volatility"] if dynamic_signal else None,
        "buy_threshold": dynamic_signal["buy_threshold"] if dynamic_signal else None,
        "sell_threshold": dynamic_signal["sell_threshold"] if dynamic_signal else None,
        "signal": dynamic_signal["signal"] if dynamic_signal else "未知",
        "signal_color": dynamic_signal["signal_color"] if dynamic_signal else "#8a8070",
        "backtest_return": f"{backtest_summary['dynamic_total']:.2f}%" if backtest_summary["dynamic_total"] is not None else "--",
        "backtest_annual": f"{backtest_summary['dynamic_annual']:.2f}%" if backtest_summary["dynamic_annual"] is not None else "--",
        "timestamp": beijing_time.strftime("%Y-%m-%d %H:%M:%S") + " (北京时间)"
    }

    with open(os.path.join(docs_dir, "dynamic_data.json"), "w", encoding="utf-8") as f:
        json.dump(dynamic_data, f, ensure_ascii=False, indent=2)
    print(f"动态信号数据已保存至 {docs_dir}/dynamic_data.json")

    # ==========================================
    # 动态注入订阅服务地址 (从环境变量)
    # ==========================================
    subscribe_worker_url = os.environ.get("SUBSCRIBE_WORKER_URL")
    formspree_id = os.environ.get("FORMSPREE_ID")
    
    index_path = os.path.join(docs_dir, "index.html")
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            updated = False
            
            # 注入 Worker URL
            if subscribe_worker_url and "__SUBSCRIBE_WORKER_URL__" in content:
                content = content.replace("__SUBSCRIBE_WORKER_URL__", subscribe_worker_url)
                print(f"已注入 Worker URL: {subscribe_worker_url}")
                updated = True
            
            # 注入 Formspree ID (备用方案)
            if formspree_id and "__FORMSPREE_ID__" in content:
                content = content.replace("__FORMSPREE_ID__", formspree_id)
                print(f"已注入 Formspree ID: {formspree_id}")
                updated = True
            
            if updated:
                with open(index_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print("index.html 更新完成")
            else:
                print("index.html 中未找到需要替换的占位符，跳过。")
        except Exception as e:
            print(f"更新 index.html 失败: {e}")

if __name__ == "__main__":
    main()
