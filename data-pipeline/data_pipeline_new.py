import argparse
import glob
import pickle
import bisect
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf


def download_data(start_day: str, end_day: str, tickers: List[str]) -> List[pd.DataFrame]:
    """Download adjusted close prices from Yahoo Finance for each ticker."""
    df_list = []
    for ticker in tickers:
        print(f"Downloading data for {ticker}")
        data = yf.download(
            ticker,
            start=start_day,
            end=end_day,
            auto_adjust=False,
            progress=False,
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if data.empty:
            print(f"Warning: no price rows for {ticker}")
            df_list.append(pd.DataFrame(columns=["date", ticker]))
            continue
        data = data.reset_index()
        price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
        data["Date"] = pd.to_datetime(data["Date"]).dt.date
        data = data[["Date", price_col]].rename(columns={"Date": "date", price_col: ticker})
        df_list.append(data)
    return df_list


def combine_dataframes(df_list: List[pd.DataFrame], output_dir: Path) -> Dict:
    """Merge ticker price dataframes into {date: {'price': {ticker: price}}}."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ticker_order = [df.columns[1] for df in df_list if len(df.columns) >= 2]
    df_dicts = [dict(zip(df["date"], df[ticker])) for df, ticker in zip(df_list, ticker_order)]
    combined_dict = {date: {"price": {}} for df_dict in df_dicts for date in df_dict}
    for i, df_dict in enumerate(df_dicts):
        for date, price in df_dict.items():
            combined_dict[date]["price"][ticker_order[i]] = float(price)
    combined_dict = dict(sorted(combined_dict.items()))

    pkl_filename = output_dir / "price.pkl"
    with pkl_filename.open("wb") as file:
        pickle.dump(combined_dict, file)
    print(f"Price data saved to: {pkl_filename}")
    return combined_dict


def _detect_symbol_column(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    for col in ["symbols", "equity", "ticker", "symbol"]:
        if col in df.columns:
            return col
    raise ValueError("Could not find symbol column in news input.")


def _detect_text_column(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    for col in ["summary", "content", "title"]:
        if col in df.columns:
            return col
    raise ValueError("Could not find text column in news input.")


def create_news_dict(
    news_input: Path,
    csv_pattern: str,
    col_name: str,
    symbol_col: str,
    output_dir: Path,
) -> Dict:
    """
    Build {date: {'news': {ticker: [text, ...]}}} from either:
    - a directory of CSV files, or
    - a single parquet file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_dict: Dict = {}

    if news_input.is_file() and news_input.suffix.lower() == ".parquet":
        news_df = pd.read_parquet(news_input)
        symbol_col = _detect_symbol_column(news_df, symbol_col)
        col_name = _detect_text_column(news_df, col_name)
        if "date" not in news_df.columns:
            if "datetime" in news_df.columns:
                news_df["date"] = pd.to_datetime(news_df["datetime"]).dt.date
            else:
                raise ValueError("News parquet must contain either 'date' or 'datetime'.")
        else:
            news_df["date"] = pd.to_datetime(news_df["date"]).dt.date

        grouped = (
            news_df.dropna(subset=[symbol_col, col_name])
            .groupby(["date", symbol_col])[col_name]
            .apply(list)
        )
        for (date, symbol), summaries in grouped.items():
            if date not in combined_dict:
                combined_dict[date] = {"news": {}}
            combined_dict[date]["news"][str(symbol)] = [str(s) for s in summaries if str(s)]
    else:
        csv_files = glob.glob(str(news_input / csv_pattern))
        for file in csv_files:
            df = pd.read_csv(file)
            symbol_col = _detect_symbol_column(df, symbol_col)
            col_name = _detect_text_column(df, col_name)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            grouped = (
                df.dropna(subset=[symbol_col, col_name])
                .groupby(["date", symbol_col])[col_name]
                .apply(list)
            )
            for (date, symbol), summaries in grouped.items():
                if date not in combined_dict:
                    combined_dict[date] = {"news": {}}
                existing = combined_dict[date]["news"].setdefault(str(symbol), [])
                existing.extend([str(s) for s in summaries if str(s)])

    pkl_filename = output_dir / "news.pkl"
    with pkl_filename.open("wb") as file:
        pickle.dump(combined_dict, file)
    print(f"News data saved to: {pkl_filename}")
    return combined_dict


def process_filing_data(
    start_day: str,
    end_day: str,
    output_dir: Path,
    filing_data: Path,
    tickers: List[str],
) -> Tuple[Dict, Dict]:
    """Create filing_q.pkl and filing_k.pkl from filing_data parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    start = pd.to_datetime(start_day)
    end = pd.to_datetime(end_day)

    filingkq = pd.read_parquet(filing_data)
    filingkq = filingkq.drop(columns=["document_url", "cik"], errors="ignore")

    if "est_timestamp" in filingkq.columns:
        filingkq = filingkq.rename(columns={"est_timestamp": "date"})
    elif "utc_timestamp" in filingkq.columns:
        filingkq = filingkq.rename(columns={"utc_timestamp": "date"})
    elif "date" not in filingkq.columns:
        raise ValueError(
            "Filing parquet must have one of these columns: est_timestamp, utc_timestamp, date."
        )

    required_cols = {"date", "ticker", "content", "type"}
    missing = required_cols - set(filingkq.columns)
    if missing:
        raise ValueError(f"Filing parquet missing required columns: {sorted(missing)}")

    filingkq = filingkq[["date", "ticker", "content", "type"]].copy()
    filingkq["date"] = pd.to_datetime(filingkq["date"], utc=True).dt.tz_convert(None)
    filingkq = filingkq[filingkq["ticker"].isin(tickers)]

    df_10k = filingkq[filingkq["type"] == "10-K"].drop(columns=["type"]).sort_values("date")
    df_10q = filingkq[filingkq["type"] == "10-Q"].drop(columns=["type"]).sort_values("date")

    df_10k = df_10k[(df_10k["date"] >= start) & (df_10k["date"] <= end)]
    df_10q = df_10q[(df_10q["date"] >= start) & (df_10q["date"] <= end)]
    df_10k["date"] = df_10k["date"].dt.date
    df_10q["date"] = df_10q["date"].dt.date

    nested_10k = {
        date: {"filing_k": df_group.set_index("ticker")["content"].to_dict()}
        for date, df_group in df_10k.groupby("date")
    }
    nested_10q = {
        date: {"filing_q": df_group.set_index("ticker")["content"].to_dict()}
        for date, df_group in df_10q.groupby("date")
    }

    q_filename = output_dir / "filing_q.pkl"
    k_filename = output_dir / "filing_k.pkl"

    with q_filename.open("wb") as file:
        pickle.dump(nested_10q, file)
    with k_filename.open("wb") as file:
        pickle.dump(nested_10k, file)
    print(f"10-K data saved to: {k_filename}")
    print(f"10-Q data saved to: {q_filename}")
    return nested_10q, nested_10k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FinMem environment data from prices/news/filings.")
    parser.add_argument("--start-day", default="2021-08-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end-day", default="2023-06-01", help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--tickers",
        default="TSLA,AMZN,NFLX,MSFT,COIN",
        help="Comma-separated ticker list.",
    )
    parser.add_argument(
        "--news-input",
        default="data/03_primary/news.parquet",
        help="News parquet file path OR directory containing CSV files.",
    )
    parser.add_argument(
        "--news-csv-pattern",
        default="*.csv",
        help="CSV pattern when --news-input is a directory.",
    )
    parser.add_argument("--news-text-column", default="summary", help="News text column name.")
    parser.add_argument(
        "--news-symbol-column",
        default="symbols",
        help="News ticker column name (auto-detected if not found).",
    )
    parser.add_argument(
        "--filing-data",
        default="data/03_primary/filing_data.parquet",
        help="Input filings parquet path.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/03_model_input",
        help="Output directory for price/news/filing/env_data pickle files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [x.strip().upper() for x in args.tickers.split(",") if x.strip()]
    output_dir = Path(args.output_dir)
    news_input = Path(args.news_input)
    filing_data = Path(args.filing_data)

    df_list = download_data(args.start_day, args.end_day, tickers)
    price = combine_dataframes(df_list, output_dir)
    news = create_news_dict(
        news_input=news_input,
        csv_pattern=args.news_csv_pattern,
        col_name=args.news_text_column,
        symbol_col=args.news_symbol_column,
        output_dir=output_dir,
    )
    q, k = process_filing_data(
        start_day=args.start_day,
        end_day=args.end_day,
        output_dir=output_dir,
        filing_data=filing_data,
        tickers=tickers,
    )

    # Shift any news on non-trading dates (weekends/holidays) to the next available
    # trading date so it is not dropped during env_data merge.
    trading_dates = sorted(price.keys())
    shifted_news: Dict = {}
    for n_date, n_payload in news.items():
        idx = bisect.bisect_left(trading_dates, n_date)
        if idx >= len(trading_dates):
            continue
        target_date = trading_dates[idx]
        tgt = shifted_news.setdefault(target_date, {"news": {}})
        for tkr, items in n_payload.get("news", {}).items():
            bucket = tgt["news"].setdefault(tkr, [])
            bucket.extend(items)
    news = shifted_news

    for date in price.keys():
        q.setdefault(date, {"filing_q": {}})
        k.setdefault(date, {"filing_k": {}})
        news.setdefault(date, {"news": {}})
        missing_tickers = [ticker for ticker in tickers if ticker not in news[date]["news"]]
        for ticker in missing_tickers:
            news[date]["news"][ticker] = []

    filled_q = dict(sorted(q.items()))
    filled_k = dict(sorted(k.items()))
    news = dict(sorted(news.items()))

    env_data = {
        key: (price[key], news[key], filled_q[key], filled_k[key]) for key in sorted(price.keys())
    }

    output_path = output_dir / "env_data.pkl"
    with output_path.open("wb") as file:
        pickle.dump(env_data, file)
    print(f"Environment data saved to: {output_path}")


if __name__ == "__main__":
    main()
