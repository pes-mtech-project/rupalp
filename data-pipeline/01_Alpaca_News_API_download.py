# Arguments
END_POINT_TEMPLATE = "https://data.alpaca.markets/v1beta1/news?start={start_date}&end={end_date}&limit=50&symbols={symbol}"
END_POINT_TEMPLATE_LINK_PAGE = "https://data.alpaca.markets/v1beta1/news?start={start_date}&end={end_date}&limit=50&symbols={symbol}&page_token={page_token}"
NUM_NEWS_PER_RECORD = 200
MAX_ATTEMPTS = 5
WAIT_TIME = 60
MAX_WORKERS = 30

# dependencies
import os
import time
import shutil
import httpx
import tenacity
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv
from rich import print
from tqdm import tqdm
from uuid import uuid4
from datetime import date, timedelta, datetime
from typing import List, Dict, Tuple, Union
from tenacity import retry, stop_after_attempt, wait_fixed

load_dotenv()
BACKFILL_MODEL = os.environ.get(
    "ALPACA_SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6"
)
BACKFILL_BATCH_SIZE = int(os.environ.get("ALPACA_SUMMARY_BATCH_SIZE", "8"))
BACKFILL_MAX_LEN = int(os.environ.get("ALPACA_SUMMARY_MAX_LEN", "100"))
BACKFILL_MIN_LEN = int(os.environ.get("ALPACA_SUMMARY_MIN_LEN", "20"))


def round_to_next_day(dt: datetime) -> datetime:
    cond = (dt.hour >= 16) and ((dt.minute > 0) or (dt.second > 0))
    new_day = dt + timedelta(days=1) if cond else dt
    return datetime(new_day.year, new_day.month, new_day.day, 9, 0, 0)


class ScraperError(Exception):
    pass


class RecordContainerFull(Exception):
    pass


def _clean_text_for_summary(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _backfill_missing_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill empty summary rows using a local summarization model and the content field.
    """
    if "summary" not in df.columns or "content" not in df.columns:
        return df

    summary_blank = df["summary"].fillna("").astype(str).str.strip().eq("")
    content_non_blank = df["content"].fillna("").astype(str).str.strip().ne("")
    target_mask = summary_blank & content_non_blank
    target_indices = list(df.index[target_mask])
    if not target_indices:
        return df

    print(
        f"Backfilling {len(target_indices)} empty summaries with local model: {BACKFILL_MODEL}"
    )
    try:
        from transformers import pipeline  # type: ignore
    except ImportError:
        print(
            "[yellow]transformers is not installed. Skipping summary backfill.[/yellow]"
        )
        return df

    summarizer = pipeline(
        "summarization",
        model=BACKFILL_MODEL,
        tokenizer=BACKFILL_MODEL,
        device=-1,
    )
    batch_size = max(1, BACKFILL_BATCH_SIZE)
    for i in range(0, len(target_indices), batch_size):
        batch_idx = target_indices[i : i + batch_size]
        batch_text = [
            _clean_text_for_summary(df.at[idx, "content"])[:4000] for idx in batch_idx
        ]
        generated = summarizer(
            batch_text,
            max_length=BACKFILL_MAX_LEN,
            min_length=BACKFILL_MIN_LEN,
            do_sample=False,
            truncation=True,
        )
        for idx, out in zip(batch_idx, generated):
            summary_text = str(out.get("summary_text", "")).strip()
            if summary_text:
                df.at[idx, "summary"] = summary_text
    return df


class ParseRecordContainer:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.record_counter = 0
        self.author_list = []
        self.content_list = []
        self.date_list = []
        self.source_list = []
        self.summary_list = []
        self.title_list = []
        self.url_list = []

    def add_records(self, records: List[Dict[str, str]]) -> None:
        for cur_record in records:
            self.author_list.append(cur_record["author"])
            self.content_list.append(cur_record["content"])
            date = cur_record["created_at"].rstrip("Z")
            self.date_list.append(datetime.fromisoformat(date))
            self.source_list.append(cur_record["source"])
            self.summary_list.append(cur_record["summary"])
            self.title_list.append(cur_record["headline"])
            self.url_list.append(cur_record["url"])
            self.record_counter += 1
            if self.record_counter == NUM_NEWS_PER_RECORD:
                raise RecordContainerFull

    def pop(self, align_next_date: bool = True) -> Union[pd.DataFrame, None]:
        if self.record_counter == 0:
            return None
        return_df = pd.DataFrame(
            {
                "author": self.author_list,
                "content": self.content_list,
                "datetime": self.date_list,
                "source": self.source_list,
                "summary": self.summary_list,
                "title": self.title_list,
                "url": self.url_list,
            }
        )
        if align_next_date:
            return_df["date"] = return_df["datetime"].apply(round_to_next_day).dt.date
        else:
            return_df["date"] = return_df["datetime"].dt.date
        return_df["equity"] = self.symbol
        return return_df


@retry(stop=stop_after_attempt(MAX_ATTEMPTS), wait=wait_fixed(WAIT_TIME))
def query_one_record(args: Tuple[date, str]) -> None:
    date, symbol = args
    next_date = date + timedelta(days=1)
    request_header = {
        "Apca-Api-Key-Id": os.environ.get("ALPACA_KEY"),
        "Apca-Api-Secret-Key": os.environ.get("ALPACA_KEY_SECRET_KEY"),
    }
    container = ParseRecordContainer(symbol)

    with httpx.Client() as client:
        # first request
        response = client.get(
            END_POINT_TEMPLATE.format(
                start_date=date.strftime("%Y-%m-%d"),
                end_date=next_date.strftime("%Y-%m-%d"),
                symbol=symbol,
            ),
            headers=request_header,
        )
        if response.status_code != 200:
            print("[red]Hit limit[/red]")
            raise ScraperError(response.text)
        result = response.json()
        next_page_token = result["next_page_token"]
        container.add_records(result["news"])

        while next_page_token:
            try:
                response = client.get(
                    END_POINT_TEMPLATE_LINK_PAGE.format(
                        start_date=date.strftime("%Y-%m-%d"),
                        end_date=next_date.strftime("%Y-%m-%d"),
                        symbol=symbol,
                        page_token=next_page_token,
                    ),
                    headers=request_header,
                )
                if response.status_code != 200:
                    raise ScraperError(response.text)
                result = response.json()
                next_page_token = result["next_page_token"]
                container.add_records(result["news"])
            except RecordContainerFull:
                break

    result = container.pop(align_next_date=True)
    if result is not None:
        result.to_parquet(os.path.join("data", "temp", f"{uuid4()}.parquet"), index=False)


def main_sync() -> None:
    # load data
    data = pd.read_parquet(os.path.join("data", "03_primary", "price_data.parquet"))
    if os.path.exists(os.path.join("data", "temp")):
        shutil.rmtree(os.path.join("data", "temp"))
    os.mkdir(os.path.join("data", "temp"))

    data["date"] = pd.to_datetime(data["est_time"]).dt.date
    data["equity"] = data["equity"].astype(str).str.upper().str.strip()
    query_data = data[["date", "equity"]].drop_duplicates()
    args_list = list(zip(query_data["date"], query_data["equity"]))
    ok_by_symbol = defaultdict(int)
    fail_by_symbol = defaultdict(int)
    with tqdm(total=len(args_list)) as pbar:
        for i, arg in enumerate(args_list):
            try:
                query_one_record(arg)
                ok_by_symbol[arg[1]] += 1
            except tenacity.RetryError as e:
                fail_by_symbol[arg[1]] += 1
                last_error = e.last_attempt.exception()
                print(
                    f"[red]Failed {arg[1]} on {arg[0]} after retries[/red]: {last_error}"
                )
            pbar.update(1)
            if (i + 1) % 3000 == 0:
                time.sleep(90)
    temp_files = os.listdir(os.path.join("data", "temp"))
    if not temp_files:
        raise RuntimeError("No news rows were downloaded. Check API keys/limits.")

    print("Download query status by ticker:")
    all_symbols = sorted(set(list(ok_by_symbol.keys()) + list(fail_by_symbol.keys())))
    for symbol in all_symbols:
        print(f"  {symbol}: ok={ok_by_symbol[symbol]}, failed={fail_by_symbol[symbol]}")

    record_dfs = [
        pd.read_parquet(os.path.join("data", "temp", f))
        for f in temp_files
    ]
    df = pd.concat(record_dfs, ignore_index=True)
    df = _backfill_missing_summaries(df)
    print("Downloaded article rows by ticker:")
    for symbol, count in df["equity"].astype(str).value_counts().sort_index().items():
        print(f"  {symbol}: rows={count}")
    df.to_parquet(os.path.join("data", "03_primary", "news.parquet"), index=False)
    print(df.shape)


if __name__ == "__main__":
    main_sync()
