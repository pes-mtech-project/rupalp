"""
Mirror the 01_SEC_API_10k10q_download.py parquet output using official SEC endpoints.

This script downloads filing indexes from data.sec.gov submissions JSON, fetches filing
documents from sec.gov Archives, extracts target sections, cleans the text, and writes:
data/03_primary/filing_data.parquet
"""

from __future__ import annotations

import argparse
import calendar
import logging
import os
import re
import time
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from unicodedata import normalize

try:
    from unidecode import unidecode  # type: ignore
except Exception:  # pragma: no cover
    unidecode = None

SEC_SUBMISSION_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_SUBMISSION_FILE_URL = "https://data.sec.gov/submissions/{name}"
SEC_ARCHIVE_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik_nozero}/{accession}/{document}"
)
SAMPLE_TICKER_CIK_MAP = {
    "TSLA": "0001318605",
    "AMZN": "0001018724",
    "NFLX": "0001065280",
    "MSFT": "0000789019",
    "COIN": "0001679788",
}

# Keep default active targets aligned with 01_SEC_API_10k10q_download.py.
DEFAULT_FORM_TO_ITEM_CODE = {"10-K": ["7"], "10-Q": ["part1item2"]}

TEN_K_ITEM_CODE = [
    "1",
    "1A",
    "1B",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "7A",
    "8",
    "9",
    "9A",
    "9B",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
]
TEN_Q_ITEM_CODE = [
    "part1item1",
    "part1item2",
    "part1item3",
    "part1item4",
    "part2item1",
    "part2item1a",
    "part2item2",
    "part2item3",
    "part2item4",
    "part2item5",
    "part2item6",
]
EIGHT_K_ITEM_CODE = [
    "1-1",
    "1-2",
    "1-3",
    "1-4",
    "2-1",
    "2-2",
    "2-3",
    "2-4",
    "2-5",
    "2-6",
    "3-1",
    "3-2",
    "3-3",
    "4-1",
    "4-2",
    "5-1",
    "5-2",
    "5-3",
    "5-4",
    "5-5",
    "5-6",
    "5-7",
    "5-8",
    "6-1",
    "6-2",
    "6-3",
    "6-4",
    "6-5",
    "6-6",
    "6-10",
    "7-1",
    "8-1",
]

SLEEP_TIME = 0.25
MAX_RETRIES = 5
DEFAULT_FINANCIAL_YEARS = 2

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(
    os.path.join("data", "03_primary", "filing_fails.log"), mode="w"
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create filing_data.parquet for TSLA/AMZN/NFLX/MSFT/COIN from SEC submissions API."
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="YYYY-MM-DD. Defaults to Jan 1 of (current year - 2).",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="YYYY-MM-DD. Defaults to today's date (UTC).",
    )
    parser.add_argument(
        "--forms",
        nargs="+",
        default=["10-K", "10-Q", "8-K"],
        help="Forms to include (default: 10-K 10-Q 8-K).",
    )
    parser.add_argument(
        "--row-per-item",
        action="store_true",
        help="Write one parquet row per extracted item while preserving sample schema.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("data", "03_primary", "filing_data.parquet"),
        help="Output parquet path.",
    )
    parser.add_argument(
        "--user-agent",
        default=os.environ.get(
            "SEC_USER_AGENT", "finmem-data-pipeline research@example.com"
        ),
        help="SEC-required User-Agent header.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=SLEEP_TIME,
        help="Sleep seconds between SEC requests.",
    )
    return parser.parse_args()


def _request_json(url: str, headers: Dict[str, str], sleep: float) -> dict:
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, headers=headers, timeout=60)
        if resp.status_code == 429:
            time.sleep(max(sleep, 1.0))
            continue
        resp.raise_for_status()
        if sleep > 0:
            time.sleep(sleep)
        return resp.json()
    raise requests.HTTPError(f"Exceeded retry limit for {url}")


def _request_text(url: str, headers: Dict[str, str], sleep: float) -> str:
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, headers=headers, timeout=60)
        if resp.status_code == 429:
            time.sleep(max(sleep, 1.0))
            continue
        resp.raise_for_status()
        if sleep > 0:
            time.sleep(sleep)
        return resp.text
    raise requests.HTTPError(f"Exceeded retry limit for {url}")


def _safe_date(year: int, month: int, day: int) -> date:
    max_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(day, max_day))


def _default_financial_year_window(
    fiscal_year_end: Optional[str], years: int = DEFAULT_FINANCIAL_YEARS
) -> Tuple[date, date]:
    today = datetime.utcnow().date()
    # Default rolling window: Jan 1 of (current year - years) through today.
    # Example: if today is 2026-03-08 and years=2, start=2024-01-01, end=2026-03-08.
    start_date = date(today.year - years, 1, 1)
    return start_date, today


def _looks_like_xml(raw_text: str) -> bool:
    head = raw_text.lstrip()[:200].lower()
    if head.startswith("<?xml"):
        return True
    if head.startswith("<xml"):
        return True
    if "<xbrl" in head or "<ix:" in head:
        return True
    return False


def _document_to_text(raw_doc: str) -> str:
    parser = "xml" if _looks_like_xml(raw_doc) else "lxml"
    return BeautifulSoup(raw_doc, parser).get_text("\n")


def _parse_datetime_utc(ts: str) -> pd.Timestamp:
    return pd.to_datetime(ts, utc=True)


def _to_est_tz(utc_dt: pd.Timestamp) -> pd.Timestamp:
    return utc_dt.tz_convert("US/Eastern")


def _rows_from_submissions_json(data: dict, ticker: str, forms: Iterable[str]) -> pd.DataFrame:
    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()
    df = pd.DataFrame(recent)
    if df.empty:
        return df
    keep = ["form", "filingDate", "acceptanceDateTime", "accessionNumber", "primaryDocument"]
    for col in keep:
        if col not in df.columns:
            df[col] = None
    df = df[keep]
    df = df[df["form"].isin(forms)].copy()
    if df.empty:
        return df
    df["ticker"] = ticker
    df["cik"] = str(data.get("cik", ""))
    return df


def _collect_index_rows(
    cik: str,
    ticker: str,
    forms: Iterable[str],
    headers: Dict[str, str],
    sleep: float,
    base_submission: Optional[dict] = None,
) -> pd.DataFrame:
    base = (
        base_submission
        if base_submission is not None
        else _request_json(SEC_SUBMISSION_URL.format(cik=cik), headers=headers, sleep=sleep)
    )
    frames = [_rows_from_submissions_json(base, ticker=ticker, forms=forms)]
    for extra in base.get("filings", {}).get("files", []):
        name = extra.get("name")
        if not name:
            continue
        extra_json = _request_json(
            SEC_SUBMISSION_FILE_URL.format(name=name), headers=headers, sleep=sleep
        )
        # extra files have recent-like arrays at root in most SEC payloads
        if "filings" in extra_json:
            frame = _rows_from_submissions_json(extra_json, ticker=ticker, forms=forms)
        else:
            frame = pd.DataFrame(extra_json)
            if not frame.empty:
                required = {
                    "form",
                    "filingDate",
                    "acceptanceDateTime",
                    "accessionNumber",
                    "primaryDocument",
                }
                if required.issubset(set(frame.columns)):
                    frame = frame[
                        [
                            "form",
                            "filingDate",
                            "acceptanceDateTime",
                            "accessionNumber",
                            "primaryDocument",
                        ]
                    ]
                    frame = frame[frame["form"].isin(forms)].copy()
                    frame["ticker"] = ticker
                    frame["cik"] = str(cik)
                else:
                    frame = pd.DataFrame()
        if not frame.empty:
            frames.append(frame)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out
    out = out.dropna(subset=["accessionNumber", "primaryDocument"])
    out = out.drop_duplicates(subset=["accessionNumber", "primaryDocument"])
    # Mirror original: prefer acceptance/fd timestamp descending.
    ts_source = out["acceptanceDateTime"].fillna(out["filingDate"])
    out["utc_timestamp"] = ts_source.apply(_parse_datetime_utc)
    out["est_timestamp"] = out["utc_timestamp"].apply(_to_est_tz)
    out["filing_date_obj"] = pd.to_datetime(out["filingDate"]).dt.date
    out = out.sort_values(by="utc_timestamp", ascending=False).reset_index(drop=True)
    return out


def _build_document_url(cik: str, accession_number: str, primary_document: str) -> str:
    accession_compact = accession_number.replace("-", "")
    cik_nozero = str(int(cik))
    return SEC_ARCHIVE_URL.format(
        cik_nozero=cik_nozero,
        accession=accession_compact,
        document=primary_document,
    )


def _item_regex_10k(item_code: str) -> re.Pattern[str]:
    return re.compile(
        rf"\bitem\s*{re.escape(item_code.lower())}\b[\.\:\-\s]*", re.IGNORECASE
    )


def _item_regex_8k(item_code: str) -> re.Pattern[str]:
    major, minor = item_code.split("-")
    marker = rf"{int(major)}\.{int(minor):02d}"
    return re.compile(rf"\bitem\s*{marker}\b[\.\:\-\s]*", re.IGNORECASE)


def _parse_10q_code(code: str) -> Tuple[int, str]:
    match = re.fullmatch(r"part([12])item([0-9]+[a-z]?)", code.lower())
    if not match:
        raise ValueError(f"Bad 10-Q item code: {code}")
    return int(match.group(1)), match.group(2)


def _part_span(text: str, part_no: int) -> Tuple[int, int]:
    part1_re = re.compile(r"\bpart\s*i\b", re.IGNORECASE)
    part2_re = re.compile(r"\bpart\s*ii\b", re.IGNORECASE)
    if part_no == 1:
        start_match = part1_re.search(text)
        if not start_match:
            return 0, len(text)
        end_match = part2_re.search(text, start_match.end())
        return start_match.start(), end_match.start() if end_match else len(text)
    start_match = part2_re.search(text)
    if not start_match:
        return 0, len(text)
    return start_match.start(), len(text)


def _next_item_code(item_codes: List[str], current: str) -> Optional[str]:
    if current not in item_codes:
        return None
    idx = item_codes.index(current)
    if idx + 1 >= len(item_codes):
        return None
    return item_codes[idx + 1]


def _slice_by_regex(
    raw_text: str,
    start_re: re.Pattern[str],
    end_res: List[re.Pattern[str]],
    lower_bound: int = 120,
    span: Optional[Tuple[int, int]] = None,
) -> Optional[str]:
    region_start, region_end = span if span else (0, len(raw_text))
    region = raw_text[region_start:region_end]
    start_match = start_re.search(region)
    if not start_match:
        return None
    s_idx = region_start + start_match.start()
    end_candidates: List[int] = []
    for end_re in end_res:
        end_match = end_re.search(raw_text, pos=s_idx + 20)
        if end_match and end_match.start() > s_idx:
            end_candidates.append(end_match.start())
    e_idx = min(end_candidates) if end_candidates else region_end
    if e_idx - s_idx < lower_bound:
        return None
    return raw_text[s_idx:e_idx]


def _extract_item_section(raw_text: str, form: str, item_code: str) -> Optional[str]:
    if form == "10-K":
        start_re = _item_regex_10k(item_code)
        next_code = _next_item_code(TEN_K_ITEM_CODE, item_code)
        end_res = [_item_regex_10k(next_code)] if next_code else []
        return _slice_by_regex(raw_text, start_re, end_res)

    if form == "10-Q":
        part_no, item_no = _parse_10q_code(item_code)
        same_part_codes = [
            c for c in TEN_Q_ITEM_CODE if c.startswith(f"part{part_no}")
        ]
        next_code = _next_item_code(same_part_codes, item_code.lower())
        span = _part_span(raw_text, part_no)
        start_re = re.compile(
            rf"\bitem\s*{re.escape(item_no)}\b[\.\:\-\s]*", re.IGNORECASE
        )
        end_res: List[re.Pattern[str]] = []
        if next_code:
            _, next_item = _parse_10q_code(next_code)
            end_res.append(
                re.compile(
                    rf"\bitem\s*{re.escape(next_item)}\b[\.\:\-\s]*",
                    re.IGNORECASE,
                )
            )
        return _slice_by_regex(raw_text, start_re, end_res, span=span)

    if form == "8-K":
        start_re = _item_regex_8k(item_code)
        next_code = _next_item_code(EIGHT_K_ITEM_CODE, item_code)
        end_res = [_item_regex_8k(next_code)] if next_code else []
        return _slice_by_regex(raw_text, start_re, end_res, lower_bound=60)

    return None


def _form_items(form: str) -> List[str]:
    if form == "10-K":
        return TEN_K_ITEM_CODE.copy()
    if form == "10-Q":
        return TEN_Q_ITEM_CODE.copy()
    if form == "8-K":
        return EIGHT_K_ITEM_CODE.copy()
    return DEFAULT_FORM_TO_ITEM_CODE.get(form, []).copy()


def _collect_items(raw_text: str, form: str) -> List[Tuple[str, str]]:
    item_codes = _form_items(form=form)
    if not item_codes:
        return []
    found_items: List[Tuple[str, str]] = []
    for code in item_codes:
        section = _extract_item_section(raw_text=raw_text, form=form, item_code=code)
        if section:
            found_items.append((code, section))
    return found_items


def _clean_section(text: str) -> str:
    txt = normalize("NFKC", text)
    if unidecode is not None:
        txt = unidecode(txt)
    else:
        txt = txt.encode("ascii", errors="ignore").decode("ascii")
    txt = " ".join(txt.split())
    return txt.lower()


def main() -> None:
    load_dotenv(os.path.join(".env"))
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    logger.info("Program starts")
    headers = {"User-Agent": args.user_agent}

    ticker_cik_map = SAMPLE_TICKER_CIK_MAP.copy()
    records: List[Dict[str, object]] = []
    for ticker, cik in ticker_cik_map.items():
        print(f"Starting ticker {ticker} ({cik})")
        base_submission = _request_json(
            SEC_SUBMISSION_URL.format(cik=cik), headers=headers, sleep=args.sleep
        )
        default_start, default_end = _default_financial_year_window(
            base_submission.get("fiscalYearEnd"), years=DEFAULT_FINANCIAL_YEARS
        )
        start_date = (
            datetime.strptime(args.start_date, "%Y-%m-%d").date()
            if args.start_date
            else default_start
        )
        end_date = (
            datetime.strptime(args.end_date, "%Y-%m-%d").date()
            if args.end_date
            else default_end
        )
        logger.info(f"{ticker}: Date range used: {start_date} to {end_date}")

        index_df = _collect_index_rows(
            cik=cik,
            ticker=ticker,
            forms=args.forms,
            headers=headers,
            sleep=args.sleep,
            base_submission=base_submission,
        )
        if index_df.empty:
            logger.info(f"{ticker}: No filing index rows found.")
            continue

        index_df = index_df[
            (index_df["filing_date_obj"] >= start_date)
            & (index_df["filing_date_obj"] <= end_date)
        ].copy()
        if index_df.empty:
            logger.info(f"{ticker}: No filings after date filtering.")
            continue
        total_filings = len(index_df)
        print(f"{ticker}: processing {total_filings} filings")

        for i, (_, row) in enumerate(index_df.iterrows(), start=1):
            if i == 1 or i % 10 == 0 or i == total_filings:
                print(f"{ticker}: {i}/{total_filings}")
            document_url = _build_document_url(
                cik=str(row["cik"]),
                accession_number=str(row["accessionNumber"]),
                primary_document=str(row["primaryDocument"]),
            )
            form = str(row["form"])
            section_code = ",".join(_form_items(form=form)) or "unknown"
            try:
                html_text = _request_text(document_url, headers=headers, sleep=args.sleep)
                raw_text = _document_to_text(html_text)
                found_items = _collect_items(raw_text=raw_text, form=form)
                if not found_items:
                    logger.info(
                        f"[red]{section_code}, file_url: {document_url}[/red], exception: section not found"
                    )
                    continue
            except Exception as exc:  # pylint: disable=broad-except
                logger.info(
                    f"[red]{section_code}, file_url: {document_url}[/red], exception: {exc}"
                )
                continue

            if args.row_per_item:
                for item_code, section_text in found_items:
                    cleaned_text = _clean_section(f"[item {item_code}] {section_text}")
                    if not cleaned_text:
                        continue
                    records.append(
                        {
                            "document_url": document_url,
                            "content": cleaned_text,
                            "ticker": str(row["ticker"]),
                            "cik": str(row["cik"]),
                            "utc_timestamp": row["utc_timestamp"],
                            "est_timestamp": row["est_timestamp"],
                            "type": form,
                        }
                    )
                continue

            section_text = " ".join([f"[item {c}] {s}" for c, s in found_items])
            cleaned_text = _clean_section(section_text)
            if not cleaned_text:
                continue

            records.append(
                {
                    "document_url": document_url,
                    "content": cleaned_text,
                    "ticker": str(row["ticker"]),
                    "cik": str(row["cik"]),
                    "utc_timestamp": row["utc_timestamp"],
                    "est_timestamp": row["est_timestamp"],
                    "type": form,
                }
            )

    if not records:
        logger.info("No rows extracted. Output parquet not written.")
        print("No rows extracted. Output parquet not written.")
        return

    filing_df = pd.DataFrame(
        records,
        columns=[
            "document_url",
            "content",
            "ticker",
            "cik",
            "utc_timestamp",
            "est_timestamp",
            "type",
        ],
    )
    filing_df.to_parquet(args.output, index=False)
    csv_output = f"{os.path.splitext(args.output)[0]}.csv"
    human_df = filing_df.copy()
    human_df["utc_timestamp"] = human_df["utc_timestamp"].astype(str)
    human_df["est_timestamp"] = human_df["est_timestamp"].astype(str)
    human_df.to_csv(csv_output, index=False)
    logger.info("Program ends")
    print(f"Wrote {len(filing_df)} rows to {args.output}")
    print(f"Wrote {len(human_df)} rows to {csv_output}")


if __name__ == "__main__":
    main()
