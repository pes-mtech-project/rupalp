import argparse
import os
import pickle
from typing import Dict, Tuple

from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def subset_symbol_dict(input_path: str, cur_symbol: str) -> Tuple[Dict, Dict]:
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    new_dict = {}
    ticker_dict_by_date = {}
    for k, v in tqdm(data.items()):
        cur_price = v[0].get("price", {})
        cur_news = v[1].get("news", {})
        cur_filing_q = v[2].get("filing_q", v[2].get("filling_q", {}))
        cur_filing_k = v[3].get("filing_k", v[3].get("filling_k", {}))

        if cur_symbol not in cur_news:
            continue

        new_dict[k] = {
            "price": {cur_symbol: cur_price[cur_symbol]} if cur_symbol in cur_price else {},
            "filing_k": {cur_symbol: cur_filing_k[cur_symbol]} if cur_symbol in cur_filing_k else {},
            "filing_q": {cur_symbol: cur_filing_q[cur_symbol]} if cur_symbol in cur_filing_q else {},
            "news": {cur_symbol: cur_news[cur_symbol]},
        }
        ticker_dict_by_date[k] = list(new_dict[k]["price"].keys())

    return new_dict, ticker_dict_by_date


def sentiment_score_finbert(text: str) -> Tuple[float, float, float]:
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch

    if not hasattr(sentiment_score_finbert, "tokenizer"):
        sentiment_score_finbert.tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        sentiment_score_finbert.model = BertForSequenceClassification.from_pretrained(
            "yiyanghkust/finbert-tone"
        )

    inputs = sentiment_score_finbert.tokenizer(
        text, return_tensors="pt", max_length=512, truncation=True
    )
    outputs = sentiment_score_finbert.model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1).tolist()[0]
    return scores[2], scores[1], scores[0]


def assign_finbert_scores(new_dict: Dict, cur_symbol: str) -> None:
    for i_date in tqdm(new_dict):
        i_date_news = new_dict[i_date]["news"].get(cur_symbol, [])
        j_new_news = []
        for j_news in i_date_news:
            pos_score, neu_score, neg_score = sentiment_score_finbert(j_news)
            j_combine_news_sentiment = (
                f"{j_news} "
                f"The positive score for this news is {pos_score}. "
                f"The neutral score for this news is {neu_score}. "
                f"The negative score for this news is {neg_score}."
            )
            j_new_news.append(j_combine_news_sentiment)
        new_dict[i_date]["news"][cur_symbol] = j_new_news


def assign_vader_scores(new_dict: Dict, cur_symbol: str) -> None:
    analyzer = SentimentIntensityAnalyzer()
    for i_date in tqdm(new_dict):
        i_date_news = new_dict[i_date]["news"].get(cur_symbol, [])
        j_new_news = []
        for j_news in i_date_news:
            j_news_sentiment = analyzer.polarity_scores(j_news)
            pos_score = j_news_sentiment["pos"]
            neu_score = j_news_sentiment["neu"]
            neg_score = j_news_sentiment["neg"]
            j_combine_news_sentiment = (
                f"{j_news} "
                f"The positive score for this news is {pos_score}. "
                f"The neutral score for this news is {neu_score}. "
                f"The negative score for this news is {neg_score}."
            )
            j_new_news.append(j_combine_news_sentiment)
        new_dict[i_date]["news"][cur_symbol] = j_new_news


def export_sub_symbol(
    cur_symbol_lst,
    senti_model_type: str,
    input_dir: str,
    output_dir: str,
) -> None:
    print("Ticker list:", cur_symbol_lst)
    os.makedirs(output_dir, exist_ok=True)
    for cur_symbol in cur_symbol_lst:
        new_dict, _ = subset_symbol_dict(input_dir, cur_symbol)
        if senti_model_type == "FinBERT":
            assign_finbert_scores(new_dict, cur_symbol)
        else:
            assign_vader_scores(new_dict, cur_symbol)

        out_path = os.path.join(output_dir, f"subset_symbols_{cur_symbol}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(new_dict, f)
        print(f"Saved: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append sentiment scores to news text by ticker.")
    parser.add_argument(
        "--symbols",
        default="TSLA,AMZN,NFLX,MSFT,COIN",
        help="Comma-separated ticker list.",
    )
    parser.add_argument(
        "--input-dir",
        default="./data/03_model_input/env_data.pkl",
        help="Input env_data.pkl path.",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/06_input",
        help="Directory for subset ticker pickle files.",
    )
    parser.add_argument(
        "--senti-model-type",
        default="Vader",
        choices=["Vader", "FinBERT"],
        help="Sentiment model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    export_sub_symbol(
        cur_symbol_lst=symbols,
        senti_model_type=args.senti_model_type,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
