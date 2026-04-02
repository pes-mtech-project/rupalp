# memory ids
short_memory_id_desc = "The id of the short-term information."
mid_memory_id_desc = "The id of the mid-term information."
long_memory_id_desc = "The id of the long-term information."
reflection_memory_id_desc = "The id of the reflection-term information."
train_memory_id_extract_prompt = "Provide the piece of information  related the most to the investment decisions from mainstream sources such as the investment suggestions major fund firms such as ARK, Two Sigma, Bridgewater Associates, etc. in the {memory_layer} memory?"
test_memory_id_extract_prompt = "Provide the piece of information related most to your investment decisions in the {memory_layer} memory?"

# trade summary
train_trade_reason_summary = "Given a professional trader's trading suggestion, can you explain to me why the trader drive such a decision with the information provided to you?"
test_trade_reason_summary = "Given the information of text and the summary of the stock price movement. Please explain the reason why you make the investment decision."
test_invest_action_choice = "Given the information, please make an investment decision: buy the stock, sell, and hold the stock"

# investment info
train_investment_info_prefix = "The current date is {cur_date}. Here are the observed financial market facts: for {symbol}, the price difference between the next trading day and the current trading day is: {future_record}\n\n"
test_investment_info_prefix = "The ticker of the stock to be analyzed is {symbol} and the current date is {cur_date}"
test_sentiment_explanation = """Summarize and interpret investment sentiment. The positive, neutral and negative scores are for understanding investment sentiments, opinions, or emotions.
        They are ratios for proportions of text that fall in each category and can be added up to be 1.
        For example, positive news about a company can lift investor sentiment, encouraging more buying activity which in turn can push stock prices higher.
        Conversely, negative news can dampen investor sentiment, leading to selling pressure and lower prices.
        """
test_momentum_explanation = """The information below provides a summary of stock price fluctuations over the previous few days, which is the "momentum" of a stock.
        It reflects the trend of a stock.
        Momentum is based on the idea that securities that have performed well in the past will continue to perform well, and conversely, securities that have performed poorly will continue to perform poorly.
        """

# prompts
train_prompt = """Given the following information, can you explain to me why the financial market fluctuation from current day to the next day behaves like this?
    You need to provide summary information and the IDs of the information to support your summary.

    ${investment_info}

    ${gr.complete_json_suffix_v2}
    Your output should strictly conform to the following json format without any additional contents: {"summary_reason": string, "short_memory_index": number, "middle_memory_index": number, "long_memory_index": number, "reflection_memory_index": number}
"""
# When cumulative returns are positive or zero, you are a risk-seeking investor.
# When cumulative returns are negative, you are a risk-averse investor.
test_prompt = """Given the information, can you make an investment decision? Please consider the short-term information, mid-term information, long-term information, and reflection information.
    Please also consider the momentum of the stock price.
    When cumulative returns are positive or zero, you are a risk-seeking investor.
    When cumulative returns are negative, you are a risk-averse investor.
    Considering how many shares of the stock you as an investor are holding now, you should provide exactly one of the following investment decisions: buy, sell, or hold.
    When it is hard to make a buy-or-sell decision, you could go with the hold option.
    You also need to provide the IDs of the information to support your decision and summarize the reason.

    ${investment_info}

    ${gr.complete_json_suffix_v2}
"""
