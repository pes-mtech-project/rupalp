import pickle

with open(
    "/Users/rupal.pursharthi/Library/CloudStorage/OneDrive-TheWaltDisneyCompany/Documents/Project_4th sem/FinMem-LLM-StockTrading-main/data-pipeline/Fake-Sample-Data/example_output/filing_q.pkl",
    "rb",
) as f:
    data = pickle.load(f)

print(f"Top-level type: {type(data)}")
for idx, (key, value) in enumerate(data.items()):
    print(f"\nRecord {idx + 1}: {key} -> {type(value)}")
    print(value)
    if idx == 1:
        break
