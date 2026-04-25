import pandas as pd

input_path = "Electronics_5.json"
output_path = "electronics_small.csv"

chunks = pd.read_json(input_path, lines=True, chunksize=10000)

first_chunk = True
rows_written = 0
max_rows = 50000   # 🔥 limit size

for chunk in chunks:
    chunk = chunk.rename(columns={
        'overall': 'rating',
        'reviewText': 'review',
        'summary': 'summary',
        'reviewerID': 'user',
        'asin': 'item'
    })

    chunk = chunk[['user', 'item', 'rating', 'review', 'summary']]
    chunk = chunk.dropna()

    # 🔥 Stop when enough data collected
    if rows_written >= max_rows:
        break

    remaining = max_rows - rows_written
    chunk = chunk.head(remaining)

    chunk['item'] = chunk['item'].astype(str)
    chunk['user'] = chunk['user'].astype(str)

    # 🔥 Write directly to CSV (no big memory)
    chunk.to_csv(output_path, mode='w' if first_chunk else 'a',
                 header=first_chunk, index=False)

    rows_written += len(chunk)
    first_chunk = False

    print(f"Processed {rows_written} rows...")

print("✅ Done! File saved as electronics_small.csv")