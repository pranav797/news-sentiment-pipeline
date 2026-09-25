# News Sentiment Pipeline

## Deduplication

News records receive a stable ID generated from the normalized headline title
and URL. Before insertion, the ingestion step compares each ID with the IDs
already stored in DuckDB and removes duplicates, including repeated records in
the same polling run. This prevents repeated RSS polls and cross-outlet copies
of the same story from inflating the dataset.