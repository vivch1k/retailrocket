import os
import gc
import time
import psutil
import pandas as pd
import numpy as np

class UserSequenceTestDataset(IterableDataset):
    def __init__(
        self,
        part_paths,
        item2idx: dict,
        seq_col: str = "items",
        user_col: str = "user_id",
        max_len: int = 100,
        test_last_n: int = 3,
        min_history_len: int = 2,
        rows_per_read: int = 50_000,
    ):
        self.part_paths = [Path(p) for p in part_paths]
        self.item2idx = item2idx

        self.seq_col = seq_col
        self.user_col = user_col

        self.max_len = max_len
        self.test_last_n = test_last_n
        self.min_history_len = min_history_len
        self.rows_per_read = rows_per_read

    def __iter__(self):
        for path in self.part_paths:
            pf = pq.ParquetFile(path)

            for batch in pf.iter_batches(
                columns=[self.user_col, self.seq_col],
                batch_size=self.rows_per_read,
            ):
                df_batch = batch.to_pandas()

                for row in df_batch.itertuples(index=False):
                    user_id = getattr(row, self.user_col)
                    raw_seq = getattr(row, self.seq_col)

                    if raw_seq is None:
                        continue

                    if isinstance(raw_seq, np.ndarray):
                        raw_seq = raw_seq.tolist()

                    raw_seq = [int(x) for x in raw_seq if pd.notna(x)]

                    if len(raw_seq) <= self.test_last_n + self.min_history_len:
                        continue

                    history_raw = raw_seq[:-self.test_last_n]
                    target_raw = raw_seq[-self.test_last_n:]

                    history_encoded = [
                        self.item2idx[item]
                        for item in history_raw
                        if item in self.item2idx
                    ]

                    if len(history_encoded) < self.min_history_len:
                        continue

                    history_encoded = history_encoded[-self.max_len:]

                    yield {
                        "user_id": user_id,
                        "history_encoded": history_encoded,
                        "history_raw": history_raw,
                        "target_raw": target_raw,
                    }


def collate_test_batch(samples):
    max_len = max(len(x["history_encoded"]) for x in samples)

    user_ids = []
    histories_encoded = []
    histories_raw = []
    targets_raw = []

    for x in samples:
        user_ids.append(x["user_id"])
        histories_raw.append(x["history_raw"])
        targets_raw.append(x["target_raw"])

        seq = x["history_encoded"]
        pad_len = max_len - len(seq)

        # Как в train: PAD слева
        padded_seq = [PAD_IDX] * pad_len + seq

        histories_encoded.append(padded_seq)

    input_ids = torch.tensor(histories_encoded, dtype=torch.long)

    return {
        "user_ids": user_ids,
        "input_ids": input_ids,
        "histories_raw": histories_raw,
        "targets_raw": targets_raw,
    }


TEST_BATCH_SIZE = 512
TOP_K = 3

test_dataset = UserSequenceTestDataset(
    part_paths=part_paths,
    item2idx=item2idx,
    seq_col=SEQ_COL,
    user_col=USER_COL,
    max_len=MAX_LEN,
    test_last_n=TEST_LAST_N,
    min_history_len=2,
    rows_per_read=50_000,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=TEST_BATCH_SIZE,
    collate_fn=collate_test_batch,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

def recommend_batch(
    model,
    input_ids,
    histories_raw,
    item2idx,
    idx2item,
    top_k: int = 3,
    filter_seen: bool = True,
):
    model.eval()

    input_ids = input_ids.to(device, non_blocking=True)

    with torch.no_grad():
        logits = model(input_ids)

        # Так как padding слева, последний реальный item находится в конце.
        # Поэтому берем logits[:, -1, :]
        next_logits = logits[:, -1, :].clone()

        # Не рекомендуем PAD
        next_logits[:, PAD_IDX] = -1e9

        if filter_seen:
            for row_idx, hist_raw in enumerate(histories_raw):
                seen_idx = [
                    item2idx[int(item)]
                    for item in hist_raw
                    if int(item) in item2idx
                ]

                if seen_idx:
                    next_logits[row_idx, seen_idx] = -1e9

        probs = torch.softmax(next_logits, dim=-1)

        top_scores, top_indices = torch.topk(
            probs,
            k=top_k,
            dim=-1,
        )

    return top_indices.cpu().numpy(), top_scores.cpu().numpy()


def build_recs_and_test_data(
    model,
    test_loader,
    item2idx,
    idx2item,
    top_k: int = 3,
    filter_seen: bool = True,
):
    rec_rows = []
    test_rows = []

    model.eval()

    for batch in tqdm(test_loader, desc="Evaluate"):
        user_ids = batch["user_ids"]
        input_ids = batch["input_ids"]
        histories_raw = batch["histories_raw"]
        targets_raw = batch["targets_raw"]

        top_indices, top_scores = recommend_batch(
            model=model,
            input_ids=input_ids,
            histories_raw=histories_raw,
            item2idx=item2idx,
            idx2item=idx2item,
            top_k=top_k,
            filter_seen=filter_seen,
        )

        # рекомендации
        for i, user_id in enumerate(user_ids):
            for rank in range(top_k):
                idx = int(top_indices[i, rank])
                score = float(top_scores[i, rank])

                item_id = idx2item[idx]

                rec_rows.append(
                    {
                        Columns.User: user_id,
                        Columns.Item: item_id,
                        Columns.Score: score,
                        Columns.Rank: rank + 1,
                    }
                )

        # ground truth
        for user_id, target_items in zip(user_ids, targets_raw):
            for item_id in target_items:
                test_rows.append(
                    {
                        Columns.User: user_id,
                        Columns.Item: int(item_id),
                        Columns.Weight: 1.0,
                    }
                )

    recs = pd.DataFrame(rec_rows)
    inner_test_data = pd.DataFrame(test_rows)

    return recs, inner_test_data


recs, inner_test_data = build_recs_and_test_data(
    model=model,
    test_loader=test_loader,
    item2idx=item2idx,
    idx2item=idx2item,
    top_k=TOP_K,
    filter_seen=True,
)

print(recs.head())
print(inner_test_data.head())


metrics = {
    "hit_rate@3": HitRate(k=3),
    "prec@3": Precision(k=3),
    "recall@3": Recall(k=3),
    "ndcg@3": NDCG(k=3),
    "map@3": MAP(k=3),
}

metrics_rnn_recs = calc_metrics(
    metrics,
    reco=recs,
    interactions=inner_test_data,
)

metrics_rnn_recs



process = psutil.Process(os.getpid())


def gb(x):
    return x / 1024 ** 3


def process_memory_gb():
    mem = process.memory_info()
    return gb(mem.rss)


def df_memory_gb(df: pd.DataFrame):
    return gb(df.memory_usage(deep=True).sum())


def log_memory(stage: str, df: pd.DataFrame | None = None):
    msg = f"[{time.strftime('%H:%M:%S')}] {stage} | process_rss={process_memory_gb():.2f} GB"

    if df is not None:
        msg += f" | df_memory={df_memory_gb(df):.2f} GB | rows={len(df):,}"

    print(msg, flush=True)


def log_top_columns_memory(df: pd.DataFrame, n: int = 10):
    mem = df.memory_usage(deep=True).sort_values(ascending=False).head(n)
    print("Top memory columns:")
    print((mem / 1024 ** 2).round(2).astype(str) + " MB")
    print()


# текущий цикл

log_memory("before export")

with engine.connect().execution_options(stream_results=True, max_row_buffer=chunksize) as conn:
    for i, chunk in enumerate(
        pd.read_sql_query(
            events_query,
            conn,
            params=params,
            chunksize=chunksize,
        )
    ):
        log_memory(f"chunk {i} fetched", chunk)
        log_top_columns_memory(chunk)

        for col in chunk.select_dtypes(include="object").columns:
            chunk[col] = chunk[col].replace("", np.nan)

        log_memory(f"chunk {i} after empty string replace", chunk)

        chunk["passport_id"] = chunk["passport_id"].astype(str)

        log_memory(f"chunk {i} before merge", chunk)

        chunk = chunk.merge(
            mapping_for_merge,
            on="passport_id",
            how="left",
        )

        log_memory(f"chunk {i} after merge", chunk)
        log_top_columns_memory(chunk)

        part_path = RAW_DATA_DIR / f"{RAW_DATA_FILE_NAME}_{i:03d}.parquet"

        chunk.to_parquet(
            part_path,
            index=False,
            compression="zstd",
        )

        log_memory(f"chunk {i} after to_parquet", chunk)

        rows = len(chunk)
        del chunk
        collected = gc.collect()

        log_memory(f"chunk {i} after del/gc, collected={collected}")

        print(f"Saved chunk {i}: {rows:,} rows -> {part_path}", flush=True)


# новый цикл

chunksize = 100_000

passport_to_canonical = (
    mapping_for_merge
    .drop_duplicates("passport_id")
    .assign(passport_id=lambda x: x["passport_id"].astype(str))
    .set_index("passport_id")["canonical_service_id"]
    .to_dict()
)

for bucket in range(SAMPLE_BUCKETS):
    print(f"\n=== Export bucket {bucket}/{SAMPLE_BUCKETS - 1} ===", flush=True)

    params = {
        "passport_ids": topk_passport_ids,
        "bucket": bucket,
        "max_rows": int(np.ceil(MAX_ROWS_SAFETY_LIMIT / SAMPLE_BUCKETS)),
    }

    with engine.connect().execution_options(
        stream_results=True,
        max_row_buffer=chunksize,
    ) as conn:
        for j, chunk in enumerate(
            pd.read_sql_query(
                events_query_by_bucket,
                conn,
                params=params,
                chunksize=chunksize,
            )
        ):
            log_memory(f"bucket={bucket}, chunk={j} fetched", chunk)

            chunk["passport_id"] = chunk["passport_id"].astype(str)
            chunk["canonical_service_id"] = chunk["passport_id"].map(passport_to_canonical)

            chunk = chunk[
                [
                    "oid_sha1",
                    "first_st_datetime",
                    "passport_id",
                    "canonical_service_id",
                ]
            ].copy()

            log_memory(f"bucket={bucket}, chunk={j} prepared", chunk)

            part_path = RAW_DATA_DIR / (
                f"{RAW_DATA_FILE_NAME}_bucket_{bucket:04d}_part_{j:03d}.parquet"
            )

            chunk.to_parquet(
                part_path,
                index=False,
                compression="zstd",
            )

            log_memory(f"bucket={bucket}, chunk={j} saved", chunk)

            rows = len(chunk)
            del chunk
            gc.collect()

            log_memory(f"bucket={bucket}, chunk={j} after del/gc")

            print(f"Saved {rows:,} rows -> {part_path}", flush=True)