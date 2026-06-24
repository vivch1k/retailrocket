class UserSequenceDataset(IterableDataset):
    """
    Один Dataset для train / val_loss / test / embeddings.

    Ожидаемый parquet-формат:
        user_id | items

    items:
        либо уже dense item_idx: [1, 5, 10, ...]
        либо raw canonical_service_id, тогда передай item2idx.
    """

    def __init__(
        self,
        part_paths,
        mode: str,
        seq_col: str = "items",
        user_col: str = "user_id",
        item2idx: dict | None = None,
        max_len: int = 100,
        val_target_n: int = 3,
        test_target_n: int = 3,
        min_history_len: int = 2,
        random_crop: bool = True,
        shuffle_files: bool = False,
        rows_per_read: int = 50_000,
        seed: int = 42,
    ):
        assert mode in {"train", "val_loss", "test", "emb"}

        self.part_paths = [Path(p) for p in part_paths]
        self.mode = mode

        self.seq_col = seq_col
        self.user_col = user_col
        self.item2idx = item2idx

        self.max_len = max_len
        self.val_target_n = val_target_n
        self.test_target_n = test_target_n
        self.holdout_n = val_target_n + test_target_n

        self.min_history_len = min_history_len
        self.random_crop = random_crop
        self.shuffle_files = shuffle_files
        self.rows_per_read = rows_per_read
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _get_worker_paths(self):
        paths = list(self.part_paths)

        rng = random.Random(self.seed + self.epoch)

        if self.shuffle_files:
            rng.shuffle(paths)

        worker = get_worker_info()

        if worker is None:
            return paths

        return paths[worker.id::worker.num_workers]

    def _to_list(self, seq):
        if seq is None:
            return []

        if isinstance(seq, np.ndarray):
            seq = seq.tolist()

        return [int(x) for x in seq if pd.notna(x)]

    def _encode(self, seq):
        seq = self._to_list(seq)

        # Если items уже dense indices, mapping не нужен.
        if self.item2idx is None:
            return seq

        return [
            self.item2idx[item]
            for item in seq
            if item in self.item2idx
        ]

    def _make_train_sample(self, user_id, raw_seq, rng):
        raw_seq = self._to_list(raw_seq)

        # train = все кроме val+test
        if len(raw_seq) < self.min_history_len + self.holdout_n:
            return None

        train_raw = raw_seq[:-self.holdout_n]
        train_seq = self._encode(train_raw)

        if len(train_seq) < self.min_history_len:
            return None

        max_items = self.max_len + 1

        if len(train_seq) > max_items:
            if self.random_crop:
                start = rng.randint(0, len(train_seq) - max_items)
                train_seq = train_seq[start:start + max_items]
            else:
                train_seq = train_seq[-max_items:]

        input_seq = train_seq[:-1]
        labels = train_seq[1:]

        if len(input_seq) == 0:
            return None

        return {
            "mode": "ce",
            "input_seq": input_seq,
            "labels": labels,
        }

    def _make_val_loss_sample(self, user_id, raw_seq, rng):
        raw_seq = self._to_list(raw_seq)

        if len(raw_seq) < self.min_history_len + self.holdout_n:
            return None

        # val_loss считаем на validation-части:
        # train = seq[:-6]
        # val   = seq[-6:-3]
        # test  = seq[-3:]
        #
        # Для входа берем train + val, test отрезаем.
        val_prefix_raw = raw_seq[:-self.test_target_n]
        val_prefix_seq = self._encode(val_prefix_raw)

        if len(val_prefix_seq) < self.min_history_len + self.val_target_n:
            return None

        input_seq = val_prefix_seq[:-1]
        labels = val_prefix_seq[1:]

        if len(input_seq) > self.max_len:
            input_seq = input_seq[-self.max_len:]
            labels = labels[-self.max_len:]

        # ВАЖНО:
        # loss считаем только на последних val_target_n позициях.
        masked_labels = [-100] * len(labels)
        masked_labels[-self.val_target_n:] = labels[-self.val_target_n:]

        return {
            "mode": "ce",
            "input_seq": input_seq,
            "labels": masked_labels,
        }

    def _make_test_sample(self, user_id, raw_seq, rng):
        raw_seq = self._to_list(raw_seq)

        if len(raw_seq) < self.min_history_len + self.test_target_n:
            return None

        # test используем только в самом конце:
        # history = train + val = seq[:-3]
        # target  = test        = seq[-3:]
        history_raw = raw_seq[:-self.test_target_n]
        target_raw = raw_seq[-self.test_target_n:]

        history_encoded = self._encode(history_raw)

        if len(history_encoded) < self.min_history_len:
            return None

        history_encoded = history_encoded[-self.max_len:]

        return {
            "mode": "eval",
            "user_id": user_id,
            "input_seq": history_encoded,
            "history_raw": history_raw,
            "target_raw": target_raw,
        }

    def _make_emb_sample(self, user_id, raw_seq, rng):
        raw_seq = self._to_list(raw_seq)

        # Для честных train-embeddings можно брать seq[:-6].
        # Если хочешь embeddings по всей истории, поменяй на raw_seq.
        if len(raw_seq) < self.min_history_len + self.holdout_n:
            return None

        history_raw = raw_seq[:-self.holdout_n]
        history_encoded = self._encode(history_raw)

        if len(history_encoded) < self.min_history_len:
            return None

        history_encoded = history_encoded[-self.max_len:]

        return {
            "mode": "emb",
            "user_id": user_id,
            "input_seq": history_encoded,
        }

    def _make_sample(self, user_id, raw_seq, rng):
        if self.mode == "train":
            return self._make_train_sample(user_id, raw_seq, rng)

        if self.mode == "val_loss":
            return self._make_val_loss_sample(user_id, raw_seq, rng)

        if self.mode == "test":
            return self._make_test_sample(user_id, raw_seq, rng)

        if self.mode == "emb":
            return self._make_emb_sample(user_id, raw_seq, rng)

        raise ValueError(self.mode)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        for path in self._get_worker_paths():
            pf = pq.ParquetFile(path)

            for batch in pf.iter_batches(
                columns=[self.user_col, self.seq_col],
                batch_size=self.rows_per_read,
            ):
                df_batch = batch.to_pandas()

                for row in df_batch.itertuples(index=False):
                    user_id = getattr(row, self.user_col)
                    raw_seq = getattr(row, self.seq_col)

                    sample = self._make_sample(user_id, raw_seq, rng)

                    if sample is not None:
                        yield sample



def collate_user_sequence_batch(samples):
    mode = samples[0]["mode"]

    max_len = max(len(x["input_seq"]) for x in samples)

    input_ids = []

    for x in samples:
        seq = x["input_seq"]
        pad_len = max_len - len(seq)

        # left padding
        input_ids.append([PAD_IDX] * pad_len + seq)

    input_ids = torch.tensor(input_ids, dtype=torch.long)

    if mode == "ce":
        labels = []

        for x in samples:
            label_seq = x["labels"]
            pad_len = max_len - len(label_seq)

            labels.append([-100] * pad_len + label_seq)

        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "mode": "ce",
            "input_ids": input_ids,
            "labels": labels,
        }

    if mode == "eval":
        return {
            "mode": "eval",
            "user_ids": [x["user_id"] for x in samples],
            "input_ids": input_ids,
            "histories_raw": [x["history_raw"] for x in samples],
            "targets_raw": [x["target_raw"] for x in samples],
        }

    if mode == "emb":
        return {
            "mode": "emb",
            "user_ids": [x["user_id"] for x in samples],
            "input_ids": input_ids,
        }

    raise ValueError(mode)


def train_one_epoch(model, train_loader, optimizer, loss_fn, n_items):
    model.train()

    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(train_loader, desc="train"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits = model(input_ids)

        loss = loss_fn(
            logits.reshape(-1, n_items),
            labels.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    return total_loss / max(total_tokens, 1)


def evaluate_loss(model, val_loader, loss_fn, n_items):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="val"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            logits = model(input_ids)

            loss = loss_fn(
                logits.reshape(-1, n_items),
                labels.reshape(-1),
            )

            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    return total_loss / max(total_tokens, 1)


best_val_loss = float("inf")
bad_epochs = 0
patience = 3

history = []

for epoch in range(EPOCHS):
    train_dataset.set_epoch(epoch)

    train_loss = train_one_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        n_items=n_items,
    )

    val_loss = evaluate_loss(
        model=model,
        val_loader=val_loader,
        loss_fn=loss_fn,
        n_items=n_items,
    )

    history.append(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
    )

    print(
        f"epoch={epoch + 1} | "
        f"train_loss={train_loss:.5f} | "
        f"val_loss={val_loss:.5f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        bad_epochs = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "n_items": n_items,
            },
            MODELS_PATH / "best_gru.pt",
        )

        print("saved best model")

    else:
        bad_epochs += 1

        if bad_epochs >= patience:
            print("early stopping")
            break\


def recommend_batch(
    model,
    input_ids,
    histories_raw,
    idx2item,
    item2idx=None,
    top_k=10,
    filter_seen=True,
):
    model.eval()

    input_ids = input_ids.to(device, non_blocking=True)

    with torch.no_grad():
        logits = model(input_ids)

        next_logits = logits[:, -1, :].clone()
        next_logits[:, PAD_IDX] = -1e9

        if filter_seen:
            for row_idx, hist_raw in enumerate(histories_raw):
                if item2idx is None:
                    seen_idx = [int(x) for x in hist_raw]
                else:
                    seen_idx = [
                        item2idx[int(x)]
                        for x in hist_raw
                        if int(x) in item2idx
                    ]

                seen_idx = [
                    x for x in seen_idx
                    if 0 <= x < next_logits.size(1)
                ]

                if seen_idx:
                    next_logits[row_idx, seen_idx] = -1e9

        top_scores, top_indices = torch.topk(
            next_logits,
            k=top_k,
            dim=-1,
        )

    return top_indices.cpu().numpy(), top_scores.cpu().numpy()



def build_user_embeddings(model, emb_loader, hidden_dim):
    model.eval()

    rows = []

    with torch.no_grad():
        for batch in tqdm(emb_loader, desc="embeddings"):
            user_ids = batch["user_ids"]
            input_ids = batch["input_ids"].to(device, non_blocking=True)

            _, _, h_n = model(input_ids, return_hidden=True)

            vectors = h_n[-1].detach().cpu().numpy()

            for user_id, vec in zip(user_ids, vectors):
                row = {"user_id": user_id}

                for i in range(hidden_dim):
                    row[f"emb_{i}"] = float(vec[i])

                rows.append(row)

    return pd.DataFrame(rows)




##########



seq_paths = sorted(PREP_DATA_DIR.glob(f"{PREP_DATA_FILE_NAME}_*.parquet"))

seq_df = pd.concat(
    [
        pd.read_parquet(path, columns=["user_id", "items"])
        for path in seq_paths
    ],
    ignore_index=True,
)

seq_df = seq_df.merge(
    user_clusters,
    on="user_id",
    how="inner",
)

seq_df.head()



def seq_len(x):
    return len(x) if x is not None else 0


def unique_cnt(x):
    return len(set(x)) if x is not None else 0


def repeat_share(x):
    if x is None or len(x) == 0:
        return 0.0

    return 1.0 - len(set(x)) / len(x)


def last_item(x):
    if x is None or len(x) == 0:
        return None

    return int(x[-1])


seq_df["seq_len"] = seq_df["items"].apply(seq_len)
seq_df["unique_items_cnt"] = seq_df["items"].apply(unique_cnt)
seq_df["repeat_share"] = seq_df["items"].apply(repeat_share)
seq_df["last_item"] = seq_df["items"].apply(last_item)


cluster_stats = (
    seq_df
    .groupby("cluster")
    .agg(
        users_cnt=("user_id", "nunique"),
        seq_len_mean=("seq_len", "mean"),
        seq_len_median=("seq_len", "median"),
        seq_len_p90=("seq_len", lambda x: x.quantile(0.90)),
        unique_items_mean=("unique_items_cnt", "mean"),
        repeat_share_mean=("repeat_share", "mean"),
    )
    .reset_index()
    .sort_values("users_cnt", ascending=False)
)

cluster_stats



rows = []

for row in seq_df[["user_id", "cluster", "items"]].itertuples(index=False):
    user_id = row.user_id
    cluster = row.cluster
    items = row.items

    if items is None:
        continue

    # set, чтобы считать долю пользователей с услугой,
    # а не количество повторов услуги
    for item_id in set(items):
        rows.append(
            {
                "user_id": user_id,
                "cluster": cluster,
                "item_id": int(item_id),
            }
        )

user_item_df = pd.DataFrame(rows)



total_users = seq_df["user_id"].nunique()

global_item_stats = (
    user_item_df
    .groupby("item_id")
    .agg(global_users=("user_id", "nunique"))
    .reset_index()
)

global_item_stats["global_share"] = (
    global_item_stats["global_users"] / total_users
)



cluster_sizes = (
    seq_df
    .groupby("cluster")
    .agg(cluster_users=("user_id", "nunique"))
    .reset_index()
)

cluster_item_stats = (
    user_item_df
    .groupby(["cluster", "item_id"])
    .agg(cluster_item_users=("user_id", "nunique"))
    .reset_index()
)

cluster_item_stats = cluster_item_stats.merge(
    cluster_sizes,
    on="cluster",
    how="left",
)

cluster_item_stats["cluster_share"] = (
    cluster_item_stats["cluster_item_users"]
    / cluster_item_stats["cluster_users"]
)



cluster_item_stats = cluster_item_stats.merge(
    global_item_stats,
    on="item_id",
    how="left",
)

cluster_item_stats["lift"] = (
    cluster_item_stats["cluster_share"]
    / cluster_item_stats["global_share"]
)


MIN_USERS_WITH_ITEM_IN_CLUSTER = 20

cluster_item_lift = (
    cluster_item_stats
    .query("cluster_item_users >= @MIN_USERS_WITH_ITEM_IN_CLUSTER")
    .sort_values(["cluster", "lift"], ascending=[True, False])
)

cluster_item_lift.head(20)


top_items_by_cluster = (
    cluster_item_lift
    .sort_values(["cluster", "lift"], ascending=[True, False])
    .groupby("cluster")
    .head(10)
    .reset_index(drop=True)
)

top_items_by_cluster



service_names = (
    service_mapping
    [["canonical_service_id", "normalized_title"]]
    .drop_duplicates("canonical_service_id")
    .rename(columns={"canonical_service_id": "item_id"})
)

top_items_by_cluster = top_items_by_cluster.merge(
    service_names,
    on="item_id",
    how="left",
)

top_items_by_cluster[
    [
        "cluster",
        "item_id",
        "normalized_title",
        "cluster_item_users",
        "cluster_share",
        "global_share",
        "lift",
    ]
]


idx2item_df = pd.DataFrame(
    [
        {"item_id": idx, "canonical_service_id": raw_id}
        for idx, raw_id in idx2item.items()
    ]
)


last_item_stats = (
    seq_df
    .dropna(subset=["last_item"])
    .groupby(["cluster", "last_item"])
    .agg(users=("user_id", "nunique"))
    .reset_index()
)

last_item_stats = last_item_stats.merge(
    cluster_sizes,
    on="cluster",
    how="left",
)

last_item_stats["share"] = (
    last_item_stats["users"] / last_item_stats["cluster_users"]
)

top_last_items = (
    last_item_stats
    .sort_values(["cluster", "share"], ascending=[True, False])
    .groupby("cluster")
    .head(10)
)

top_last_items



for cluster_id in sorted(seq_df["cluster"].unique()):
    print("=" * 80)
    print(f"CLUSTER {cluster_id}")

    stats = cluster_stats[cluster_stats["cluster"] == cluster_id].iloc[0]

    print(f"users_cnt: {stats['users_cnt']:,}")
    print(f"seq_len_mean: {stats['seq_len_mean']:.2f}")
    print(f"seq_len_median: {stats['seq_len_median']:.2f}")
    print(f"unique_items_mean: {stats['unique_items_mean']:.2f}")
    print(f"repeat_share_mean: {stats['repeat_share_mean']:.3f}")

    print("\nTop distinctive services by lift:")

    display_cols = [
        "item_id",
        "cluster_item_users",
        "cluster_share",
        "global_share",
        "lift",
    ]

    if "normalized_title" in top_items_by_cluster.columns:
        display_cols.insert(1, "normalized_title")

    display(
        top_items_by_cluster[
            top_items_by_cluster["cluster"] == cluster_id
        ][display_cols].head(10)
    )



emb_cols = [col for col in user_embeddings_df.columns if col.startswith("emb_")]

cluster_centers = (
    user_embeddings_df
    .groupby("cluster")[emb_cols]
    .mean()
)

rows = []

for cluster_id, group in user_embeddings_df.groupby("cluster"):
    center = cluster_centers.loc[cluster_id].values

    vectors = group[emb_cols].values

    distances = np.linalg.norm(vectors - center, axis=1)

    tmp = group[["user_id", "cluster"]].copy()
    tmp["distance_to_center"] = distances

    rows.append(tmp)

user_distances = pd.concat(rows, ignore_index=True)

typical_users = (
    user_distances
    .sort_values(["cluster", "distance_to_center"])
    .groupby("cluster")
    .head(10)
)

typical_users


typical_users_with_seq = typical_users.merge(
    seq_df[["user_id", "items"]],
    on="user_id",
    how="left",
)

typical_users_with_seq.head(20)