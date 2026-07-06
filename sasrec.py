import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRecTorch(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_len: int = 100,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.num_items = num_items
        self.max_len = max_len
        self.pad_idx = pad_idx

        self.item_emb = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=d_model,
            padding_idx=pad_idx,
        )

        self.pos_emb = nn.Embedding(
            num_embeddings=max_len,
            embedding_dim=d_model,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=n_layers,
        )

        self.out_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        """
        input_ids: [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if seq_len > self.max_len:
            input_ids = input_ids[:, -self.max_len:]
            seq_len = self.max_len

        key_padding_mask = input_ids.eq(self.pad_idx)  # [B, T]

        item_emb = self.item_emb(input_ids)

        # Важно:
        # твой collate делает left padding до max_len внутри batch,
        # а не до глобального self.max_len.
        # Поэтому якорим позиции к правому краю:
        # последняя позиция всегда получает позицию self.max_len - 1.
        pos_ids = torch.arange(seq_len, device=device)
        pos_ids = pos_ids + (self.max_len - seq_len)
        pos_ids = pos_ids.clamp(min=0, max=self.max_len - 1)

        pos_emb = self.pos_emb(pos_ids).unsqueeze(0)

        x = item_emb + pos_emb
        x = self.dropout(x)

        # зануляем padding-позиции
        x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        # causal mask: запрещаем смотреть в будущие позиции
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

        h = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )

        h = self.out_norm(h)
        h = h.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        # tied output projection: logits через item embedding matrix
        logits = h @ self.item_emb.weight.T  # [B, T, num_items]

        # PAD не должен рекомендоваться
        logits[..., self.pad_idx] = -1e9

        return logits, h

    @torch.no_grad()
    def sequence_embedding(self, input_ids):
        """
        User/profile embedding = hidden state последнего непаддингового item.
        """
        logits, h = self.forward(input_ids)

        nonpad = input_ids.ne(self.pad_idx)

        # Для left padding последний валидный item обычно справа,
        # но сделаем устойчиво.
        reversed_nonpad = torch.flip(nonpad, dims=[1])
        last_from_right = reversed_nonpad.int().argmax(dim=1)
        last_idx = input_ids.size(1) - 1 - last_from_right

        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)

        return h[batch_idx, last_idx]



    from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm


PAD_IDX = 0
MAX_LEN = 100
BATCH_SIZE = 256
NUM_ITEMS = n_items + 1  # если 0 зарезервирован под PAD, item ids: 1..n_items

train_ds = UserSequenceDataset(
    part_paths=train_part_paths,
    mode="train",
    item2idx=item2idx,
    max_len=MAX_LEN,
    val_target_n=3,
    test_target_n=3,
    random_crop=True,
    shuffle_files=True,
)

val_ds = UserSequenceDataset(
    part_paths=train_part_paths,
    mode="val_loss",
    item2idx=item2idx,
    max_len=MAX_LEN,
    val_target_n=3,
    test_target_n=3,
    random_crop=False,
    shuffle_files=False,
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    collate_fn=collate_user_sequence_batch,
    num_workers=4,
    pin_memory=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    collate_fn=collate_user_sequence_batch,
    num_workers=4,
    pin_memory=True,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SASRecTorch(
    num_items=NUM_ITEMS,
    max_len=MAX_LEN,
    d_model=128,
    n_heads=4,
    n_layers=2,
    dropout=0.2,
    pad_idx=PAD_IDX,
).to(device)

optimizer = AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)


def run_train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    loader.dataset.set_epoch(epoch)

    total_loss = 0.0
    total_steps = 0

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits, _ = model(input_ids)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_steps += 1

    return total_loss / max(total_steps, 1)


@torch.no_grad()
def run_val_loss(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_steps = 0

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits, _ = model(input_ids)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        total_loss += loss.item()
        total_steps += 1

    return total_loss / max(total_steps, 1)


for epoch in range(10):
    train_loss = run_train_epoch(model, train_loader, optimizer, device, epoch)
    val_loss = run_val_loss(model, val_loader, device)

    print(
        f"epoch={epoch} "
        f"train_loss={train_loss:.4f} "
        f"val_loss={val_loss:.4f}"
    )



same_cnt = 0
copy_pred_cnt = 0
copy_when_not_same_cnt = 0
total = 0

for pos in range(len_batch.item() - 1):
    curr_item = int(input_ids[0, pos].item())
    true_next_item = int(input_ids[0, pos + 1].item())

    item_probs = probs[0, pos].detach().cpu().numpy()
    pred_item = int(item_probs.argmax())

    total += 1
    # повторы реально есть в данных
    if curr_item == true_next_item:
        same_cnt += 1
    # модель часто копирует текущий item
    if pred_item == curr_item:
        copy_pred_cnt += 1
    #  модель копирует даже когда не надо, это проблема
    if pred_item == curr_item and curr_item != true_next_item:
        copy_when_not_same_cnt += 1

print("total transitions:", total)
print("true_next == curr share:", same_cnt / max(total, 1))
print("pred == curr share:", copy_pred_cnt / max(total, 1))
print(
    "pred == curr while true_next != curr share:",
    copy_when_not_same_cnt / max(total, 1),
)


import ast
import numpy as np
import torch


def parse_embedding(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)

    if isinstance(x, list):
        return np.asarray(x, dtype=np.float32)

    if isinstance(x, str):
        return np.asarray(ast.literal_eval(x), dtype=np.float32)

    raise TypeError(f"Unknown embedding type: {type(x)}")




def build_text_embedding_matrix(
    service_mapping,
    n_items: int,
    emb_col: str = "text_embedding",
    item_col: str = "canonical_service_id",
    item2idx: dict | None = None,
    pad_idx: int = 0,
    normalize: bool = True,
):
    sm = (
        service_mapping[[item_col, emb_col]]
        .dropna(subset=[item_col, emb_col])
        .drop_duplicates(subset=[item_col])
        .copy()
    )

    first_vec = parse_embedding(sm[emb_col].iloc[0])
    text_emb_dim = len(first_vec)

    matrix = np.zeros(
        (n_items, text_emb_dim),
        dtype=np.float32,
    )

    filled = 0
    skipped = 0

    for row in sm.itertuples(index=False):
        raw_item_id = int(getattr(row, item_col))
        vec = parse_embedding(getattr(row, emb_col))

        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

        if item2idx is None:
            item_idx = raw_item_id
        else:
            if raw_item_id not in item2idx:
                skipped += 1
                continue

            item_idx = item2idx[raw_item_id]

        if item_idx == pad_idx:
            continue

        if 0 <= item_idx < n_items:
            matrix[item_idx] = vec
            filled += 1
        else:
            skipped += 1

    print(f"text_emb_dim: {text_emb_dim}")
    print(f"filled item embeddings: {filled}")
    print(f"skipped: {skipped}")
    print(f"matrix shape: {matrix.shape}")

    return torch.tensor(matrix, dtype=torch.float32)

service_mapping = pd.read_parquet("data/service_mapping.parquet")

text_embedding_matrix = build_text_embedding_matrix(
    service_mapping=service_mapping,
    n_items=n_items,
    emb_col="text_embedding",          # поменяй на свое название колонки
    item_col="canonical_service_id",
    item2idx=None,                     # если items уже dense indices
    pad_idx=PAD_IDX,
    normalize=True,
)




import torch
from torch import nn


class GRUWithTextFeatures(nn.Module):
    def __init__(
        self,
        num_items: int,
        item_emb_dim: int,
        hidden_dim: int,
        text_embedding_matrix: torch.Tensor,
        num_layers: int = 1,
        dropout: float = 0.1,
        freeze_text_embeddings: bool = True,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.pad_idx = pad_idx

        self.item_emb = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=item_emb_dim,
            padding_idx=pad_idx,
        )

        self.text_emb = nn.Embedding.from_pretrained(
            embeddings=text_embedding_matrix,
            freeze=freeze_text_embeddings,
            padding_idx=pad_idx,
        )

        text_emb_dim = text_embedding_matrix.shape[1]

        self.text_proj = nn.Linear(
            text_emb_dim,
            item_emb_dim,
        )

        self.input_norm = nn.LayerNorm(item_emb_dim)
        self.dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=item_emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.out_norm = nn.LayerNorm(hidden_dim)

        # Я бы сначала использовал отдельную голову,
        # а не tied output через item_emb.weight,
        # чтобы уменьшить copy-bias текущего item.
        self.output = nn.Linear(hidden_dim, num_items)

    def forward(self, input_ids, return_hidden: bool = False):
        key_padding_mask = input_ids.eq(self.pad_idx)

        item_x = self.item_emb(input_ids)

        text_x = self.text_emb(input_ids)
        text_x = self.text_proj(text_x)

        x = item_x + text_x
        x = self.input_norm(x)
        x = self.dropout(x)

        # Обнуляем PAD-позиции
        x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        h, h_n = self.gru(x)

        h = self.out_norm(h)
        h = h.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        logits = self.output(h)

        if return_hidden:
            return logits, h, h_n

        return logits




class SASRecWithTextFeatures(nn.Module):
    def __init__(
        self,
        num_items: int,
        d_model: int,
        max_len: int,
        text_embedding_matrix: torch.Tensor,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        freeze_text_embeddings: bool = True,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.num_items = num_items
        self.d_model = d_model
        self.max_len = max_len
        self.pad_idx = pad_idx

        self.item_emb = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=d_model,
            padding_idx=pad_idx,
        )

        self.text_emb = nn.Embedding.from_pretrained(
            embeddings=text_embedding_matrix,
            freeze=freeze_text_embeddings,
            padding_idx=pad_idx,
        )

        text_emb_dim = text_embedding_matrix.shape[1]

        self.text_proj = nn.Linear(
            text_emb_dim,
            d_model,
        )

        self.pos_emb = nn.Embedding(
            num_embeddings=max_len,
            embedding_dim=d_model,
        )

        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.out_norm = nn.LayerNorm(d_model)

        # Отдельная голова — проще и безопаснее для MVP.
        self.output = nn.Linear(d_model, num_items)

    def forward(self, input_ids, return_hidden: bool = False):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if seq_len > self.max_len:
            raise ValueError(
                f"seq_len={seq_len} > max_len={self.max_len}. "
                f"Увеличь max_len в модели или обрезай sequence в Dataset."
            )

        key_padding_mask = input_ids.eq(self.pad_idx)

        item_x = self.item_emb(input_ids)

        text_x = self.text_emb(input_ids)
        text_x = self.text_proj(text_x)

        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_x = self.pos_emb(pos_ids)

        x = item_x + text_x + pos_x
        x = self.input_norm(x)
        x = self.dropout(x)

        # Обнуляем PAD-позиции
        x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        h = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )

        h = self.out_norm(h)
        h = h.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        logits = self.output(h)

        if return_hidden:
            return logits, h

        return logits


# Emeddings
class SASRecWithTextFeatures(nn.Module):
    def __init__(
        self,
        num_items: int,
        d_model: int,
        max_len: int,
        text_embedding_matrix: torch.Tensor,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        freeze_text_embeddings: bool = True,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.num_items = num_items
        self.d_model = d_model
        self.max_len = max_len
        self.pad_idx = pad_idx

        self.item_emb = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=d_model,
            padding_idx=pad_idx,
        )

        self.text_emb = nn.Embedding.from_pretrained(
            embeddings=text_embedding_matrix,
            freeze=freeze_text_embeddings,
            padding_idx=pad_idx,
        )

        text_emb_dim = text_embedding_matrix.shape[1]

        self.text_proj = nn.Linear(
            text_emb_dim,
            d_model,
        )

        self.pos_emb = nn.Embedding(
            num_embeddings=max_len,
            embedding_dim=d_model,
        )

        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.out_norm = nn.LayerNorm(d_model)

        # Отдельная голова — проще и безопаснее для MVP.
        self.output = nn.Linear(d_model, num_items)

    def forward(self, input_ids, return_hidden: bool = False):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if seq_len > self.max_len:
            raise ValueError(
                f"seq_len={seq_len} > max_len={self.max_len}. "
                f"Увеличь max_len в модели или обрезай sequence в Dataset."
            )

        key_padding_mask = input_ids.eq(self.pad_idx)

        item_x = self.item_emb(input_ids)

        text_x = self.text_emb(input_ids)
        text_x = self.text_proj(text_x)

        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_x = self.pos_emb(pos_ids)

        x = item_x + text_x + pos_x
        x = self.input_norm(x)
        x = self.dropout(x)

        # Обнуляем PAD-позиции
        x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        h = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )

        h = self.out_norm(h)
        h = h.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        logits = self.output(h)

        if return_hidden:
            return logits, h

        return logits