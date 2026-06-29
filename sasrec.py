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