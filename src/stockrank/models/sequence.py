from __future__ import annotations

import numpy as np
import pandas as pd

from stockrank.models.base import BaseForecaster
from stockrank.utils.logging import get_logger

logger = get_logger(__name__)


class PanelTensor:

    def __init__(self, frame: pd.DataFrame, feature_names: list[str]) -> None:
        self.feature_names = list(feature_names)
        self.dates = pd.DatetimeIndex(np.sort(frame["date"].unique()))
        self.tickers = np.array(sorted(frame["ticker"].astype(str).unique()))
        self.date_pos = {d: i for i, d in enumerate(self.dates)}
        self.tick_pos = {t: i for i, t in enumerate(self.tickers)}

        n_d, n_t, n_f = len(self.dates), len(self.tickers), len(feature_names)
        self.cube = np.zeros((n_d, n_t, n_f), dtype=np.float32)
        self.valid = np.zeros((n_d, n_t), dtype=bool)

        di = frame["date"].map(self.date_pos).to_numpy()
        ti = frame["ticker"].astype(str).map(self.tick_pos).to_numpy()
        self.cube[di, ti, :] = np.nan_to_num(
            frame[feature_names].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        self.valid[di, ti] = True
        self.nbytes = self.cube.nbytes

    def positions(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        di = frame["date"].map(self.date_pos).to_numpy()
        ti = frame["ticker"].astype(str).map(self.tick_pos).to_numpy()
        return di.astype(np.int32), ti.astype(np.int32)

    def gather(self, di: np.ndarray, ti: np.ndarray, seq_len: int) -> np.ndarray:
        offsets = np.arange(-seq_len + 1, 1)
        rows = di[:, None] + offsets[None, :]
        np.clip(rows, 0, len(self.dates) - 1, out=rows)
        return self.cube[rows, ti[:, None], :]


def _require_torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for sequence models. Install with: pip install -e '.[deep]'"
        ) from exc


def _build_module(kind: str, n_features: int, params: dict):
    torch = _require_torch()
    nn = torch.nn

    hidden = int(params.get("hidden_size", 64))
    layers = int(params.get("num_layers", 1))
    dropout = float(params.get("dropout", 0.2))

    class RecurrentHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            rnn_cls = nn.LSTM if kind == "lstm" else nn.GRU
            self.rnn = rnn_cls(
                input_size=n_features,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.norm = nn.LayerNorm(hidden)
            self.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.GELU(),
                nn.Linear(hidden // 2, 1)
            )

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(self.norm(out[:, -1, :])).squeeze(-1)

    class AttentionHead(nn.Module):

        def __init__(self, seq_len: int) -> None:
            super().__init__()
            d_model = hidden
            self.proj = nn.Linear(n_features, d_model)
            self.pos = nn.Parameter(0.02 * torch.randn(1, seq_len, d_model))
            enc = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=max(2, d_model // 32),
                dim_feedforward=2 * d_model,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

        def forward(self, x):
            h = self.encoder(self.proj(x) + self.pos[:, -x.shape[1] :, :])
            return self.head(h[:, -1, :]).squeeze(-1)

    if kind == "transformer":
        return AttentionHead(int(params.get("sequence_length", 40)))
    return RecurrentHead()


class SequenceForecaster(BaseForecaster):

    supports_importance = False

    def __init__(self, feature_names: list[str], kind: str = "gru", **params) -> None:
        super().__init__(feature_names, **params)
        self.kind = kind
        self.name = kind
        self.tensor_: PanelTensor | None = None

    def attach_tensor(self, tensor: PanelTensor) -> SequenceForecaster:
        self.tensor_ = tensor
        return self

    def _ensure_tensor(self, frame: pd.DataFrame) -> PanelTensor:
        if self.tensor_ is None:
            self.tensor_ = PanelTensor(frame, self.feature_names)
        return self.tensor_

    def fit(self, train: pd.DataFrame, y_col: str = "target") -> SequenceForecaster:
        torch = _require_torch()
        p = self.params
        seq_len = int(p.get("sequence_length", 40))
        batch_size = int(p.get("batch_size", 1024))
        epochs = int(p.get("epochs", 6))
        lr = float(p.get("lr", 1e-3))
        wd = float(p.get("weight_decay", 1e-5))
        max_seq = int(p.get("max_train_sequences", 100_000))
        patience = int(p.get("patience", 2))
        seed = int(p.get("seed", 0))

        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        tensor = self._ensure_tensor(train)

        y = train[y_col].to_numpy(dtype=np.float32)
        ok = np.isfinite(y)
        sub = train.loc[ok]
        y = y[ok]
        di, ti = tensor.positions(sub)

        order = np.argsort(di, kind="stable")
        di, ti, y = di[order], ti[order], y[order]
        cut = int(len(y) * 0.88)
        if len(y) - cut < 500:
            cut = len(y)

        tr_idx = np.arange(cut)
        if len(tr_idx) > max_seq:
            tr_idx = rng.choice(tr_idx, size=max_seq, replace=False)
            tr_idx.sort()
        va_idx = np.arange(cut, len(y))
        if va_idx.size > 40_000:
            va_idx = rng.choice(va_idx, size=40_000, replace=False)
            va_idx.sort()

        self.y_scale_ = float(np.std(y[tr_idx])) or 1.0

        model = _build_module(self.kind, len(self.feature_names), p)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        n_steps = max(1, (len(tr_idx) // batch_size) * epochs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=n_steps)
        loss_fn = torch.nn.HuberLoss(delta=1.0)

        best_val, best_state, bad = np.inf, None, 0
        step = 0
        for epoch in range(epochs):
            model.train()
            perm = rng.permutation(len(tr_idx))
            running = 0.0
            nb = 0
            for s in range(0, len(perm), batch_size):
                sel = tr_idx[perm[s : s + batch_size]]
                xb = torch.from_numpy(tensor.gather(di[sel], ti[sel], seq_len))
                yb = torch.from_numpy(y[sel] / self.y_scale_)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if step < n_steps:
                    sched.step()
                step += 1
                running += float(loss.item())
                nb += 1

            val = np.nan
            if va_idx.size > 0:
                model.eval()
                preds, targs = [], []
                with torch.no_grad():
                    for s in range(0, va_idx.size, 4096):
                        sel = va_idx[s : s + 4096]
                        xb = torch.from_numpy(tensor.gather(di[sel], ti[sel], seq_len))
                        preds.append(model(xb).numpy())
                        targs.append(y[sel] / self.y_scale_)
                pv, tv = np.concatenate(preds), np.concatenate(targs)
                val = float(np.mean((pv - tv) ** 2))

            logger.info(
                "  %s epoch %d/%d train_loss=%.5f val_mse=%.5f",
                self.kind, epoch + 1, epochs, running / max(nb, 1), val
            )

            if np.isfinite(val) and val < best_val - 1e-7:
                best_val, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    logger.info("  early stopping at epoch %d", epoch + 1)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        self.seq_len_ = seq_len
        self.fitted_ = True
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        torch = _require_torch()
        tensor = self._ensure_tensor(frame)
        di, ti = tensor.positions(frame)
        out = np.empty(len(frame), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(frame), 8192):
                sl = slice(s, min(s + 8192, len(frame)))
                xb = torch.from_numpy(tensor.gather(di[sl], ti[sl], self.seq_len_))
                out[sl] = self.model_(xb).numpy()
        return out * self.y_scale_
