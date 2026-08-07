"""Leakage-safe cross-validation for overlapping financial labels.

The problem
-----------
A label with a 5-day horizon formed on Monday is still being realised on Friday.
If Monday lands in the training set and Wednesday lands in the test set, the two
observations share four days of the same price path. Ordinary k-fold, and even
plain chronological splitting, therefore leak information across the boundary and
inflate every metric that follows.

The fix, following Lopez de Prado, has two parts:

* **Purging** removes training observations whose label window overlaps the test
  window at all.
* **Embargo** additionally drops training observations for a short period *after*
  the test window, because serial correlation in features means an observation
  immediately after the test set is still partly informative about it.

``PurgedWalkForward`` is the default because it also respects the direction of
time: every model is only ever fitted on the past and scored on the future, which
is the only arrangement that answers the question an investor actually asks.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockrank.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Fold:
    """One train/test split, described both by row indices and by dates."""

    index: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_purged: int = 0

    def describe(self) -> dict:
        return {
            "fold": self.index,
            "train": f"{self.train_start.date()} to {self.train_end.date()}",
            "test": f"{self.test_start.date()} to {self.test_end.date()}",
            "n_train": int(self.train_idx.size),
            "n_test": int(self.test_idx.size),
            "n_purged": int(self.n_purged),
        }


class PurgedWalkForward:
    """Rolling or expanding walk-forward splits with purge and embargo.

    Parameters
    ----------
    n_splits:
        Number of test windows, laid out to end at the end of the sample.
    train_window_days, test_window_days:
        Calendar-agnostic: measured in *trading days* taken from the data itself.
    label_horizon:
        Length of the label window. Training rows whose label extends into the
        test window are purged.
    embargo_days:
        Extra buffer applied after the test window.
    expanding:
        If True, each fold trains on all history up to the purge boundary.
    """

    def __init__(
        self,
        n_splits: int = 6,
        train_window_days: int = 1260,
        test_window_days: int = 252,
        label_horizon: int = 5,
        embargo_days: int = 10,
        expanding: bool = False,
    ) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        self.n_splits = n_splits
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.label_horizon = label_horizon
        self.embargo_days = embargo_days
        self.expanding = expanding

    def _unique_dates(self, dates: pd.Series) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(np.sort(pd.unique(pd.to_datetime(dates))))

    def split(self, dates: pd.Series) -> Iterator[Fold]:
        dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
        uniq = self._unique_dates(dates)
        n = len(uniq)

        needed = self.train_window_days + self.n_splits * self.test_window_days
        if n < needed:
            # Shrink the training window rather than failing: short samples are
            # common in smoke tests and quick experiments.
            available = n - self.n_splits * self.test_window_days
            if available < 60:
                raise ValueError(
                    f"Not enough history: {n} trading days for {self.n_splits} folds of "
                    f"{self.test_window_days} days"
                )
            logger.warning(
                "Only %d trading days available; shrinking train window from %d to %d",
                n,
                self.train_window_days,
                available,
            )
            self.train_window_days = available

        date_to_pos = pd.Series(np.arange(n), index=uniq)
        pos = date_to_pos.reindex(dates.to_numpy()).to_numpy()

        for k in range(self.n_splits):
            test_end_pos = n - (self.n_splits - 1 - k) * self.test_window_days - 1
            test_start_pos = test_end_pos - self.test_window_days + 1
            if test_start_pos <= 0:
                continue

            # Purge: a training label formed at position p spans p+1 .. p+1+h,
            # so anything within (horizon + 1) of the test start must go.
            purge = self.label_horizon + 1
            train_end_pos = test_start_pos - purge - 1
            train_start_pos = (
                0 if self.expanding else max(0, train_end_pos - self.train_window_days + 1)
            )
            if train_end_pos - train_start_pos < 60:
                continue

            in_train = (pos >= train_start_pos) & (pos <= train_end_pos)
            in_test = (pos >= test_start_pos) & (pos <= test_end_pos)

            # Embargo after the test window (only relevant for expanding folds
            # that would otherwise train on the immediate aftermath).
            if self.embargo_days > 0:
                emb_lo, emb_hi = test_end_pos + 1, test_end_pos + self.embargo_days
                in_train &= ~((pos >= emb_lo) & (pos <= emb_hi))

            n_purged = int(
                ((pos > train_end_pos) & (pos < test_start_pos)).sum()
                + ((pos >= test_end_pos + 1) & (pos <= test_end_pos + self.embargo_days)).sum()
            )

            yield Fold(
                index=k,
                train_idx=np.flatnonzero(in_train),
                test_idx=np.flatnonzero(in_test),
                train_start=uniq[train_start_pos],
                train_end=uniq[train_end_pos],
                test_start=uniq[test_start_pos],
                test_end=uniq[test_end_pos],
                n_purged=n_purged,
            )

    def get_folds(self, dates: pd.Series) -> list[Fold]:
        folds = list(self.split(dates))
        if not folds:
            raise ValueError("No valid folds could be constructed from the given dates")
        logger.info(
            "Built %d purged walk-forward folds (train=%dd test=%dd purge=%dd embargo=%dd)",
            len(folds),
            self.train_window_days,
            self.test_window_days,
            self.label_horizon + 1,
            self.embargo_days,
        )
        return folds


class PurgedKFold:
    """Purged k-fold for hyperparameter search inside a single training window.

    Unlike walk-forward this does train on data that comes after some of the test
    data, so it is used only for tuning within a fold, never for the headline
    out-of-sample numbers.
    """

    def __init__(self, n_splits: int = 5, label_horizon: int = 5, embargo_days: int = 10) -> None:
        self.n_splits = n_splits
        self.label_horizon = label_horizon
        self.embargo_days = embargo_days

    def split(self, dates: pd.Series) -> Iterator[Fold]:
        dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
        uniq = pd.DatetimeIndex(np.sort(pd.unique(dates)))
        n = len(uniq)
        pos = pd.Series(np.arange(n), index=uniq).reindex(dates.to_numpy()).to_numpy()
        bounds = np.linspace(0, n, self.n_splits + 1).astype(int)
        purge = self.label_horizon + 1

        for k in range(self.n_splits):
            lo, hi = bounds[k], bounds[k + 1] - 1
            in_test = (pos >= lo) & (pos <= hi)
            in_train = ~in_test
            in_train &= ~((pos > hi) & (pos <= hi + purge + self.embargo_days))
            in_train &= ~((pos >= lo - purge) & (pos < lo))
            yield Fold(
                index=k,
                train_idx=np.flatnonzero(in_train),
                test_idx=np.flatnonzero(in_test),
                train_start=uniq[0],
                train_end=uniq[max(lo - purge - 1, 0)],
                test_start=uniq[lo],
                test_end=uniq[hi],
                n_purged=int(n - in_train.sum() - in_test.sum()),
            )
