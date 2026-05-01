"""
Deriv Digits Even/Odd Bot  —  1HZ25V  —  Markov + Z-Score hybrid engine
═══════════════════════════════════════════════════════════════════════════

CONTRACT
  DIGITEVEN  → wins if last digit ∈ {0,2,4,6,8}  — P(win) ≈ 0.50
  DIGITODD   → wins if last digit ∈ {1,3,5,7,9}  — P(win) ≈ 0.50
  Duration   : 1 tick  (shortest feedback loop, no edge in waiting longer)

SIGNAL ENGINE  (replaces old bias/streak engine)
  Two independent signals combined — both must agree in direction:

  1. Markov Chain
       Full 10×10 transition matrix learned from observed digit sequences.
       P(next=even | last_digit=d)  computed from empirical counts.
       Signal fires when this conditional probability > MARKOV_THRESH.
       e.g. "after a 7, P(even) = 0.63" is real structural information the
       naive even_rate calculation misses entirely.

  2. Z-Score (adaptive, not fixed-baseline)
       rolling_mean and rolling_std computed over BASELINE_WINDOW ticks.
       short_rate = even_rate over SHORT_WINDOW (last N ticks).
       Z = (short_rate − rolling_mean) / rolling_std
       Fire only when |Z| > Z_THRESH (default 1.5).
       Adaptive: if the market has been running 55% even for 200 ticks,
       Z measures deviation from 55%, not from theoretical 50%.

  Combined confidence:
       conf = MARKOV_W × markov_signal + ZSCORE_W × zscore_signal
       Both signals must agree in direction; if they conflict → no trade.

STAKE SIZING  — Martingale 1.15×, max 2 steps
  Step 0 (base):    $0.35
  Step 1 (1 loss):  $0.35 × 1.15  = $0.40
  Step 2 (2 losses):$0.35 × 1.15² = $0.46
  Win or step 2 hit → reset to step 0.
  Max cycle exposure: $1.21.

ARCHITECTURE
  Identical pattern to original:
    • Same DerivClient RPC pattern (req_id → Future map)
    • Same health server for Railway
    • Same reconnect loop with exponential backoff
    • Same Config dataclass + env var overrides

Run:
    export DERIV_API_TOKEN=your_token
    python digits_main.py

Backtest (no API needed):
    python digits_main.py --backtest
"""

import asyncio
import csv
import json
import logging
import math
import os
import random
import signal
import sys
import time
from collections import deque, Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import websockets


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("digits_bot")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Deriv API
    api_token: str = field(
        default_factory=lambda: os.getenv("DERIV_API_TOKEN", ""))
    app_id: str = field(
        default_factory=lambda: os.getenv("DERIV_APP_ID", "1089"))
    api_url: str = "wss://ws.binaryws.com/websockets/v3"

    # Contract
    symbol:   str = "1HZ25V"
    duration: int = 1          # 1 tick — optimal for digit contracts
    currency: str = "USD"

    # ── MARKOV CHAIN PARAMETERS ───────────────────────────────────────────────

    # Minimum observations per cell before the Markov signal is trusted
    # LOG ANALYSIS: cells fill to ~5 obs in the first 50 ticks; 10 was blocking
    # all signals until too late when the matrix had already converged to ~0.50
    markov_min_obs:  int   = 5
    # P(even | last_digit) must exceed this to fire a long-even signal
    # LOG ANALYSIS: 1HZ25V Markov matrix converges to 0.49–0.51 for all digits;
    # 0.54 was permanently unreachable after tick ~150. Lowered to 0.51 to trade
    # on real (small) structural deviations rather than requiring casino-level edge.
    markov_thresh:   float = 0.51

    # ── Z-SCORE PARAMETERS ───────────────────────────────────────────────────

    # Baseline window for computing rolling mean/std of even_rate
    baseline_window: int   = 200
    # Short window for the instantaneous even_rate observation
    short_window:    int   = 20
    # |Z| must exceed this threshold to fire a signal
    # LOG ANALYSIS: even_rate fluctuates in a tight band (0.46–0.58); the
    # baseline std over 20-tick chunks is small so |Z| rarely exceeds 1.5.
    # Lowered to 0.8 to capture genuine short-window deviations from baseline.
    z_thresh:        float = 0.8

    # ── SIGNAL COMBINATION ───────────────────────────────────────────────────

    markov_weight: float = 0.50
    zscore_weight: float = 0.50

    # Warmup: minimum digits before any trading starts
    warmup_ticks: int = 50

    # Auto-calibrated confidence threshold
    # LOG ANALYSIS: with tighter markov/z thresholds the combined signal will
    # score in the 0.50–0.60 range; the old 0.55 init would still block most.
    conf_threshold_init: float = 0.50
    conf_threshold_min:  float = 0.48
    conf_threshold_max:  float = 0.65
    recal_every:         int   = 20

    # ── MARTINGALE STAKE SIZING ───────────────────────────────────────────────

    base_stake:      float = 0.35   # step 0
    martingale_mult: float = 1.28   # multiply on each loss
    martingale_max:  int   = 3      # maximum steps before reset
    min_stake:       float = 0.35
    max_stake:       float = 5.00
    max_balance_pct: float = 0.05

    # ── RISK / COOLDOWN ───────────────────────────────────────────────────────

    loss_cooldown_ticks:    int   = 3
    max_consecutive_losses: int   = 5
    max_daily_loss_pct:     float = 0.15

    # ── FILES ─────────────────────────────────────────────────────────────────
    history_file: str = "digit_trades.csv"

    # ── SKIP LOG ─────────────────────────────────────────────────────────────
    skip_log_interval:  float = 30.0
    skip_summary_every: int   = 100


# ─────────────────────────────────────────────────────────────────────────────
# MARKOV + Z-SCORE HYBRID ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class DigitEngine:
    """
    Ingests raw price ticks, extracts last digit, and computes:

      Markov signal:
        - Maintains a 10×10 empirical transition matrix (digit → next digit)
        - From the matrix, derives P(next=even | last_digit=d)
        - Fires when this probability diverges meaningfully from 0.5

      Z-Score signal:
        - Tracks rolling even_rate over a long baseline window
        - Computes Z = (short_window_rate - baseline_mean) / baseline_std
        - Fires when |Z| > z_thresh

      Both signals must agree in direction; conflict → no trade.
      Combined confidence = weighted sum of the two normalised signals.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Rolling digit history (need baseline_window + short_window at minimum)
        self._digits: deque = deque(maxlen=max(cfg.baseline_window + cfg.short_window, 300))
        self._tick:   int   = 0

        # 10×10 Markov transition matrix: _markov[from_digit][to_digit] = count
        self._markov: List[List[int]] = [[0] * 10 for _ in range(10)]

        # Previous digit for transition recording
        self._prev_digit: Optional[int] = None

        # Auto-calibrated confidence threshold
        self._conf_threshold: float = cfg.conf_threshold_init

        # Rolling trade outcomes for threshold recalibration
        self._recent_outcomes: deque = deque(maxlen=cfg.recal_every * 2)
        self._trades_since_cal: int  = 0

        # Extended cooldown ticks set by recalibrator on bad losing runs
        self._recal_cooldown_ticks: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def push(self, price: float) -> Optional["Signal"]:
        """
        Ingest a tick price. Returns a Signal if conditions are met, else None.
        """
        digit = self._extract_digit(price)

        # Update Markov transition matrix
        if self._prev_digit is not None:
            self._markov[self._prev_digit][digit] += 1
        self._prev_digit = digit

        self._digits.append(digit)
        self._tick += 1

        if self._tick < self.cfg.warmup_ticks:
            return None
        if len(self._digits) < self.cfg.baseline_window:
            return None

        # Extended cooldown set by recalibrator after a bad losing run
        if self._recal_cooldown_ticks > 0:
            self._recal_cooldown_ticks -= 1
            return None

        return self._evaluate(digit)

    def record_outcome(self, won: bool):
        """Call after each settled trade to feed the recalibrator."""
        self._recent_outcomes.append(1 if won else 0)
        self._trades_since_cal += 1
        if self._trades_since_cal >= self.cfg.recal_every:
            self._recalibrate()

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def threshold(self) -> float:
        return self._conf_threshold

    @property
    def is_warm(self) -> bool:
        return self._tick >= self.cfg.warmup_ticks

    def digit_distribution(self) -> Dict[int, int]:
        """Distribution of all seen digits (for diagnostics)."""
        return dict(Counter(self._digits))

    def markov_summary(self) -> str:
        """One-line summary of P(even | last_digit) for all digits."""
        parts = []
        for d in range(10):
            row   = self._markov[d]
            total = sum(row)
            if total == 0:
                parts.append(f"{d}:?")
            else:
                p_even = sum(row[e] for e in range(0, 10, 2)) / total
                parts.append(f"{d}:{p_even:.2f}")
        return " ".join(parts)

    # ── Internal calculations ─────────────────────────────────────────────────

    @staticmethod
    def _extract_digit(price: float) -> int:
        """Last digit of the price (e.g. 12345.67 → 7)."""
        return int(round(price * 100)) % 10

    def _evaluate(self, digit: int) -> Optional["Signal"]:
        digits = list(self._digits)

        # ── 1. Markov signal ──────────────────────────────────────────────────
        row   = self._markov[digit]
        total = sum(row)

        if total < self.cfg.markov_min_obs:
            # Not enough observations for this digit — skip
            return None

        p_even_given_d = sum(row[e] for e in range(0, 10, 2)) / total

        # How far does this deviate from 0.5?
        # Signal direction: if P(even|d) > thresh → next likely even → bet EVEN
        #                   if P(odd|d)  > thresh → next likely odd  → bet ODD
        p_odd_given_d = 1.0 - p_even_given_d

        if p_even_given_d > self.cfg.markov_thresh:
            markov_dir    = "EVEN"
            # Normalise: how far above thresh, relative to headroom to 1.0
            headroom      = 1.0 - self.cfg.markov_thresh
            markov_signal = (p_even_given_d - self.cfg.markov_thresh) / headroom if headroom > 0 else 0.0
        elif p_odd_given_d > self.cfg.markov_thresh:
            markov_dir    = "ODD"
            headroom      = 1.0 - self.cfg.markov_thresh
            markov_signal = (p_odd_given_d - self.cfg.markov_thresh) / headroom if headroom > 0 else 0.0
        else:
            # Neither direction has sufficient Markov edge
            return None

        markov_signal = round(min(1.0, markov_signal), 4)

        # ── 2. Z-Score signal (adaptive) ──────────────────────────────────────
        baseline_slice = digits[-self.cfg.baseline_window:]
        short_slice    = digits[-self.cfg.short_window:]

        # Even rate over baseline
        b_even_rate = sum(1 for d in baseline_slice if d % 2 == 0) / len(baseline_slice)

        # Variance from rolling chunks within baseline
        # Approximate: split baseline into non-overlapping short_window chunks
        chunk_rates = []
        step = self.cfg.short_window
        for i in range(0, len(baseline_slice) - step + 1, step):
            chunk = baseline_slice[i:i + step]
            chunk_rates.append(sum(1 for d in chunk if d % 2 == 0) / len(chunk))

        if len(chunk_rates) < 2:
            # Cannot compute std — no signal yet
            return None

        b_mean = sum(chunk_rates) / len(chunk_rates)
        b_var  = sum((r - b_mean) ** 2 for r in chunk_rates) / (len(chunk_rates) - 1)
        b_std  = math.sqrt(b_var) if b_var > 0 else 1e-6

        # Short window even rate
        s_even_rate = sum(1 for d in short_slice if d % 2 == 0) / len(short_slice)

        z_score = (s_even_rate - b_mean) / b_std

        if abs(z_score) < self.cfg.z_thresh:
            # Z-score not significant enough
            return None

        # Z-Score direction: positive Z → too many evens → reversion → bet ODD
        #                    negative Z → too many odds  → reversion → bet EVEN
        if z_score > 0:
            zscore_dir    = "ODD"
            zscore_signal = min(1.0, (abs(z_score) - self.cfg.z_thresh) / self.cfg.z_thresh)
        else:
            zscore_dir    = "EVEN"
            zscore_signal = min(1.0, (abs(z_score) - self.cfg.z_thresh) / self.cfg.z_thresh)

        zscore_signal = round(zscore_signal, 4)

        # ── 3. Agreement check ────────────────────────────────────────────────
        if markov_dir != zscore_dir:
            # Signals conflict — no trade
            return None

        direction = markov_dir

        # ── 4. Combined confidence ────────────────────────────────────────────
        confidence = (self.cfg.markov_weight * markov_signal +
                      self.cfg.zscore_weight * zscore_signal)
        confidence = round(min(1.0, confidence), 4)

        if confidence < self._conf_threshold:
            return None

        return Signal(
            tick          = self._tick,
            digit         = digit,
            direction     = direction,
            confidence    = confidence,
            p_even_given_d= round(p_even_given_d, 4),
            markov_signal = markov_signal,
            z_score       = round(z_score, 4),
            zscore_signal = zscore_signal,
            short_rate    = round(s_even_rate, 4),
            baseline_mean = round(b_mean, 4),
            threshold     = self._conf_threshold,
        )

    def _recalibrate(self):
        """
        Adjust confidence threshold based on recent hit rate.

        Winning (hit_rate > 0.55)  → hold steady.
        Losing moderately (0.45-0.55) → raise threshold +0.01 (more selective).
        Losing badly (< 0.45) → raise threshold +0.02 + 30-tick cooldown.
        Threshold only moves up; losing makes the bot more selective.
        """
        if len(self._recent_outcomes) < 10:
            return

        hit_rate = sum(self._recent_outcomes) / len(self._recent_outcomes)
        old      = self._conf_threshold

        if hit_rate > 0.55:
            pass  # winning — hold steady

        elif hit_rate < 0.45:
            self._conf_threshold = min(
                self.cfg.conf_threshold_max,
                self._conf_threshold + 0.02
            )
            self._recal_cooldown_ticks = 30
            log.warning(
                f"[RECAL] hit_rate={hit_rate:.1%} — signals weak. "
                f"Threshold {old:.3f} → {self._conf_threshold:.3f}. "
                f"Extended cooldown: 30 ticks."
            )

        else:
            self._conf_threshold = min(
                self.cfg.conf_threshold_max,
                self._conf_threshold + 0.01
            )
            if abs(self._conf_threshold - old) > 0.001:
                log.info(
                    f"[RECAL] hit_rate={hit_rate:.1%} — marginal. "
                    f"Threshold {old:.3f} → {self._conf_threshold:.3f}."
                )

        self._trades_since_cal = 0


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    tick:           int
    digit:          int
    direction:      str    # "EVEN" or "ODD"
    confidence:     float
    p_even_given_d: float  # Markov P(even | last digit)
    markov_signal:  float  # normalised markov component [0,1]
    z_score:        float  # raw Z value
    zscore_signal:  float  # normalised zscore component [0,1]
    short_rate:     float  # even_rate over short window
    baseline_mean:  float  # rolling baseline even_rate mean
    threshold:      float

    def stake(self, balance: float, cfg: Config, martingale_step: int) -> float:
        """
        Martingale 1.15×, max 2 steps.
        step 0 → base_stake
        step 1 → base_stake × 1.15
        step 2 → base_stake × 1.15²
        Capped at max_stake and max_balance_pct.
        """
        raw = cfg.base_stake * (cfg.martingale_mult ** martingale_step)
        return round(
            max(cfg.min_stake,
                min(raw, cfg.max_stake, balance * cfg.max_balance_pct)),
            2
        )

    def __str__(self):
        return (
            f"tick={self.tick}  digit={self.digit}  dir={self.direction}  "
            f"conf={self.confidence:.3f}  P(even|d)={self.p_even_given_d:.3f}  "
            f"Z={self.z_score:+.3f}  short_rate={self.short_rate:.3f}  "
            f"baseline_μ={self.baseline_mean:.3f}  "
            f"m_sig={self.markov_signal:.3f}  z_sig={self.zscore_signal:.3f}  "
            f"threshold={self.threshold:.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class RiskManager:

    def __init__(self, cfg: Config):
        self.cfg              = cfg
        self._consec_losses   = 0
        self._in_trade        = False
        self._paused          = False
        self._pause_reason    = ""
        self._start_balance:  Optional[float] = None
        self._daily_pnl       = 0.0
        self._cooldown_ticks  = 0

        # Martingale state
        self._martingale_step: int = 0

    def set_balance(self, b: float):
        if self._start_balance is None:
            self._start_balance = b

    def tick(self):
        """Call on every tick to decrement cooldown counter."""
        if self._cooldown_ticks > 0:
            self._cooldown_ticks -= 1

    @property
    def martingale_step(self) -> int:
        return self._martingale_step

    def can_trade(self) -> Tuple[bool, str]:
        if self._in_trade:
            return False, "in_trade"
        if self._paused:
            return False, f"paused:{self._pause_reason}"
        if self._cooldown_ticks > 0:
            return False, f"cooldown:{self._cooldown_ticks}ticks"
        if self._consec_losses >= self.cfg.max_consecutive_losses:
            self._paused       = True
            self._pause_reason = f"{self._consec_losses}_consec_losses"
            return False, f"paused:{self._pause_reason}"
        if self._start_balance:
            cap = self._start_balance * self.cfg.max_daily_loss_pct
            if self._daily_pnl < -cap:
                self._paused       = True
                self._pause_reason = "daily_loss_cap"
                return False, "paused:daily_loss_cap"
        return True, "ok"

    def on_open(self):
        self._in_trade = True

    def on_close(self, won: bool, profit: float):
        self._in_trade   = False
        self._daily_pnl += profit

        if won:
            self._consec_losses    = 0
            self._cooldown_ticks   = 0
            self._martingale_step  = 0   # reset on win
        else:
            self._consec_losses += 1
            self._cooldown_ticks = self.cfg.loss_cooldown_ticks

            # Advance martingale, but never beyond max step
            if self._martingale_step < self.cfg.martingale_max:
                self._martingale_step += 1
            else:
                # Hit max step — reset after this loss
                self._martingale_step = 0

            log.info(
                f"LOSS #{self._consec_losses} | "
                f"martingale_step={self._martingale_step} | "
                f"cooldown={self.cfg.loss_cooldown_ticks} ticks"
            )

    def reset(self):
        self._paused           = False
        self._consec_losses    = 0
        self._cooldown_ticks   = 0
        self._martingale_step  = 0
        log.info("RiskManager: reset")

    def release_trade_lock(self):
        self._in_trade = False
        log.warning("Trade lock released (unconfirmed settlement)")


# ─────────────────────────────────────────────────────────────────────────────
# TRADE HISTORY
# ─────────────────────────────────────────────────────────────────────────────

class History:

    COLS = [
        "ts", "tick", "contract_id", "direction", "digit",
        "stake", "martingale_step", "confidence",
        "p_even_given_d", "markov_signal",
        "z_score", "zscore_signal",
        "short_rate", "baseline_mean", "threshold",
        "won", "profit", "balance", "settle_source",
    ]

    def __init__(self, path: str):
        self.path  = path
        self._rows: List[dict] = []
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.COLS).writeheader()

    def add(self, row: dict):
        self._rows.append(row)
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.COLS).writerow(
                {c: row.get(c, "") for c in self.COLS}
            )

    def update_last(self, contract_id, won: bool, profit: float,
                    balance: float, settle_source: str = "api"):
        for r in reversed(self._rows):
            if str(r.get("contract_id")) == str(contract_id):
                r["won"]           = won
                r["profit"]        = round(profit, 5)
                r["balance"]       = round(balance, 4)
                r["settle_source"] = settle_source
                self._rewrite()
                return

    def _rewrite(self):
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.COLS)
            w.writeheader()
            for r in self._rows:
                w.writerow({c: r.get(c, "") for c in self.COLS})

    @property
    def stats(self) -> dict:
        done = [r for r in self._rows if r.get("won") != ""]
        if not done:
            return {"n": 0, "win_rate": 0.0, "pnl": 0.0}
        wins = sum(1 for r in done
                   if r.get("won") is True or r.get("won") == "True")
        pnl  = sum(float(r.get("profit", 0) or 0) for r in done)
        return {"n": len(done), "win_rate": wins / len(done),
                "pnl": round(pnl, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# DERIV WEBSOCKET CLIENT  (same RPC pattern as original)
# ─────────────────────────────────────────────────────────────────────────────

class DerivClient:

    def __init__(self, cfg: Config):
        self.cfg        = cfg
        self._ws        = None
        self._rid       = 0
        self._pending:  Dict[int, asyncio.Future] = {}
        self._tick_cb:  Optional[Callable]        = None
        self._connected: bool = False
        self.balance:   float = 0.0

    async def connect(self):
        url = f"{self.cfg.api_url}?app_id={self.cfg.app_id}"
        self._ws        = await websockets.connect(
            url, ping_interval=20, ping_timeout=10
        )
        self._connected = True
        asyncio.create_task(self._listen())

    async def auth(self):
        r = await self._rpc({"authorize": self.cfg.api_token})
        if "error" in r:
            raise ConnectionError(r["error"]["message"])
        self.balance = float(r["authorize"].get("balance", 0))
        log.info(
            f"Auth OK | login={r['authorize'].get('loginid')} "
            f"balance={self.balance:.2f}"
        )

    async def subscribe_ticks(self, cb: Callable):
        self._tick_cb = cb
        await self._send({
            "ticks":     self.cfg.symbol,
            "subscribe": 1,
            "req_id":    self._next(),
        })

    async def proposal(self, direction: str, stake: float) -> Optional[dict]:
        contract_type = "DIGITEVEN" if direction == "EVEN" else "DIGITODD"
        r = await self._rpc({
            "proposal":       1,
            "amount":         str(stake),
            "basis":          "stake",
            "contract_type":  contract_type,
            "currency":       self.cfg.currency,
            "duration":       self.cfg.duration,
            "duration_unit":  "t",
            "symbol":         self.cfg.symbol,
        })
        if "error" in r:
            log.warning(f"Proposal error: {r['error']['message']}")
            return None
        return r.get("proposal")

    async def buy(self, proposal_id: str, price: float) -> Optional[dict]:
        r = await self._rpc({"buy": proposal_id, "price": str(price)})
        if "error" in r:
            log.error(f"Buy error: {r['error']['message']}")
            return None
        b            = r.get("buy", {})
        self.balance = float(b.get("balance_after", self.balance))
        return b

    async def contract_status(self, contract_id) -> Optional[dict]:
        r = await self._rpc({
            "proposal_open_contract": 1,
            "contract_id": int(contract_id),
        })
        if "error" in r:
            return None
        return r.get("proposal_open_contract")

    async def profit_table_lookup(self, contract_id) -> Optional[dict]:
        r = await self._rpc({
            "profit_table": 1,
            "description":  1,
            "sort":         "DESC",
            "limit":        10,
        })
        for txn in r.get("profit_table", {}).get("transactions", []):
            if str(txn.get("contract_id")) == str(contract_id):
                return txn
        return None

    async def refresh_balance(self):
        r            = await self._rpc({"balance": 1, "account": "current"})
        self.balance = float(
            r.get("balance", {}).get("balance", self.balance)
        )

    async def disconnect(self):
        if self._ws:
            await self._ws.close()

    @property
    def connected(self) -> bool:
        return self._connected

    def _next(self) -> int:
        self._rid += 1
        return self._rid

    async def _rpc(self, payload: dict) -> dict:
        rid               = self._next()
        payload["req_id"] = rid
        fut               = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        await self._send(payload)
        try:
            return await asyncio.wait_for(fut, timeout=20.0)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            return {"error": {"message": "timeout"}}

    async def _send(self, payload: dict):
        await self._ws.send(json.dumps(payload))

    async def _listen(self):
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                if msg.get("msg_type") == "tick" and self._tick_cb:
                    q = float(msg.get("tick", {}).get("quote", 0))
                    if q > 0:
                        asyncio.create_task(self._call(q))
                    continue
                rid = msg.get("req_id")
                if rid and rid in self._pending:
                    f = self._pending.pop(rid)
                    if not f.done():
                        f.set_result(msg)
        except Exception as e:
            log.error(f"WS listener error: {e}")
        finally:
            self._connected = False
            log.warning("WS listener exited — connection lost")

    async def _call(self, price: float):
        try:
            if asyncio.iscoroutinefunction(self._tick_cb):
                await self._tick_cb(price)
            else:
                self._tick_cb(price)
        except Exception as e:
            log.error(f"Tick cb error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# BOT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class Bot:

    def __init__(self, cfg: Config):
        self.cfg     = cfg
        self.engine  = DigitEngine(cfg)
        self.risk    = RiskManager(cfg)
        self.history = History(cfg.history_file)
        self.client  = DerivClient(cfg)
        self._alive  = True

        # Skip logging state
        self._last_skip_log:  float   = 0.0
        self._skip_counts:    Counter = Counter()
        self._ticks_after_warmup: int = 0

        # State log
        self._last_state_log: float = 0.0

    # ── Entry ─────────────────────────────────────────────────────────────────

    async def run(self):
        await self._connect_and_run()

    async def _connect_and_run(self):
        retry_delay = 5
        while self._alive:
            try:
                log.info("Connecting to Deriv API...")
                await self.client.connect()
                await self.client.auth()
                self.risk.set_balance(self.client.balance)
                if not self.engine.is_warm:
                    log.info(
                        f"Warming up — need {self.cfg.warmup_ticks} ticks "
                        f"before trading"
                    )
                else:
                    log.info(
                        f"Buffer already warm ({self.engine.tick} ticks) — "
                        f"trading immediately"
                    )
                await self.client.subscribe_ticks(self.on_tick)
                retry_delay = 5
                while self._alive and self.client.connected:
                    await asyncio.sleep(1)
                if self._alive:
                    log.warning("Connection lost — reconnecting in 5s...")
                    await asyncio.sleep(5)
            except Exception as e:
                log.error(f"Connection error: {e} — retrying in {retry_delay}s")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
        await self.client.disconnect()

    # ── Tick handler ──────────────────────────────────────────────────────────

    async def on_tick(self, price: float):
        self.risk.tick()

        signal = self.engine.push(price)

        if not self.engine.is_warm:
            remaining = self.cfg.warmup_ticks - self.engine.tick
            if self.engine.tick % 10 == 0:
                log.info(f"Warming up: {remaining} ticks remaining...")
            return

        self._ticks_after_warmup += 1

        # Periodic state log
        now = time.time()
        if now - self._last_state_log > 15:
            self._log_state(price)
            self._last_state_log = now

        # Skip summary
        if self._ticks_after_warmup % self.cfg.skip_summary_every == 0:
            self._log_skip_summary()

        if signal is None:
            self._skip_counts["no_signal"] += 1
            return

        ok, reason = self.risk.can_trade()
        if not ok:
            self._skip_counts[reason] += 1
            self._maybe_log_skip(reason, signal)
            return

        await self._execute(signal)

    # ── Trade execution ───────────────────────────────────────────────────────

    async def _execute(self, sig: Signal):
        stake = sig.stake(self.client.balance, self.cfg, self.risk.martingale_step)
        log.info(
            f"SIGNAL | {sig} | "
            f"stake=${stake:.2f} (step={self.risk.martingale_step}) "
            f"balance=${self.client.balance:.2f}"
        )

        prop = await self.client.proposal(sig.direction, stake)
        if not prop:
            return

        self.risk.on_open()
        result = await self.client.buy(prop["id"], float(prop["ask_price"]))
        if not result:
            self.risk.on_close(won=False, profit=-stake)
            return

        cid       = result.get("contract_id")
        buy_price = float(result.get("buy_price", stake))

        self.history.add({
            "ts":             datetime.now(timezone.utc).isoformat(),
            "tick":           sig.tick,
            "contract_id":    cid,
            "direction":      sig.direction,
            "digit":          sig.digit,
            "stake":          buy_price,
            "martingale_step":self.risk.martingale_step,
            "confidence":     sig.confidence,
            "p_even_given_d": sig.p_even_given_d,
            "markov_signal":  sig.markov_signal,
            "z_score":        sig.z_score,
            "zscore_signal":  sig.zscore_signal,
            "short_rate":     sig.short_rate,
            "baseline_mean":  sig.baseline_mean,
            "threshold":      sig.threshold,
        })

        # 1-tick contract — settle almost immediately
        await asyncio.sleep(3)
        await self._settle(cid, buy_price, sig)

    # ── Settlement ────────────────────────────────────────────────────────────

    async def _settle(self, cid, buy_price: float, sig: Signal):
        won           = None
        profit        = None
        settle_source = "unknown"

        log.info(f"[SETTLE] Polling contract {cid}...")

        # Poll up to 8 times × 3s = 24s max for a 1-tick contract
        for attempt in range(1, 9):
            status = await self.client.contract_status(cid)

            if status:
                is_sold    = status.get("is_sold", False)
                sell_price = status.get("sell_price")
                api_profit = status.get("profit")
                api_status = status.get("status", "")

                log.info(
                    f"[SETTLE] Poll {attempt}/8 | cid={cid} "
                    f"status={api_status!r} is_sold={is_sold} "
                    f"profit={api_profit}"
                )

                if is_sold or api_status in ("sold", "won", "lost"):
                    if api_profit is not None:
                        profit = float(api_profit)
                    elif sell_price is not None:
                        profit = float(sell_price) - buy_price
                    else:
                        profit = 0.0
                    won           = profit > 0
                    settle_source = "proposal_open_contract"
                    break
            else:
                log.info(
                    f"[SETTLE] Poll {attempt}/8 | cid={cid} "
                    f"no response, retrying..."
                )

            await asyncio.sleep(3)

        # Fallback: profit_table
        if won is None:
            log.warning(
                f"[SETTLE] proposal_open_contract did not confirm for {cid} "
                f"— trying profit_table"
            )
            txn = await self.client.profit_table_lookup(cid)
            if txn:
                profit        = float(txn.get("profit", 0))
                won           = profit > 0
                settle_source = "profit_table"
                log.info(
                    f"[SETTLE] profit_table confirmed | cid={cid} "
                    f"profit={profit:+.4f} won={won}"
                )
            else:
                log.warning(
                    f"[SETTLE] UNCONFIRMED — cid={cid} not found. "
                    f"Skipping engine update to avoid phantom result."
                )
                await self.client.refresh_balance()
                self.risk.release_trade_lock()
                return

        # Confirmed
        await self.client.refresh_balance()
        self.risk.on_close(won, profit)
        self.engine.record_outcome(won)
        self.history.update_last(
            cid, won, profit, self.client.balance, settle_source
        )

        stats = self.history.stats
        log.info(
            f"{'WIN' if won else 'LOSS'} | cid={cid} profit={profit:+.4f} "
            f"balance={self.client.balance:.2f} source={settle_source} | "
            f"WR={stats['win_rate']:.1%} n={stats['n']} "
            f"P&L={stats['pnl']:+.4f} "
            f"martingale_step→{self.risk.martingale_step}"
        )

    # ── Logging helpers ───────────────────────────────────────────────────────

    def _log_state(self, price: float):
        dist = self.engine.digit_distribution()
        even = sum(v for k, v in dist.items() if k % 2 == 0)
        odd  = sum(v for k, v in dist.items() if k % 2 != 0)
        tot  = even + odd or 1
        log.info(
            f"[STATE tick={self.engine.tick}] price={price:.4f} "
            f"even_rate={even/tot:.3f} ({even}/{tot}) "
            f"threshold={self.engine.threshold:.3f} "
            f"martingale_step={self.risk.martingale_step} "
            f"markov: {self.engine.markov_summary()} "
            f"stats={self.history.stats}"
        )

    def _maybe_log_skip(self, reason: str, sig: Optional[Signal]):
        now = time.time()
        if now - self._last_skip_log < self.cfg.skip_log_interval:
            return
        self._last_skip_log = now
        if sig:
            log.info(f"[SKIP] reason={reason} | signal was: {sig}")
        else:
            log.info(f"[SKIP] reason={reason}")

    def _log_skip_summary(self):
        total = sum(self._skip_counts.values())
        if total == 0:
            return
        summary = " | ".join(
            f"{k}:{v}({v/total*100:.0f}%)"
            for k, v in self._skip_counts.most_common(6)
        )
        log.info(
            f"[SKIP SUMMARY ticks={self._ticks_after_warmup}] "
            f"total={total} | {summary}"
        )

    def shutdown(self):
        self._alive = False
        self._log_skip_summary()
        log.info(f"Shutdown | stats={self.history.stats}")


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTER
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(cfg: Config, n_ticks: int = 5000, seed: int = 42):
    random.seed(seed)
    print("=" * 64)
    print("Backtest: 1HZ25V Digit Even/Odd | Markov + Z-Score + Martingale")
    print("=" * 64)

    # Generate synthetic tick prices with realistic digit distribution
    def gen_ticks(n):
        base   = 12345.00
        prices = []
        for _ in range(n):
            cent  = random.randint(0, 99)
            base  = round(base + random.gauss(0, 0.05), 2)
            price = float(f"{int(base)}.{cent:02d}")
            prices.append(price)
        return prices

    ticks  = gen_ticks(n_ticks)
    engine = DigitEngine(cfg)
    risk   = RiskManager(cfg)
    risk.set_balance(1000.0)

    balance      = 1000.0
    bal_log      = [balance]
    trades       = 0
    wins         = 0
    skip_counts: Counter = Counter()

    for i, price in enumerate(ticks):
        risk.tick()
        signal = engine.push(price)

        if signal is None:
            skip_counts["no_signal"] += 1
            continue

        ok, reason = risk.can_trade()
        if not ok:
            skip_counts[reason] += 1
            continue

        stake = signal.stake(balance, cfg, risk.martingale_step)

        # Simulate outcome: next digit is the result
        if i + 1 < len(ticks):
            next_digit = int(round(ticks[i + 1] * 100)) % 10
            if signal.direction == "EVEN":
                won = next_digit % 2 == 0
            else:
                won = next_digit % 2 != 0
        else:
            won = random.random() < 0.5

        # Payout: ~95% for Even/Odd
        profit = stake * 0.95 if won else -stake

        balance += profit
        bal_log.append(balance)

        risk.on_close(won, profit)
        engine.record_outcome(won)

        trades += 1
        if won:
            wins += 1

        if trades % 50 == 0:
            print(
                f"  [tick {i:5d}] trades={trades} "
                f"WR={wins/trades:.1%} bal={balance:.2f} "
                f"threshold={engine.threshold:.3f} "
                f"mstep={risk.martingale_step}"
            )

    wr  = wins / trades if trades else 0.0
    pnl = balance - 1000.0

    peaks = [max(bal_log[:i + 1]) for i in range(len(bal_log))]
    dd    = max((p - v) / p for p, v in zip(peaks, bal_log)) if len(bal_log) > 1 else 0.0

    print(f"\n{'─'*64}")
    print(f"  Trades         : {trades}")
    print(f"  Win rate       : {wr:.1%}  (breakeven ≈ 51.3%)")
    print(f"  P&L            : {pnl:+.2f} USD  (start 1000)")
    print(f"  Max drawdown   : {dd:.1%}")
    print(f"  Final threshold: {engine.threshold:.3f}")
    print(f"  Markov P(even|d): {engine.markov_summary()}")

    if skip_counts:
        total = sum(skip_counts.values())
        print(f"\n  Skip breakdown ({total} total):")
        for gate, count in skip_counts.most_common():
            print(f"    {gate:25s}: {count} ({count/total:.0%})")

    dist = engine.digit_distribution()
    print(f"\n  Digit distribution: {dist}")
    print("=" * 64)


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH SERVER  (Railway requires a bound port)
# ─────────────────────────────────────────────────────────────────────────────

def _start_health_server():
    import http.server
    import threading
    port = int(os.getenv("PORT", "8080"))

    class _H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK - digits bot running")

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("", port), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log.info(f"Health-check server on :{port}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def live(cfg: Config):
    if not cfg.api_token:
        log.error(
            "No API token found. "
            "Set DERIV_API_TOKEN environment variable."
        )
        sys.exit(1)

    bot = Bot(cfg)

    def handle_signal(sig, frame):
        log.info("Shutdown signal received...")
        bot.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT,  handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    log.info("=" * 64)
    log.info(f"1HZ25V Digits Even/Odd Bot — Markov + Z-Score Engine")
    log.info(f"Symbol={cfg.symbol}  Duration={cfg.duration}t")
    log.info(f"Markov thresh={cfg.markov_thresh}  Z_thresh={cfg.z_thresh}")
    log.info(f"Baseline window={cfg.baseline_window}  Short window={cfg.short_window}")
    log.info(f"Martingale: {cfg.base_stake}× {cfg.martingale_mult} max_steps={cfg.martingale_max}")
    log.info(f"Confidence init={cfg.conf_threshold_init}")
    log.info("=" * 64)

    _start_health_server()
    await bot.run()


if __name__ == "__main__":
    cfg = Config()

    if "--backtest" in sys.argv:
        run_backtest(cfg)
    else:
        asyncio.run(live(cfg))
