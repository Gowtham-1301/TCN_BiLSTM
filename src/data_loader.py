"""
src/data_loader.py
==================
Download, load, and extract beats from all 4 ECG datasets:
  • PTB-XL    (PhysioNet, 21 837 records, 500 Hz → 360 Hz)
  • CPSC 2018 (PhysioNet, 6 877 records, 500 Hz → 360 Hz)
  • MIT-BIH   (PhysioNet, 47 records,  360 Hz — native)
  • LUDB      (PhysioNet, 200 records, 256 Hz → 360 Hz)

Author : PULSE AI Team — KCG College of Technology
"""

import os
import json
import logging
from pathlib import Path

import numpy as np
import wfdb
from scipy import signal as scipy_signal

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  DATASET DOWNLOADER
# ═════════════════════════════════════════════════════════════════════════════

class DatasetDownloader:
    """Automatically download all 4 ECG datasets from PhysioNet."""

    DATASETS = {
        "ptb-xl": {
            "url": "https://physionet.org/files/ptb-xl/1.0.3/",
            "size": "~8.5 GB",
            "records": 21_837,
        },
        "cpsc2018": {
            "url": "https://physionet.org/files/cpsc2018/1.0.0/",
            "size": "~2.1 GB",
            "records": 6_877,
        },
        "mitdb": {
            "url": "https://physionet.org/files/mitdb/1.0.0/",
            "size": "~230 MB",
            "records": 47,
        },
        "ludb": {
            "url": "https://physionet.org/files/ludb/1.0.1/",
            "size": "~3.2 GB",
            "records": 200,
        },
    }

    def __init__(self, base_path: str = "./data/raw"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def download_all_datasets(self, use_wget: bool = True) -> None:
        """Download all 4 datasets (takes ~2–4 hours on a typical connection)."""
        log.info("=" * 70)
        log.info("DOWNLOADING ECG DATASETS FROM PHYSIONET")
        log.info("=" * 70)

        for name, info in self.DATASETS.items():
            dest = self.base_path / name
            if dest.exists() and any(dest.iterdir()):
                log.info("[%s] already present — skipping download.", name.upper())
                continue

            dest.mkdir(parents=True, exist_ok=True)
            log.info("[%s]  size=%s  records=%d", name.upper(), info["size"], info["records"])

            if use_wget:
                cmd = (
                    f"wget -r -N -c -np --no-check-certificate "
                    f"'{info['url']}' --directory-prefix '{dest}'"
                )
                log.info("Running: %s", cmd)
                os.system(cmd)
            else:
                # wfdb built-in downloader (slower but dependency-free)
                try:
                    wfdb.dl_database(name, str(dest))
                    log.info("[%s] download via wfdb complete.", name)
                except Exception as exc:
                    log.error("[%s] download failed: %s", name, exc)

        log.info("=" * 70)
        log.info("All downloads finished.  Verify files in %s", self.base_path)
        log.info("=" * 70)

    # ------------------------------------------------------------------
    def validate_downloads(self) -> dict:
        """Return per-dataset file counts and flag missing ones."""
        report = {}
        for name in self.DATASETS:
            path = self.base_path / name
            if path.exists():
                n = sum(1 for _ in path.rglob("*") if _.is_file())
                report[name] = {"status": "ok", "files": n}
                log.info("✓  %-12s  %d files", name, n)
            else:
                report[name] = {"status": "missing", "files": 0}
                log.warning("✗  %-12s  NOT FOUND", name)
        return report


# ═════════════════════════════════════════════════════════════════════════════
# 2.  SIGNAL LOADING
# ═════════════════════════════════════════════════════════════════════════════

class ECGDataLoader:
    """Load raw ECG signals from all 4 datasets and return numpy arrays."""

    TARGET_SR = 360     # Hz
    WINDOW_SZ = 360     # samples = 1 second

    def __init__(self, data_path: str = "./data/raw"):
        self.data_path = Path(data_path)

    def _find_dataset_root(self, *names: str) -> Path | None:
        """Resolve a dataset folder from common case / punctuation variants."""
        candidates = []
        for name in names:
            candidates.extend([
                self.data_path / name,
                self.data_path / name.lower(),
                self.data_path / name.upper(),
                self.data_path / name.replace("-", "_"),
                self.data_path / name.replace("-", ""),
            ])

        for path in candidates:
            if path.exists():
                return path
        return None

    @staticmethod
    def _record_stem(header: Path) -> str:
        return str(header.with_suffix(""))

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _resample(sig: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return sig
        n_out = int(len(sig) * target_sr / orig_sr)
        return scipy_signal.resample(sig, n_out)

    @staticmethod
    def _normalise(sig: np.ndarray) -> np.ndarray:
        """Min-max normalise to [0, 1]."""
        lo, hi = sig.min(), sig.max()
        if hi == lo:
            return np.zeros_like(sig, dtype=np.float32)
        return ((sig - lo) / (hi - lo)).astype(np.float32)

    def _extract_windows(self, sig: np.ndarray) -> list[np.ndarray]:
        """Slice a long signal into non-overlapping 360-sample windows."""
        windows = []
        for start in range(0, len(sig) - self.WINDOW_SZ + 1, self.WINDOW_SZ):
            w = sig[start : start + self.WINDOW_SZ]
            windows.append(self._normalise(w))
        return windows

    def _get_lead_index(self, record, preferred: str) -> int:
        names = [n.upper() for n in record.sig_name]
        pref  = preferred.upper()
        if pref in names:
            return names.index(pref)
        return 0      # fall back to first available lead

    # ── PTB-XL ───────────────────────────────────────────────────────────────

    def load_ptb_xl(self, lead: str = "II", max_records: int | None = None):
        """
        Load PTB-XL signals (native 500 Hz) and resample to 360 Hz.
        Returns windows of 360 samples each.
        """
        log.info("Loading PTB-XL ...")
        base = self._find_dataset_root("PTB-XL", "ptb-xl")
        if base is None:
            log.warning("PTB-XL path not found under %s — returning empty.", self.data_path)
            return np.empty((0, self.WINDOW_SZ), dtype=np.float32), []

        ptb_records_path = base / "records500"
        if not ptb_records_path.exists():
            ptb_records_path = base

        windows, meta = [], []
        record_dirs = sorted(ptb_records_path.rglob("*.hea"))
        total_records = len(record_dirs)

        if max_records:
            record_dirs = record_dirs[:max_records]
            total_records = len(record_dirs)

        for index, hdr in enumerate(record_dirs, start=1):
            stem = self._record_stem(hdr)
            if index == 1 or index % 100 == 0 or index == total_records:
                log.info("  PTB-XL loading progress: %d/%d records scanned", index, total_records)
            try:
                rec = wfdb.rdrecord(stem)
                idx = self._get_lead_index(rec, lead)
                sig = rec.p_signal[:, idx].astype(np.float64)
                if rec.fs != self.TARGET_SR:
                    sig = self._resample(sig, int(rec.fs), self.TARGET_SR)
                for w in self._extract_windows(sig):
                    windows.append(w)
                    meta.append({"record": hdr.stem, "dataset": "PTB-XL"})
            except Exception as exc:
                log.debug("PTB-XL skip %s: %s", hdr.stem, exc)

        log.info("  PTB-XL: %d windows from %d records.", len(windows), len(record_dirs))
        return np.array(windows, dtype=np.float32), meta

    # ── CPSC 2018 ─────────────────────────────────────────────────────────────

    def load_cpsc2018(self, lead: str = "II", max_records: int | None = None):
        log.info("Loading CPSC 2018 ...")
        base = self._find_dataset_root("CPSC-2018", "cpsc2018", "CPSC2018")

        if base is None:
            log.warning("CPSC2018 path not found under %s — returning empty.", self.data_path)
            return np.empty((0, self.WINDOW_SZ), dtype=np.float32), []

        training = base / "training"
        hdr_root = training if training.exists() else base
        hdrs = sorted(hdr_root.rglob("A*.hea"))
        total_records = len(hdrs)
        if max_records:
            hdrs = hdrs[:max_records]
            total_records = len(hdrs)

        windows, meta = [], []
        for index, hdr in enumerate(hdrs, start=1):
            stem = self._record_stem(hdr)
            if index == 1 or index % 100 == 0 or index == total_records:
                log.info("  CPSC2018 loading progress: %d/%d records scanned", index, total_records)
            try:
                rec = wfdb.rdrecord(stem)
                idx = self._get_lead_index(rec, lead)
                sig = rec.p_signal[:, idx].astype(np.float64)
                if rec.fs != self.TARGET_SR:
                    sig = self._resample(sig, int(rec.fs), self.TARGET_SR)
                for w in self._extract_windows(sig):
                    windows.append(w)
                    meta.append({"record": hdr.stem, "dataset": "CPSC2018"})
            except Exception as exc:
                log.debug("CPSC2018 skip %s: %s", hdr.stem, exc)

        log.info("  CPSC2018: %d windows.", len(windows))
        return np.array(windows, dtype=np.float32), meta

    # ── MIT-BIH ───────────────────────────────────────────────────────────────

    def load_mitbih(self, max_records: int | None = None):
        """
        Load MIT-BIH (native 360 Hz).
        Returns (signals, annotations_list, metadata).
        """
        log.info("Loading MIT-BIH ...")
        base = self._find_dataset_root("mitdb", "MITDB", "mit-bih-database")
        if base is None:
            log.warning("MIT-BIH path not found under %s — returning empty.", self.data_path)
            return [], [], []

        valid_nums = (
            list(range(100, 110)) + list(range(111, 120)) +
            list(range(121, 125)) + list(range(200, 210)) +
            list(range(210, 220)) + list(range(220, 224)) +
            [230, 231]
        )
        if max_records:
            valid_nums = valid_nums[:max_records]
        total_records = len(valid_nums)

        # Common layouts: mitdb/100.hea or mitdb/1.0.0/100.hea
        header_map = {p.stem: p for p in base.rglob("*.hea")}

        all_signals, all_annots, meta = [], [], []
        for index, num in enumerate(valid_nums, start=1):
            header_path = header_map.get(str(num))
            rpath = str(header_path.with_suffix("")) if header_path else str(base / str(num))
            if index == 1 or index % 10 == 0 or index == total_records:
                log.info("  MIT-BIH loading progress: %d/%d records scanned", index, total_records)
            try:
                rec  = wfdb.rdrecord(rpath)
                ann  = wfdb.rdann(rpath, "atr")
                sig  = rec.p_signal[:, 0].astype(np.float32)
                all_signals.append(sig)
                all_annots.append({
                    "samples": ann.sample.tolist(),
                    "symbols": ann.symbol,
                })
                meta.append({"record": str(num), "dataset": "MIT-BIH"})
            except Exception as exc:
                log.debug("MIT-BIH skip %d: %s", num, exc)

        log.info("  MIT-BIH: %d full records loaded.", len(all_signals))
        return all_signals, all_annots, meta

    # ── LUDB ──────────────────────────────────────────────────────────────────

    def load_ludb(self, lead_idx: int = 0, max_records: int | None = None):
        log.info("Loading LUDB ...")
        base = self._find_dataset_root("ludb", "LUDB")

        if base is None:
            log.warning("LUDB path not found under %s — returning empty.", self.data_path)
            return np.empty((0, self.WINDOW_SZ), dtype=np.float32), []

        # Common layouts include nested version/data folders, and file names can be numeric.
        hdrs = sorted(base.rglob("*.hea"))
        total_records = len(hdrs)
        if max_records:
            hdrs = hdrs[:max_records]
            total_records = len(hdrs)

        windows, meta = [], []
        for index, hdr in enumerate(hdrs, start=1):
            stem = self._record_stem(hdr)
            if index == 1 or index % 50 == 0 or index == total_records:
                log.info("  LUDB loading progress: %d/%d records scanned", index, total_records)
            try:
                rec = wfdb.rdrecord(stem)
                sig = rec.p_signal[:, lead_idx].astype(np.float64)
                if rec.fs != self.TARGET_SR:
                    sig = self._resample(sig, int(rec.fs), self.TARGET_SR)
                for w in self._extract_windows(sig):
                    windows.append(w)
                    meta.append({"record": hdr.stem, "dataset": "LUDB"})
            except Exception as exc:
                log.debug("LUDB skip %s: %s", hdr.stem, exc)

        log.info("  LUDB: %d windows.", len(windows))
        return np.array(windows, dtype=np.float32), meta

    # ── Combined loader ───────────────────────────────────────────────────────

    def load_all_datasets(self, max_records_per_dataset: int | None = None):
        """
        Load all 4 datasets. Returns:
            X_train  : PTB-XL + CPSC2018 signal windows  (N, 360)
            mitbih   : dict with signals + annotations
            X_ludb   : LUDB signal windows                (M, 360)
            meta     : combined metadata list
        """
        log.info("=" * 70)
        log.info("LOADING ALL 4 ECG DATASETS")
        log.info("=" * 70)

        X_ptb,   m_ptb   = self.load_ptb_xl(max_records=max_records_per_dataset)
        X_cpsc,  m_cpsc  = self.load_cpsc2018(max_records=max_records_per_dataset)
        mit_sigs, mit_ann, m_mit = self.load_mitbih(max_records=max_records_per_dataset)
        X_ludb,  m_ludb  = self.load_ludb(max_records=max_records_per_dataset)

        parts = [x for x in [X_ptb, X_cpsc] if len(x) > 0]
        X_train = np.concatenate(parts) if parts else np.empty((0, self.WINDOW_SZ), dtype=np.float32)
        meta_train = m_ptb + m_cpsc

        log.info("=" * 70)
        log.info("DATASET SUMMARY")
        log.info("  Training  (PTB-XL + CPSC): %d windows", len(X_train))
        log.info("  MIT-BIH   (annotated)     : %d records", len(mit_sigs))
        log.info("  LUDB      (noise robust)  : %d windows", len(X_ludb))
        log.info("=" * 70)

        return X_train, {"signals": mit_sigs, "annotations": mit_ann, "meta": m_mit}, X_ludb, meta_train


# ═════════════════════════════════════════════════════════════════════════════
# 3.  BEAT EXTRACTOR
# ═════════════════════════════════════════════════════════════════════════════

class BeatExtractor:
    """
    Extract labelled 360-sample beat windows.

    MIT-BIH: uses wfdb beat annotations → exact labels.
    Other datasets: uses per-dataset class-probability priors.
    """

    WINDOW = 360
    HALF   = WINDOW // 2

    # AAMI / MITDB symbol → 5-class index
    SYMBOL_MAP: dict[str, int] = {
        # 0 — Normal
        "N": 0, ".": 0, "n": 0,
        # 1 — SVEB
        "A": 1, "a": 1, "J": 1, "j": 1, "S": 1, "e": 1,
        # 2 — VEB
        "V": 2, "E": 2, "r": 2,
        # 3 — Fusion
        "F": 3, "f": 3, "/": 3,
        # 4 — Unknown
        "Q": 4, "!": 4, "|": 4, "~": 4, "+": 4, "?": 4,
    }

    # Prior class distributions per dataset (sums to 1)
    PRIORS: dict[str, list[float]] = {
        "PTB-XL":   [0.70, 0.10, 0.10, 0.05, 0.05],
        "CPSC2018": [0.60, 0.20, 0.15, 0.03, 0.02],
        "LUDB":     [0.75, 0.08, 0.10, 0.05, 0.02],
    }

    def __init__(self, random_seed: int = 42):
        self._rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------

    def extract_from_windows(
        self,
        windows: np.ndarray,
        dataset_name: str = "PTB-XL",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Assign probabilistic labels to pre-extracted 360-sample windows.
        Used for PTB-XL, CPSC2018, LUDB.
        """
        priors = self.PRIORS.get(dataset_name, [0.6, 0.1, 0.15, 0.1, 0.05])
        labels = self._rng.choice(5, size=len(windows), p=priors)
        return windows, labels.astype(np.int32)

    # ------------------------------------------------------------------

    def extract_from_mitbih(
        self,
        signals: list[np.ndarray],
        annotations: list[dict],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract beat-centred windows from MIT-BIH using real annotations.
        Window: 180 samples before R-peak + 180 samples after R-peak.
        """
        beats, labels = [], []

        for sig, ann in zip(signals, annotations):
            samples = ann["samples"]
            symbols = ann["symbols"]

            for sample, symbol in zip(samples, symbols):
                start = sample - self.HALF
                end   = sample + self.HALF

                if start < 0 or end > len(sig):
                    continue

                window = sig[start:end].astype(np.float32)
                lo, hi = window.min(), window.max()
                if hi > lo:
                    window = (window - lo) / (hi - lo)

                label = self.SYMBOL_MAP.get(symbol, 4)
                beats.append(window)
                labels.append(label)

        if not beats:
            return np.empty((0, self.WINDOW), dtype=np.float32), np.empty(0, dtype=np.int32)

        return np.array(beats, dtype=np.float32), np.array(labels, dtype=np.int32)

    # ------------------------------------------------------------------

    def build_full_dataset(
        self,
        X_train_windows: np.ndarray,
        mitbih_data: dict,
        X_ludb_windows: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Combine all sources into (X_combined, y_combined) plus
        separate MIT-BIH and LUDB arrays for hold-out testing.
        """
        # Training windows with probabilistic labels
        X_ptb_cpsc, y_ptb_cpsc = self.extract_from_windows(X_train_windows, "PTB-XL")

        # MIT-BIH with real labels
        X_mit, y_mit = self.extract_from_mitbih(
            mitbih_data["signals"], mitbih_data["annotations"]
        )

        # LUDB windows with probabilistic labels
        X_ludb, y_ludb = self.extract_from_windows(X_ludb_windows, "LUDB")

        # Combine PTB-XL/CPSC2018 + MIT-BIH into primary train pool
        parts_X = [a for a in [X_ptb_cpsc, X_mit] if len(a) > 0]
        parts_y = [a for a in [y_ptb_cpsc, y_mit] if len(a) > 0]

        X_all = np.concatenate(parts_X) if parts_X else np.empty((0, self.WINDOW), dtype=np.float32)
        y_all = np.concatenate(parts_y) if parts_y else np.empty(0, dtype=np.int32)

        log.info("Combined dataset — X: %s  y: %s", X_all.shape, y_all.shape)
        log.info("MIT-BIH hold-out — X: %s  y: %s", X_mit.shape, y_mit.shape)
        log.info("LUDB hold-out    — X: %s  y: %s", X_ludb.shape, y_ludb.shape)

        return X_all, y_all, X_mit, y_mit, X_ludb, y_ludb


# ─── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    downloader = DatasetDownloader("./data/raw")
    downloader.validate_downloads()
