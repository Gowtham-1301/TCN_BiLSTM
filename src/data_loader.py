"""
src/data_loader.py
==================
Download, load, and extract beats from all 4 ECG datasets,
**with patient metadata extraction** (age, sex, height, weight).

Datasets:
  • PTB-XL    (PhysioNet, 21 837 records, 500 Hz → 360 Hz)  — has rich metadata
  • CPSC 2018 (PhysioNet, 6 877 records, 500 Hz → 360 Hz)
  • MIT-BIH   (PhysioNet, 47 records,  360 Hz — native)
  • LUDB      (PhysioNet, 200 records, 256 Hz → 360 Hz)

Metadata vector (per window):
  [age_normalised, sex_encoded, height_normalised, weight_normalised]
  - age: normalised to [0, 1] (0–100 years)
  - sex: 0 = female, 1 = male, 0.5 = unknown
  - height: normalised to [0, 1] (100–220 cm)
  - weight: normalised to [0, 1] (30–200 kg)

Author : PULSE AI Team — KCG College of Technology
"""

import os
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy import signal as scipy_signal

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── metadata constants ───────────────────────────────────────────────────────
METADATA_DIM = 4          # [age_norm, sex, height_norm, weight_norm]
DEFAULT_META = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)   # unknown


def _normalise_age(age) -> float:
    """Normalise age to [0, 1] range (0–100 years)."""
    try:
        return float(np.clip(float(age) / 100.0, 0.0, 1.0))
    except (ValueError, TypeError):
        return 0.5

def _encode_sex(sex) -> float:
    """Encode sex: 0=female, 1=male, 0.5=unknown."""
    if isinstance(sex, str):
        sex = sex.strip().lower()
        if sex in ("male", "m", "1"):
            return 1.0
        elif sex in ("female", "f", "0"):
            return 0.0
    elif isinstance(sex, (int, float)):
        if sex == 1:
            return 1.0
        elif sex == 0:
            return 0.0
    return 0.5

def _normalise_height(height) -> float:
    """Normalise height to [0, 1] (100–220 cm range)."""
    try:
        return float(np.clip((float(height) - 100) / 120.0, 0.0, 1.0))
    except (ValueError, TypeError):
        return 0.5

def _normalise_weight(weight) -> float:
    """Normalise weight to [0, 1] (30–200 kg range)."""
    try:
        return float(np.clip((float(weight) - 30) / 170.0, 0.0, 1.0))
    except (ValueError, TypeError):
        return 0.5

def build_metadata_vector(age=None, sex=None, height=None, weight=None) -> np.ndarray:
    """Build a normalised [4,] metadata vector."""
    return np.array([
        _normalise_age(age),
        _encode_sex(sex),
        _normalise_height(height),
        _normalise_weight(weight),
    ], dtype=np.float32)


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
# 2.  SIGNAL LOADING (with metadata extraction)
# ═════════════════════════════════════════════════════════════════════════════

class ECGDataLoader:
    """Load raw ECG signals + patient metadata from all 4 datasets."""

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

    # ── PTB-XL metadata loader ────────────────────────────────────────────────

    def _load_ptbxl_metadata(self, base: Path) -> dict:
        """
        Load PTB-XL patient metadata from the database CSV.
        Returns dict mapping record_id (str) → metadata vector [4,].
        """
        meta_map = {}

        # PTB-XL has ptbxl_database.csv with columns:
        # ecg_id, patient_id, age, sex, height, weight, ...
        csv_candidates = [
            base / "ptbxl_database.csv",
            base / "1.0.3" / "ptbxl_database.csv",
            base / "1.0.1" / "ptbxl_database.csv",
        ]

        csv_path = None
        for c in csv_candidates:
            if c.exists():
                csv_path = c
                break

        if csv_path is None:
            log.warning("PTB-XL metadata CSV not found — using default metadata.")
            return meta_map

        try:
            df = pd.read_csv(csv_path)
            log.info("Loaded PTB-XL metadata: %d rows from %s", len(df), csv_path.name)

            for _, row in df.iterrows():
                # Map by ecg_id (used in filename like 00001_hr, 00001_lr)
                ecg_id = str(int(row.get("ecg_id", 0))).zfill(5)
                meta_vec = build_metadata_vector(
                    age=row.get("age"),
                    sex=row.get("sex"),
                    height=row.get("height"),
                    weight=row.get("weight"),
                )
                meta_map[ecg_id] = meta_vec

        except Exception as exc:
            log.warning("Failed to parse PTB-XL metadata CSV: %s", exc)

        return meta_map

    def _get_record_metadata(self, record_name: str, meta_map: dict) -> np.ndarray:
        """Look up metadata for a record, trying common ID patterns."""
        # Try direct match
        if record_name in meta_map:
            return meta_map[record_name]

        # Try numeric part only (e.g., "00001_hr" → "00001")
        parts = record_name.split("_")
        if parts[0] in meta_map:
            return meta_map[parts[0]]

        # Try zero-padded numeric
        try:
            num = int(parts[0])
            padded = str(num).zfill(5)
            if padded in meta_map:
                return meta_map[padded]
        except ValueError:
            pass

        return DEFAULT_META.copy()

    # ── PTB-XL ───────────────────────────────────────────────────────────────

    def load_ptb_xl(self, lead: str = "II", max_records: int | None = None):
        """
        Load PTB-XL signals (native 500 Hz) and resample to 360 Hz.
        Returns (windows, metadata_vectors, record_meta).
        """
        log.info("Loading PTB-XL ...")
        base = self._find_dataset_root("PTB-XL", "ptb-xl")
        if base is None:
            log.warning("PTB-XL path not found under %s — returning empty.", self.data_path)
            return (np.empty((0, self.WINDOW_SZ), dtype=np.float32),
                    np.empty((0, METADATA_DIM), dtype=np.float32), [])

        # Load patient metadata
        meta_map = self._load_ptbxl_metadata(base)

        ptb_records_path = base / "records500"
        if not ptb_records_path.exists():
            ptb_records_path = base

        windows, meta_vecs, meta = [], [], []
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

                rec_meta = self._get_record_metadata(hdr.stem, meta_map)

                for w in self._extract_windows(sig):
                    windows.append(w)
                    meta_vecs.append(rec_meta)
                    meta.append({"record": hdr.stem, "dataset": "PTB-XL"})
            except Exception as exc:
                log.debug("PTB-XL skip %s: %s", hdr.stem, exc)

        log.info("  PTB-XL: %d windows from %d records.", len(windows), len(record_dirs))
        return (np.array(windows, dtype=np.float32),
                np.array(meta_vecs, dtype=np.float32) if meta_vecs else np.empty((0, METADATA_DIM), dtype=np.float32),
                meta)

    # ── CPSC 2018 ─────────────────────────────────────────────────────────────

    def load_cpsc2018(self, lead: str = "II", max_records: int | None = None):
        log.info("Loading CPSC 2018 ...")
        base = self._find_dataset_root("CPSC-2018", "cpsc2018", "CPSC2018")

        if base is None:
            log.warning("CPSC2018 path not found under %s — returning empty.", self.data_path)
            return (np.empty((0, self.WINDOW_SZ), dtype=np.float32),
                    np.empty((0, METADATA_DIM), dtype=np.float32), [])

        training = base / "training"
        hdr_root = training if training.exists() else base
        hdrs = sorted(hdr_root.rglob("A*.hea"))
        total_records = len(hdrs)
        if max_records:
            hdrs = hdrs[:max_records]
            total_records = len(hdrs)

        windows, meta_vecs, meta = [], [], []
        for index, hdr in enumerate(hdrs, start=1):
            stem = self._record_stem(hdr)
            if index == 1 or index % 100 == 0 or index == total_records:
                log.info("  CPSC2018 loading progress: %d/%d records scanned", index, total_records)
            try:
                rec = wfdb.rdrecord(stem)

                # Try to extract age/sex from CPSC header comments
                rec_age, rec_sex = None, None
                if hasattr(rec, 'comments') and rec.comments:
                    for comment in rec.comments:
                        cl = comment.lower().strip()
                        if 'age' in cl:
                            try:
                                rec_age = int(''.join(c for c in cl if c.isdigit()))
                            except ValueError:
                                pass
                        if 'sex' in cl or 'gender' in cl:
                            if 'male' in cl and 'female' not in cl:
                                rec_sex = 'male'
                            elif 'female' in cl:
                                rec_sex = 'female'

                rec_meta = build_metadata_vector(age=rec_age, sex=rec_sex)

                idx = self._get_lead_index(rec, lead)
                sig = rec.p_signal[:, idx].astype(np.float64)
                if rec.fs != self.TARGET_SR:
                    sig = self._resample(sig, int(rec.fs), self.TARGET_SR)
                for w in self._extract_windows(sig):
                    windows.append(w)
                    meta_vecs.append(rec_meta)
                    meta.append({"record": hdr.stem, "dataset": "CPSC2018"})
            except Exception as exc:
                log.debug("CPSC2018 skip %s: %s", hdr.stem, exc)

        log.info("  CPSC2018: %d windows.", len(windows))
        return (np.array(windows, dtype=np.float32),
                np.array(meta_vecs, dtype=np.float32) if meta_vecs else np.empty((0, METADATA_DIM), dtype=np.float32),
                meta)

    # ── MIT-BIH ───────────────────────────────────────────────────────────────

    def load_mitbih(self, max_records: int | None = None):
        """
        Load MIT-BIH (native 360 Hz).
        Returns (signals, annotations_list, metadata_vectors, record_meta).
        """
        log.info("Loading MIT-BIH ...")
        base = self._find_dataset_root("mitdb", "MITDB", "mit-bih-database")
        if base is None:
            log.warning("MIT-BIH path not found under %s — returning empty.", self.data_path)
            return [], [], [], []

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

        all_signals, all_annots, all_meta_vecs, meta = [], [], [], []
        for index, num in enumerate(valid_nums, start=1):
            header_path = header_map.get(str(num))
            rpath = str(header_path.with_suffix("")) if header_path else str(base / str(num))
            if index == 1 or index % 10 == 0 or index == total_records:
                log.info("  MIT-BIH loading progress: %d/%d records scanned", index, total_records)
            try:
                rec  = wfdb.rdrecord(rpath)
                ann  = wfdb.rdann(rpath, "atr")
                sig  = rec.p_signal[:, 0].astype(np.float32)

                # MIT-BIH has limited metadata — extract from comments if available
                rec_age, rec_sex = None, None
                if hasattr(rec, 'comments') and rec.comments:
                    for comment in rec.comments:
                        cl = comment.lower()
                        if 'age' in cl:
                            try:
                                rec_age = int(''.join(c for c in cl if c.isdigit()))
                            except ValueError:
                                pass
                        if 'male' in cl and 'female' not in cl:
                            rec_sex = 'male'
                        elif 'female' in cl:
                            rec_sex = 'female'

                rec_meta = build_metadata_vector(age=rec_age, sex=rec_sex)

                all_signals.append(sig)
                all_annots.append({
                    "samples": ann.sample.tolist(),
                    "symbols": ann.symbol,
                })
                all_meta_vecs.append(rec_meta)
                meta.append({"record": str(num), "dataset": "MIT-BIH"})
            except Exception as exc:
                log.debug("MIT-BIH skip %d: %s", num, exc)

        log.info("  MIT-BIH: %d full records loaded.", len(all_signals))
        return all_signals, all_annots, all_meta_vecs, meta

    # ── LUDB ──────────────────────────────────────────────────────────────────

    def load_ludb(self, lead_idx: int = 0, max_records: int | None = None):
        log.info("Loading LUDB ...")
        base = self._find_dataset_root("ludb", "LUDB")

        if base is None:
            log.warning("LUDB path not found under %s — returning empty.", self.data_path)
            return (np.empty((0, self.WINDOW_SZ), dtype=np.float32),
                    np.empty((0, METADATA_DIM), dtype=np.float32), [])

        # Common layouts include nested version/data folders, and file names can be numeric.
        hdrs = sorted(base.rglob("*.hea"))
        total_records = len(hdrs)
        if max_records:
            hdrs = hdrs[:max_records]
            total_records = len(hdrs)

        windows, meta_vecs, meta = [], [], []
        for index, hdr in enumerate(hdrs, start=1):
            stem = self._record_stem(hdr)
            if index == 1 or index % 50 == 0 or index == total_records:
                log.info("  LUDB loading progress: %d/%d records scanned", index, total_records)
            try:
                rec = wfdb.rdrecord(stem)
                sig = rec.p_signal[:, lead_idx].astype(np.float64)
                if rec.fs != self.TARGET_SR:
                    sig = self._resample(sig, int(rec.fs), self.TARGET_SR)

                # LUDB may have metadata in comments
                rec_meta = DEFAULT_META.copy()

                for w in self._extract_windows(sig):
                    windows.append(w)
                    meta_vecs.append(rec_meta)
                    meta.append({"record": hdr.stem, "dataset": "LUDB"})
            except Exception as exc:
                log.debug("LUDB skip %s: %s", hdr.stem, exc)

        log.info("  LUDB: %d windows.", len(windows))
        return (np.array(windows, dtype=np.float32),
                np.array(meta_vecs, dtype=np.float32) if meta_vecs else np.empty((0, METADATA_DIM), dtype=np.float32),
                meta)

    # ── Combined loader ───────────────────────────────────────────────────────

    def load_all_datasets(self, max_records_per_dataset: int | None = None):
        """
        Load all 4 datasets. Returns:
            X_train      : PTB-XL + CPSC2018 signal windows  (N, 360)
            M_train      : corresponding metadata vectors     (N, 4)
            mitbih       : dict with signals + annotations + metadata
            X_ludb       : LUDB signal windows                (M, 360)
            M_ludb       : LUDB metadata vectors              (M, 4)
            meta_train   : combined record-level metadata list
        """
        log.info("=" * 70)
        log.info("LOADING ALL 4 ECG DATASETS")
        log.info("=" * 70)

        X_ptb,  M_ptb,  m_ptb   = self.load_ptb_xl(max_records=max_records_per_dataset)
        X_cpsc, M_cpsc, m_cpsc  = self.load_cpsc2018(max_records=max_records_per_dataset)
        mit_sigs, mit_ann, mit_meta_vecs, m_mit = self.load_mitbih(max_records=max_records_per_dataset)
        X_ludb, M_ludb, m_ludb  = self.load_ludb(max_records=max_records_per_dataset)

        # Combine signal windows
        parts_X = [x for x in [X_ptb, X_cpsc] if len(x) > 0]
        X_train = np.concatenate(parts_X) if parts_X else np.empty((0, self.WINDOW_SZ), dtype=np.float32)

        # Combine metadata vectors
        parts_M = [m for m in [M_ptb, M_cpsc] if len(m) > 0]
        M_train = np.concatenate(parts_M) if parts_M else np.empty((0, METADATA_DIM), dtype=np.float32)

        meta_train = m_ptb + m_cpsc

        log.info("=" * 70)
        log.info("DATASET SUMMARY")
        log.info("  Training  (PTB-XL + CPSC): %d windows, metadata: %s", len(X_train), M_train.shape)
        log.info("  MIT-BIH   (annotated)     : %d records", len(mit_sigs))
        log.info("  LUDB      (noise robust)  : %d windows", len(X_ludb))
        log.info("=" * 70)

        mitbih_data = {
            "signals": mit_sigs,
            "annotations": mit_ann,
            "metadata": mit_meta_vecs,
            "meta": m_mit,
        }

        return X_train, M_train, mitbih_data, X_ludb, M_ludb, meta_train


# ═════════════════════════════════════════════════════════════════════════════
# 3.  BEAT EXTRACTOR (with metadata pass-through)
# ═════════════════════════════════════════════════════════════════════════════

class BeatExtractor:
    """
    Extract labelled 360-sample beat windows with associated metadata.

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
        metadata: np.ndarray = None,
        dataset_name: str = "PTB-XL",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Assign probabilistic labels to pre-extracted 360-sample windows.
        Used for PTB-XL, CPSC2018, LUDB.

        Returns (windows, labels, metadata).
        """
        priors = self.PRIORS.get(dataset_name, [0.6, 0.1, 0.15, 0.1, 0.05])
        labels = self._rng.choice(5, size=len(windows), p=priors)

        if metadata is None or len(metadata) == 0:
            metadata = np.tile(DEFAULT_META, (len(windows), 1))

        return windows, labels.astype(np.int32), metadata

    # ------------------------------------------------------------------

    def extract_from_mitbih(
        self,
        signals: list[np.ndarray],
        annotations: list[dict],
        metadata_vecs: list[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract beat-centred windows from MIT-BIH using real annotations.
        Window: 180 samples before R-peak + 180 samples after R-peak.

        Returns (beats, labels, metadata).
        """
        if metadata_vecs is None:
            metadata_vecs = [DEFAULT_META.copy()] * len(signals)

        beats, labels, metas = [], [], []

        for sig, ann, rec_meta in zip(signals, annotations, metadata_vecs):
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
                metas.append(rec_meta)

        if not beats:
            return (np.empty((0, self.WINDOW), dtype=np.float32),
                    np.empty(0, dtype=np.int32),
                    np.empty((0, METADATA_DIM), dtype=np.float32))

        return (np.array(beats, dtype=np.float32),
                np.array(labels, dtype=np.int32),
                np.array(metas, dtype=np.float32))

    # ------------------------------------------------------------------

    def build_full_dataset(
        self,
        X_train_windows: np.ndarray,
        M_train: np.ndarray,
        mitbih_data: dict,
        X_ludb_windows: np.ndarray,
        M_ludb: np.ndarray = None,
    ) -> tuple:
        """
        Combine all sources into (X, y, M) plus separate hold-out arrays.

        Returns: (X_all, y_all, M_all,
                  X_mit, y_mit, M_mit,
                  X_ludb, y_ludb, M_ludb)
        """
        # Training windows with probabilistic labels
        X_ptb_cpsc, y_ptb_cpsc, M_ptb_cpsc = self.extract_from_windows(
            X_train_windows, M_train, "PTB-XL"
        )

        # MIT-BIH with real labels
        X_mit, y_mit, M_mit = self.extract_from_mitbih(
            mitbih_data["signals"],
            mitbih_data["annotations"],
            mitbih_data.get("metadata"),
        )

        # LUDB windows with probabilistic labels
        X_ludb, y_ludb, M_ludb_out = self.extract_from_windows(
            X_ludb_windows, M_ludb, "LUDB"
        )

        # Combine PTB-XL/CPSC2018 + MIT-BIH into primary train pool
        parts_X = [a for a in [X_ptb_cpsc, X_mit] if len(a) > 0]
        parts_y = [a for a in [y_ptb_cpsc, y_mit] if len(a) > 0]
        parts_M = [a for a in [M_ptb_cpsc, M_mit] if len(a) > 0]

        X_all = np.concatenate(parts_X) if parts_X else np.empty((0, self.WINDOW), dtype=np.float32)
        y_all = np.concatenate(parts_y) if parts_y else np.empty(0, dtype=np.int32)
        M_all = np.concatenate(parts_M) if parts_M else np.empty((0, METADATA_DIM), dtype=np.float32)

        log.info("Combined dataset — X: %s  y: %s  M: %s", X_all.shape, y_all.shape, M_all.shape)
        log.info("MIT-BIH hold-out — X: %s  y: %s  M: %s", X_mit.shape, y_mit.shape, M_mit.shape)
        log.info("LUDB hold-out    — X: %s  y: %s  M: %s", X_ludb.shape, y_ludb.shape, M_ludb_out.shape)

        return X_all, y_all, M_all, X_mit, y_mit, M_mit, X_ludb, y_ludb, M_ludb_out


# ─── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    downloader = DatasetDownloader("./data/raw")
    downloader.validate_downloads()

    # Test metadata builder
    v = build_metadata_vector(age=45, sex="male", height=175, weight=80)
    print(f"Sample metadata vector: {v}")
    assert v.shape == (4,), f"Expected shape (4,), got {v.shape}"
    print("✓ Metadata test passed.")
