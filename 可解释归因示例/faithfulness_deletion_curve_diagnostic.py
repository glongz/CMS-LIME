#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMS-LIME deletion/insertion diagnostic: L2 norms, P(preictal) vs k/K, moment-match on/off.

Produces:
  - Console: per-segment summary (Del./Ins. AUC, max-min of curves, degenerate flags)
  - CSV: one row per (segment, k) with P_del, P_ins (mm on), P_ins (mm off)
  - JSON: run config + one-line summary per segment
  - Optional PNG: deletion / insertion curves

Example:
  python 可解释归因示例/faithfulness_deletion_curve_diagnostic.py ^
    --data_dir D:/public_data/CHBMIT/1_data_clean/chb01 --out_dir 可解释归因示例/deletion_diag_out ^
    --max_segments 6 --cms_unit_types microstate,shapelet,timefreq --fast
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
for p in (str(_REPO), str(_THIS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plot_cms_attribution_panels as pan  # noqa: E402
from chb_loocv_weights import apply_loocv_model_cli  # noqa: E402
from chb_paths import chb_patient_data_dir  # noqa: E402
from faithfulness_protocol_experiment import (  # noqa: E402
    RankedRegion,
    conf_preictal,
    deletion_insertion_curves,
    explain_cms_lime_regions,
    hybrid_noise_on_mask,
    union_masks,
)

try:
    from cms_lime_explainer import CMSLimeConfig, CMSLimeExplainer
except Exception as e:  # pragma: no cover
    raise SystemExit(f"Need cms_lime_explainer: {e}")


def _deletion_path_l2_maxabs(
    x_ct: np.ndarray,
    regions: List[RankedRegion],
    noise_scale: float,
    rng: np.random.Generator,
) -> List[Tuple[float, float]]:
    """
    Match ``deletion_insertion_curves`` deletion branch: for k=1..K, hybrid on union of first k masks
    (same rng consumption order) and report ||Δ||_2 and max |Δ| vs original.
    """
    if not regions:
        return []
    K = len(regions)
    masks = [r.mask for r in regions]
    out: List[Tuple[float, float]] = []
    for k in range(1, K + 1):
        mk = union_masks(masks[:k])
        if not np.any(mk):
            out.append((0.0, 0.0))
            continue
        xp = hybrid_noise_on_mask(x_ct, mk, rng, noise_scale=float(noise_scale))
        d = (xp.astype(np.float64) - x_ct.astype(np.float64)).ravel()
        out.append((float(np.linalg.norm(d)), float(np.max(np.abs(d)))))
    return out


def _l2_fully_perturbed_protocol(
    x: np.ndarray,
    regions: List[RankedRegion],
    rng_seed: int,
    noise_scale: float,
) -> float:
    """
    与 ``deletion_insertion_curves`` 中插入阶段起点 **同一条 RNG 流**：先做 K 步 deletion
    的 hybrid 扰动，再对 **整段 union 掩模** 做一次 hybrid。该向量即插入曲线在 k=0 处的输入。
    （Deletion 路径本身**不**做 per-channel 矩匹配；矩匹配只作用于 insertion 的逐步恢复。）
    """
    masks = [r.mask for r in regions]
    rng = np.random.default_rng(int(rng_seed))
    hkw = float(noise_scale)
    for k in range(1, len(masks) + 1):
        mk = union_masks(masks[:k])
        hybrid_noise_on_mask(x, mk, rng, noise_scale=hkw)
    full_m = union_masks(masks)
    xpp = hybrid_noise_on_mask(x, full_m, rng, noise_scale=hkw)
    d = xpp.astype(np.float64).ravel() - x.astype(np.float64).ravel()
    return float(np.linalg.norm(d))


def _l2_fully_perturbed_fresh(
    x: np.ndarray,
    regions: List[RankedRegion],
    rng_seed: int,
    noise_scale: float,
) -> float:
    """仅对全 union 掩模扰动（新 RNG），用于看「掩模本身」能造成多大 L2，不受前面 K 步消耗影响。"""
    full_m = union_masks([r.mask for r in regions])
    rng = np.random.default_rng(int(rng_seed))
    xpp = hybrid_noise_on_mask(x, full_m, rng, noise_scale=float(noise_scale))
    d = xpp.astype(np.float64).ravel() - x.astype(np.float64).ravel()
    return float(np.linalg.norm(d))


def _setup_cms(
    npy_path: str,
    wrapper: Any,
    args: argparse.Namespace,
) -> CMSLimeExplainer:
    n_pert = int(getattr(args, "cms_n_perturbations", 0) or 0) or (120 if args.fast else 400)
    if bool(getattr(args, "cms_half_pert", False)):
        n_pert = max(32, n_pert // 2)
    n_feat = max(1, min(int(getattr(args, "cms_dpp_k", 10)), int(args.max_regions), 16))
    cfg = CMSLimeConfig(
        verbose=False,
        n_perturbations=n_pert,
        n_shapelets=12 if args.fast else 60,
        n_microstates=3 if args.fast else 4,
        n_features=n_feat,
        regression_method=str(getattr(args, "cms_regression", "ridge")),
    )
    ex = CMSLimeExplainer(cfg)
    data = pan.load_long_npy(npy_path)
    X_list, y_list = [], []
    for w, s, e in pan.iter_windows(data, 1280, int(args.stride)):
        if len(X_list) >= int(args.cms_fit_windows):
            break
        w = np.array(w, dtype=np.float32, copy=True)
        pr = wrapper.predict_proba(w)[0]
        y_list.append(int(np.argmax(pr)))
        X_list.append(w[0])
    ex.fit(np.stack(X_list, axis=0), np.asarray(y_list, dtype=int), wrapper)
    return ex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Directory with .npy (1,C,T). If empty, require --chb_patient",
    )
    ap.add_argument(
        "--chb_patient",
        type=int,
        default=0,
        help="1..99: use .../1_data_clean/chb%%02d as data_dir (see chb_paths).",
    )
    ap.add_argument("--npy", type=str, default="", help="If set, use only this stem (e.g. chb01_03) from data_dir")
    ap.add_argument("--out_dir", type=str, default=str(_THIS / "deletion_curve_diag_out"))
    ap.add_argument("--model_path", type=str, default="")
    ap.add_argument(
        "--loocv_model_name",
        type=str,
        default="",
        help="e.g. eeginception: set model_path from 组合留一法 (requires --chb_patient)",
    )
    ap.add_argument("--loocv_select_loop", type=int, default=1)
    ap.add_argument(
        "--loocv_sync_data",
        action="store_true",
        help="Set data_dir to remapped chb id (12->13) to match training script",
    )
    ap.add_argument("--loocv_weight_root", type=str, default="", help="Optional LOOCV root directory")
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--target_class", type=int, default=1)
    ap.add_argument("--max_regions", type=int, default=12)
    ap.add_argument("--max_segments", type=int, default=6, help="Number of 1280-sample windows to evaluate")
    ap.add_argument("--stride", type=int, default=20000)
    ap.add_argument("--hybrid_noise_scale", type=float, default=2.0)
    ap.add_argument("--base_seed", type=int, default=42, help="deletion_insertion uses base_seed+seg_idx per segment")
    ap.add_argument("--cms_fit_windows", type=int, default=24)
    ap.add_argument(
        "--cms_unit_types",
        type=str,
        default="microstate,shapelet,timefreq",
        help="Comma list passed to explain_instance, e.g. microstate,shapelet,timefreq or microstate only",
    )
    ap.add_argument("--cms_dpp_k", type=int, default=10)
    ap.add_argument("--cms_n_perturbations", type=int, default=0, help="0=auto 120 (fast) or 400")
    ap.add_argument("--cms_half_pert", action="store_true")
    ap.add_argument("--cms_regression", type=str, default="ridge")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--simple_model", action="store_true")
    ap.add_argument(
        "--conf_mode",
        type=str,
        default="prob",
        choices=["prob", "logit"],
    )
    ap.add_argument("--no_plot", action="store_true")
    ap.add_argument("--degenerate_eps", type=float, default=1e-6, help="max(y)-min(y) below this => flat")
    ap.add_argument("--no_progress", action="store_true", help="Disable tqdm on segments")
    args = ap.parse_args()

    if int(getattr(args, "chb_patient", 0) or 0) > 0:
        p = int(args.chb_patient)
        if p < 1 or p > 99:
            raise SystemExit("--chb_patient must be 1..99")
        args.data_dir = str(chb_patient_data_dir(p))
    apply_loocv_model_cli(args)

    if not (getattr(args, "data_dir", "") or "").strip():
        raise SystemExit("Set --data_dir or --chb_patient")

    data_dir = Path(args.data_dir)
    all_npy = sorted(data_dir.glob("*.npy"), key=lambda p: p.name)
    if (args.npy or "").strip():
        stem = (args.npy or "").replace(".npy", "").strip()
        all_npy = [p for p in all_npy if p.stem == stem]
    if not all_npy:
        raise SystemExit("No .npy files to use")

    sample = pan.load_long_npy(str(all_npy[0]))
    if sample.ndim != 3 or sample.shape[0] != 1:
        raise SystemExit("Need (1,C,T) npy")
    n_ch = int(sample.shape[1])
    device = torch.device(args.device)
    mp, _ = pan.resolve_model_path((args.model_path or "").strip() or None)
    if bool(args.simple_model):
        mp = ""
    model = pan.load_model_robust(mp, n_chans=n_ch, n_classes=2, device=device)
    wrapper = pan.EEGModelWrapper(model, device, n_chans=n_ch)

    def predict_proba_np(x_bct: np.ndarray) -> np.ndarray:
        if args.conf_mode == "prob":
            return wrapper.predict_proba(x_bct)
        x_bct = np.array(x_bct, dtype=np.float32, copy=True, order="C")
        xt = torch.from_numpy(x_bct).to(device)
        if xt.dim() == 2:
            xt = xt.unsqueeze(0)
        with torch.no_grad():
            out = model(xt)
            if out.dim() == 1:
                out = out.unsqueeze(0)
            if out.min() >= 0 and out.max() <= 1 and out.sum(dim=-1).min() > 0.99:
                out = torch.logit(torch.clamp(out, 1e-6, 1 - 1e-6))
        return out.cpu().numpy()

    ex = _setup_cms(str(all_npy[0]), wrapper, args)
    ut = [s.strip() for s in str(args.cms_unit_types).split(",") if s.strip()]
    hkw = float(args.hybrid_noise_scale)
    degen_eps = float(args.degenerate_eps)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_csv: List[Dict[str, Any]] = []
    seg_summaries: List[Dict[str, Any]] = []

    windows: List[Tuple[np.ndarray, int, int, str]] = []
    for p in all_npy:
        data = pan.load_long_npy(str(p))
        for w, s, e in pan.iter_windows(data, 1280, int(args.stride)):
            windows.append((np.array(w[0], dtype=np.float32), s, e, p.stem))
            if len(windows) >= int(args.max_segments):
                break
        if len(windows) >= int(args.max_segments):
            break

    _nw = len(windows)
    _use_pb = (tqdm is not None) and (not bool(args.no_progress)) and _nw > 0
    _pbar = (
        tqdm(
            total=_nw,
            desc="Deletion diagnostic",
            unit="seg",
            dynamic_ncols=True,
            mininterval=0.25,
            file=sys.stdout,
        )
        if _use_pb
        else None
    )
    try:
        for seg_i, (x, start, end, stem) in enumerate(windows):
            rseed = int(args.base_seed) + seg_i * 1_000_003
            rng = np.random.default_rng(rseed)
            regions = explain_cms_lime_regions(
                ex, x, int(args.target_class), int(args.max_regions), unit_types=ut, exp_out=None
            )
            regions = regions[: int(args.max_regions)]
            n_reg = len(regions)
            if n_reg == 0:
                print(f"[skip] {stem} [{start},{end}): no CMS regions")
                if _pbar is not None:
                    _pbar.update(1)
                    _pbar.set_postfix_str(f"{stem}:skip", refresh=False)
                continue

            del_on, ins_on, x_depth, y_del, y_ins_on = deletion_insertion_curves(
                x, regions, predict_proba_np, rng, int(args.target_class),
                insertion_moment_match=True, hybrid_noise_scale=hkw,
            )
            rng2 = np.random.default_rng(rseed)
            del_off, ins_off, _, y_del2, y_ins_off = deletion_insertion_curves(
                x, regions, predict_proba_np, rng2, int(args.target_class),
                insertion_moment_match=False, hybrid_noise_scale=hkw,
            )
            if not (np.allclose(y_del, y_del2) or np.max(np.abs(y_del - y_del2)) < 1e-5):
                print("[warn] deletion curves differ between mm runs;_rng state?", stem, start)

            p0 = conf_preictal(x, predict_proba_np, int(args.target_class))
            mm = float(np.max(y_del) - np.min(y_del))
            mi = float(np.max(y_ins_on) - np.min(y_ins_on))
            mio = float(np.max(y_ins_off) - np.min(y_ins_off))
            flags: List[str] = []
            if mm < degen_eps:
                flags.append("deletion_flat")
            if mi < degen_eps:
                flags.append("insertion_mm_on_flat")
            if mio < degen_eps:
                flags.append("insertion_mm_off_flat")

            l2_rows = _deletion_path_l2_maxabs(
                x, regions, hkw, np.random.default_rng(int(rseed))
            )
            l2k1 = l2_rows[0][0] if l2_rows else 0.0
            l2kK = l2_rows[-1][0] if l2_rows else 0.0

            y_del = np.asarray(y_del, dtype=float)
            l2_full_protocol = _l2_fully_perturbed_protocol(x, regions, rseed, hkw)
            l2_full_fresh = _l2_fully_perturbed_fresh(x, regions, rseed, hkw)
            print(
                f"\n=== {stem} start={start} end={end} n_regions={n_reg} P_orig={p0:.6f} ==="
            )
            print(
                f"  P_preictal along DELETION (k=0..K, K={n_reg}): "
                + "[" + ", ".join(f"{float(v):.6f}" for v in y_del) + "]"
            )
            print(
                f"  ||X-X''||_2  fully perturbed, protocol RNG (insertion k=0 point): {l2_full_protocol:.8g}"
            )
            print(
                f"  ||X-X''||_2  fully perturbed, fresh RNG union-only:         {l2_full_fresh:.8g}"
            )
            print(
                f"  (Deletion 不做矩匹配; insertion 的矩匹配 on/off 见 Ins AUC 与 CSV; "
                f"逐步 union 的 L2: k=1 {l2k1:.8g}, k=K {l2kK:.8g})"
            )
            print(
                f"  Del AUC={del_on:.6f}  Ins AUC (mm on)={ins_on:.6f}  Ins AUC (mm off)={ins_off:.6f}"
            )
            print(
                f"  max-min del={mm:.2e}  ins mm_on={mi:.2e}  ins mm_off={mio:.2e}  flags={','.join(flags) or 'none'}"
            )
            print(
                f"  L2 union mask k=1: {l2k1:.8g}  k=K: {l2kK:.8g}  (from hybrid_noise_on_mask)"
            )
            for i in range(len(x_depth)):
                kn = float(x_depth[i]) if i < len(x_depth) else np.nan
                pdel = float(y_del[i]) if i < len(y_del) else np.nan
                pio = float(y_ins_on[i]) if i < len(y_ins_on) else np.nan
                pif = float(y_ins_off[i]) if i < len(y_ins_off) else np.nan
                rows_csv.append(
                    {
                        "stem": stem,
                        "win_start": start,
                        "win_end": end,
                        "seg_index": seg_i,
                        "k_index": i,
                        "k_norm": kn,
                        "P_preictal_deletion": pdel,
                        "P_preictal_insertion_moment_match_on": pio,
                        "P_preictal_insertion_moment_match_off": pif,
                    }
                )
            seg_summaries.append(
                {
                    "stem": stem,
                    "win_start": start,
                    "win_end": end,
                    "n_regions": n_reg,
                    "P_preictal_original": p0,
                    "deletion_auc": del_on,
                    "insertion_auc_moment_match_on": ins_on,
                    "insertion_auc_moment_match_off": ins_off,
                    "max_min_deletion": mm,
                    "max_min_insertion_mm_on": mi,
                    "max_min_insertion_mm_off": mio,
                    "flags": flags,
                    "l2_k1": l2k1,
                    "l2_kK": l2kK,
                    "P_preictal_deletion_k0_to_K": [float(v) for v in y_del],
                    "l2_fully_perturbed_protocol": l2_full_protocol,
                    "l2_fully_perturbed_fresh_union": l2_full_fresh,
                    "l2_per_k": [
                        {"k_step": j + 1, "l2": t[0], "max_abs": t[1]}
                        for j, t in enumerate(l2_rows)
                    ],
                }
            )
            if _pbar is not None:
                _pbar.update(1)
                _pbar.set_postfix_str(f"{stem}[{start}]", refresh=False)
    finally:
        if _pbar is not None:
            _pbar.close()

    csv_path = out_dir / "deletion_insertion_k_curves.csv"
    if rows_csv:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            w.writeheader()
            for r in rows_csv:
                w.writerow(r)
    (out_dir / "deletion_insertion_per_segment.json").write_text(
        json.dumps(
            {
                "config": {k: v for k, v in vars(args).items()},
                "cms_unit_types": ut,
                "segments": seg_summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if rows_csv:
        print(f"\n[done] wrote {csv_path}")
    else:
        print("\n[warn] no curve rows (no regions or no windows); CSV skipped")
    print(f"[done] wrote {out_dir / 'deletion_insertion_per_segment.json'}")

    if not args.no_plot and seg_summaries:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[warn] matplotlib not available, skip plot: {e}")
            return

        n_seg = len(seg_summaries)
        fig, axes = plt.subplots(n_seg, 2, figsize=(10, 2.5 * n_seg), squeeze=False)
        for i, s in enumerate(seg_summaries):
            sub = [r for r in rows_csv if r["stem"] == s["stem"] and r["win_start"] == s["win_start"]]
            sub = sorted(sub, key=lambda r: r["k_index"])
            kx = [r["k_norm"] for r in sub]
            yd = [r["P_preictal_deletion"] for r in sub]
            yio = [r["P_preictal_insertion_moment_match_on"] for r in sub]
            yif = [r["P_preictal_insertion_moment_match_off"] for r in sub]
            ax0 = axes[i][0]
            ax0.plot(kx, yd, "b.-", label="deletion")
            ax0.set_ylabel("P(preictal)")
            ax0.set_xlabel("k/K")
            ax0.set_title(f"{s['stem']} [{s['win_start']},{s['win_end']}) del")
            ax0.grid(True, alpha=0.3)
            ax0.legend(loc="best", fontsize=8)

            ax1 = axes[i][1]
            ax1.plot(kx, yio, "g.-", label="ins mm on")
            ax1.plot(kx, yif, "m.--", label="ins mm off")
            ax1.set_xlabel("k/K")
            ax1.set_title("insertion (moment match compare)")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="best", fontsize=8)
        fig.tight_layout()
        ppath = out_dir / "deletion_insertion_curves.png"
        fig.savefig(ppath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[done] figure {ppath}")


if __name__ == "__main__":
    main()
