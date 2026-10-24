"""Binary-Jaccard PCoA and all-pair PERMANOVA; retain every JN1 sample.

Python produces all figures. R/vegan is used only for non-visual statistical
validation and spatial-median PERMDISP. No graphics device is opened in R.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import scipy
from scipy.stats import chi2

OUT = Path(__file__).resolve().parent
ORDER = ["CON", "DSS", "DSS+VPP", "DSS+IPP", "DSS+FMT-V"]
COLORS = dict(zip(ORDER, ["#000000", "#FF0000", "#0080FF", "#41F0AE", "#FF8000"]))
MARKERS = dict(zip(ORDER, ["o", "s", "^", "D", "v"]))
SEED = 20261011
N_PERM = 9999


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    if not rows:
        raise ValueError("Refusing to save an empty result table")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_distance(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.reader(handle, delimiter="\t") if row]
    ids = [row[0] for row in rows[1:]]
    columns = rows[0][1:]
    d = np.array([row[1:] for row in rows[1:]], dtype=float)
    if len(set(ids)) != len(ids) or ids != columns:
        raise ValueError("Duplicate identifiers or row/column mismatch")
    if d.shape != (len(ids), len(ids)) or not np.isfinite(d).all():
        raise ValueError("Non-square matrix or non-finite values")
    if not np.allclose(d, d.T, atol=1e-12, rtol=0):
        raise ValueError("Distance matrix is asymmetric")
    if not np.allclose(np.diag(d), 0, atol=1e-12, rtol=0):
        raise ValueError("Nonzero diagonal")
    if d.min() < 0 or d.max() > 1:
        raise ValueError("Binary-Jaccard distances outside [0,1]")
    # Author-confirmed design filter: only JN1 original-experiment animals.
    keep = np.array([sample.startswith("JN1") for sample in ids])
    excluded = [sample for sample, retain in zip(ids, keep) if not retain]
    if set(excluded) != {"FenVPP1", "FenVPP2", "FenIPP1", "FenIPP2"}:
        raise ValueError("Unexpected non-JN1 samples; review the design filter")
    return ids, d, keep, excluded


def group_of(sample):
    code = {"C": "CON", "D": "DSS", "V": "DSS+VPP", "I": "DSS+IPP", "F": "DSS+FMT-V"}
    if not sample.startswith("JN1") or len(sample) < 5 or sample[3] not in code:
        raise ValueError(f"Unknown sample identifier: {sample}")
    if not sample[4:].isdigit():
        raise ValueError(f"Nonstandard sample identifier: {sample}")
    return code[sample[3]]


def center(d):
    squared = d * d
    return -0.5 * (squared - squared.mean(axis=0)[None, :]
                   - squared.mean(axis=1)[:, None] + squared.mean())


def pcoa(d, labels):
    b = center(d)
    eigenvalues, eigenvectors = np.linalg.eigh(b)
    indices = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[indices], eigenvectors[:, indices]
    positive = eigenvalues > max(eigenvalues[0], 1.0) * 1e-12
    scores = eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])
    if scores.shape[1] < 2:
        raise ValueError("Fewer than two positive ordination axes")
    # Preserve the earlier figure's orientation; axis signs are arbitrary.
    if scores[labels == "CON", 0].mean() > 0:
        scores[:, 0] *= -1
    if scores[labels == "CON", 1].mean() < 0:
        scores[:, 1] *= -1
    explained = eigenvalues[positive] / eigenvalues[positive].sum()
    return scores, eigenvalues, explained, b


def permanova_stat(b, labels):
    levels = np.unique(labels)
    n = len(labels)
    ss_total = float(np.trace(b))
    ss_between = sum(b[np.ix_(labels == level, labels == level)].sum()
                     / np.count_nonzero(labels == level) for level in levels)
    ss_within = ss_total - ss_between
    df1, df2 = len(levels) - 1, n - len(levels)
    return float((ss_between / df1) / (ss_within / df2)), float(ss_between / ss_total), df1, df2


def monte_carlo_permanova(b, labels):
    observed = permanova_stat(b, labels)
    rng = np.random.default_rng(SEED)
    ge = sum(permanova_stat(b, rng.permutation(labels))[0] >= observed[0] - 1e-12
             for _ in range(N_PERM))
    return observed, (ge + 1) / (N_PERM + 1), ge


def exact_pair_permanova(d, labels, group1):
    b = center(d)
    observed = permanova_stat(b, labels)
    n, n1 = len(labels), int(np.count_nonzero(labels == group1))
    n2 = n - n1
    ss_total = np.trace(b)
    total = math.comb(n, n1)
    ge = 0
    combinations = itertools.combinations(range(n), n1)
    while batch := list(itertools.islice(combinations, 1000)):
        indicator = np.zeros((len(batch), n), dtype=float)
        indicator[np.arange(len(batch))[:, None], np.asarray(batch)] = 1
        # Centered B has zero row/column sums; complement's quadratic form
        # equals the first group's quadratic form.
        between = np.einsum("bi,ij,bj->b", indicator, b, indicator, optimize=True) * (1 / n1 + 1 / n2)
        fp = between / ((ss_total - between) / (n - 2))
        ge += int(np.count_nonzero(fp >= observed[0] - 1e-12))
    # Exhaustive enumeration includes the observed allocation: no +1 needed.
    return observed, ge / total, total, ge


def bh_adjust(values):
    values = np.asarray(values, float)
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum.accumulate((values[order] * len(values)
                                     / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty_like(values)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def add_data_ellipse(ax, xy, color):
    covariance = np.cov(xy, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    radius = np.sqrt(chi2.ppf(0.95, 2))
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    ax.add_patch(Ellipse(xy.mean(axis=0), *(2 * radius * np.sqrt(np.maximum(values, 0))),
                         angle=angle, facecolor=color, edgecolor=color,
                         linewidth=0.85, alpha=0.10, zorder=1))


def draw_figure(scores, labels, explained, global_rows, pairs):
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 8, "axes.linewidth": 0.8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.right": False, "axes.spines.top": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    # 183-mm double-column width; square 80-mm ordination panel.
    fig = plt.figure(figsize=(7.204724, 4.330709))
    ax = fig.add_axes([0.10, 0.19, 80 / 183, 80 / 110])
    side = fig.add_axes([0.59, 0.11, 0.39, 0.83])
    side.set_axis_off()
    handles = []
    for group in ORDER:
        xy = scores[labels == group, :2]
        add_data_ellipse(ax, xy, COLORS[group])
        handles.append(ax.scatter(xy[:, 0], xy[:, 1], s=29, marker=MARKERS[group],
                                  facecolors="white", edgecolors=COLORS[group],
                                  linewidths=1.0, zorder=3, label=group))
    ax.axhline(0, color="#D9D9D9", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#D9D9D9", linewidth=0.6, zorder=0)
    ax.set_xlabel(f"PCoA1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PCoA2 ({explained[1] * 100:.1f}%)")
    ax.set_title("Binary Jaccard", fontsize=8, pad=6)
    ax.set_box_aspect(1)
    ax.margins(0.08)
    side.legend(handles=handles, labels=ORDER, loc="upper left", frameon=False,
                borderaxespad=0, handletextpad=0.7, labelspacing=0.65)
    perma = global_rows[0]
    disp = global_rows[1]
    side.text(0, 0.735,
              f"Overall PERMANOVA\nF = {perma['pseudo_F']:.2f}; R² = {perma['R2']:.3f}; P = {perma['p_raw']:.4f}\n"
              f"PERMDISP: F = {float(disp['pseudo_F']):.2f}; P = {float(disp['p_raw']):.4f}",
              ha="left", va="top", fontsize=6.7, linespacing=1.5)
    side.text(0, 0.565, "All pairwise PERMANOVA", fontsize=7, ha="left", va="top")
    cell_text = [[row["Group1"], row["Group2"], f"{row['p_raw']:.4f}", f"{row['p_BH']:.4f}"]
                 for row in pairs]
    table = side.table(cellText=cell_text, colLabels=["Group 1", "Group 2", "P", "BH adj. P"],
                       colWidths=[0.25, 0.30, 0.20, 0.25],
                       bbox=[-0.015, 0.055, 1.015, 0.47], cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(6.1)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.PAD = 0.025
        if row == 0:
            cell.set_facecolor("#F1F1F1")
            cell.set_text_props(weight="bold")
        elif pairs[row - 1]["p_BH"] < 0.05:
            cell.set_text_props(weight="bold")
    side.text(0, 0.022, "BH correction across all 10 comparisons", fontsize=6.1,
              va="top", ha="left")
    fig.canvas.draw()
    box = ax.get_window_extent()
    if abs(box.width - box.height) > 0.5:
        raise AssertionError("Ordination axes are not square")
    base = OUT / "microbiome_BinaryJaccard_PCoA_allpairs"
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=600)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUT / "preview.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--rscript", required=True, type=Path)
    args = parser.parse_args()
    source_ids, source_d, keep, excluded = load_distance(args.input)
    ids = np.asarray(source_ids)[keep]
    d = source_d[np.ix_(keep, keep)]
    labels = np.asarray([group_of(sample) for sample in ids])
    counts = {group: int(np.count_nonzero(labels == group)) for group in ORDER}
    if counts != dict(zip(ORDER, [7, 7, 8, 8, 6])):
        raise ValueError(f"Unexpected group sizes: {counts}")
    scores, eigenvalues, explained, b = pcoa(d, labels)
    write_csv(OUT / "sample_metadata.csv", [{"SampleID": sample, "Group": group}
                                           for sample, group in zip(ids, labels)])
    write_csv(OUT / "PCoA_coordinates.csv", [{"SampleID": sample, "Group": group,
              **{f"PCoA{k + 1}": value for k, value in enumerate(row)}}
              for sample, group, row in zip(ids, labels, scores)])
    write_csv(OUT / "analysed_binary_jaccard_distance.csv",
              [{"SampleID": sample, **dict(zip(ids, row))} for sample, row in zip(ids, d)])
    write_csv(OUT / "eigenvalues.csv", [{"Axis": k + 1, "Eigenvalue": value,
              "Positive_inertia_fraction": float(value / eigenvalues[eigenvalues > 1e-12].sum())
              if value > 1e-12 else ""} for k, value in enumerate(eigenvalues)])
    stat, p, ge = monte_carlo_permanova(b, labels)
    global_perma = {"Test": "PERMANOVA", "pseudo_F": stat[0], "R2": stat[1],
                    "df1": stat[2], "df2": stat[3], "p_raw": p,
                    "Permutation_method": "Monte Carlo unrestricted sample-label permutation",
                    "Permutations_or_allocations": N_PERM,
                    "Extreme_permutations": ge, "Seed": SEED}
    pairs = []
    for group1, group2 in itertools.combinations(ORDER, 2):
        retain = np.isin(labels, [group1, group2])
        observed, pp, allocations, extreme = exact_pair_permanova(
            d[np.ix_(retain, retain)], labels[retain], group1)
        pairs.append({"Group1": group1, "Group2": group2, "n1": counts[group1],
                      "n2": counts[group2], "pseudo_F": observed[0], "R2": observed[1],
                      "df1": observed[2], "df2": observed[3], "p_raw": pp,
                      "Permutation_method": "Exhaustive fixed-size group-label allocations",
                      "Allocations": allocations, "Extreme_allocations": extreme})
    for row, adjusted in zip(pairs, bh_adjust([row["p_raw"] for row in pairs])):
        row["p_BH"] = float(adjusted)
        row["Significant_BH_0.05"] = bool(adjusted < 0.05)
    write_csv(OUT / "pairwise_PERMANOVA.csv", pairs)
    write_csv(OUT / "overall_PERMANOVA.csv", [global_perma])
    completed = subprocess.run([str(args.rscript), "--vanilla", str(OUT / "dispersion_and_validation.R"),
                                str(OUT)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    (OUT / "statistical_validation.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    dispersion_rows = read_csv(OUT / "overall_PERMDISP.csv")
    global_rows = [global_perma, dispersion_rows[0]]
    common = ["Test", "pseudo_F", "R2", "df1", "df2", "p_raw", "Permutation_method",
              "Permutations_or_allocations", "Extreme_permutations", "Seed"]
    write_csv(OUT / "overall_statistics.csv", [{key: row.get(key, "") for key in common}
                                              for row in global_rows], common)
    dispersion_pairs = read_csv(OUT / "pairwise_PERMDISP.csv")
    dispersion_lookup = {(row["Group1"], row["Group2"]): row for row in dispersion_pairs}
    combined = [{**row,
                 "PERMDISP_F": float(dispersion_lookup[row["Group1"], row["Group2"]]["pseudo_F"]),
                 "PERMDISP_p_raw": float(dispersion_lookup[row["Group1"], row["Group2"]]["p_raw"]),
                 "PERMDISP_p_BH": float(dispersion_lookup[row["Group1"], row["Group2"]]["p_BH"])}
                for row in pairs]
    write_csv(OUT / "all_pairwise_results.csv", combined)
    draw_figure(scores, labels, explained, global_rows, pairs)
    negative = eigenvalues[eigenvalues < -1e-12]
    qa = {
        "input_SHA256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "input_samples": len(source_ids), "analysed_samples": len(ids),
        "included_ids": ids.tolist(), "excluded_ids": excluded,
        "exclusion_reason": "Author confirmed non-JN1 samples do not belong to the original experiment",
        "post_hoc_distance_exclusions": [], "group_counts": counts,
        "negative_eigenvalues_count": len(negative),
        "negative_inertia_absolute_fraction_of_positive": float(-negative.sum() / eigenvalues[eigenvalues > 1e-12].sum()),
        "PCoA_display": "Uncorrected original dissimilarity; axes expressed as fractions of positive inertia",
        "PCoA_first_two_axis_percent": (explained[:2] * 100).tolist(),
        "ellipses": "95% chi-square-scaled covariance data ellipses; NOT confidence intervals of centroids",
        "inference_space": "Original complete distance matrix, not the two displayed PCoA axes",
        "independence_assumption": "Each JN1 identifier is one independent animal sampled once; no cage/batch metadata provided",
        "pairwise_correction": "Benjamini-Hochberg across all 10 PERMANOVA comparisons; dispersion tests form a separate family",
        "PERMDISP": "vegan spatial median, bias.adjust=TRUE, signed positive/negative axes retained, 9999 residual permutations",
        "backend": "Python/matplotlib exclusively for figures; R/vegan for non-visual dispersion and numerical cross-checks",
        "source_preflight_warnings": {
            "random_generator": "Only sample labels are permuted for PERMANOVA; no artificial measurements are generated",
            "Rscript_reference": "R companion performs non-visual PERMDISP and adonis2 validation only; no graphics devices or plots",
        },
        "versions": {"Python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                     "matplotlib": mpl.__version__, **json.loads((OUT / "R_versions.json").read_text())},
        "design_contract": {
            "conclusion": "Visualize five-group microbiome beta diversity and quantify every between-group contrast without sample selection",
            "archetype": "quantitative grid with square ordination hero and side statistics",
            "exports": "183x110 mm; editable SVG/PDF; 600-dpi PNG/TIFF; CSV source/statistics",
            "style_inheritance": "Earlier five-group palette, hollow markers, square axes, side annotations; rebuilt statistical implementation",
        },
    }
    (OUT / "QA.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"group_counts": counts, "overall": global_rows, "pairs": combined}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
