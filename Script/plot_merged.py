"""Layout-only revision: one plot, no sample-count labels, explicit line styles.

Contract: pooled associations shown on their original numerical scales;
15 original rows and verified statistics retained, with no exclusions.
Python quantitative single-panel chart, 110 x 117 mm; editable SVG/PDF,
600-dpi PNG/TIFF and 300-dpi preview. Colours identify treatment groups;
markers and line styles identify metabolites. Counts remain in metadata.
"""
import json
from pathlib import Path
import numpy as np
from plot_correlation import DATA, GROUPS, rank_correlation, plot_overlay, write_metadata

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'merged_no_n'
OUT.mkdir(exist_ok=True)
results = json.loads((ROOT / 'statistics.json').read_text(encoding='utf-8'))
assert DATA.shape == (15, 3)
assert int(np.sum(GROUPS == 'DSS')) == 7
assert int(np.sum(GROUPS == 'DSS+VPP')) == 8
for j, result in enumerate(results, 1):
    assert np.isclose(rank_correlation(DATA[:, 0], DATA[:, j]), result['Spearman_rho'])
write_metadata(OUT, results)
qa_path = OUT / 'QA.json'
qa = json.loads(qa_path.read_text(encoding='utf-8'))
qa.update({
    'archetype': 'Single-panel quantitative overlay',
    'revision': 'Display-only: all n labels removed; line legend explicitly MMIA (solid) and 5-HIA (dashed)',
    'counts_display': 'Not on figure, retained in metadata and manuscript legend',
    'statistics_changed': False,
    'dimensions_mm': [110, 117],
})
qa_path.write_text(json.dumps(qa, indent=2), encoding='utf-8')
plot_overlay(OUT, results, show_n=False)
print(OUT / 'correlation_merged.svg')
