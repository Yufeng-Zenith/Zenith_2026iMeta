"""Group-labelled pooled Spearman associations with descriptive OLS fits.

All supplied observations are retained. Group colours do not imply that the
pooled correlations have been adjusted for treatment group. No between-
correlation hypothesis test is performed. No measurement unit is invented.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy import stats
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

RAW = '''0.000291 40.45872 12.8818
0.000989 46.99507 7.304209
0.000337 42.89613 9.427132
0.014263 136.8881 29.55893
0.004033 55.017 21.15309
0.007763 79.56886 11.57885
0.00117 36.94698 14.91957
0.007388 87.93879 23.9168
0.029295 104.1712 27.25742
0.014121 95.11508 28.80617
0.026492 97.12626 27.87309
0.036658 138.1775 35.0925
0.001306 38.6893 28.76227
0.03809 162.9231 44.31837
0.001502 78.01526 34.45488'''
TOKENS = [row.split() for row in RAW.splitlines()]
DATA = np.array(TOKENS, dtype=float)
GROUPS = np.array(['DSS']*7 + ['DSS+VPP']*8)
COLOURS = {'DSS': '#FF3030', 'DSS+VPP': '#1685FF'}
SEED = 20261012
BOOTSTRAP_RESAMPLES = 50000
FIG_WIDTH_MM = 183

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .7, 'ytick.major.width': .7,
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'mathtext.fontset': 'dejavusans', 'figure.facecolor': 'white',
    'savefig.facecolor': 'white', 'legend.frameon': False,
})

def exact_spearman_distribution(n: int, cap: int) -> np.ndarray:
    """Count permutations by sum of squared rank differences up to cap.

    A subset dynamic program enumerates assignments algebraically; its counts
    are integers, not Monte Carlo estimates. It is valid for untied ranks.
    Counts remain below 15!, hence fit in uint64 for the supplied sample size.
    """
    assert n <= 15
    dp = np.zeros((1 << n, cap+1), dtype=np.uint64)
    dp[0, 0] = 1
    for mask in range((1 << n)-1):
        i = mask.bit_count()+1
        row = dp[mask]
        if not row.any():
            continue
        remaining = ((1 << n)-1)^mask
        while remaining:
            bit = remaining & -remaining
            j = bit.bit_length()
            remaining -= bit
            cost = (i-j)**2
            if cost <= cap:
                dp[mask | bit, cost:] += row[:cap+1-cost]
    return dp[-1].copy()

def rank_correlation(x, y, axis=-1):
    rx = stats.rankdata(x, axis=axis)
    ry = stats.rankdata(y, axis=axis)
    rx = rx - np.mean(rx, axis=axis, keepdims=True)
    ry = ry - np.mean(ry, axis=axis, keepdims=True)
    return np.sum(rx*ry, axis=axis) / np.sqrt(np.sum(rx*rx, axis=axis)*np.sum(ry*ry, axis=axis))

def validate_exact_engine():
    n = 5
    cap = n*(n*n-1)//3
    dist = exact_spearman_distribution(n, cap)
    direct = np.zeros(cap+1, np.uint64)
    for p in itertools.permutations(range(1,n+1)):
        d = sum((i-j)**2 for i,j in enumerate(p,1))
        direct[d] += 1
    assert np.array_equal(dist, direct)
    assert int(dist.sum()) == math.factorial(n)
    assert np.array_equal(dist, dist[::-1])

def analyse():
    assert DATA.shape == (15,3) and np.isfinite(DATA).all()
    assert np.all(DATA[:,0] >= 0)
    assert list(GROUPS[:7]) == ['DSS']*7 and list(GROUPS[7:]) == ['DSS+VPP']*8
    assert all(len(np.unique(DATA[:,j])) == 15 for j in range(3))
    validate_exact_engine()
    x = DATA[:,0]
    d_values = [int(np.sum((stats.rankdata(x)-stats.rankdata(DATA[:,j]))**2)) for j in (1,2)]
    dist = exact_spearman_distribution(15, max(d_values))
    output = []
    for j, (name,d) in enumerate(zip(['MMIA','5-HIA'],d_values),1):
        rho = float(rank_correlation(x, DATA[:,j]))
        assert rho > 0
        lower_tail_count = int(dist[:d+1].sum())
        p_exact = 2*lower_tail_count/math.factorial(15)
        boot = stats.bootstrap((x,DATA[:,j]),rank_correlation,
            paired=True, vectorized=True, axis=-1, method='BCa',
            n_resamples=BOOTSTRAP_RESAMPLES, batch=4096,
            confidence_level=.95, rng=np.random.default_rng(SEED+j))
        assert np.isfinite(boot.bootstrap_distribution).all()
        lr = stats.linregress(x,DATA[:,j])
        result = {'metabolite':name, 'n':15, 'DSS_n':7, 'DSS_VPP_n':8,
            'Spearman_rho':rho, 'Spearman_exact_two_sided_P':p_exact,
            'Spearman_S_squared_rank_difference_sum':d,
            'one_tail_permutation_count':lower_tail_count,
            'total_permutations':math.factorial(15),
            'rho_CI95_low_BCa':float(boot.confidence_interval.low),
            'rho_CI95_high_BCa':float(boot.confidence_interval.high),
            'rho_bootstrap_resamples':BOOTSTRAP_RESAMPLES,
            'rho_bootstrap_seed':SEED+j,
            'OLS_R_squared':float(lr.rvalue**2), 'OLS_slope':float(lr.slope),
            'OLS_intercept':float(lr.intercept), 'OLS_slope_SE':float(lr.stderr),
            'OLS_intercept_SE':float(lr.intercept_stderr),
            'asymptotic_Scipy_Spearman_P_for_reference_only':float(stats.spearmanr(x,DATA[:,j]).pvalue),
            'P_adjustment_on_figure':'None; two raw correlation P values',
            'association_scope':'Pooled DSS and DSS+VPP; not group-adjusted',
            'unit_of_n':'15 paired sample observations; independence assumed',
        }
        output.append(result)
    order = np.argsort([r['Spearman_exact_two_sided_P'] for r in output])
    running = 0.
    for rank,k in enumerate(order):
        running = max(running, (len(output)-rank)*output[k]['Spearman_exact_two_sided_P'])
        output[k]['Holm_adjusted_P_two_tests_reference_only'] = min(1.,running)
    return output

def ols_curve(j):
    x = DATA[:,0]; y = DATA[:,j]
    grid = np.linspace(x.min(),x.max(),250)
    lr = stats.linregress(x,y)
    fitted = lr.intercept + lr.slope*grid
    residual = y-(lr.intercept+lr.slope*x)
    mse = np.sum(residual**2)/(len(x)-2)
    sxx = np.sum((x-x.mean())**2)
    se_mean = np.sqrt(mse*(1/len(x)+(grid-x.mean())**2/sxx))
    delta = stats.t.ppf(.975,len(x)-2)*se_mean
    return grid, fitted, fitted-delta, fitted+delta

def p_text(p):
    assert p > 0
    if p < .0001:
        exponent = int(math.floor(math.log10(p)))
        mantissa = p / 10**exponent
        return rf'$P = {mantissa:.2f}\times10^{{{exponent}}}$'
    return rf'$P = {p:.5f}$'

def stats_text(result, show_n=True):
    return '\n'.join([
        rf"Spearman $\rho = {result['Spearman_rho']:.4f}$",
        p_text(result['Spearman_exact_two_sided_P']) + (r'; $n = 15$' if show_n else ''),
        f"95% CI [{result['rho_CI95_low_BCa']:.3f}, {result['rho_CI95_high_BCa']:.3f}]",
        rf"OLS $R^2 = {result['OLS_R_squared']:.4f}$",
    ])

def finish_axis(ax):
    ax.set_xlim(0,.04)
    ax.set_xticks(np.arange(0,.041,.01))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel(r'$\it{D.\ newyorkensis}$ relative abundance',labelpad=5)
    ax.tick_params(direction='out')

def group_handles(marker='o', show_n=True):
    return [Line2D([],[],marker=marker,ls='none',mfc='white',mec=COLOURS[g],
        mew=.9,ms=4.2,label=(f'{g} (n = {7 if g=="DSS" else 8})' if show_n else g)) for g in COLOURS]

def save_figure(fig, out, stem):
    fig.savefig(out/f'{stem}.svg')
    fig.savefig(out/f'{stem}.pdf')
    fig.savefig(out/f'{stem}.png',dpi=600)
    fig.savefig(out/f'{stem}.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(out/f'{stem}_preview.png',dpi=300)

def plot_panels(out, results):
    # Quantitative grid; two equally important associations, not a comparison
    # of correlation strength. User group colours and hollow markers retained.
    fig = plt.figure(figsize=(FIG_WIDTH_MM/25.4,99/25.4))
    axes = [fig.add_axes([.085,.22,.385,.64]),fig.add_axes([.59,.22,.385,.64])]
    for j,(ax,res) in enumerate(zip(axes,results),1):
        grid,fit,low,high = ols_curve(j)
        ax.fill_between(grid,low,high,color='#A4A4A4',alpha=.24,lw=0,zorder=1)
        ax.plot(grid,fit,color='#333333',lw=1.05,zorder=2)
        for g in COLOURS:
            keep=GROUPS==g
            ax.scatter(DATA[keep,0],DATA[keep,j],s=22,marker='o',
                facecolors='white',edgecolors=COLOURS[g],linewidths=.9,zorder=3)
        finish_axis(ax)
        ax.set_title(res['metabolite'],pad=7,fontweight='bold')
        ax.set_ylabel('Metabolomic intensity',labelpad=5)
        ax.set_ylim(0,205 if j==1 else 60)
        ax.set_yticks([0,50,100,150,200] if j==1 else [0,15,30,45,60])
        ax.text(.035,.965,stats_text(res),transform=ax.transAxes,
            ha='left',va='top',fontsize=7,linespacing=1.35)
    fig.legend(handles=group_handles(),loc='lower center',bbox_to_anchor=(.53,.06),
        ncol=2,columnspacing=2,handletextpad=.45)
    fig.text(.53,.012,'Pooled exact Spearman tests; lines and shaded bands: OLS fits and 95% mean-response CIs.',
        ha='center',va='bottom',fontsize=6)
    save_figure(fig,out,'correlation_two_panels')
    # Optional numbered copy provides a reversible row-to-point audit trail.
    label_positions={1:(.0032,29),2:(.0032,67),3:(.0032,43),7:(.0032,18),13:(.0032,8)}
    for j,ax in enumerate(axes,1):
        for i,(x,y) in enumerate(zip(DATA[:,0],DATA[:,j]),1):
            if j==1 and i in label_positions:
                ax.annotate(str(i),(x,y),xytext=label_positions[i],textcoords='data',fontsize=5.5,
                    color='#333333',ha='left',va='bottom',
                    arrowprops={'arrowstyle':'-','lw':.35,'color':'#9A9A9A'})
                continue
            if j==2 and i in (1,2,3,7):
                position={1:(.0026,12),2:(.0026,5),3:(.0026,9),7:(.0026,16)}[i]
                ax.annotate(str(i),(x,y),xytext=position,textcoords='data',fontsize=5.5,
                    color='#333333',ha='left',va='bottom',
                    arrowprops={'arrowstyle':'-','lw':.35,'color':'#9A9A9A'})
                continue
            offset=(4,4)
            if j==2 and i in (4,10,11):offset={4:(5,6),10:(-13,5),11:(5,-10)}[i]
            ax.annotate(str(i),(x,y),xytext=offset,textcoords='offset points',fontsize=5.5,
                color='#333333',ha='left',va='bottom',
                arrowprops=None)
    save_figure(fig,out,'correlation_two_panels_numbered')
    plt.close(fig)

def plot_overlay(out,results,show_n=True):
    fig=plt.figure(figsize=(110/25.4,117/25.4))
    ax=fig.add_axes([.17,.22,.78,.67])
    for j,marker,ls,col in [(1,'o','-','#333333'),(2,'^','--','#7C7C7C')]:
        grid,fit,low,high=ols_curve(j)
        ax.fill_between(grid,low,high,color=col,alpha=.12,lw=0,zorder=1)
        ax.plot(grid,fit,ls=ls,color=col,lw=1,zorder=2)
        for g in COLOURS:
            keep=GROUPS==g
            ax.scatter(DATA[keep,0],DATA[keep,j],s=20 if j==1 else 24,marker=marker,
                facecolors='white',edgecolors=COLOURS[g],linewidths=.9,zorder=3)
    finish_axis(ax);ax.set_ylabel('Metabolomic intensity',labelpad=5)
    ax.set_ylim(0,250);ax.set_yticks([0,50,100,150,200,250])
    ax.text(.04,.965,'MMIA\n'+stats_text(results[0],show_n=show_n),transform=ax.transAxes,ha='left',va='top',fontsize=6.5,linespacing=1.2)
    ax.text(.53,.965,'5-HIA\n'+stats_text(results[1],show_n=show_n),transform=ax.transAxes,ha='left',va='top',fontsize=6.5,linespacing=1.2)
    fig.legend(handles=group_handles(show_n=show_n),loc='upper center',bbox_to_anchor=(.55,.985),
        ncol=2,columnspacing=1,handletextpad=.4)
    mh=[Line2D([],[],color=c,ls=ls,lw=1,marker=m,mfc='white',mec=c,ms=4,
        label=(n if show_n else f'{n} ({"solid" if ls=="-" else "dashed"})'))
        for n,m,ls,c in [('MMIA','o','-','#333333'),('5-HIA','^','--','#7C7C7C')]]
    fig.legend(handles=mh,loc='lower center',bbox_to_anchor=(.55,.075),ncol=2,columnspacing=1.7)
    fig.text(.55,.023,'Exact Spearman P; 95% CI for rho: paired BCa bootstrap.\nLines / bands: OLS fits / 95% mean-response CIs.',
        ha='center',fontsize=5.8,linespacing=1.25)
    save_figure(fig,out,'correlation_overlay' if show_n else 'correlation_merged')
    plt.close(fig)

def write_metadata(out, results):
    with (out/'source_data.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['Sample_row','Group','D_newyorkensis_relative_abundance','MMIA','5-HIA'])
        for i,(tok,g) in enumerate(zip(TOKENS,GROUPS),1):w.writerow([i,g,*tok])
    with (out/'statistics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    (out/'statistics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    qa={'claim':'Pooled sample-level association of D. newyorkensis with MMIA and 5-HIA; no causal or stronger-correlation claim',
        'archetype':'Quantitative two-panel grid; optional overlay mirrors original plot family',
        'backend':'Python, verified saved preference', 'rows_received':15,'rows_used':15,'rows_excluded':0,
        'groups':{'DSS':7,'DSS+VPP':8},'group_assignment':'Rows 1-7 DSS; rows 8-15 DSS+VPP',
        'x_scale':'Original supplied values, no percentage conversion',
        'y_units':'Author confirmed normalized peak intensity / relative abundance; original y-axis name Metabolomic intensity retained at author request',
        'n_definition':'Paired sample observations; assumed independent; animal identifiers not supplied',
        'tests':'Two-sided exact untied Spearman null permutation distribution via integer subset dynamic programming',
        'exact_engine_validation':'Distribution for n=5 cross-checked against all 120 permutations; full n=15 distribution independently checked: sum=15!=1307674368000 and reflection symmetry; both tail counts agree with truncated computation',
        'rho_CI':'50,000 paired BCa bootstrap resamples, unstratified, fixed seeds',
        'fit':'Pooled untransformed OLS with intercept; Student-t(13) pointwise 95% CI for mean response, not prediction interval',
        'R_squared_source':'OLS coefficient of determination; never rho squared',
        'multiplicity':'Figure shows raw P values; Holm across two correlations included in statistics file only',
        'group_adjustment':'None; group-coded scatter does not remove group confounding',
        'formal_comparison_of_correlations':'Not performed',
        'reuse':'Style-only inheritance from supplied scatter references, analysis rebuilt from supplied data',
        'exports':'SVG editable text, PDF embedded TrueType, PNG/TIFF 600dpi, PNG previews 300dpi',
        'static_preflight_notes':'RNG is used only for paired resampling of real input data, never to fabricate plotted observations. Logarithm is used only to format strictly positive P values. Overlay width 110 mm is an intentional supplemental layout, main width 183 mm.',
        'software':{'Python':sys.version.split()[0],'NumPy':np.__version__,'SciPy':scipy.__version__,'Matplotlib':mpl.__version__}}
    (out/'QA.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    results=analyse();write_metadata(args.output,results)
    plot_panels(args.output,results);plot_overlay(args.output,results)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
