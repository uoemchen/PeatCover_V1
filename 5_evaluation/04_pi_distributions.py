"""Distributions of the PEATMAP peat fraction and of the Peatland Index (PI) at peat and non-peat field points.

Points (data/evaluation/pi_distribution_points.csv.gz): the Peat-DBase points (one point per 90 m pixel), with
  peatmap_fraction_pct  Canada: PEAT_AREA / AREA (%) of the PEATMAP Canada polygon that holds the point
  pi_revised            Canada: PI of the regression network inside the PEATMAP polygons; Eurasia: classification PI
  pi_submitted          the same for the submitted map
(values read at the pixel that contains each point; empty = outside the PEATMAP polygons, open water or no data).
For each variable: number, mean and standard deviation at peat and non-peat points, and the Kolmogorov-Smirnov statistic
D = max |F_peat - F_nonpeat| (scipy ks_2samp); one figure per panel with Gaussian kernel densities (bandwidth factor 0.15).
Output: results/tables/pi_distributions_ks.csv, results/figures/pdf_<panel>.png
Usage:  python 04_pi_distributions.py [--points data/evaluation/pi_distribution_points.csv.gz]
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, ks_2samp

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
VARIABLES = [('Canada', 'peatmap_fraction_pct', 'PEATMAP peat fraction (%)'),
             ('Canada', 'pi_submitted', 'PI, submitted map'), ('Canada', 'pi_revised', 'PI, revised map'),
             ('Eurasia', 'pi_submitted', 'PI, submitted map'), ('Eurasia', 'pi_revised', 'PI, revised map')]
PANELS = [('a_canada_peatmap', 'Canada', 'peatmap_fraction_pct', 'PEATMAP Fraction'),
          ('b_canada_pi', 'Canada', 'pi_revised', 'Peatland Index'), ('c_eurasia_pi', 'Eurasia', 'pi_revised', 'Peatland Index')]
COLOURS = {0: '#0072B2', 1: '#E69F00'}           # non-peat, peat


def panel(d, col, xlabel, out):
    fig, ax = plt.subplots(figsize=(2.4, 2.2))
    xs = np.linspace(0, 100, 401)
    x = d[[col, 'label']].dropna()
    top = 0
    for lab, name in ((0, 'Non-Peat'), (1, 'Peat')):
        v = x[x.label == lab][col].values
        y = gaussian_kde(v, bw_method=0.15)(xs)
        top = max(top, y.max())
        ax.fill_between(xs, y, color=COLOURS[lab], alpha=0.3, lw=0)
        ax.plot(xs, y, color=COLOURS[lab], lw=1.2, label=f'{name} (n={len(v):,})')
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, top * 1.3)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel('Probability Density', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(loc='upper right', fontsize=6, frameon=True, edgecolor='0.8', fancybox=False)
    fig.tight_layout()
    fig.savefig(out, dpi=400)
    plt.close(fig)


def main(points):
    P = pd.read_csv(points)
    rows = []
    for region, col, what in VARIABLES:
        x = P[P.region == region][[col, 'label']].dropna()
        p, q = x[x.label == 1][col], x[x.label == 0][col]
        rows.append({'region': region, 'variable': what, 'n_peat': len(p), 'n_nonpeat': len(q), 'mean_peat': p.mean(),
                     'sd_peat': p.std(), 'mean_nonpeat': q.mean(), 'sd_nonpeat': q.std(), 'KS_D': ks_2samp(p, q).statistic})
    K = pd.DataFrame(rows)
    tab, fig = REPO / 'results/tables', REPO / 'results/figures'
    tab.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    K.to_csv(tab / 'pi_distributions_ks.csv', index=False, float_format='%.3f')
    print(K.to_string(index=False, float_format='%.3f'))
    for name, region, col, xlabel in PANELS:
        panel(P[P.region == region], col, xlabel, fig / f'pdf_{name}.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--points', default=str(REPO / 'data/evaluation/pi_distribution_points.csv.gz'))
    main(ap.parse_args().points)
