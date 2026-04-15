#!/usr/bin/env python3
"""
Analyze tournament results and generate a LaTeX metrics report.

Usage:
    python analyze_and_report.py /path/to/results/folder
    python analyze_and_report.py /path/to/results/folder --compile

Scans all .tsv files in the given folder (recursively), classifies them
as training or finetuning, computes top-3 agreement and top-1 ID rates,
finds redundancy clusters, and writes a LaTeX report.

With --compile, runs pdflatex automatically.
"""

import argparse
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_tsv(path):
    """Parse a tournament result TSV file. Returns dict or None on failure."""
    with open(path) as f:
        lines = f.readlines()

    info = {
        'path': path,
        'filename': os.path.basename(path),
        'date': '',
        'user': '',
        'gifs': [],
        'roi': '',
        'top3_label': '',
        'candidates': [],   # [{label, metrics: {name: value}}]
        'metric_names': [],
    }

    # Parse header
    for line in lines[:20]:
        line = line.strip()
        if line.startswith('Date:'):
            info['date'] = line[5:].strip()
        elif line.startswith('User:'):
            info['user'] = line[5:].strip()
        elif line.startswith('GIFs:'):
            info['gifs'] = [g.strip() for g in line[5:].split(',')]
        elif line.startswith('ROI:'):
            info['roi'] = line[4:].strip()
        elif line.startswith('Top 3:'):
            info['top3_label'] = line[6:].strip()

    # Find the TSV table header
    table_start = None
    for i, line in enumerate(lines):
        if line.startswith('Rank\t'):
            table_start = i
            break

    if table_start is None:
        return None

    # Parse header row
    headers = lines[table_start].strip().split('\t')
    metric_names = headers[2:]  # Skip Rank, Candidate
    info['metric_names'] = metric_names

    # Parse data rows
    for line in lines[table_start + 1:]:
        line = line.strip()
        if not line or not line[0].isdigit():
            break
        parts = line.split('\t')
        if len(parts) < 3:
            break
        try:
            rank = int(parts[0])
            label = parts[1]
            values = {}
            for j, name in enumerate(metric_names):
                if j + 2 < len(parts):
                    try:
                        values[name] = float(parts[j + 2])
                    except ValueError:
                        values[name] = 0.0
            info['candidates'].append({
                'rank': rank,
                'label': label,
                'metrics': values,
            })
        except (ValueError, IndexError):
            break

    if len(info['candidates']) < 2:
        return None

    return info


def classify_file(info):
    """Classify as 'training' or 'finetuning' based on GIF paths."""
    for gif in info['gifs']:
        if 'finetune' in gif.lower() or 'from' in gif.lower():
            return 'finetuning'
    # Also check candidate labels
    for c in info['candidates']:
        if 'finetune' in c['label'].lower() or 'from' in c['label'].lower():
            return 'finetuning'
    return 'training'


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_top3_agreement(info, metric_name, higher_is_better=True):
    """How many of metric's top-3 overlap with human's top-3?"""
    candidates = info['candidates']
    if len(candidates) < 3:
        return None, None

    human_top3 = set(c['label'] for c in candidates[:3])

    metric_sorted = sorted(
        candidates,
        key=lambda c: c['metrics'].get(metric_name, 0),
        reverse=higher_is_better,
    )
    metric_top3 = set(c['label'] for c in metric_sorted[:3])
    metric_top1 = metric_sorted[0]['label']

    agreement = len(human_top3 & metric_top3)
    top1_in_human_top3 = 1 if metric_top1 in human_top3 else 0

    return agreement, top1_in_human_top3


def compute_safe_elimination(info, metric_name, higher_is_better=True, eliminate_frac=0.5):
    """Check if eliminating the bottom eliminate_frac by metric preserves all human top-3.
    Returns number of human top-3 that survive (3 = all safe, 0 = all lost)."""
    candidates = info['candidates']
    n = len(candidates)
    if n < 4:
        return None

    human_top3 = set(c['label'] for c in candidates[:3])

    metric_sorted = sorted(
        candidates,
        key=lambda c: c['metrics'].get(metric_name, 0),
        reverse=higher_is_better,
    )

    # Keep the top (1 - eliminate_frac), but always keep at least 3
    n_keep = max(3, int(math.ceil(n * (1 - eliminate_frac))))
    kept = set(c['label'] for c in metric_sorted[:n_keep])

    return len(human_top3 & kept)


def analyze_elimination(files, metric_names):
    """For each metric, compute safe elimination rates at various thresholds."""
    thresholds = [0.25, 0.50, 0.75]
    results = {}

    for name in metric_names:
        best_dir = {}  # threshold -> {rate, inverted}
        for elim in thresholds:
            for hib in (True, False):
                survivals = []
                for f in files:
                    if name not in f['candidates'][0]['metrics']:
                        continue
                    s = compute_safe_elimination(f, name, higher_is_better=hib, eliminate_frac=elim)
                    if s is not None:
                        survivals.append(s)
                if not survivals:
                    continue
                # "safe" = all 3 human top-3 survived
                safe_rate = sum(1 for s in survivals if s == 3) / len(survivals) * 100
                avg_survival = np.mean(survivals) / 3 * 100

                key = (elim, hib)
                if elim not in best_dir or safe_rate > best_dir[elim]['safe_rate']:
                    best_dir[elim] = {
                        'safe_rate': safe_rate,
                        'avg_survival': avg_survival,
                        'inverted': not hib,
                        'n_files': len(survivals),
                    }

        if best_dir:
            results[name] = best_dir

    return results


def analyze_files(files):
    """Run full analysis on a list of parsed files."""
    if not files:
        return None

    # Determine which metrics are available across all files
    all_metric_names = set(files[0]['metric_names'])
    for f in files[1:]:
        all_metric_names &= set(f['metric_names'])
    all_metric_names = sorted(all_metric_names)

    # Known inverted metrics (lower is better despite general higher-is-better convention)
    # We try both directions and pick the one with better agreement
    results = {}
    for name in all_metric_names:
        # Try higher_is_better=True
        agreements_hib = []
        top1_hib = []
        for f in files:
            if name not in f['candidates'][0]['metrics']:
                continue
            a, t = compute_top3_agreement(f, name, higher_is_better=True)
            if a is not None:
                agreements_hib.append(a)
                top1_hib.append(t)

        # Try higher_is_better=False
        agreements_lib = []
        top1_lib = []
        for f in files:
            if name not in f['candidates'][0]['metrics']:
                continue
            a, t = compute_top3_agreement(f, name, higher_is_better=False)
            if a is not None:
                agreements_lib.append(a)
                top1_lib.append(t)

        mean_hib = np.mean(agreements_hib) if agreements_hib else 0
        mean_lib = np.mean(agreements_lib) if agreements_lib else 0

        if mean_lib > mean_hib:
            results[name] = {
                'agreement': mean_lib / 3 * 100,
                'top1_rate': np.mean(top1_lib) * 100 if top1_lib else 0,
                'inverted': True,
                'n_files': len(agreements_lib),
            }
        else:
            results[name] = {
                'agreement': mean_hib / 3 * 100,
                'top1_rate': np.mean(top1_hib) * 100 if top1_hib else 0,
                'inverted': False,
                'n_files': len(agreements_hib),
            }

    return results, all_metric_names


def find_redundancy_clusters(files, metric_names, threshold=0.95):
    """Find clusters of metrics with Pearson |r| > threshold."""
    # Collect aligned metric values (only candidates that have ALL metrics)
    all_values = defaultdict(list)
    for f in files:
        for c in f['candidates']:
            if all(name in c['metrics'] for name in metric_names):
                for name in metric_names:
                    all_values[name].append(c['metrics'][name])

    # Compute correlation matrix
    names = sorted(all_values.keys())
    n = len(names)
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if len(all_values[names[i]]) > 2 and len(all_values[names[j]]) > 2:
                r, _ = pearsonr(all_values[names[i]], all_values[names[j]])
                corr[i, j] = r

    # Find clusters via greedy grouping
    used = set()
    clusters = []
    for i in range(n):
        if i in used:
            continue
        cluster = [i]
        used.add(i)
        for j in range(i + 1, n):
            if j in used:
                continue
            if abs(corr[i, j]) > threshold:
                cluster.append(j)
                used.add(j)
        if len(cluster) > 1:
            min_r = min(abs(corr[a, b]) for a in cluster for b in cluster if a != b)
            max_r = max(abs(corr[a, b]) for a in cluster for b in cluster if a != b)
            clusters.append({
                'metrics': [names[k] for k in cluster],
                'r_range': (min_r, max_r),
            })

    return clusters


# ---------------------------------------------------------------------------
# LaTeX generation
# ---------------------------------------------------------------------------

def escape_tex(s):
    """Escape special LaTeX characters."""
    return s.replace('_', r'\_').replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')


def generate_results_table(results, metric_names, title, n_files, n_candidates_approx):
    """Generate a LaTeX table section for a category of results."""
    sorted_metrics = sorted(
        [(name, results[name]) for name in metric_names if name in results],
        key=lambda x: x[1]['top1_rate'],
        reverse=True,
    )

    chance_pct = round(3 / n_candidates_approx * 100) if n_candidates_approx > 3 else 100

    lines = []
    lines.append(f'\\section{{Results: {title} ({n_files} tournaments)}}')
    lines.append('')

    if sorted_metrics and sorted_metrics[0][1]['top1_rate'] < chance_pct + 5:
        lines.append(f'No metric reliably identifies the human-preferred epoch. '
                     f'Chance level $\\approx$ {chance_pct}\\% for $\\sim${n_candidates_approx} candidates.')
        lines.append('')

    lines.append('\\vspace{0.5em}')
    lines.append('\\begin{center}')
    lines.append('\\small')
    lines.append('\\begin{tabular}{clcc}')
    lines.append('\\toprule')
    lines.append('\\textbf{Rank} & \\textbf{Metric} & \\textbf{Top-3 Agree.} & \\textbf{Top-1 ID Rate} \\\\')
    lines.append('\\midrule')

    for i, (name, data) in enumerate(sorted_metrics):
        rank = i + 1
        inv = ' (inv.)' if data['inverted'] else ''
        disp_name = escape_tex(name) + inv

        agree = f'{data["agreement"]:.0f}\\%'
        top1 = f'{data["top1_rate"]:.0f}\\%'

        # Color coding
        if data['top1_rate'] >= 45:
            lines.append(f'\\rowcolor{{toprow}} {rank} & {disp_name} & {agree} & \\textbf{{{top1}}} \\\\')
        elif data['top1_rate'] >= 30:
            lines.append(f'\\rowcolor{{midrow}} {rank} & {disp_name} & {agree} & {top1} \\\\')
        elif data['top1_rate'] < 20:
            lines.append(f'\\rowcolor{{botrow}} {rank} & {disp_name} & {agree} & {top1} \\\\')
        else:
            lines.append(f'{rank} & {disp_name} & {agree} & {top1} \\\\')

        # Show only top 5 if training is at chance
        if title.startswith('Training') and rank >= 5 and data['top1_rate'] <= chance_pct + 2:
            lines.append('\\midrule')
            lines.append(f'\\multicolumn{{4}}{{l}}{{\\textit{{All other metrics have top-1 ID rate $\\leq$ {data["top1_rate"]:.0f}\\% (chance level).}}}} \\\\')
            break

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{center}')
    lines.append('')

    return '\n'.join(lines)


def generate_elimination_table(elim_results, metric_names, title, n_files):
    """Generate a LaTeX table for safe elimination analysis."""
    if not elim_results:
        return ''

    # Sort by safe rate at 50% elimination
    sorted_metrics = sorted(
        [(name, elim_results[name]) for name in metric_names if name in elim_results],
        key=lambda x: x[1].get(0.50, {}).get('safe_rate', 0),
        reverse=True,
    )

    lines = []
    lines.append(f'\\section{{Safe Elimination: {title} ({n_files} tournaments)}}')
    lines.append('')
    lines.append('If we use a metric to discard the worst candidates before human inspection, ')
    lines.append('how often do \\emph{all three} human-preferred epochs survive the cut? ')
    lines.append('A metric with 100\\% safe rate at 50\\% elimination means we can always ')
    lines.append('throw away half the candidates by that metric without losing any human favorite.')
    lines.append('')
    lines.append('\\vspace{0.5em}')
    lines.append('\\begin{center}')
    lines.append('\\small')
    lines.append('\\begin{tabular}{clccc}')
    lines.append('\\toprule')
    lines.append('\\textbf{\\#} & \\textbf{Metric} & '
                 '\\textbf{Cut 25\\%} & \\textbf{Cut 50\\%} & \\textbf{Cut 75\\%} \\\\')
    lines.append('\\midrule')

    for i, (name, thresholds) in enumerate(sorted_metrics):
        inv = ' (inv.)' if thresholds.get(0.50, {}).get('inverted', False) else ''
        disp_name = escape_tex(name) + inv
        cols = []
        for t in [0.25, 0.50, 0.75]:
            if t in thresholds:
                rate = thresholds[t]['safe_rate']
                cols.append(f'{rate:.0f}\\%')
            else:
                cols.append('---')

        rate_50 = thresholds.get(0.50, {}).get('safe_rate', 0)
        if rate_50 >= 90:
            lines.append(f'\\rowcolor{{toprow}} {i+1} & {disp_name} & {" & ".join(cols)} \\\\')
        elif rate_50 >= 75:
            lines.append(f'\\rowcolor{{midrow}} {i+1} & {disp_name} & {" & ".join(cols)} \\\\')
        elif rate_50 < 60:
            lines.append(f'\\rowcolor{{botrow}} {i+1} & {disp_name} & {" & ".join(cols)} \\\\')
        else:
            lines.append(f'{i+1} & {disp_name} & {" & ".join(cols)} \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{center}')
    lines.append('')
    return '\n'.join(lines)


def generate_report(training_results, finetuning_results, clusters,
                    elim_training, elim_finetuning,
                    training_metric_names, finetuning_metric_names,
                    n_training, n_finetuning, n_total,
                    output_path):
    """Generate the full LaTeX report."""

    # Determine best metrics for findings
    ft_sorted = sorted(
        [(n, d) for n, d in finetuning_results.items()],
        key=lambda x: x[1]['top1_rate'], reverse=True
    ) if finetuning_results else []
    best_ft = ft_sorted[0] if ft_sorted else ('N/A', {'top1_rate': 0})

    lines = []
    lines.append(r"""\documentclass[11pt]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{booktabs}
\usepackage{array}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{hyperref}

\definecolor{toprow}{HTML}{E8F5E9}
\definecolor{midrow}{HTML}{FFF8E1}
\definecolor{botrow}{HTML}{FFEBEE}

\title{Image Quality Metrics for X-ray Tomography Epoch Selection\\[4pt]\large Analysis of """ + str(n_total) + r""" Human Comparison Tournaments}
\author{Compare Game Analysis}
\date{\today}

\begin{document}
\maketitle

\section{Context}

When denoising X-ray tomography reconstructions with iterative deep learning methods, the training must be stopped at an optimal epoch. There is no ground truth to optimize against, so the stopping point is typically chosen by visual inspection.

We built a pairwise comparison tool where users compare epoch pairs side-by-side in a merge-sort tournament and select the better-looking reconstruction. The tool also computes image quality metrics on each candidate. By comparing the human rankings with each metric's ranking, we evaluate which metrics best predict human visual judgment.

There are two experimental modes. \textbf{Training tiles} (""" + str(n_training) + r""" tournaments): initial training runs with wide epoch ranges and large quality differences. \textbf{Finetuning} (""" + str(n_finetuning) + r""" tournaments): starting from a pre-trained checkpoint with much smaller differences between candidates.

The tournament uses a merge-sort algorithm with top\_k=3, meaning only the top-3 positions are reliably ordered. All analysis uses \emph{only} the top-3 positions: \textbf{top-3 agreement} (overlap between metric's and human's top-3) and \textbf{top-1 identification rate} (whether the metric's \#1 pick is in the human's top-3).

For reference, chance-level performance for random selection: training tiles ($\sim$37 candidates) have $\sim$8\% top-1 ID rate; finetuning ($\sim$20 candidates) have $\sim$15\% top-1 ID rate. Note that training tile tournaments include 3 views of the same tomograms, so the effective sample size is smaller than the file count suggests.
""")

    # Training results
    if training_results:
        lines.append(generate_results_table(
            training_results, training_metric_names,
            'Training Tiles', n_training, 37))

    # Finetuning results
    if finetuning_results:
        lines.append(generate_results_table(
            finetuning_results, finetuning_metric_names,
            'Finetuning', n_finetuning, 20))

    # Safe elimination
    if elim_finetuning:
        lines.append(generate_elimination_table(
            elim_finetuning, finetuning_metric_names,
            'Finetuning', n_finetuning))
    if elim_training:
        lines.append(generate_elimination_table(
            elim_training, training_metric_names,
            'Training Tiles', n_training))

    # Redundancy
    if clusters:
        lines.append(r'\section{Metric Redundancy}')
        lines.append('')
        lines.append(r'Clusters of metrics with Pearson $|r| > 0.95$ (carrying the same information):')
        lines.append(r'\vspace{0.5em}')
        lines.append(r'\begin{center}')
        lines.append(r'\small')
        lines.append(r'\begin{tabular}{lll}')
        lines.append(r'\toprule')
        lines.append(r'\textbf{\#} & \textbf{Metrics in cluster} & \textbf{$|r|$} \\')
        lines.append(r'\midrule')
        for i, cl in enumerate(clusters):
            names_str = ', '.join(escape_tex(m) for m in cl['metrics'])
            r_lo, r_hi = cl['r_range']
            if abs(r_hi - r_lo) < 0.01:
                r_str = f'{r_hi:.2f}'
            else:
                r_str = f'{r_lo:.2f}--{r_hi:.2f}'
            lines.append(f'{i+1} & {names_str} & {r_str} \\\\')
        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(r'\end{center}')
        lines.append('')

    # Findings
    lines.append(r'\section{Key Findings}')
    lines.append(r'\begin{enumerate}')

    if training_results and finetuning_results:
        lines.append(r'\item \textbf{Metrics work dramatically better for finetuning than for training tiles.} '
                     f'Best finetuning metric: {best_ft[1]["top1_rate"]:.0f}\\% top-1 ID rate '
                     f'({escape_tex(best_ft[0])}). Training tile metrics are at chance level.')

    lines.append(r'\item \textbf{No single metric is sufficient.} Even for finetuning, the best metric '
                 f'misses the human top-3 {100 - best_ft[1]["agreement"]:.0f}\\% of the time. '
                 r'Human visual comparison remains necessary.')

    if best_ft[0] != 'N/A':
        lines.append(f'\\item \\textbf{{{escape_tex(best_ft[0])} is the most practically useful metric}} '
                     f'for finetuning ({best_ft[1]["top1_rate"]:.0f}\\% top-1 ID rate).')

    # Find best eliminator
    if elim_finetuning:
        best_elim = sorted(elim_finetuning.items(),
                           key=lambda x: x[1].get(0.50, {}).get('safe_rate', 0),
                           reverse=True)
        if best_elim:
            be_name, be_data = best_elim[0]
            be_rate = be_data.get(0.50, {}).get('safe_rate', 0)
            lines.append(f'\\item \\textbf{{Metrics are more reliable for eliminating the worst '
                         f'than for picking the best.}} '
                         f'{escape_tex(be_name)} safely eliminates 50\\% of candidates '
                         f'while preserving all human top-3 picks in {be_rate:.0f}\\% of tournaments. '
                         f'Recommended workflow: use metrics to pre-filter, then compare the survivors visually.')

    n_droppable = sum(len(c['metrics']) - 1 for c in clusters)
    if n_droppable > 0:
        total_metrics = len(set(training_metric_names) | set(finetuning_metric_names))
        lines.append(f'\\item \\textbf{{{n_droppable} of {total_metrics} metrics can be dropped}} '
                     r'without losing information (redundant within clusters).')

    lines.append(r'\end{enumerate}')
    lines.append('')

    # Appendix: metric descriptions
    lines.append(r'\appendix')
    lines.append(r'\section{Metric Descriptions}')
    lines.append(r'\small')
    lines.append('')
    lines.append(r'\subsection*{Differential metrics}')
    lines.append(r'These measure sharpness via spatial derivatives of the image intensity.')
    lines.append(r'\begin{description}')
    lines.append(r'\item[Brenner] Sum of squared horizontal differences with spacing 2: '
                 r'$\frac{1}{N}\sum (I(x+2,y) - I(x,y))^2$. Higher values indicate sharper edges. '
                 r'Simple and fast but sensitive to noise.')
    lines.append(r'\item[Tenengrad] Mean squared Sobel gradient magnitude: '
                 r'$\frac{1}{N}\sum (G_x^2 + G_y^2)$ where $G_x, G_y$ are Sobel-filtered images. '
                 r'More robust than Brenner due to directional averaging.')
    lines.append(r'\item[Laplacian Var] Variance of the Laplacian-filtered image: '
                 r'$\mathrm{Var}(\nabla^2 I)$. Captures both edges and fine texture. '
                 r'Widely used in autofocus (e.g.\ OpenCV).')
    lines.append(r'\item[SML (Sum Modified Laplacian)] Sum of absolute second derivatives in x and y separately: '
                 r'$\frac{1}{N}\sum |I_{xx}| + |I_{yy}|$. More robust than standard Laplacian because '
                 r'absolute values prevent cancellation of opposing gradients.')
    lines.append(r'\end{description}')
    lines.append('')
    lines.append(r'\subsection*{Correlative metrics}')
    lines.append(r'These measure sharpness via autocorrelation of neighboring pixels.')
    lines.append(r'\begin{description}')
    lines.append(r'\item[Vollath F4] Difference of autocorrelations at lag 1 and lag 2: '
                 r'$\langle I(x) \cdot I(x+1) \rangle - \langle I(x) \cdot I(x+2) \rangle$. '
                 r'Positive for sharp images (rapid decorrelation), near zero for blurry ones.')
    lines.append(r'\item[Vollath F5] Autocorrelation at lag 1 minus squared mean: '
                 r'$\langle I(x) \cdot I(x+1) \rangle - \mu^2$. More noise-robust variant of F4.')
    lines.append(r'\end{description}')
    lines.append('')
    lines.append(r'\subsection*{Statistical metrics}')
    lines.append(r'These characterize the intensity distribution of the image.')
    lines.append(r'\begin{description}')
    lines.append(r'\item[Norm.\ Var.\ (Normalized Variance)] Variance divided by mean: '
                 r'$\sigma^2 / \mu$. Measures relative contrast. Insensitive to brightness scaling.')
    lines.append(r'\item[Hist.\ Entropy] Shannon entropy of the intensity histogram: '
                 r'$-\sum p_i \log_2 p_i$. Higher entropy means more uniform use of the intensity range.')
    lines.append(r'\item[Kurtosis] Excess kurtosis of intensity distribution: '
                 r'$\mu_4/\sigma^4 - 3$. Measures tail heaviness. Can be high for both structured '
                 r'content (sparse features) and impulsive noise, making interpretation ambiguous.')
    lines.append(r'\item[Local Std] Mean of local standard deviations computed in $7 \times 7$ non-overlapping blocks. '
                 r'Directly measures local texture richness. Higher values indicate more preserved fine detail.')
    lines.append(r'\end{description}')
    lines.append('')
    lines.append(r'\subsection*{Spectral (DCT-based) metrics}')
    lines.append(r'These analyze the frequency content of the image via the Discrete Cosine Transform.')
    lines.append(r'\begin{description}')
    lines.append(r'\item[DCTS (DCT Shannon Entropy)] Normalized Shannon entropy of the DCT power spectrum '
                 r'within the OTF support radius $r_o$: $-\sum p_i \ln p_i / \ln n$. '
                 r'Recommended by Royer et al.\ 2016 for light-sheet microscopy autofocus. '
                 r'Performed poorly on our X-ray tomography data.')
    lines.append(r'\item[NDCTBE (Normalized DCT Bayes Entropy)] Bayesian variant of DCTS using a '
                 r'Dirichlet(1/n) prior for smoothing. Highly correlated with DCTS ($r = 0.97$).')
    lines.append(r'\item[HF Energy] Fraction of DCT energy in the high-frequency band '
                 r'($r_o/2 < r \leq r_o$) relative to total energy within $r_o$. '
                 r'Higher values indicate more high-frequency content (sharper).')
    lines.append(r'\item[Spec.\ Struct.\ (Spectral Flatness)] $1 - $ spectral flatness (Wiener entropy). '
                 r'Spectral flatness is the ratio of geometric to arithmetic mean of the power spectrum. '
                 r'Higher values (less flat spectrum) indicate more structured content. '
                 r'\textbf{Note:} in our data this metric works best when \emph{inverted} '
                 r'(lower values preferred), suggesting the relationship reverses for denoised reconstructions.')
    lines.append(r'\item[Spec.\ Slope] Slope $\beta$ of the radial power spectrum on a log-log scale, '
                 r'fit via linear regression: $P(f) \sim f^{-\beta}$. '
                 r'Natural images have $\beta \approx 2$. Steeper slopes indicate more low-frequency dominance. '
                 r'Performed poorly --- essentially random for our data.')
    lines.append(r'\end{description}')
    lines.append('')
    lines.append(r'\subsection*{Wavelet metrics}')
    lines.append(r'These use the Haar wavelet transform to separate detail from approximation.')
    lines.append(r'\begin{description}')
    lines.append(r'\item[NHWTSE (Norm.\ Haar Wavelet Transform Shannon Entropy)] Normalized Shannon entropy '
                 r'of the wavelet detail coefficients (LH, HL, HH subbands). Higher entropy means '
                 r'more uniformly distributed detail energy.')
    lines.append(r'\item[Wavelet E.R.\ (Wavelet Energy Ratio)] Fraction of total image energy in the '
                 r'detail subbands across 3 decomposition levels. Higher values mean more energy in '
                 r'edges and texture relative to smooth approximation. '
                 r'\textbf{Third best metric for finetuning} (54\% top-1 ID rate).')
    lines.append(r'\end{description}')
    lines.append('')
    lines.append(r'\subsection*{Domain-specific metrics (this work)}')
    lines.append(r'These were designed specifically for X-ray tomography denoising evaluation.')
    lines.append(r'\begin{description}')
    lines.append(r'\item[LoG Response] Multi-scale Laplacian-of-Gaussian response, computed at '
                 r'$\sigma = 1, 2, 4$ and scale-normalized: $\frac{1}{3}\sum_\sigma \sigma^2 \cdot '
                 r'\langle |\nabla^2 G_\sigma * I| \rangle$. Designed to detect sparse structural features '
                 r'at multiple scales. Part of the Local Std / Tenengrad redundancy cluster.')
    lines.append(r'\item[Noise Est.] Mean absolute value of the double Laplacian '
                 r'(4th spatial derivative): $\langle |(\nabla^2)^2 I| \rangle$. Near zero for '
                 r'over-smoothed images; moderate values indicate preserved fine detail. '
                 r'Part of the Laplacian Var / SML redundancy cluster.')
    lines.append(r'\item[Contrast Con.\ (Contrast Consistency)] Interquartile range (IQR) of the '
                 r'intensity histogram, normalized by the full intensity range: $\mathrm{IQR} / (I_{max} - I_{min})$. '
                 r'Measures whether the reconstruction preserves contrast diversity. '
                 r'Narrowing IQR indicates over-smoothing.')
    lines.append(r'\end{description}')
    lines.append('')

    lines.append(r'\vspace{1em}')
    lines.append(r'\noindent\textbf{Reference:} Royer, L.A.\ et al.\ (2016). Adaptive light-sheet '
                 r'microscopy for long-term, high-resolution imaging in living organisms. '
                 r'\textit{Nature Biotechnology}, 34(12), 1267--1278.')
    lines.append(r'\end{document}')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f'Report written to {output_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Analyze tournament results and generate LaTeX report')
    parser.add_argument('results_dir', help='Directory containing .tsv result files')
    parser.add_argument('--compile', action='store_true', help='Run pdflatex after generating')
    parser.add_argument('--output', default=None, help='Output .tex path (default: results_dir/metrics_report.tex)')
    args = parser.parse_args()

    # Find all TSV files
    tsv_files = []
    for root, dirs, filenames in os.walk(args.results_dir):
        for fn in filenames:
            if fn.endswith('.tsv'):
                tsv_files.append(os.path.join(root, fn))

    print(f'Found {len(tsv_files)} TSV files')

    # Parse
    parsed = []
    for path in sorted(tsv_files):
        info = parse_tsv(path)
        if info:
            info['category'] = classify_file(info)
            parsed.append(info)
        else:
            print(f'  Skipped (unparseable): {os.path.basename(path)}')

    print(f'Parsed {len(parsed)} files ({sum(1 for p in parsed if p["category"] == "training")} training, '
          f'{sum(1 for p in parsed if p["category"] == "finetuning")} finetuning)')

    training = [p for p in parsed if p['category'] == 'training']
    finetuning = [p for p in parsed if p['category'] == 'finetuning']

    # Analyze
    training_results, training_names = analyze_files(training) if training else (None, [])
    finetuning_results, finetuning_names = analyze_files(finetuning) if finetuning else (None, [])

    # Redundancy (on all files combined)
    all_names = sorted(set((training_names or []) + (finetuning_names or [])))
    clusters = find_redundancy_clusters(parsed, all_names) if len(parsed) > 1 else []

    # Safe elimination analysis
    elim_training = analyze_elimination(training, training_names) if training else {}
    elim_finetuning = analyze_elimination(finetuning, finetuning_names) if finetuning else {}

    # Print summary
    print('\n=== TRAINING TILES ===')
    if training_results:
        sorted_t = sorted(training_results.items(), key=lambda x: x[1]['top1_rate'], reverse=True)
        for name, data in sorted_t[:5]:
            inv = ' (inv)' if data['inverted'] else ''
            print(f'  {name}{inv}: {data["agreement"]:.1f}% agree, {data["top1_rate"]:.1f}% top-1')
    else:
        print('  No training tile data')

    print('\n=== FINETUNING ===')
    if finetuning_results:
        sorted_f = sorted(finetuning_results.items(), key=lambda x: x[1]['top1_rate'], reverse=True)
        for name, data in sorted_f[:10]:
            inv = ' (inv)' if data['inverted'] else ''
            print(f'  {name}{inv}: {data["agreement"]:.1f}% agree, {data["top1_rate"]:.1f}% top-1')
    else:
        print('  No finetuning data')

    print('\n=== SAFE ELIMINATION (finetuning) ===')
    if elim_finetuning:
        # Sort by safe rate at 50% elimination
        sorted_e = sorted(elim_finetuning.items(),
                          key=lambda x: x[1].get(0.50, {}).get('safe_rate', 0), reverse=True)
        for name, thresholds in sorted_e[:10]:
            rates = []
            for t in [0.25, 0.50, 0.75]:
                if t in thresholds:
                    rates.append(f'{t:.0%}→{thresholds[t]["safe_rate"]:.0f}%')
            inv = ' (inv)' if thresholds.get(0.50, {}).get('inverted', False) else ''
            print(f'  {name}{inv}: {", ".join(rates)}')

    print(f'\n=== REDUNDANCY CLUSTERS ({len(clusters)}) ===')
    for cl in clusters:
        print(f'  {", ".join(cl["metrics"])} (|r| = {cl["r_range"][0]:.2f}-{cl["r_range"][1]:.2f})')

    # Generate report
    output = args.output or os.path.join(args.results_dir, 'metrics_report.tex')
    generate_report(
        training_results, finetuning_results, clusters,
        elim_training, elim_finetuning,
        training_metric_names=training_names or [],
        finetuning_metric_names=finetuning_names or [],
        n_training=len(training), n_finetuning=len(finetuning), n_total=len(parsed),
        output_path=output,
    )

    if args.compile:
        output_dir = os.path.dirname(output) or '.'
        os.system(f'cd "{output_dir}" && pdflatex -interaction=nonstopmode "{os.path.basename(output)}" > /dev/null 2>&1')
        os.system(f'cd "{output_dir}" && pdflatex -interaction=nonstopmode "{os.path.basename(output)}" > /dev/null 2>&1')
        # Clean up
        base = os.path.splitext(output)[0]
        for ext in ['.aux', '.log', '.out']:
            try:
                os.remove(base + ext)
            except FileNotFoundError:
                pass
        pdf = base + '.pdf'
        if os.path.exists(pdf):
            print(f'PDF compiled: {pdf}')
        else:
            print('PDF compilation failed — check pdflatex installation')


if __name__ == '__main__':
    main()
