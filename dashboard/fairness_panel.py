"""
fairness_panel.py
-----------------
Builds fairness audit visualizations for the Responsible AI tab.

Components:
  - build_fairness_banner_html() : Big status banner (FAIR / BIAS DETECTED)
  - build_metrics_html()         : 5 metrics with pass/fail indicators
  - build_tardiness_chart()      : Grouped bar chart by product type
  - build_energy_chart()         : Pie chart of energy share
  - build_wait_time_chart()      : Bar chart of avg wait times
  - load_fairness_data()         : Load from logs/fairness_report.json
"""

import json
import os
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────
#  COLORS
# ─────────────────────────────────────────────────────────────────

TYPE_COLORS = {
    'A': '#2563EB',   # Blue
    'B': '#059669',   # Green
    'C': '#D97706',   # Amber
}

PASS_COLOR = '#059669'  # Green
FAIL_COLOR = '#DC2626'  # Red

PASS_BG    = '#ECFDF5'
FAIL_BG    = '#FEF2F2'


# ─────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────

def load_fairness_data(report_path: str = 'logs/fairness_report.json') -> dict:
    """
    Load fairness report from JSON file.
    Returns empty dict if file doesn't exist.
    """
    if not os.path.exists(report_path):
        return {}
    try:
        with open(report_path) as f:
            return json.load(f)
    except Exception:
        return {}


def load_audit_csv(csv_path: str = 'logs/fairness_audit.csv') -> pd.DataFrame:
    """
    Load per-job fairness audit CSV.
    Returns empty DataFrame if file doesn't exist.
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────
#  FAIRNESS BANNER
# ─────────────────────────────────────────────────────────────────

def build_fairness_banner_html(report: dict) -> str:
    """
    Build a large, visually prominent fairness status banner.
    Green for FAIR, Red for BIAS DETECTED.
    """
    if not report:
        return _placeholder_banner()

    status    = report.get('fairness_status', 'N/A')
    n_eps     = report.get('episodes_evaluated', 0)
    n_jobs    = report.get('total_jobs_audited', 0)
    flags     = report.get('flags', [])

    if status == 'FAIR':
        icon        = '✅'
        title       = 'SYSTEM IS FAIR'
        subtitle    = 'All 5 fairness checks passed — no systematic bias detected'
        bg_gradient = 'linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%)'
        border_col  = '#059669'
        text_col    = '#065F46'
        badge_bg    = '#059669'
    elif status == 'BIAS_DETECTED':
        icon        = '⚠️'
        title       = 'BIAS DETECTED'
        subtitle    = f'{len(flags)} fairness check(s) failed — review flags below'
        bg_gradient = 'linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%)'
        border_col  = '#DC2626'
        text_col    = '#991B1B'
        badge_bg    = '#DC2626'
    else:
        return _placeholder_banner()

    return (
        f'<div style="background:{bg_gradient};border:2px solid {border_col}33;'
        f'border-radius:16px;padding:20px 24px;text-align:center;'
        f'box-shadow:0 4px 12px rgba(0,0,0,0.08);margin-bottom:16px">'
        f'<div style="font-size:36px;margin-bottom:6px">{icon}</div>'
        f'<div style="font-size:24px;font-weight:800;color:{text_col};'
        f'letter-spacing:1px;margin-bottom:6px">{title}</div>'
        f'<div style="font-size:13px;color:{text_col};opacity:0.85;'
        f'margin-bottom:12px">{subtitle}</div>'
        f'<div style="display:flex;gap:16px;justify-content:center">'
        f'<div style="background:white;border-radius:8px;padding:6px 16px;'
        f'font-size:12px;color:#475569">'
        f'📊 <b style="color:{text_col}">{n_eps}</b> episodes evaluated</div>'
        f'<div style="background:white;border-radius:8px;padding:6px 16px;'
        f'font-size:12px;color:#475569">'
        f'💼 <b style="color:{text_col}">{n_jobs}</b> jobs audited</div>'
        f'</div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────
#  METRICS TABLE
# ─────────────────────────────────────────────────────────────────

def build_metrics_html(report: dict) -> str:
    """
    Build a styled table of all 5 fairness metrics with pass/fail status.
    """
    if not report:
        return _placeholder_metrics()

    metrics    = report.get('metrics', {})
    thresholds = report.get('thresholds', {})
    pass_fail  = report.get('pass_fail', {})
    flags      = report.get('flags', [])

    METRIC_DEFS = [
        {
            'key':       'tardiness_std_dev',
            'label':     'Tardiness Std Dev',
            'icon':      '📉',
            'pf_key':    'tardiness_std_dev',
            'threshold': f"< {thresholds.get('tardiness_std_dev', 0.15):.2f}",
            'desc':      'Equal late rates across Product types A, B, C',
            'value_fmt': lambda v: f'{v:.4f}' if v else 'N/A',
        },
        {
            'key':       'max_disparity_ratio',
            'label':     'Max Disparity Ratio',
            'icon':      '⚖️',
            'pf_key':    'max_disparity_ratio',
            'threshold': f"< {thresholds.get('max_disparity_ratio', 2.0):.1f}x",
            'desc':      'Worst-to-best tardiness ratio across types',
            'value_fmt': lambda v: f'{v:.3f}x' if v else 'N/A',
        },
        {
            'key':       'energy_share',
            'label':     'Energy Share Equity',
            'icon':      '⚡',
            'pf_key':    'energy_share_equity',
            'threshold': f"No type > {thresholds.get('energy_share_max', 0.75):.0%}",
            'desc':      'No single product type monopolises energy resources',
            'value_fmt': lambda v: (
                f"A:{v.get('A',0):.0%} B:{v.get('B',0):.0%} C:{v.get('C',0):.0%}"
                if isinstance(v, dict) else 'N/A'
            ),
        },
        {
            'key':       'priority_fairness',
            'label':     'Priority Fairness',
            'icon':      '🚀',
            'pf_key':    'priority_fairness',
            'threshold': f"> {thresholds.get('priority_fairness_min', 1.0):.1f}",
            'desc':      'Rush orders complete faster than normal-priority jobs',
            'value_fmt': lambda v: f'{v:.3f}' if v else 'N/A',
        },
        {
            'key':       'wait_time_std_dev',
            'label':     'Wait Time Equity',
            'icon':      '⏳',
            'pf_key':    'wait_time_equity',
            'threshold': f"< {thresholds.get('wait_time_std_dev', 3.0):.1f} steps",
            'desc':      'All product types wait similar amounts before processing',
            'value_fmt': lambda v: f'{v:.3f} steps' if v else 'N/A',
        },
    ]

    rows = []
    for md in METRIC_DEFS:
        val       = metrics.get(md['key'], None)
        passed    = pass_fail.get(md['pf_key'], False)
        badge_bg  = PASS_BG if passed else FAIL_BG
        badge_col = PASS_COLOR if passed else FAIL_COLOR
        badge_txt = '✓ PASS' if passed else '✗ FAIL'

        val_str = md['value_fmt'](val) if val is not None else 'N/A'

        rows.append(
            f'<div style="display:flex;align-items:center;gap:12px;'
            f'padding:10px 14px;background:{"#FAFBFC" if len(rows)%2==0 else "white"};'
            f'border-radius:8px;margin-bottom:4px;'
            f'border:1px solid #F1F5F9">'
            # Icon + label
            f'<div style="min-width:160px">'
            f'<div style="font-size:13px;font-weight:600;color:#1E293B">'
            f'{md["icon"]} {md["label"]}</div>'
            f'<div style="font-size:10px;color:#94A3B8;margin-top:2px">'
            f'{md["desc"]}</div>'
            f'</div>'
            # Value
            f'<div style="flex:1;font-size:12px;color:#475569;'
            f'font-family:monospace">{val_str}</div>'
            # Threshold
            f'<div style="font-size:11px;color:#94A3B8;min-width:80px;'
            f'text-align:center">{md["threshold"]}</div>'
            # Badge
            f'<div style="background:{badge_bg};color:{badge_col};'
            f'font-size:11px;font-weight:700;padding:3px 12px;'
            f'border-radius:20px;min-width:70px;text-align:center;'
            f'border:1px solid {badge_col}44">{badge_txt}</div>'
            f'</div>'
        )

    # Flags section
    flag_html = ''
    if flags:
        flag_items = ''.join(
            f'<div style="padding:6px 10px;background:#FEF3C7;border-left:3px solid #D97706;'
            f'margin-bottom:4px;border-radius:0 6px 6px 0;font-size:11px;color:#92400E">'
            f'⚠ {f}</div>'
            for f in flags
        )
        flag_html = (
            f'<div style="margin-top:12px">'
            f'<div style="font-size:12px;font-weight:600;color:#1E293B;margin-bottom:6px">'
            f'🚩 Detected Bias Flags</div>'
            + flag_items +
            '</div>'
        )

    return (
        '<div style="background:white;border:1px solid #E2E8F0;border-radius:12px;'
        'padding:16px">'
        '<div style="font-size:13px;font-weight:700;color:#1E293B;margin-bottom:10px">'
        '📋 Fairness Metrics — Detailed Results'
        '</div>'
        + ''.join(rows)
        + flag_html
        + '</div>'
    )


# ─────────────────────────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────────────────────────

def build_tardiness_chart(report: dict) -> go.Figure:
    """
    Grouped bar chart: tardiness rate per product type.
    Shows both tardiness_rate and avg_tardiness.
    """
    if not report:
        return _empty_figure("Load fairness report to see tardiness breakdown")

    per_type = report.get('per_type_breakdown', {})
    if not per_type:
        return _empty_figure("No per-type data available")

    types  = ['A', 'B', 'C']
    labels = ['Product A (Short)', 'Product B (Medium)', 'Product C (Long)']
    colors = [TYPE_COLORS[t] for t in types]

    tard_rates = [
        per_type.get(t, {}).get('tardiness_rate', 0) * 100
        for t in types
    ]
    avg_tards  = [
        per_type.get(t, {}).get('avg_tardiness', 0)
        for t in types
    ]
    counts = [
        per_type.get(t, {}).get('count', 0)
        for t in types
    ]

    fig = go.Figure()

    # Tardiness rate bars
    fig.add_trace(go.Bar(
        name='Tardiness Rate (%)',
        x=labels,
        y=tard_rates,
        marker=dict(color=colors, opacity=0.85),
        text=[f'{v:.1f}%' for v in tard_rates],
        textposition='outside',
        textfont=dict(size=11),
        hovertemplate=(
            '<b>%{x}</b><br>'
            'Late jobs: %{y:.1f}%<br>'
            '<extra></extra>'
        ),
        yaxis='y',
    ))

    # Count annotations
    for i, (label, count, tr) in enumerate(zip(labels, counts, tard_rates)):
        fig.add_annotation(
            x=label, y=max(tard_rates) + 5,
            text=f'n={count}',
            showarrow=False,
            font=dict(size=10, color='#94A3B8'),
            yref='y',
        )

    fig.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,250,252,0.6)',
        showlegend=False,
        xaxis=dict(
            tickfont=dict(size=11, color='#475569'),
            showgrid=False,
            showline=True,
            linecolor='#E2E8F0',
        ),
        yaxis=dict(
            title=dict(text='% Jobs Late', font=dict(size=11, color='#475569')),
            tickfont=dict(size=10, color='#64748B'),
            gridcolor='#E2E8F0',
            range=[0, max(max(tard_rates) * 1.3, 10)],
            showline=False,
        ),
        bargap=0.3,
        hoverlabel=dict(bgcolor='white', font_size=12),
    )

    return fig


def build_energy_chart(report: dict) -> go.Figure:
    """
    Donut chart: energy share consumed by each product type.
    """
    if not report:
        return _empty_figure("No energy data available")

    metrics = report.get('metrics', {})
    energy  = metrics.get('energy_share', {})
    if not energy:
        energy = {
            t: report.get('per_type_breakdown', {}).get(t, {}).get('energy_share', 0)
            for t in ['A', 'B', 'C']
        }

    types  = ['A', 'B', 'C']
    labels = ['Product A\n(Short)', 'Product B\n(Medium)', 'Product C\n(Long)']
    values = [energy.get(t, 0) * 100 for t in types]
    colors = [TYPE_COLORS[t] for t in types]

    fig = go.Figure(go.Pie(
        labels=['Type A', 'Type B', 'Type C'],
        values=values,
        hole=0.50,
        marker=dict(
            colors=colors,
            line=dict(color='white', width=2.5),
        ),
        textfont=dict(size=11, color='white'),
        hovertemplate='<b>%{label}</b><br>Energy share: %{value:.1f}%<extra></extra>',
        textposition='inside',
        insidetextorientation='radial',
    ))

    # Center label
    fig.add_annotation(
        text='Energy<br>Share',
        x=0.5, y=0.5,
        font=dict(size=12, color='#475569'),
        showarrow=False,
    )

    fig.update_layout(
        height=240,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.05,
            xanchor='center',
            x=0.5,
            font=dict(size=11, color='#475569'),
        ),
        hoverlabel=dict(bgcolor='white', font_size=12),
    )

    return fig


def build_wait_time_chart(report: dict) -> go.Figure:
    """
    Horizontal bar chart comparing average wait times per product type.
    """
    if not report:
        return _empty_figure("No wait time data available")

    per_type = report.get('per_type_breakdown', {})
    if not per_type:
        return _empty_figure("No per-type data available")

    types    = ['A', 'B', 'C']
    labels   = ['Product A<br>(Short)', 'Product B<br>(Medium)', 'Product C<br>(Long)']
    waits    = [per_type.get(t, {}).get('avg_wait_time', 0) for t in types]
    colors   = [TYPE_COLORS[t] for t in types]

    fig = go.Figure(go.Bar(
        x=waits,
        y=['Type A', 'Type B', 'Type C'],
        orientation='h',
        marker=dict(color=colors, opacity=0.80),
        text=[f'{w:.1f} steps' for w in waits],
        textposition='outside',
        textfont=dict(size=11),
        hovertemplate='<b>%{y}</b><br>Avg wait: %{x:.1f} steps<extra></extra>',
    ))

    # Add equitable line
    mean_wait = np.mean(waits) if waits else 0
    fig.add_vline(
        x=mean_wait,
        line_width=2,
        line_dash='dash',
        line_color='#94A3B8',
        annotation_text=f' Mean: {mean_wait:.1f}',
        annotation_font=dict(size=10, color='#94A3B8'),
    )

    fig.update_layout(
        height=200,
        margin=dict(l=0, r=80, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,250,252,0.6)',
        showlegend=False,
        xaxis=dict(
            title=dict(text='Avg Wait Time (steps)', font=dict(size=11)),
            tickfont=dict(size=10, color='#64748B'),
            gridcolor='#E2E8F0',
            showline=False,
        ),
        yaxis=dict(
            tickfont=dict(size=11, color='#1E293B'),
            showgrid=False,
            showline=False,
        ),
        hoverlabel=dict(bgcolor='white', font_size=12),
    )

    return fig


def build_per_type_summary_html(report: dict) -> str:
    """
    Build a 3-card summary row for per-type breakdown.
    """
    if not report:
        return ''

    per_type = report.get('per_type_breakdown', {})
    if not per_type:
        return ''

    TYPE_LABELS = {
        'A': ('Product A', 'Short jobs · Low energy', '#EFF6FF', '#2563EB', '🔵'),
        'B': ('Product B', 'Medium jobs · Moderate energy', '#ECFDF5', '#059669', '🟢'),
        'C': ('Product C', 'Long jobs · High energy', '#FFFBEB', '#D97706', '🟡'),
    }

    cards = []
    for t, (name, desc, bg, color, icon) in TYPE_LABELS.items():
        data = per_type.get(t, {})
        late = data.get('tardiness_rate', 0)
        wait = data.get('avg_wait_time', 0)
        n    = data.get('count', 0)

        cards.append(
            f'<div style="flex:1;background:{bg};border:1.5px solid {color}22;'
            f'border-radius:12px;padding:14px;'
            f'box-shadow:0 2px 6px rgba(0,0,0,0.05)">'
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:8px">'
            f'<span style="font-size:18px">{icon}</span>'
            f'<div>'
            f'<div style="font-weight:700;color:{color};font-size:13px">{name}</div>'
            f'<div style="font-size:10px;color:#64748B">{desc}</div>'
            f'</div>'
            f'</div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">'
            f'<div style="background:white;border-radius:6px;padding:6px;text-align:center">'
            f'<div style="font-size:16px;font-weight:800;color:{color}">{late:.0%}</div>'
            f'<div style="font-size:10px;color:#64748B">late rate</div>'
            f'</div>'
            f'<div style="background:white;border-radius:6px;padding:6px;text-align:center">'
            f'<div style="font-size:16px;font-weight:800;color:{color}">{wait:.1f}</div>'
            f'<div style="font-size:10px;color:#64748B">avg wait</div>'
            f'</div>'
            f'</div>'
            f'<div style="margin-top:8px;font-size:10px;color:#94A3B8;text-align:center">'
            f'{n} jobs evaluated'
            f'</div>'
            f'</div>'
        )

    return (
        '<div style="display:flex;gap:10px;margin:10px 0">'
        + ''.join(cards) +
        '</div>'
    )


# ─────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────

def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5, y=0.5,
        xref='paper', yref='paper',
        showarrow=False,
        font=dict(size=12, color='#94A3B8'),
    )
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,250,252,0.6)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _placeholder_banner() -> str:
    return (
        '<div style="background:#F8FAFC;border:2px dashed #CBD5E1;'
        'border-radius:16px;padding:30px;text-align:center;'
        'color:#94A3B8">'
        '<div style="font-size:36px;margin-bottom:8px">📊</div>'
        '<div style="font-size:16px;font-weight:600;margin-bottom:6px">'
        'Fairness Report Not Yet Available</div>'
        '<div style="font-size:13px">'
        'Run <code style="background:#F1F5F9;padding:2px 6px;border-radius:4px">'
        'python run_fairness_eval.py</code> to generate the audit report'
        '</div>'
        '</div>'
    )


def _placeholder_metrics() -> str:
    return (
        '<div style="background:#F8FAFC;border:1px solid #E2E8F0;'
        'border-radius:12px;padding:24px;text-align:center;color:#94A3B8">'
        'No metrics available — run fairness evaluation first.'
        '</div>'
    )