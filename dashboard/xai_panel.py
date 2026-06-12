"""
xai_panel.py
------------
Builds SHAP explanation visualizations and hypothesis verification badges
for the Explainability tab of the dashboard.

Components:
  - build_shap_chart()      : Horizontal bar chart of feature importance
  - build_hypothesis_html() : Color-coded H1/H2/H3 verification badges
  - build_narrative_html()  : Styled narrative explanation card
  - build_action_card_html(): Styled card showing agent's chosen action
"""

import plotly.graph_objects as go
import numpy as np


# ─────────────────────────────────────────────────────────────────
#  COLORS
# ─────────────────────────────────────────────────────────────────

POSITIVE_COLOR = '#2563EB'   # Blue  — feature supports the action
NEGATIVE_COLOR = '#DC2626'   # Red   — feature opposes the action
BAR_BG         = '#F1F5F9'   # Light slate for zero bars

GROUP_COLORS = {
    'Machine Status':      '#7C3AED',   # Purple
    'Machine Load':        '#2563EB',   # Blue
    'Job 0 (Most Urgent)': '#DC2626',   # Red
    'Job 1':               '#D97706',   # Amber
    'Jobs 2-5':            '#059669',   # Green
    'Time Context':        '#475569',   # Slate
    'Energy Price':        '#F59E0B',   # Yellow
}

ACTION_COLORS = {
    0: ('#2563EB', '#EFF6FF', '🔵'),
    1: ('#059669', '#ECFDF5', '🟢'),
    2: ('#D97706', '#FFFBEB', '🟡'),
    3: ('#7C3AED', '#F5F3FF', '🟣'),
    4: ('#DB2777', '#FDF2F8', '🔴'),
    5: ('#64748B', '#F8FAFC', '⚫'),
    6: ('#94A3B8', '#F8FAFC', '⏸'),
}


# ─────────────────────────────────────────────────────────────────
#  SHAP BAR CHART
# ─────────────────────────────────────────────────────────────────

def build_shap_chart(explanation: dict) -> go.Figure:
    """
    Build a horizontal bar chart showing top feature importance.

    Args:
        explanation : Output of SHAPExplainer.explain()

    Returns:
        plotly.graph_objects.Figure
    """
    if explanation is None:
        return _empty_shap_figure("Run the simulation to generate an explanation")

    top_features = explanation.get('top_features', [])
    if not top_features:
        return _empty_shap_figure("No feature data available")

    # Limit to top 6 features
    top_features = top_features[:6]
    names  = [f[0] for f in top_features]
    values = [f[1] for f in top_features]

    # Color: positive SHAP = blue, negative or neutral = gradient from amber
    # Since we use absolute values, use group-based colors for variety
    shap_raw = explanation.get('shap_values', None)
    bar_colors = []
    for name in names:
        # Determine group for color coding
        assigned_color = POSITIVE_COLOR
        for group_name, color in GROUP_COLORS.items():
            # Check if feature name starts with a keyword from the group
            if 'Job 0' in name:
                assigned_color = GROUP_COLORS['Job 0 (Most Urgent)']
                break
            elif 'Job 1' in name:
                assigned_color = GROUP_COLORS['Job 1']
                break
            elif 'Job' in name:
                assigned_color = GROUP_COLORS['Jobs 2-5']
                break
            elif 'M1' in name or 'M2' in name or 'M3' in name:
                if 'Remaining' in name:
                    assigned_color = GROUP_COLORS['Machine Load']
                else:
                    assigned_color = GROUP_COLORS['Machine Status']
                break
            elif 'Energy Price' in name:
                assigned_color = GROUP_COLORS['Energy Price']
                break
            elif 'Clock' in name:
                assigned_color = GROUP_COLORS['Time Context']
                break
        bar_colors.append(assigned_color)

    # Shorten feature names for display
    display_names = [_shorten_name(n) for n in names]

    fig = go.Figure(go.Bar(
        x=values,
        y=display_names,
        orientation='h',
        marker=dict(
            color=bar_colors,
            opacity=0.85,
            line=dict(width=0),
        ),
        text=[f'{v:.1f}%' for v in values],
        textposition='outside',
        textfont=dict(size=12, color='#1E293B', family='monospace'),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.1f}%<extra></extra>',
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=0, r=70, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,250,252,0.6)',
        xaxis=dict(
            title=dict(text='Feature Importance (%)', font=dict(size=11, color='#475569')),
            tickfont=dict(size=10, color='#64748B'),
            gridcolor='#E2E8F0',
            range=[0, max(values) * 1.25 if values else 100],
            showline=False,
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(size=11, color='#1E293B'),
            autorange='reversed',
            showgrid=False,
            showline=False,
        ),
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            bordercolor='#E2E8F0',
        ),
        showlegend=False,
    )

    return fig


def build_group_chart(explanation: dict) -> go.Figure:
    """
    Build a donut chart showing feature group importance.
    """
    if explanation is None:
        return _empty_shap_figure("No explanation available")

    groups = explanation.get('groups', {})
    if not groups:
        return _empty_shap_figure("No group data available")

    # Filter out very small groups
    filtered = {k: v for k, v in groups.items() if v > 1.0}
    if not filtered:
        filtered = groups

    labels = list(filtered.keys())
    values = list(filtered.values())
    colors = [GROUP_COLORS.get(lbl, '#94A3B8') for lbl in labels]

    # Shorten labels
    short_labels = [lbl.replace('(Most Urgent)', '').strip() for lbl in labels]

    fig = go.Figure(go.Pie(
        labels=short_labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textfont=dict(size=11),
        hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>',
        textposition='outside',
    ))

    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            font=dict(size=10, color='#475569'),
            orientation='v',
        ),
        hoverlabel=dict(bgcolor='white', font_size=12),
    )

    return fig


# ─────────────────────────────────────────────────────────────────
#  HYPOTHESIS BADGES
# ─────────────────────────────────────────────────────────────────

def build_hypothesis_html(hypothesis_results: dict) -> str:
    """
    Build HTML for H1, H2, H3 verification badges.

    Args:
        hypothesis_results : Output of HypothesisTester.run_all()
                             loaded from logs/hypothesis_results.json

    Returns:
        HTML string for st.markdown(unsafe_allow_html=True)
    """
    if not hypothesis_results:
        return _placeholder_hypothesis_html()

    HYPO_META = {
        'H1': {
            'title': 'H1 — Urgency Priority',
            'icon':  '⏱️',
            'desc':  'Agent prioritises the most urgent job when deadline is imminent',
        },
        'H2': {
            'title': 'H2 — Energy Awareness',
            'icon':  '⚡',
            'desc':  'Agent responds to energy price spikes — avoids expensive operations',
        },
        'H3': {
            'title': 'H3 — Slot Ordering',
            'icon':  '🎯',
            'desc':  'Agent weights urgency correctly across different job queue slots',
        },
    }

    cards = []
    for hid, meta in HYPO_META.items():
        result_data = hypothesis_results.get(hid, {})
        status      = result_data.get('result', 'NOT VERIFIED')
        verified    = status == 'VERIFIED'

        if verified:
            bg_color   = '#ECFDF5'
            border     = '#059669'
            badge_bg   = '#059669'
            badge_text = '✓ VERIFIED'
            text_color = '#065F46'
        else:
            bg_color   = '#FEF2F2'
            border     = '#DC2626'
            badge_bg   = '#DC2626'
            badge_text = '✗ NOT VERIFIED'
            text_color = '#991B1B'

        # Build detail line
        detail = ''
        if hid == 'H1' and 'urgency_pct' in result_data:
            pct = result_data['urgency_pct']
            detail = f'<div style="color:#64748B;font-size:11px;margin-top:4px">Urgency feature: {pct:.1f}%</div>'
        elif hid == 'H2' and 'delta_pp' in result_data:
            delta = result_data.get('delta_pp', 0)
            detail = f'<div style="color:#64748B;font-size:11px;margin-top:4px">Energy Δ spike: +{delta:.1f}pp</div>'
        elif hid == 'H3' and 'ratio' in result_data:
            ratio = result_data.get('ratio', 0)
            detail = f'<div style="color:#64748B;font-size:11px;margin-top:4px">Urgency ratio: {ratio:.2f}x</div>'

        cards.append(
            f'<div style="flex:1;background:{bg_color};border:1.5px solid {border}33;'
            f'border-radius:12px;padding:14px;box-shadow:0 2px 6px rgba(0,0,0,0.06)">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
            f'<span style="font-size:20px">{meta["icon"]}</span>'
            f'<span style="font-weight:700;font-size:13px;color:#1E293B">{meta["title"]}</span>'
            f'</div>'
            f'<div style="background:{badge_bg};color:white;font-size:11px;'
            f'font-weight:600;padding:3px 10px;border-radius:20px;display:inline-block;'
            f'letter-spacing:0.5px">{badge_text}</div>'
            f'<div style="color:#475569;font-size:11px;margin-top:6px;line-height:1.4">'
            f'{meta["desc"]}</div>'
            f'{detail}'
            f'</div>'
        )

    return (
        '<div style="display:flex;gap:10px;margin:8px 0">'
        + ''.join(cards) +
        '</div>'
    )


# ─────────────────────────────────────────────────────────────────
#  ACTION CARD AND NARRATIVE
# ─────────────────────────────────────────────────────────────────

def build_action_card_html(explanation: dict) -> str:
    """
    Build a styled card showing the agent's chosen action and confidence.
    """
    if explanation is None:
        return (
            '<div style="background:#F8FAFC;border:1.5px solid #E2E8F0;'
            'border-radius:12px;padding:16px;text-align:center;color:#94A3B8">'
            '⏳  Run the simulation to see live agent decisions here'
            '</div>'
        )

    action    = explanation.get('chosen_action', 6)
    label     = explanation.get('action_label', 'WAIT')
    prob      = explanation.get('action_prob', 0.0)
    probs     = explanation.get('action_probs', [])

    color, bg, icon = ACTION_COLORS.get(action, ('#64748B', '#F8FAFC', '⚪'))

    # Build mini probability bar
    prob_bars = ''
    if probs:
        action_names = ['J0', 'J1', 'J2', 'J3', 'J4', 'J5', 'WAIT']
        prob_bars = '<div style="display:flex;gap:3px;margin-top:8px;align-items:flex-end;height:30px">'
        for i, p in enumerate(probs[:7]):
            bar_h  = max(int(p * 100), 2)
            bc     = color if i == action else '#CBD5E1'
            prob_bars += (
                f'<div style="flex:1;display:flex;flex-direction:column;'
                f'align-items:center;justify-content:flex-end">'
                f'<div style="width:100%;height:{bar_h}px;background:{bc};'
                f'border-radius:3px 3px 0 0;min-height:2px"></div>'
                f'<div style="font-size:9px;color:#94A3B8;margin-top:2px">'
                f'{action_names[i] if i < len(action_names) else i}</div>'
                f'</div>'
            )
        prob_bars += '</div>'

    return (
        f'<div style="background:{bg};border:2px solid {color}33;'
        f'border-radius:14px;padding:16px 20px;'
        f'box-shadow:0 2px 8px rgba(0,0,0,0.07)">'
        f'<div style="display:flex;align-items:center;justify-content:space-between">'
        f'<div>'
        f'<div style="font-size:11px;font-weight:600;color:{color};'
        f'letter-spacing:1px;text-transform:uppercase">Agent Decision</div>'
        f'<div style="font-size:17px;font-weight:700;color:#0F172A;margin:4px 0">'
        f'{icon} {label}</div>'
        f'</div>'
        f'<div style="text-align:right">'
        f'<div style="font-size:11px;color:#64748B">Confidence</div>'
        f'<div style="font-size:26px;font-weight:800;color:{color}">'
        f'{prob*100:.0f}%</div>'
        f'</div>'
        f'</div>'
        f'{prob_bars}'
        f'</div>'
    )


def build_narrative_html(explanation: dict) -> str:
    """
    Build a styled card for the SHAP narrative text.
    """
    if explanation is None:
        return ''

    narrative = explanation.get('narrative', '')
    if not narrative:
        return ''

    lines = narrative.split('\n')
    html_lines = []
    for line in lines:
        if line.startswith('Decision:'):
            continue  # Shown in action card already
        elif line.startswith('Confidence:'):
            continue
        elif line.startswith('Top reasons'):
            html_lines.append(
                f'<div style="font-weight:600;color:#1E293B;'
                f'font-size:12px;margin:10px 0 6px 0">{line}</div>'
            )
        elif line.strip().startswith(('1.', '2.', '3.', '4.')):
            parts = line.strip().split(maxsplit=1)
            num   = parts[0] if parts else ''
            rest  = parts[1] if len(parts) > 1 else ''
            # Split rest on whitespace to get feature name and percentage
            rest_parts = rest.rsplit(maxsplit=1)
            fname = rest_parts[0].strip() if rest_parts else rest
            pct   = rest_parts[1] if len(rest_parts) > 1 else ''
            html_lines.append(
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;padding:4px 8px;margin:2px 0;'
                f'background:#F1F5F9;border-radius:6px;">'
                f'<span style="color:#475569;font-size:11px">'
                f'<b style="color:#1E3A8A">{num}</b> {fname}</span>'
                f'<span style="font-weight:700;color:#1E3A8A;font-size:12px;'
                f'font-family:monospace">{pct}</span>'
                f'</div>'
            )
        elif line.startswith('Importance by group:'):
            html_lines.append(
                f'<div style="font-weight:600;color:#1E293B;'
                f'font-size:12px;margin:10px 0 6px 0">{line}</div>'
            )
        elif line.strip() and line.startswith('  '):
            # Group importance line
            parts = line.strip().rsplit(maxsplit=1)
            gname = parts[0].strip() if parts else line.strip()
            gpct  = parts[1] if len(parts) > 1 else ''
            try:
                pct_val = float(gpct.replace('%', ''))
                bar_w   = min(pct_val * 2, 100)
            except Exception:
                bar_w = 20
            color = '#1E3A8A'
            html_lines.append(
                f'<div style="margin:3px 0">'
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:11px;color:#475569;margin-bottom:2px">'
                f'<span>{gname}</span>'
                f'<span style="font-weight:600;color:{color}">{gpct}</span>'
                f'</div>'
                f'<div style="background:#E2E8F0;border-radius:4px;height:5px">'
                f'<div style="width:{bar_w}%;background:{color};'
                f'border-radius:4px;height:5px"></div>'
                f'</div>'
                f'</div>'
            )

    return (
        '<div style="background:#FAFBFC;border:1px solid #E2E8F0;'
        'border-radius:12px;padding:14px 16px;margin-top:6px">'
        + ''.join(html_lines) +
        '</div>'
    )


# ─────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────

def _shorten_name(name: str) -> str:
    """Shorten feature names for chart display."""
    replacements = {
        'Deadline Urgency': 'Deadline',
        'Processing Time':  'Proc Time',
        '(Most Urgent)':    '',
        'Remaining Steps':  'Remaining',
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name.strip()


def _empty_shap_figure(message: str) -> go.Figure:
    """Return an empty figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5, y=0.5,
        xref='paper', yref='paper',
        showarrow=False,
        font=dict(size=13, color='#94A3B8'),
    )
    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,250,252,0.6)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _placeholder_hypothesis_html() -> str:
    """Return placeholder HTML when hypothesis results are not yet available."""
    cards = []
    for hid, title, icon in [
        ('H1', 'Urgency Priority', '⏱️'),
        ('H2', 'Energy Awareness', '⚡'),
        ('H3', 'Slot Ordering', '🎯'),
    ]:
        cards.append(
            f'<div style="flex:1;background:#F8FAFC;border:1.5px solid #E2E8F0;'
            f'border-radius:12px;padding:14px;text-align:center">'
            f'<div style="font-size:22px">{icon}</div>'
            f'<div style="font-weight:700;color:#64748B;font-size:13px;margin:4px 0">'
            f'{hid} — {title}</div>'
            f'<div style="color:#94A3B8;font-size:11px">'
            f'Run hypothesis_tester.py first</div>'
            f'</div>'
        )
    return (
        '<div style="display:flex;gap:10px;margin:8px 0">'
        + ''.join(cards) +
        '</div>'
    )