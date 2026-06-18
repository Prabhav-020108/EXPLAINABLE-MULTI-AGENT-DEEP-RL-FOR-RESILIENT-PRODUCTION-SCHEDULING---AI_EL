"""
xai_panel.py  (redesigned)
-----------
SHAP explanation visualizations with plain-English translations.
Every feature name and chart label is human-readable.
"""

import plotly.graph_objects as go
import numpy as np

# ─────────────────────────────────────────────────────────────────
#  PLAIN-ENGLISH FEATURE TRANSLATIONS
# ─────────────────────────────────────────────────────────────────

# Maps raw SHAP feature names → what a human understands
PLAIN_ENGLISH = {
    # Machine Status
    'M1: Idle':            'Machine 1 is free',
    'M1: Busy':            'Machine 1 is working',
    'M1: Broken':          'Machine 1 is broken',
    'M2: Idle':            'Machine 2 is free',
    'M2: Busy':            'Machine 2 is working',
    'M2: Broken':          'Machine 2 is broken',
    'M3: Idle':            'Machine 3 is free',
    'M3: Busy':            'Machine 3 is working',
    'M3: Broken':          'Machine 3 is broken',
    # Machine Load
    'M1: Remaining Steps': 'Machine 1 — steps left on current job',
    'M2: Remaining Steps': 'Machine 2 — steps left on current job',
    'M3: Remaining Steps': 'Machine 3 — steps left on current job',
    # Job features
    'Job0: Type':              'Most urgent job — job type',
    'Job0: Proc Time':         'Most urgent job — how long it takes',
    'Job0: Deadline Urgency':  'Most urgent job — deadline closeness',
    'Job0: Deadline':          'Most urgent job — deadline closeness',
    'Job0: Energy Cost':       'Most urgent job — electricity usage',
    'Job1: Type':              '2nd job — job type',
    'Job1: Proc Time':         '2nd job — how long it takes',
    'Job1: Deadline Urgency':  '2nd job — deadline closeness',
    'Job1: Deadline':          '2nd job — deadline closeness',
    'Job1: Energy Cost':       '2nd job — electricity usage',
    'Job2: Type':              '3rd job — job type',
    'Job2: Proc Time':         '3rd job — how long it takes',
    'Job2: Deadline Urgency':  '3rd job — deadline closeness',
    'Job2: Deadline':          '3rd job — deadline closeness',
    'Job2: Energy Cost':       '3rd job — electricity usage',
    'Job3: Type':              '4th job — type',
    'Job3: Proc Time':         '4th job — duration',
    'Job3: Deadline Urgency':  '4th job — deadline closeness',
    'Job3: Deadline':          '4th job — deadline closeness',
    'Job3: Energy Cost':       '4th job — energy usage',
    'Job4: Type':              '5th job — type',
    'Job4: Proc Time':         '5th job — duration',
    'Job4: Deadline Urgency':  '5th job — deadline closeness',
    'Job4: Deadline':          '5th job — deadline closeness',
    'Job4: Energy Cost':       '5th job — energy usage',
    'Job5: Type':              '6th job — type',
    'Job5: Proc Time':         '6th job — duration',
    'Job5: Deadline Urgency':  '6th job — deadline closeness',
    'Job5: Deadline':          '6th job — deadline closeness',
    'Job5: Energy Cost':       '6th job — energy usage',
    # Context
    'Global Clock':  'How far into the shift we are',
    'Energy Price':  'Current electricity price level',
}

# Group labels in plain English
GROUP_PLAIN = {
    'Machine Status':       'Which machines are free / working / broken',
    'Machine Load':         'How busy each machine is right now',
    'Job 0 (Most Urgent)':  'The most urgent job in queue',
    'Job 1':                'The 2nd job in queue',
    'Jobs 2-5':             'Remaining jobs in queue',
    'Time Context':         'Time remaining in the shift',
    'Energy Price':         'Current electricity cost',
}

def _plain(name: str) -> str:
    """Return plain-English label for a SHAP feature name."""
    return PLAIN_ENGLISH.get(name, name)

def _plain_group(name: str) -> str:
    return GROUP_PLAIN.get(name, name)

# ─────────────────────────────────────────────────────────────────
#  COLOR PALETTE
# ─────────────────────────────────────────────────────────────────

GROUP_COLORS = {
    'Machine Status':       '#7C3AED',
    'Machine Load':         '#2563EB',
    'Job 0 (Most Urgent)':  '#DC2626',
    'Job 1':                '#D97706',
    'Jobs 2-5':             '#059669',
    'Time Context':         '#475569',
    'Energy Price':         '#F59E0B',
}

ACTION_COLORS = {
    0: ('#2563EB', '#EFF6FF', '🔵'),
    1: ('#059669', '#ECFDF5', '🟢'),
    2: ('#D97706', '#FFFBEB', '🟡'),
    3: ('#7C3AED', '#F5F3FF', '🟣'),
    4: ('#DB2777', '#FDF2F8', '🩷'),
    5: ('#64748B', '#F8FAFC', '⚫'),
    6: ('#94A3B8', '#F8FAFC', '⏸'),
}

ACTION_LABELS = {
    0: 'Assign the most urgent job',
    1: 'Assign the 2nd job in queue',
    2: 'Assign the 3rd job in queue',
    3: 'Assign the 4th job in queue',
    4: 'Assign the 5th job in queue',
    5: 'Assign the 6th job in queue',
    6: 'Wait — no assignment yet',
}

# ─────────────────────────────────────────────────────────────────
#  DECISION STORY CARD
# ─────────────────────────────────────────────────────────────────

def build_decision_story_html(explanation: dict) -> str:
    """
    Build a plain-English 'story' card explaining the AI's decision.
    This is the first thing the user sees on the AI Decisions tab.
    """
    if explanation is None:
        return (
            '<div style="background:#F8FAFC;border:2px dashed #E2E8F0;border-radius:16px;'
            'padding:28px;text-align:center;color:#94A3B8">'
            '<div style="font-size:32px;margin-bottom:10px">🧠</div>'
            '<div style="font-size:15px;font-weight:600">No decision recorded yet</div>'
            '<div style="font-size:13px;margin-top:6px">Run the simulation to see the AI making decisions in real time.</div>'
            '</div>'
        )

    action   = explanation.get('chosen_action', 6)
    prob     = explanation.get('action_prob', 0.0)
    features = explanation.get('top_features', [])
    groups   = explanation.get('groups', {})

    label = ACTION_LABELS.get(action, f'Action {action}')
    color, bg, icon = ACTION_COLORS.get(action, ('#64748B', '#F8FAFC', '⚪'))

    # Build reason bullets
    reasons_html = ''
    for rank, (fname, pct) in enumerate(features[:3], 1):
        plain = _plain(fname)
        medal = ['🥇', '🥈', '🥉'][rank - 1]
        reasons_html += (
            f'<div style="display:flex;align-items:center;gap:10px;'
            f'padding:8px 12px;background:white;border-radius:8px;margin-bottom:6px;'
            f'border:1px solid #E2E8F0">'
            f'<span style="font-size:18px">{medal}</span>'
            f'<div style="flex:1">'
            f'<div style="font-size:13px;font-weight:600;color:#1E293B">{plain}</div>'
            f'</div>'
            f'<div style="background:{color};color:white;font-size:11px;font-weight:700;'
            f'padding:2px 10px;border-radius:20px">{pct:.1f}%</div>'
            f'</div>'
        )

    # Top group plain name
    top_group_name = max(groups, key=groups.get) if groups else ''
    top_group_plain = _plain_group(top_group_name)
    top_group_pct   = groups.get(top_group_name, 0)

    return (
        f'<div style="background:{bg};border:2px solid {color}33;border-radius:16px;'
        f'padding:20px 24px;box-shadow:0 4px 16px rgba(0,0,0,0.06)">'
        # Header
        f'<div style="display:flex;align-items:flex-start;justify-content:space-between;'
        f'margin-bottom:16px">'
        f'<div>'
        f'<div style="font-size:11px;font-weight:700;letter-spacing:1px;'
        f'text-transform:uppercase;color:{color};margin-bottom:4px">AI Decision</div>'
        f'<div style="font-size:20px;font-weight:800;color:#0F172A">{icon} {label}</div>'
        f'<div style="font-size:12px;color:#64748B;margin-top:3px">'
        f'The AI is {prob*100:.0f}% confident in this choice</div>'
        f'</div>'
        f'<div style="background:white;border-radius:12px;padding:8px 16px;'
        f'text-align:center;border:1.5px solid {color}44">'
        f'<div style="font-size:24px;font-weight:800;color:{color}">{prob*100:.0f}%</div>'
        f'<div style="font-size:10px;color:#94A3B8;font-weight:600">CONFIDENCE</div>'
        f'</div>'
        f'</div>'
        # Reasons
        f'<div style="font-size:12px;font-weight:700;color:#475569;margin-bottom:8px;'
        f'text-transform:uppercase;letter-spacing:0.5px">Top 3 Reasons for this Decision:</div>'
        f'{reasons_html}'
        # Summary
        f'<div style="margin-top:12px;padding:10px 14px;background:white;border-radius:8px;'
        f'border-left:3px solid {color};font-size:12px;color:#475569">'
        f'<b style="color:#1E293B">In summary:</b> The biggest influence was '
        f'<b style="color:{color}">{top_group_plain}</b>, '
        f'which contributed {top_group_pct:.1f}% of the decision weight.'
        f'</div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────
#  SHAP BAR CHART  (with plain-English labels)
# ─────────────────────────────────────────────────────────────────

def build_shap_chart(explanation: dict) -> go.Figure:
    if explanation is None:
        return _empty_figure("Run the simulation to see feature importance")

    top_features = explanation.get('top_features', [])
    if not top_features:
        return _empty_figure("No feature data yet")

    top_features = top_features[:6]
    # Use plain English names
    names  = [_plain(f[0]) for f in top_features]
    values = [f[1] for f in top_features]

    # Color by group
    bar_colors = []
    for orig_name, _ in top_features:
        color = '#2563EB'  # default
        if 'M1' in orig_name or 'M2' in orig_name or 'M3' in orig_name:
            if 'Remaining' in orig_name:
                color = GROUP_COLORS['Machine Load']
            else:
                color = GROUP_COLORS['Machine Status']
        elif 'Job0' in orig_name or 'Job 0' in orig_name:
            color = GROUP_COLORS['Job 0 (Most Urgent)']
        elif 'Job1' in orig_name or 'Job 1' in orig_name:
            color = GROUP_COLORS['Job 1']
        elif 'Job' in orig_name:
            color = GROUP_COLORS['Jobs 2-5']
        elif 'Energy Price' in orig_name:
            color = GROUP_COLORS['Energy Price']
        elif 'Clock' in orig_name:
            color = GROUP_COLORS['Time Context']
        bar_colors.append(color)

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation='h',
        marker=dict(color=bar_colors, opacity=0.88, line=dict(width=0)),
        text=[f'  {v:.1f}%' for v in values],
        textposition='outside',
        textfont=dict(size=12, color='#1E293B', family='Inter'),
        hovertemplate='<b>%{y}</b><br>Influence: %{x:.1f}%<extra></extra>',
    ))

    fig.update_layout(
        height=260,
        margin=dict(l=0, r=70, t=8, b=8),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,250,252,0.5)',
        xaxis=dict(
            title=dict(text='How much this influenced the AI\'s decision (%)',
                       font=dict(size=11, color='#64748B')),
            tickfont=dict(size=10, color='#94A3B8'),
            gridcolor='#E2E8F0',
            range=[0, max(values) * 1.3 if values else 100],
            showline=False, zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(size=11, color='#1E293B'),
            autorange='reversed',
            showgrid=False, showline=False,
        ),
        hoverlabel=dict(bgcolor='white', font_size=12, bordercolor='#E2E8F0'),
        showlegend=False,
    )
    return fig


def build_group_chart(explanation: dict) -> go.Figure:
    """Donut chart of feature group importance."""
    if explanation is None:
        return _empty_figure("No data yet")

    groups = explanation.get('groups', {})
    if not groups:
        return _empty_figure("No group data")

    filtered = {k: v for k, v in groups.items() if v > 1.0}
    if not filtered:
        filtered = groups

    labels = list(filtered.keys())
    values = list(filtered.values())
    colors = [GROUP_COLORS.get(lbl, '#94A3B8') for lbl in labels]
    plain_labels = [_plain_group(lbl) for lbl in labels]

    fig = go.Figure(go.Pie(
        labels=plain_labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='white', width=2.5)),
        textfont=dict(size=10),
        hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>',
        textposition='outside',
    ))

    fig.update_layout(
        height=240,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(font=dict(size=10, color='#475569'), orientation='v'),
        hoverlabel=dict(bgcolor='white', font_size=12),
    )
    return fig


# ─────────────────────────────────────────────────────────────────
#  HYPOTHESIS BADGES
# ─────────────────────────────────────────────────────────────────

def build_hypothesis_html(hypothesis_results: dict) -> str:
    if not hypothesis_results:
        return _placeholder_hypothesis_html()

    HYPO_META = {
        'H1': {
            'title': 'H1 — Urgency Priority',
            'icon':  '⏱️',
            'question': 'Does the AI prioritize jobs with urgent deadlines?',
            'desc':  'When a job\'s deadline is almost up, does the AI pick it first?',
        },
        'H2': {
            'title': 'H2 — Energy Awareness',
            'icon':  '⚡',
            'question': 'Does the AI react to electricity price spikes?',
            'desc':  'When electricity costs spike, does the AI change its scheduling?',
        },
        'H3': {
            'title': 'H3 — Smart Ordering',
            'icon':  '🎯',
            'question': 'Does the AI correctly rank jobs by urgency?',
            'desc':  'Does the AI give more weight to the most-urgent job over the second-most-urgent?',
        },
    }

    cards = []
    for hid, meta in HYPO_META.items():
        result_data = hypothesis_results.get(hid, {})
        status   = result_data.get('result', 'NOT VERIFIED')
        verified = status == 'VERIFIED'

        bg     = '#ECFDF5' if verified else '#FEF2F2'
        border = '#059669' if verified else '#DC2626'
        badge  = '#059669' if verified else '#DC2626'
        badge_text = '✓ VERIFIED' if verified else '✗ NOT VERIFIED'
        text   = '#065F46' if verified else '#991B1B'

        detail = ''
        if hid == 'H1':
            pct = result_data.get('job0_group_pct', result_data.get('urgency_pct', 0))
            detail = f'<div style="color:#64748B;font-size:11px;margin-top:4px">Urgency influence: {pct:.1f}%</div>'
        elif hid == 'H2':
            pct = result_data.get('energy_pct', 0)
            detail = f'<div style="color:#64748B;font-size:11px;margin-top:4px">Energy influence: {pct:.1f}%</div>'
        elif hid == 'H3':
            ratio = result_data.get('ratio', 0)
            detail = f'<div style="color:#64748B;font-size:11px;margin-top:4px">Urgency ratio: {ratio:.1f}×</div>'

        cards.append(
            f'<div style="flex:1;background:{bg};border:1.5px solid {border}33;'
            f'border-radius:14px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.05)">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
            f'<span style="font-size:22px">{meta["icon"]}</span>'
            f'<span style="font-weight:700;font-size:13px;color:#1E293B">{meta["title"]}</span>'
            f'</div>'
            f'<div style="background:{badge};color:white;font-size:11px;font-weight:700;'
            f'padding:3px 12px;border-radius:20px;display:inline-block">{badge_text}</div>'
            f'<div style="color:#475569;font-size:12px;margin-top:8px;'
            f'font-style:italic;line-height:1.5">"{meta["question"]}"</div>'
            f'<div style="color:#64748B;font-size:11px;margin-top:4px">{meta["desc"]}</div>'
            f'{detail}'
            f'</div>'
        )

    return (
        '<div style="display:flex;gap:10px;margin:8px 0">'
        + ''.join(cards) +
        '</div>'
    )


# ─────────────────────────────────────────────────────────────────
#  ACTION CARD
# ─────────────────────────────────────────────────────────────────

def build_action_card_html(explanation: dict) -> str:
    """Alias kept for backward-compatibility — now delegates to story."""
    return build_decision_story_html(explanation)


def build_narrative_html(explanation: dict) -> str:
    """Plain-English narrative below the SHAP chart."""
    if not explanation:
        return ''
    top = explanation.get('top_features', [])
    if not top:
        return ''

    lines = ['<div style="font-weight:600;color:#1E293B;font-size:12px;margin:10px 0 6px">Detailed feature influence:</div>']
    for rank, (fname, pct) in enumerate(top[:4], 1):
        plain = _plain(fname)
        lines.append(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:5px 10px;background:#F1F5F9;border-radius:6px;margin-bottom:3px">'
            f'<span style="color:#475569;font-size:12px"><b style="color:#1E3A8A">#{rank}</b> {plain}</span>'
            f'<span style="font-weight:700;color:#1E3A8A;font-size:12px;font-family:monospace">{pct:.1f}%</span>'
            f'</div>'
        )

    groups = explanation.get('groups', {})
    if groups:
        lines.append('<div style="font-weight:600;color:#1E293B;font-size:12px;margin:10px 0 6px">Influence by category:</div>')
        for gname, pct in sorted(groups.items(), key=lambda x: x[1], reverse=True)[:4]:
            if pct > 0.5:
                plain_g = _plain_group(gname)
                bar_w = min(pct * 2, 100)
                lines.append(
                    f'<div style="margin:4px 0">'
                    f'<div style="display:flex;justify-content:space-between;font-size:11px;'
                    f'color:#475569;margin-bottom:2px">'
                    f'<span>{plain_g}</span><span style="font-weight:600;color:#1E3A8A">{pct:.1f}%</span>'
                    f'</div>'
                    f'<div style="background:#E2E8F0;border-radius:4px;height:5px">'
                    f'<div style="width:{bar_w}%;background:#2563EB;border-radius:4px;height:5px"></div>'
                    f'</div>'
                    f'</div>'
                )

    return (
        '<div style="background:#FAFBFC;border:1px solid #E2E8F0;border-radius:12px;'
        'padding:14px 16px;margin-top:8px">'
        + ''.join(lines) +
        '</div>'
    )


# ─────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────

def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref='paper', yref='paper',
        showarrow=False, font=dict(size=13, color='#94A3B8'),
    )
    fig.update_layout(
        height=260, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(248,250,252,0.5)',
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def _placeholder_hypothesis_html() -> str:
    cards = []
    for hid, title, icon in [
        ('H1', 'Urgency Priority', '⏱️'),
        ('H2', 'Energy Awareness', '⚡'),
        ('H3', 'Smart Ordering',   '🎯'),
    ]:
        cards.append(
            f'<div style="flex:1;background:#F8FAFC;border:1.5px solid #E2E8F0;'
            f'border-radius:14px;padding:16px;text-align:center">'
            f'<div style="font-size:24px">{icon}</div>'
            f'<div style="font-weight:700;color:#64748B;font-size:13px;margin:6px 0">'
            f'{hid} — {title}</div>'
            f'<div style="color:#94A3B8;font-size:11px">'
            f'Run <code>python run_hypothesis_tests.py</code> first</div>'
            f'</div>'
        )
    return '<div style="display:flex;gap:10px;margin:8px 0">' + ''.join(cards) + '</div>'