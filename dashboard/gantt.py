"""
gantt.py
--------
Builds Plotly Gantt charts for the factory scheduling dashboard.

Shows:
  - Completed jobs as solid colored bars (by job type)
  - In-progress jobs as semi-transparent animated bars
  - Broken machine periods as gray striped overlays
  - Current time step as a vertical dashed line
  - Deadline markers for in-progress jobs
"""

import plotly.graph_objects as go
import numpy as np

# ─────────────────────────────────────────────────────────────────
#  COLOR PALETTE
# ─────────────────────────────────────────────────────────────────

JOB_COLORS = {
    'A':      '#2563EB',  # Blue 600 — Short/fast jobs
    'B':      '#059669',  # Emerald 600 — Medium jobs
    'C':      '#D97706',  # Amber 600 — Long/heavy jobs
    'rush':   '#DC2626',  # Red 600 — Rush orders
    'broken': '#94A3B8',  # Slate 400 — Broken/unavailable
}

JOB_COLORS_LIGHT = {
    'A':    'rgba(37, 99, 235, 0.35)',
    'B':    'rgba(5, 150, 105, 0.35)',
    'C':    'rgba(217, 119, 6, 0.35)',
    'rush': 'rgba(220, 38, 38, 0.35)',
}

TYPE_LABELS = {
    'A': 'Type A — Short',
    'B': 'Type B — Medium',
    'C': 'Type C — Long',
    'rush': 'Rush Order',
    'broken': 'Breakdown',
}

MACHINE_LABELS = ['Machine 1', 'Machine 2', 'Machine 3']


# ─────────────────────────────────────────────────────────────────
#  MAIN BUILDER
# ─────────────────────────────────────────────────────────────────

def build_gantt(render_state: dict, env=None) -> go.Figure:
    """
    Build a Plotly horizontal bar Gantt chart from env render state.

    Args:
        render_state : Output of env.render()
        env          : FactoryGym instance (for in-progress job details)

    Returns:
        plotly.graph_objects.Figure
    """
    current_step = render_state.get('step', 0)
    max_steps    = render_state.get('max_steps', 100)
    gantt_jobs   = render_state.get('gantt_jobs', [])
    machines     = render_state.get('machines', [])
    spike_active = render_state.get('spike_active', False)

    fig = go.Figure()

    # ── Ghost anchor traces ───────────────────────────────────────
    # These invisible zero-width bars force Plotly to render the y-axis
    # as categorical (Machine 1/2/3) even before any real jobs complete.
    # Without these, Plotly defaults to a numeric y-axis (0,1,2,3...).
    for ml in MACHINE_LABELS:
        fig.add_trace(go.Bar(
            name='_anchor',
            x=[0],
            y=[ml],
            base=[0],
            orientation='h',
            marker=dict(color='rgba(0,0,0,0)', line=dict(width=0)),
            showlegend=False,
            hoverinfo='none',
        ))

    # Track which job types we've already added a legend entry for
    legend_shown = set()

    # ── Completed jobs ────────────────────────────────────────────
    for job in gantt_jobs:
        job_type = job['type']
        priority = job.get('priority', 'normal')

        # Rush orders get special color regardless of type
        color_key = 'rush' if priority == 'high' else job_type
        color     = JOB_COLORS.get(color_key, '#64748B')
        label     = TYPE_LABELS.get(color_key, f'Type {color_key}')

        show_legend = color_key not in legend_shown
        if show_legend:
            legend_shown.add(color_key)

        machine_label = MACHINE_LABELS[int(job['machine'])] \
            if job['machine'] is not None else 'Machine ?'

        tardiness = job.get('tardiness', 0)
        deadline  = job.get('deadline', 0)
        on_time   = tardiness == 0

        # Tardiness indicator: add small red marker if late
        border_color = '#DC2626' if not on_time else 'rgba(255,255,255,0.8)'
        line_width   = 2.5 if not on_time else 1

        hover_text = (
            f"<b>Job {job['id']}  ·  {label}</b><br>"
            f"Machine: {machine_label}<br>"
            f"Start: Step {job['start']}<br>"
            f"End:   Step {job['start'] + job['duration']}<br>"
            f"Deadline: Step {deadline}<br>"
            f"Status: {'✓ On Time' if on_time else f'⚠ {tardiness} steps late'}"
        )

        fig.add_trace(go.Bar(
            name=label,
            x=[job['duration']],
            y=[machine_label],
            base=[job['start']],
            orientation='h',
            marker=dict(
                color=color,
                line=dict(color=border_color, width=line_width),
                opacity=0.88,
            ),
            text=f"J{job['id']}",
            textposition='inside',
            textfont=dict(color='white', size=11, family='monospace'),
            hovertemplate=hover_text + '<extra></extra>',
            showlegend=show_legend,
            legendgroup=color_key,
        ))

    # ── In-progress jobs ──────────────────────────────────────────
    if env is not None:
        for i, machine in enumerate(machines):
            if machine['status'] == 'busy' and machine.get('current_job') is not None:
                # Access env directly for full job object
                try:
                    env_machine = env.machines[i]
                    job_obj     = env_machine.get('current_job')
                    if job_obj and job_obj.start_time is not None:
                        elapsed  = current_step - job_obj.start_time
                        color_lk = (
                            'rush' if job_obj.priority == 'high'
                            else job_obj.job_type
                        )
                        light_color = JOB_COLORS_LIGHT.get(
                            color_lk, 'rgba(100,116,139,0.25)'
                        )
                        machine_label = MACHINE_LABELS[i]

                        fig.add_trace(go.Bar(
                            name='In Progress',
                            x=[elapsed],
                            y=[machine_label],
                            base=[job_obj.start_time],
                            orientation='h',
                            marker=dict(
                                color=light_color,
                                line=dict(
                                    color=JOB_COLORS.get(color_lk, '#64748B'),
                                    width=2,
                                ),
                                pattern=dict(shape='/', size=6,
                                             solidity=0.4,
                                             fgcolor=JOB_COLORS.get(
                                                 color_lk, '#64748B')
                                             ),
                            ),
                            text=f'J{job_obj.job_id} ▶',
                            textposition='inside',
                            textfont=dict(size=10, color='#475569'),
                            hovertemplate=(
                                f"<b>Job {job_obj.job_id} — In Progress</b><br>"
                                f"Machine: {machine_label}<br>"
                                f"Elapsed: {elapsed} / {job_obj.processing_time} steps<br>"
                                f"Remaining: {machine['remaining']} steps<br>"
                                f"Deadline: Step {job_obj.deadline}"
                                "<extra></extra>"
                            ),
                            showlegend='In Progress' not in legend_shown,
                            legendgroup='inprogress',
                        ))
                        legend_shown.add('In Progress')
                except Exception:
                    pass

    # ── Broken machine overlays ───────────────────────────────────
    for i, machine in enumerate(machines):
        if machine['status'] == 'broken':
            machine_label = MACHINE_LABELS[i]
            repair_left   = machine.get('repair', 0)
            # Show gray overlay from now to now+repair_countdown
            fig.add_trace(go.Bar(
                name='Breakdown',
                x=[max(repair_left, 1)],
                y=[machine_label],
                base=[current_step],
                orientation='h',
                marker=dict(
                    color='rgba(148, 163, 184, 0.40)',
                    line=dict(color='#94A3B8', width=1.5),
                    pattern=dict(shape='x', size=5,
                                 solidity=0.35, fgcolor='#CBD5E1'),
                ),
                text='⚠ BREAKDOWN',
                textposition='inside',
                textfont=dict(size=10, color='#64748B'),
                hovertemplate=(
                    f"<b>Machine {i+1} — BREAKDOWN</b><br>"
                    f"Repair in: {repair_left} steps<extra></extra>"
                ),
                showlegend='Breakdown' not in legend_shown,
                legendgroup='broken',
            ))
            legend_shown.add('Breakdown')

    # ── Current step line ─────────────────────────────────────────
    fig.add_vline(
        x=current_step,
        line_width=2.5,
        line_dash='dash',
        line_color='#1E3A8A',
        annotation_text=f'  Step {current_step}',
        annotation_position='top right',
        annotation_font=dict(color='#1E3A8A', size=11),
    )

    # ── Energy spike background tint ──────────────────────────────
    if spike_active:
        fig.add_vrect(
            x0=current_step - 1,
            x1=current_step + 1,
            fillcolor='rgba(251, 191, 36, 0.12)',
            line_width=0,
        )

    # ── Layout ───────────────────────────────────────────────────
    x_range_end = max(max_steps, current_step + 5)

    fig.update_layout(
        barmode='overlay',
        height=260,
        margin=dict(l=0, r=30, t=20, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#F8FAFC',
        xaxis=dict(
            title='Time Steps',
            range=[0, x_range_end],
            tickfont=dict(size=11, color='#475569'),
            gridcolor='#E2E8F0',
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor='#CBD5E1',
        ),
        yaxis=dict(
            title=None,
            type='category',
            tickfont=dict(size=12, color='#1E293B'),
            categoryorder='array',
            categoryarray=MACHINE_LABELS[::-1],  # reversed = Machine 1 at top
            gridcolor='rgba(0,0,0,0)',
            showline=False,
            fixedrange=True,
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
            font=dict(size=11, color='#475569'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#E2E8F0',
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family='system-ui',
            bordercolor='#E2E8F0',
        ),
    )

    # Placeholder text when simulation hasn't started
    if not gantt_jobs and not any(m['status'] == 'busy' for m in machines):
        fig.add_annotation(
            text='⏳  Press ▶ Run in the sidebar to start the simulation',
            x=max_steps / 2,
            y=0.5,
            xref='x',
            yref='paper',
            showarrow=False,
            font=dict(size=13, color='#94A3B8'),
        )

    return fig


def build_machine_status_html(render_state: dict) -> str:
    """
    Build HTML for 3 machine status indicator cards.
    Returns an HTML string for use with st.markdown(unsafe_allow_html=True).
    """
    machines = render_state.get('machines', [])

    STATUS_STYLES = {
        'idle':   ('🟢', '#059669', '#ECFDF5', 'Idle'),
        'busy':   ('🔵', '#2563EB', '#EFF6FF', 'Working'),
        'broken': ('🔴', '#DC2626', '#FEF2F2', 'Breakdown!'),
    }

    cards = []
    for i, m in enumerate(machines):
        status = m.get('status', 'idle')
        icon, color, bg, label = STATUS_STYLES.get(
            status, ('⚪', '#64748B', '#F8FAFC', 'Unknown')
        )
        remaining = m.get('remaining', 0)
        repair    = m.get('repair', 0)

        sub = ''
        if status == 'busy' and remaining > 0:
            sub = f'<span style="color:#64748B;font-size:11px">{remaining} steps left</span>'
        elif status == 'broken' and repair > 0:
            sub = f'<span style="color:#DC2626;font-size:11px">Repair in {repair} steps</span>'
        elif status == 'idle':
            sub = f'<span style="color:#059669;font-size:11px">Ready for jobs</span>'

        cards.append(
            f'<div style="flex:1;background:{bg};border:1.5px solid {color}33;'
            f'border-radius:10px;padding:12px 14px;text-align:center;'
            f'box-shadow:0 1px 4px rgba(0,0,0,0.06)">'
            f'<div style="font-size:20px;margin-bottom:2px">{icon}</div>'
            f'<div style="font-weight:600;color:{color};font-size:13px;'
            f'margin:2px 0">Machine {i+1}</div>'
            f'<div style="font-size:12px;color:#475569;font-weight:500">{label}</div>'
            f'{sub}'
            f'</div>'
        )

    return (
        '<div style="display:flex;gap:10px;margin:10px 0">'
        + ''.join(cards) +
        '</div>'
    )