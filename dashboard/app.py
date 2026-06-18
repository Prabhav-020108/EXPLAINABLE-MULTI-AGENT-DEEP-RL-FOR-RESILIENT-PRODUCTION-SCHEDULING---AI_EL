"""
app.py  —  Explainable MAPPO Factory Scheduler  (Redesigned)
=============================================================
Premium dashboard with:
  🏭 Factory Floor  — Animated live simulation + Gantt chart
  🧠 AI Brain       — Plain-English SHAP explanations + hypothesis tests
  ⚖️  Fairness       — Live session fairness + historical audit

Run with:
    streamlit run dashboard/app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import sys
import os
import time
import json
import numpy as np

# ─────────────────────────────────────────────────────────────────
#  PATH SETUP
# ─────────────────────────────────────────────────────────────────
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(DASHBOARD_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dashboard.gantt     import build_gantt
from dashboard.xai_panel import (
    build_shap_chart, build_group_chart,
    build_hypothesis_html, build_narrative_html,
    build_decision_story_html,
)
from dashboard.fairness_panel import (
    build_fairness_banner_html, build_metrics_html,
    build_tardiness_chart, build_energy_chart,
    build_wait_time_chart, build_per_type_summary_html,
    load_fairness_data, build_live_fairness_html,
    build_bias_explanation_html,
)

# ─────────────────────────────────────────────────────────────────
#  ENVIRONMENT CONFIG  (must match training)
# ─────────────────────────────────────────────────────────────────
ENV_CONFIG = {
    'n_machines': 3, 'n_jobs': 6, 'max_steps': 100,
    'breakdown_rate': 0.003, 'rush_rate': 0.005, 'energy_spike_rate': 0.05,
    'reward_weights': {'completion': 0.30, 'tardiness': 0.40, 'energy': 0.20, 'idle': 0.10},
    'max_energy_per_step': 9.0,
}

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='AI Factory Scheduler',
    page_icon='🏭',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ─────────────────────────────────────────────────────────────────
#  PREMIUM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600;700&display=swap');

/* ── Reset & Base ───────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
.stApp { background: #F8FAFC !important; font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; padding-bottom: 1rem !important; max-width: 100% !important; }

/* ── Sidebar ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid #1E293B !important;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: white !important; }
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="stSidebar"] .block-container { padding: 1rem !important; }
[data-testid="stSidebar"] label { color: #94A3B8 !important; font-size: 11px !important; font-weight: 600 !important; letter-spacing: 0.5px !important; text-transform: uppercase !important; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] > div { color: #94A3B8 !important; }

/* ── Sidebar buttons ─────────────────────────────────────────── */
[data-testid="stSidebar"] .stButton > button {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    color: #E2E8F0 !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    width: 100% !important;
    padding: 8px 12px !important;
    transition: all 0.18s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #334155 !important;
    border-color: #475569 !important;
    transform: translateY(-1px) !important;
}

/* ── Primary action buttons ──────────────────────────────────── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    transition: all 0.18s ease !important;
    border: none !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important; }

/* ── Tabs ─────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #F1F5F9 !important;
    border-radius: 14px !important;
    padding: 6px !important;
    gap: 4px !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    color: #64748B !important;
    border: none !important;
    transition: all 0.2s ease !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #0F172A !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.10) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.25rem !important; }

/* ── Metric cards ─────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: white !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
}
[data-testid="stMetric"] label {
    font-size: 10px !important; font-weight: 700 !important;
    letter-spacing: 1px !important; text-transform: uppercase !important;
    color: #94A3B8 !important;
}
[data-testid="stMetricValue"] {
    font-size: 22px !important; font-weight: 800 !important;
    color: #0F172A !important; font-family: 'JetBrains Mono', monospace !important;
}

/* ── Expanders ───────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 12px !important;
    background: white !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: 13px !important;
    color: #1E293B !important;
}

/* ── Slider ──────────────────────────────────────────────────── */
[data-testid="stSlider"] [data-testid="stWidgetLabel"] { font-size: 11px !important; color: #94A3B8 !important; }

/* ── Toggle ──────────────────────────────────────────────────── */
[data-testid="stToggle"] label { font-size: 11px !important; color: #94A3B8 !important; }

/* ── Plotly charts ────────────────────────────────────────────── */
[data-testid="stPlotlyChart"] { border-radius: 12px !important; overflow: hidden !important; }

/* ── Scrollbar ────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }

/* ── Live pulse dot ───────────────────────────────────────────── */
@keyframes live-pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(0.8)} }
.live-dot { display:inline-block;width:8px;height:8px;background:#10B981;border-radius:50%;
  margin-right:5px;animation:live-pulse 1.4s ease-in-out infinite;vertical-align:middle; }

/* ── Section divider ──────────────────────────────────────────── */
hr { border:none!important; border-top:1px solid #E2E8F0!important; margin:10px 0!important; }

/* ── Custom section label ─────────────────────────────────────── */
.sec-label { font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
  color:#94A3B8;margin:14px 0 6px 2px; }

/* ── Card shell ───────────────────────────────────────────────── */
.card { background:white;border:1.5px solid #E2E8F0;border-radius:16px;padding:18px;
  box-shadow:0 2px 10px rgba(0,0,0,0.05); }
.card-header { font-size:12px;font-weight:700;color:#1E293B;
  letter-spacing:0.3px;margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  PLOTLY RENDER HELPER
# ─────────────────────────────────────────────────────────────────
def _plot(fig, key=None):
    cfg = {'displayModeBar': False}
    kw  = {'key': key} if key else {}
    try:
        st.plotly_chart(fig, width='stretch', config=cfg, **kw)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=cfg, **kw)


# ─────────────────────────────────────────────────────────────────
#  ANIMATED FACTORY FLOOR  (HTML component — the WOW element)
# ─────────────────────────────────────────────────────────────────
def _build_factory_floor_html(state: dict) -> str:
    """
    Build a fully animated factory-floor visualization as self-contained HTML.
    Shows 3 machine pods with animated progress rings, job queue, and step bar.
    """
    machines  = state.get('machines', [])
    available = state.get('available_jobs', [])
    step      = state.get('step', 0)
    max_steps = state.get('max_steps', 100)
    spike     = state.get('spike_active', False)
    energy_px = state.get('energy_price', 1.0)
    metrics   = state.get('metrics', {})
    completed = metrics.get('completed', 0)
    total     = metrics.get('total', 6)
    breakdowns = metrics.get('breakdowns', 0)

    # ── Build machine pods ────────────────────────────────────────
    machine_html = ''
    for m in machines:
        status    = m.get('status', 'idle')
        mid       = m.get('id', 0)
        current   = m.get('current_job', None)
        remaining = m.get('remaining', 0)
        repair    = m.get('repair', 0)

        if status == 'idle':
            color = '#10B981'; bg = '#022C22'; badge_col = '#6EE7B7'
            status_txt = 'READY'; job_txt = 'Awaiting job...'
            icon = '⚙️'; pct = 0; anim = ''; pulse = ''
        elif status == 'busy':
            color = '#3B82F6'; bg = '#1E3A8A'; badge_col = '#93C5FD'
            status_txt = 'WORKING'; job_txt = f'Job #{current}' if current is not None else 'Processing...'
            icon = '⚙️'; pct = max(5, min(95, int((1 - remaining / max(1, 20)) * 100)))
            anim = 'class="spin"'
            pulse = f'<div class="pulse-ring" style="border-color:{color}"></div>'
        else:  # broken
            color = '#EF4444'; bg = '#450A0A'; badge_col = '#FCA5A5'
            status_txt = f'REPAIR ({repair}s)'; job_txt = 'Machine down!'
            icon = '🔧'; pct = 0; anim = ''; pulse = ''

        r      = 28
        circ   = 6.2832 * r
        offset = circ * (1 - pct / 100)
        broken_anim = 'blink' if status == 'broken' else ''

        machine_html += f"""
        <div class="mpod {broken_anim}" style="border-color:{color}33">
          {pulse}
          <div class="mlabel">Machine {mid + 1}</div>
          <div class="ring-wrap">
            <svg width="76" height="76" viewBox="0 0 76 76">
              <circle cx="38" cy="38" r="{r}" fill="none" stroke="#1E293B" stroke-width="6"/>
              <circle cx="38" cy="38" r="{r}" fill="none" stroke="{color}" stroke-width="6"
                stroke-linecap="round" transform="rotate(-90 38 38)"
                stroke-dasharray="{circ:.2f}" stroke-dashoffset="{offset:.2f}"
                style="transition:stroke-dashoffset 0.6s ease"/>
            </svg>
            <div class="ring-center">
              <span {anim} style="font-size:20px">{icon}</span>
              <div style="font-size:10px;color:{color};font-weight:700;margin-top:1px">{pct}%</div>
            </div>
          </div>
          <div style="background:{bg};color:{badge_col};font-size:9px;font-weight:700;
            padding:3px 12px;border-radius:20px;letter-spacing:0.8px">{status_txt}</div>
          <div style="font-size:9px;color:#94A3B8;margin-top:4px;text-align:center;
            max-width:88px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{job_txt}</div>
        </div>"""

    # ── Build queue column ─────────────────────────────────────────
    if not available:
        queue_html = '<div style="color:#10B981;font-size:11px;font-weight:700;text-align:center;line-height:1.5">✅ All Jobs<br>Assigned!</div>'
    else:
        queue_html = ''
        for j in available[:5]:
            jt  = j.get('type', 'A')
            jid = j.get('id', 0)
            pri = j.get('priority', 'normal')
            if pri == 'high':
                col = '#DC2626'; lbl = f'🔴 J{jid} RUSH'
            else:
                col = {'A': '#2563EB', 'B': '#059669', 'C': '#D97706'}.get(jt, '#64748B')
                tlbl = {'A': 'Short', 'B': 'Med', 'C': 'Long'}.get(jt, jt)
                lbl = f'J{jid} · {tlbl}'
            queue_html += f'<div class="qjob" style="background:{col}22;border:1px solid {col}55;color:{col}">{lbl}</div>'
        if len(available) > 5:
            queue_html += f'<div style="color:#475569;font-size:9px;text-align:center;margin-top:2px">+{len(available)-5} more</div>'

    # ── Spike badge ────────────────────────────────────────────────
    spike_badge = ''
    if spike:
        spike_badge = f'<div class="spike">⚡ ENERGY SPIKE ×{energy_px:.1f}</div>'

    # ── Bottom stats bar ───────────────────────────────────────────
    prog_pct = int(step / max(max_steps, 1) * 100)
    td_rate  = metrics.get('tardiness_rate', 0)
    td_col   = '#10B981' if td_rate < 0.10 else ('#F59E0B' if td_rate < 0.20 else '#EF4444')

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;font-family:system-ui,sans-serif}}
body{{background:linear-gradient(135deg,#0F172A 0%,#1E293B 100%);
  padding:14px 14px 36px 14px;height:260px;overflow:hidden;position:relative;color:white}}
.outer{{display:flex;align-items:center;height:100%;gap:12px}}
/* Queue */
.qcol{{display:flex;flex-direction:column;gap:5px;min-width:78px}}
.qtitle{{font-size:9px;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:3px}}
.qjob{{padding:4px 8px;border-radius:6px;font-size:9px;font-weight:700;
  text-align:center;animation:qpulse 2s ease-in-out infinite}}
/* Arrow */
.arrow{{font-size:18px;color:#334155;animation:flow 1.5s ease-in-out infinite;flex-shrink:0}}
/* Machines */
.mrow{{display:flex;gap:10px;flex:1;justify-content:center;align-items:center}}
.mpod{{display:flex;flex-direction:column;align-items:center;gap:5px;
  background:#1E293B;border-radius:18px;padding:12px 10px;border:1.5px solid #334155;
  min-width:96px;position:relative}}
.mlabel{{font-size:9px;font-weight:700;color:#64748B;letter-spacing:1px;text-transform:uppercase}}
.ring-wrap{{position:relative;width:76px;height:76px}}
.ring-center{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center}}
.pulse-ring{{position:absolute;top:0;left:0;width:76px;height:76px;border-radius:50%;
  border:2px solid;animation:ringpulse 2s ease-out infinite}}
/* Spike badge */
.spike{{position:absolute;top:10px;right:10px;background:#FEF3C7;color:#D97706;
  font-size:9px;font-weight:700;padding:3px 10px;border-radius:20px;border:1px solid #D97706;
  animation:spikepulse 1s ease-in-out infinite}}
/* Bottom bar */
.bot{{position:absolute;bottom:0;left:0;right:0;height:30px;background:rgba(15,23,42,0.95);
  border-top:1px solid #1E293B;display:flex;align-items:center;gap:10px;padding:0 14px}}
.prog{{flex:1;height:4px;background:#334155;border-radius:2px}}
.prog-fill{{height:4px;background:linear-gradient(90deg,#2563EB,#0D9488);border-radius:2px;transition:width 0.5s}}
.stat{{font-size:9px;color:#64748B;font-family:monospace;white-space:nowrap}}
.blink{{animation:blinkanim 1s ease-in-out infinite}}
@keyframes ringpulse{{0%{{transform:scale(1);opacity:0.8}}100%{{transform:scale(1.5);opacity:0}}}}
@keyframes blinkanim{{0%,100%{{opacity:1}}50%{{opacity:0.5}}}}
@keyframes flow{{0%,100%{{opacity:0.3;transform:translateX(0)}}50%{{opacity:1;transform:translateX(5px)}}}}
@keyframes qpulse{{0%,100%{{opacity:1}}50%{{opacity:0.7}}}}
@keyframes spin{{from{{transform:rotate(0deg)}}to{{transform:rotate(360deg)}}}}
@keyframes spikepulse{{0%,100%{{transform:scale(1)}}50%{{transform:scale(1.04)}}}}
.spin{{animation:spin 1.8s linear infinite;display:inline-block}}
</style></head><body>
<div class="outer">
  <div class="qcol">
    <div class="qtitle">📋 Queue</div>
    {queue_html}
  </div>
  <div class="arrow">→</div>
  <div class="mrow">{machine_html}</div>
</div>
{spike_badge}
<div class="bot">
  <div class="stat">✅ {completed}/{total} jobs</div>
  <div class="prog"><div class="prog-fill" style="width:{prog_pct}%"></div></div>
  <div class="stat" style="color:{td_col}">📉 {td_rate:.0%} late</div>
  <div class="stat">⏱ Step {step}/{max_steps}</div>
  {f'<div class="stat" style="color:#EF4444">⚠ {breakdowns} breakdown(s)</div>' if breakdowns > 0 else ''}
</div>
</body></html>"""
    return html


# ─────────────────────────────────────────────────────────────────
#  JOB QUEUE  (plain-English cards)
# ─────────────────────────────────────────────────────────────────
def _render_job_queue(available_jobs: list, current_step: int):
    """Render the job queue with clear, human-readable job cards."""
    if not available_jobs:
        st.markdown(
            '<div style="background:#ECFDF5;border:1.5px solid #059669;border-radius:12px;'
            'padding:16px;text-align:center;color:#065F46;font-weight:600;font-size:14px">'
            '✅  All jobs have been assigned to machines — simulation running!'
            '</div>', unsafe_allow_html=True)
        return

    TYPE_INFO = {
        'A': ('⚡ Short Job',  'Completes in 3–6 steps. Fast & low energy.',    '#2563EB', '#EFF6FF'),
        'B': ('📦 Medium Job', 'Completes in 7–12 steps. Balanced workload.',   '#059669', '#ECFDF5'),
        'C': ('🏭 Long Job',   'Completes in 13–20 steps. High energy job.',    '#D97706', '#FFFBEB'),
    }

    for job in available_jobs:
        jt   = job.get('type', 'A')
        jid  = job.get('id', 0)
        proc = job.get('proc', 0)
        dl   = job.get('deadline', 0)
        nrg  = job.get('energy', 0)
        pri  = job.get('priority', 'normal')
        steps_left = dl - current_step

        if pri == 'high':
            col = '#DC2626'; bg = '#FEF2F2'
            type_label = '🔴 RUSH ORDER'
            type_desc  = 'Urgent job! Must be done ASAP.'
            urgency_badge = '<span style="background:#DC2626;color:white;font-size:10px;font-weight:700;padding:2px 10px;border-radius:20px;margin-left:8px">⚠ CRITICAL</span>'
        else:
            label, desc, col, bg = TYPE_INFO.get(jt, ('Unknown', '', '#64748B', '#F8FAFC'))
            type_label = label
            type_desc  = desc
            if steps_left <= 2:
                urgency_badge = '<span style="background:#DC2626;color:white;font-size:10px;font-weight:700;padding:2px 10px;border-radius:20px;margin-left:8px">🔴 CRITICAL</span>'
            elif steps_left <= 5:
                urgency_badge = '<span style="background:#D97706;color:white;font-size:10px;font-weight:700;padding:2px 10px;border-radius:20px;margin-left:8px">🟡 URGENT</span>'
            else:
                urgency_badge = '<span style="background:#059669;color:white;font-size:10px;font-weight:700;padding:2px 10px;border-radius:20px;margin-left:8px">🟢 On Time</span>'

        nrg_label = 'Low ⚡' if nrg < 0.3 else ('Medium ⚡⚡' if nrg < 0.6 else 'High ⚡⚡⚡')

        st.markdown(f"""
        <div style="background:{bg};border:1.5px solid {col}44;border-left:4px solid {col};
        border-radius:12px;padding:12px 16px;margin-bottom:8px;
        box-shadow:0 2px 6px rgba(0,0,0,0.04)">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
            <div style="display:flex;align-items:center;gap:8px">
              <span style="font-size:15px;font-weight:800;color:{col}">Job #{jid}</span>
              <span style="font-size:13px;font-weight:600;color:{col}">{type_label}</span>
              {urgency_badge}
            </div>
            <span style="font-size:11px;color:#94A3B8">{type_desc}</span>
          </div>
          <div style="display:flex;gap:20px;flex-wrap:wrap">
            <div style="font-size:12px;color:#475569">
              <span style="color:#94A3B8">⏱ Processing time:</span>
              <b style="color:#1E293B;margin-left:4px">{proc} steps</b>
            </div>
            <div style="font-size:12px;color:#475569">
              <span style="color:#94A3B8">📅 Deadline:</span>
              <b style="color:#1E293B;margin-left:4px">Step {dl}</b>
              <span style="color:{col};font-size:11px;margin-left:4px">({steps_left} steps left)</span>
            </div>
            <div style="font-size:12px;color:#475569">
              <span style="color:#94A3B8">🔋 Energy:</span>
              <b style="color:#1E293B;margin-left:4px">{nrg_label}</b>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  PERFORMANCE COMPARISON  (MAPPO vs baselines)
# ─────────────────────────────────────────────────────────────────
def _load_benchmark_data() -> dict | None:
    """Load benchmark results CSV for AI vs baseline comparison."""
    csv_path = os.path.join(PROJECT_ROOT, 'logs', 'benchmark_results.csv')
    if not os.path.exists(csv_path):
        return None
    try:
        import csv
        rows = {}
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                rows[row['method']] = row
        return rows
    except Exception:
        return None


def _build_performance_comparison_html(data: dict) -> str:
    """Build a visual MAPPO vs baselines comparison card."""
    if not data:
        return ''

    METHODS = [
        ('FCFS',        '🔴', 'First Come First Served (old way)'),
        ('SPT',         '🟡', 'Shortest Processing Time (old way)'),
        ('EDD',         '🟠', 'Earliest Due Date (old way)'),
        ('MAPPO_best',  '🟢', 'Our AI (MAPPO) — best checkpoint'),
    ]

    rows_html = ''
    best_tard = float(data.get('MAPPO_best', {}).get('mean_tardiness', 0.5) or 0.5)

    for method, dot, label in METHODS:
        d = data.get(method)
        if not d:
            continue
        tard = float(d.get('mean_tardiness', 0))
        ms   = float(d.get('mean_makespan', 0))
        enrg = float(d.get('mean_energy', 0))
        is_ai = 'MAPPO' in method

        tard_col = '#059669' if is_ai else '#64748B'
        bg = 'linear-gradient(135deg,#ECFDF5,#D1FAE5)' if is_ai else 'transparent'
        border = '1.5px solid #059669' if is_ai else '1px solid #F1F5F9'
        weight = '700' if is_ai else '400'

        improvement = ''
        if is_ai and best_tard > 0:
            fcfs_tard = float(data.get('FCFS', {}).get('mean_tardiness', 0.35) or 0.35)
            if fcfs_tard > 0:
                impr = (fcfs_tard - best_tard) / fcfs_tard * 100
                improvement = f'<span style="background:#059669;color:white;font-size:9px;font-weight:700;padding:1px 8px;border-radius:20px;margin-left:8px">▲ {impr:.0f}% better than FCFS</span>'

        rows_html += f"""
        <div style="display:flex;align-items:center;gap:12px;padding:9px 12px;
          background:{bg};border:{border};border-radius:10px;margin-bottom:5px">
          <span style="font-size:14px;width:20px">{dot}</span>
          <div style="flex:1">
            <div style="font-size:12px;font-weight:{weight};color:#1E293B">{label} {improvement}</div>
          </div>
          <div style="font-size:11px;color:{tard_col};font-weight:{weight};font-family:monospace;min-width:60px;text-align:right">{tard:.1%} late</div>
          <div style="font-size:11px;color:#64748B;font-family:monospace;min-width:50px;text-align:right">{ms:.1f}s</div>
        </div>"""

    if not rows_html:
        return ''

    return (
        '<div style="background:white;border:1.5px solid #E2E8F0;border-radius:14px;padding:14px">'
        '<div style="font-size:12px;font-weight:700;color:#1E293B;margin-bottom:10px">'
        '🏆 AI vs Traditional Methods</div>'
        '<div style="display:flex;justify-content:flex-end;gap:16px;margin-bottom:6px">'
        '<div style="font-size:10px;color:#94A3B8;font-family:monospace">Late Jobs</div>'
        '<div style="font-size:10px;color:#94A3B8;font-family:monospace">Speed</div>'
        '</div>'
        + rows_html + '</div>'
    )


# ─────────────────────────────────────────────────────────────────
#  CACHED RESOURCE LOADING
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_resources():
    """Load MAPPO model + SHAP explainer once, shared across all reruns."""
    from agents.mappo_agent import MAPPOAgent
    from xai.shap_explainer import SHAPExplainer

    agent    = MAPPOAgent()
    model_ok = False

    for path in ['models/mappo_factory_best.pth', 'models/mappo_factory_final.pth']:
        full = os.path.join(PROJECT_ROOT, path)
        if os.path.exists(full):
            agent.load(full)
            model_ok = True
            break

    if not model_ok:
        return agent, None, False

    agent.set_eval_mode()

    try:
        explainer = SHAPExplainer(agent.actors[0], ENV_CONFIG, n_background=50, verbose=False)
    except Exception:
        explainer = None

    return agent, explainer, model_ok


# ─────────────────────────────────────────────────────────────────
#  SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────
def _load_json_safe(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _init_session():
    from env.factory_gym import FactoryGym
    from fairness.auditor import FairnessAuditor

    if 'session_ready' not in st.session_state:
        env = FactoryGym(ENV_CONFIG)
        obs, _ = env.reset(seed=0)

        st.session_state.env              = env
        st.session_state.obs              = obs
        st.session_state.state            = env.render()
        st.session_state.auditor          = FairnessAuditor()
        st.session_state.running          = False
        st.session_state.step_delay       = 0.30
        st.session_state.episode_done     = False
        st.session_state.episode_count    = 0
        st.session_state.total_steps      = 0
        st.session_state.last_explanation = None
        st.session_state.enable_shap      = True
        st.session_state.pending_disruption = None
        st.session_state.decision_log     = []   # recent AI decisions
        st.session_state.fairness_report  = _load_json_safe(
            os.path.join(PROJECT_ROOT, 'logs', 'fairness_report.json'))
        st.session_state.hypothesis_results = _load_json_safe(
            os.path.join(PROJECT_ROOT, 'logs', 'hypothesis_results.json'))
        st.session_state.session_ready    = True


# ─────────────────────────────────────────────────────────────────
#  SIMULATION HELPERS
# ─────────────────────────────────────────────────────────────────
def _do_step(agent, explainer):
    """Run one simulation step and update all session state."""
    env     = st.session_state.env
    obs     = st.session_state.obs
    auditor = st.session_state.auditor

    if st.session_state.pending_disruption:
        etype, edata = st.session_state.pending_disruption
        _apply_disruption(env, etype, edata)
        st.session_state.pending_disruption = None

    masks   = {f'machine_{i}': env.get_action_mask(i) for i in range(env.n_machines)}
    actions, _, probs_dict = agent.predict(obs, masks)

    count_before = len(env.completed_jobs)
    next_obs, _, terms, truncs, _ = env.step(actions)

    for job in env.completed_jobs[count_before:]:
        if job.completion_time is not None:
            auditor.record_job_completion(
                job,
                machine_id=(job.assigned_machine if job.assigned_machine is not None else 0),
            )

    # Log AI decision for the decision feed
    log_entry = {
        'step':    env.current_step,
        'actions': {k: int(v) for k, v in actions.items()},
    }
    log = st.session_state.decision_log
    log.append(log_entry)
    if len(log) > 8:
        log.pop(0)

    # SHAP explanation on job completion
    new_completions = env.completed_jobs[count_before:]
    if new_completions and explainer and st.session_state.enable_shap:
        try:
            result = explainer.explain(obs.get('machine_0', np.zeros(38)))
            st.session_state.last_explanation = result
        except Exception:
            pass

    st.session_state.obs         = next_obs
    st.session_state.state       = env.render()
    st.session_state.total_steps += 1

    if terms.get('__all__', False) or truncs.get('__all__', False):
        st.session_state.episode_done  = True
        st.session_state.running       = False
        st.session_state.episode_count += 1


def _apply_disruption(env, evt_type: str, evt_data: dict):
    from env.job_generator import create_rush_order

    if evt_type == 'breakdown':
        mid = evt_data.get('machine_id', 0)
        m   = env.machines[mid]
        if m['status'] != 'broken':
            if m['current_job'] is not None:
                paused = m['current_job']
                paused.reset_assignment()
                env.available_jobs.append(paused)
                env.available_jobs.sort(key=lambda j: j.deadline)
                m['current_job'] = None
            m['status']           = 'broken'
            m['remaining_steps']  = 0
            m['repair_countdown'] = 10

    elif evt_type == 'rush_order':
        rush = create_rush_order(env.current_step, env._rush_id_counter)
        env._rush_id_counter += 1
        env.available_jobs.append(rush)
        env.available_jobs.sort(key=lambda j: j.deadline)

    elif evt_type == 'energy_spike':
        env.disruption_mgr.current_spike_factor  = 2.5
        env.disruption_mgr.spike_remaining_steps = 15


def _reset_episode():
    from env.factory_gym import FactoryGym
    from fairness.auditor import FairnessAuditor

    seed    = st.session_state.episode_count + 1
    new_env = FactoryGym(ENV_CONFIG)
    obs, _  = new_env.reset(seed=seed)

    st.session_state.env              = new_env
    st.session_state.obs              = obs
    st.session_state.state            = new_env.render()
    st.session_state.auditor          = FairnessAuditor()
    st.session_state.running          = False
    st.session_state.episode_done     = False
    st.session_state.last_explanation = None
    st.session_state.pending_disruption = None
    st.session_state.decision_log     = []


# ─────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────
def _render_header(state: dict, model_ok: bool):
    step    = state.get('step', 0) if state else 0
    mx      = state.get('max_steps', 100) if state else 100
    metrics = state.get('metrics', {}) if state else {}
    spike   = state.get('spike_active', False) if state else False
    e_price = state.get('energy_price', 1.0) if state else 1.0

    tard    = metrics.get('tardiness_rate', 0.0)
    energy  = metrics.get('episode_energy', 0.0)
    done    = metrics.get('completed', 0)
    total   = metrics.get('total', 6)

    running  = st.session_state.get('running', False)
    ep_done  = st.session_state.get('episode_done', False)
    ep_count = st.session_state.get('episode_count', 0)
    tot_steps = st.session_state.get('total_steps', 0)

    # Status badge
    if running:
        status_html = '<span class="live-dot"></span><span style="color:#10B981;font-size:11px;font-weight:700;letter-spacing:0.8px">LIVE</span>'
    elif ep_done:
        status_html = '<span style="color:#D97706;font-size:11px;font-weight:700">✓ COMPLETE</span>'
    else:
        status_html = '<span style="color:#64748B;font-size:11px;font-weight:600">PAUSED</span>'

    spike_badge = ''
    if spike:
        spike_badge = (
            '<span style="background:#FFFBEB;border:1.5px solid #D97706;color:#D97706;'
            'font-size:10px;font-weight:700;padding:3px 10px;border-radius:20px;margin-left:10px">'
            '⚡ ENERGY SPIKE</span>'
        )

    model_badge = (
        '<span style="background:#ECFDF5;border:1px solid #059669;color:#065F46;'
        'font-size:10px;font-weight:600;padding:2px 10px;border-radius:20px">✓ AI Loaded</span>'
        if model_ok else
        '<span style="background:#FEF2F2;border:1px solid #DC2626;color:#991B1B;'
        'font-size:10px;font-weight:600;padding:2px 10px;border-radius:20px">⚠ Model Missing</span>'
    )

    tard_col = '#DC2626' if tard > 0.15 else ('#D97706' if tard > 0.08 else '#059669')
    prog_pct = int(step / max(mx, 1) * 100)

    st.markdown(f"""
    <div style="background:white;border-bottom:2px solid #F1F5F9;
    padding:12px 24px;display:flex;align-items:center;justify-content:space-between;
    margin-bottom:16px;box-shadow:0 2px 12px rgba(0,0,0,0.06)">

      <!-- Left: Brand -->
      <div style="display:flex;align-items:center;gap:14px">
        <div style="background:linear-gradient(135deg,#0F172A,#2563EB);width:42px;height:42px;
        border-radius:12px;display:flex;align-items:center;justify-content:center;
        font-size:22px;box-shadow:0 4px 12px rgba(37,99,235,0.35)">🏭</div>
        <div>
          <div style="font-size:17px;font-weight:900;color:#0F172A;letter-spacing:-0.5px">
            AI Factory Scheduler</div>
          <div style="font-size:11px;color:#94A3B8;margin-top:1px">
            MAPPO · Multi-Agent Deep RL · SHAP Explainability · Industry 4.0</div>
        </div>
        <div style="margin-left:6px">{status_html}{spike_badge}</div>
      </div>

      <!-- Centre: Progress -->
      <div style="text-align:center">
        <div style="font-size:11px;color:#64748B;font-weight:600;margin-bottom:4px">
          Episode {ep_count + 1}  ·  Step {step} / {mx}
        </div>
        <div style="width:180px;height:6px;background:#F1F5F9;border-radius:3px;overflow:hidden">
          <div style="height:6px;background:{'#DC2626' if ep_done else 'linear-gradient(90deg,#2563EB,#0D9488)'};
          border-radius:3px;width:{prog_pct}%;transition:width 0.4s"></div>
        </div>
      </div>

      <!-- Right: KPIs -->
      <div style="display:flex;gap:20px;align-items:center">
        <div style="text-align:center">
          <div style="font-size:20px;font-weight:800;color:{tard_col};font-family:'JetBrains Mono',monospace">{tard:.0%}</div>
          <div style="font-size:9px;color:#94A3B8;font-weight:700;letter-spacing:0.5px;text-transform:uppercase">Tardiness</div>
        </div>
        <div style="width:1px;height:32px;background:#E2E8F0"></div>
        <div style="text-align:center">
          <div style="font-size:20px;font-weight:800;color:#1E3A8A;font-family:'JetBrains Mono',monospace">{energy:.1f}</div>
          <div style="font-size:9px;color:#94A3B8;font-weight:700;letter-spacing:0.5px;text-transform:uppercase">Energy</div>
        </div>
        <div style="width:1px;height:32px;background:#E2E8F0"></div>
        <div style="text-align:center">
          <div style="font-size:20px;font-weight:800;color:#059669;font-family:'JetBrains Mono',monospace">{done}/{total}</div>
          <div style="font-size:9px;color:#94A3B8;font-weight:700;letter-spacing:0.5px;text-transform:uppercase">Jobs Done</div>
        </div>
        <div style="width:1px;height:32px;background:#E2E8F0"></div>
        <div style="text-align:center">
          <div style="font-size:20px;font-weight:800;color:#7C3AED;font-family:'JetBrains Mono',monospace">{tot_steps:,}</div>
          <div style="font-size:9px;color:#94A3B8;font-weight:700;letter-spacing:0.5px;text-transform:uppercase">Total Steps</div>
        </div>
        <div style="margin-left:6px">{model_badge}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
def _render_sidebar():
    with st.sidebar:
        # Brand
        st.markdown("""
        <div style="text-align:center;padding:10px 0 18px">
          <div style="font-size:32px">🏭</div>
          <div style="font-size:14px;font-weight:800;color:white;margin-top:4px">MAPPO Scheduler</div>
          <div style="font-size:10px;color:#64748B;margin-top:2px">Multi-Agent Deep RL · XAI</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-label">⚙️ Simulation Controls</div>', unsafe_allow_html=True)

        running  = st.session_state.running
        ep_done  = st.session_state.episode_done

        c1, c2 = st.columns(2)
        with c1:
            if not running and not ep_done:
                if st.button('▶ Run', key='btn_run', help='Start auto simulation'):
                    st.session_state.running = True
            elif running:
                if st.button('⏸ Pause', key='btn_pause'):
                    st.session_state.running = False
            else:
                st.button('▶ Run', key='btn_run2', disabled=True)
        with c2:
            if st.button('↺ Reset', key='btn_reset', help='Start a new episode'):
                _reset_episode()

        if not running and not ep_done:
            if st.button('→ Step Once', key='btn_step', help='Advance 1 time step'):
                agent, explainer, _ = _load_resources()
                _do_step(agent, explainer)

        st.session_state.step_delay = st.slider(
            'Speed (lower = faster)', min_value=0.05, max_value=2.0,
            value=st.session_state.step_delay, step=0.05, format='%.2fs')

        st.session_state.enable_shap = st.toggle(
            '🔍 Live SHAP Analysis', value=st.session_state.enable_shap,
            help='Disable for faster simulation')

        st.markdown('<hr/><div class="sec-label">⚡ Trigger Disruptions</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:10px;color:#475569;margin-bottom:8px;line-height:1.6">'
            'Inject a disruption to test AI resilience. Watch the AI adapt in real-time!</div>',
            unsafe_allow_html=True)

        for i in range(3):
            if st.button(f'🔧 Break Machine {i+1}', key=f'brk_{i}',
                         help=f'Force Machine {i+1} into a 10-step repair'):
                st.session_state.pending_disruption = ('breakdown', {'machine_id': i})
                if not st.session_state.running:
                    agent, explainer, _ = _load_resources()
                    _do_step(agent, explainer)

        if st.button('🔴 Add Rush Order', key='btn_rush',
                     help='Insert a high-priority urgent job immediately'):
            st.session_state.pending_disruption = ('rush_order', {})
            if not st.session_state.running:
                agent, explainer, _ = _load_resources()
                _do_step(agent, explainer)

        if st.button('💰 Spike Energy ×2.5', key='btn_spike',
                     help='Electricity cost jumps for 15 steps'):
            st.session_state.pending_disruption = ('energy_spike', {})
            if not st.session_state.running:
                agent, explainer, _ = _load_resources()
                _do_step(agent, explainer)

        # Session stats
        st.markdown('<hr/><div class="sec-label">📊 Session Stats</div>', unsafe_allow_html=True)
        state   = st.session_state.state or {}
        metrics = state.get('metrics', {})

        stat_rows = [
            ('Episodes run',   st.session_state.episode_count),
            ('Total steps',    f"{st.session_state.total_steps:,}"),
            ('Jobs done',      f"{metrics.get('completed',0)}/{metrics.get('total',6)}"),
            ('Breakdowns',     metrics.get('breakdowns', 0)),
            ('Rush orders',    metrics.get('rush_orders', 0)),
        ]
        for lbl, val in stat_rows:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:5px 8px;'
                f'background:#1E293B;border-radius:6px;margin-bottom:3px">'
                f'<span style="font-size:11px;color:#94A3B8">{lbl}</span>'
                f'<span style="font-size:11px;font-weight:700;color:#E2E8F0;font-family:monospace">{val}</span>'
                f'</div>', unsafe_allow_html=True)

        # What is this project?
        st.markdown('<hr/>', unsafe_allow_html=True)
        with st.expander("📖 What is this project?"):
            st.markdown("""
**MAPPO Factory Scheduler** is a university EL project demonstrating AI-driven production scheduling.

**The Setup:**
- A virtual factory has **3 machines** and **6 jobs** to process
- Jobs arrive with **deadlines**, **processing times**, and **energy costs**
- The AI (MAPPO) decides in real-time which job to assign to which machine

**The AI:**
- Uses **Multi-Agent Deep RL** — one AI agent per machine
- Learned by running **1 million simulation episodes**
- Outperforms traditional methods like FCFS and EDD

**The Innovations:**
- ⚡ **SHAP** explains every AI decision
- ⚖️ **Fairness Monitor** detects scheduling bias
- 🚨 **Disruption recovery** — AI adapts in milliseconds
            """, unsafe_allow_html=True)

        st.markdown(
            '<div style="text-align:center;padding-top:8px">'
            '<div style="font-size:9px;color:#334155;line-height:1.8">'
            'EL Project · Semester 2<br>Industry 4.0 · Smart Manufacturing</div></div>',
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  TAB 1 — FACTORY FLOOR
# ─────────────────────────────────────────────────────────────────
def _render_tab_schedule():
    state = st.session_state.state or {}
    env   = st.session_state.env

    # ── Animated Factory Floor ─────────────────────────────────────
    st.markdown(
        '<div style="background:white;border:1.5px solid #E2E8F0;border-radius:16px;'
        'overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.07);margin-bottom:16px">'
        '<div style="padding:12px 16px 0;display:flex;align-items:center;gap:8px">'
        '<div style="font-size:13px;font-weight:700;color:#1E293B">🏭 Live Factory Floor</div>'
        '<div style="font-size:11px;color:#94A3B8;margin-left:auto">Machines processing jobs in real-time</div>'
        '</div>',
        unsafe_allow_html=True)

    floor_html = _build_factory_floor_html(state)
    components.html(floor_html, height=262, scrolling=False)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Gantt Chart ────────────────────────────────────────────────
    st.markdown(
        '<div style="background:white;border:1.5px solid #E2E8F0;border-radius:16px;'
        'padding:16px;box-shadow:0 2px 10px rgba(0,0,0,0.05);margin-bottom:16px">'
        '<div style="font-size:13px;font-weight:700;color:#1E293B;margin-bottom:4px">📅 Production Gantt Chart — Job Timeline</div>'
        '<div style="font-size:11px;color:#94A3B8;margin-bottom:10px">'
        'Each bar = one job being processed. Blue = Short job, Green = Medium job, Orange = Long job, Red = Rush order.</div>',
        unsafe_allow_html=True)

    gantt_fig = build_gantt(state, env=env)
    _plot(gantt_fig, key='main_gantt')
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Key Metrics Row ────────────────────────────────────────────
    metrics  = state.get('metrics', {})
    step     = state.get('step', 0)
    e_price  = state.get('energy_price', 1.0)
    spike    = state.get('spike_active', False)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        tard = metrics.get('tardiness_rate', 0)
        st.metric('🎯 Late Jobs', f'{tard:.1%}',
                  delta='Good ✓' if tard < 0.10 else 'High ⚠',
                  delta_color='normal' if tard < 0.10 else 'inverse')
    with c2:
        st.metric('⚡ Energy Used', f"{metrics.get('episode_energy',0):.1f}")
    with c3:
        st.metric('✅ Jobs Done', f"{metrics.get('completed',0)} / {metrics.get('total',6)}")
    with c4:
        st.metric('⏱ Step', f'{step}')
    with c5:
        lbl = f'×{e_price:.1f}' + (' 🔥' if spike else '')
        st.metric('💰 Energy Price', lbl)

    # ── Performance Comparison ─────────────────────────────────────
    benchmark_data = _load_benchmark_data()
    if benchmark_data:
        st.markdown('<div style="margin-top:16px">', unsafe_allow_html=True)
        st.markdown(
            _build_performance_comparison_html(benchmark_data),
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Job Queue (plain English) ──────────────────────────────────
    available = state.get('available_jobs', [])
    st.markdown('<div style="margin-top:16px">', unsafe_allow_html=True)

    with st.expander(
        f"📋 Production Queue — {len(available)} job(s) waiting to be assigned"
        if available else "📋 Production Queue — All jobs assigned ✅",
        expanded=bool(available)
    ):
        st.markdown(
            '<div style="background:#F0F9FF;border:1px solid #BAE6FD;border-radius:10px;'
            'padding:12px 14px;margin-bottom:12px;font-size:12px;color:#0C4A6E">'
            '<b>ℹ️ What is this?</b> These are manufacturing jobs waiting to be assigned to one of the 3 machines. '
            'The AI (MAPPO) automatically picks the best machine for each job every time step. '
            'Jobs are sorted by deadline — most urgent first.'
            '</div>', unsafe_allow_html=True)
        _render_job_queue(available, step)

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  TAB 2 — AI BRAIN
# ─────────────────────────────────────────────────────────────────
def _render_tab_xai(explainer):
    explanation = st.session_state.last_explanation
    hypo        = st.session_state.hypothesis_results

    # ── What is this? ──────────────────────────────────────────────
    with st.expander("❓ What is the 'AI Brain' tab?", expanded=False):
        st.markdown("""
**This tab shows you HOW and WHY the AI makes its scheduling decisions.**

- The AI picks which job to assign to which machine every single time step
- **SHAP** (SHapley Additive exPlanations) is a mathematical method that shows which factors influenced the AI's decision and by how much
- Think of it as opening the "black box" of the AI — you can see its reasoning

**How to read the charts:**
- Longer bar = that factor was MORE important for the decision
- "Machine 2 is free" being important means: the AI noticed Machine 2 was available
- "Most urgent job — deadline closeness" being high means: the AI is prioritising jobs that are almost overdue

**The hypothesis badges** test whether the AI is reasoning correctly:
- ✅ Urgency Priority: Does the AI pick jobs with urgent deadlines first?
- ✅ Energy Awareness: Does the AI react when electricity prices spike?
- ✅ Smart Ordering: Does the AI rank more-urgent jobs higher than less-urgent ones?
        """)

    # ── Decision Story (plain English) ────────────────────────────
    st.markdown(
        '<div style="font-size:13px;font-weight:700;color:#1E293B;margin-bottom:10px">'
        '🤖 Latest AI Decision — Plain English</div>',
        unsafe_allow_html=True)

    st.markdown(build_decision_story_html(explanation), unsafe_allow_html=True)

    # ── SHAP Charts ─────────────────────────────────────────────────
    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    col_bars, col_donut = st.columns([3, 2], gap='medium')

    with col_bars:
        st.markdown(
            '<div class="card"><div class="card-header">📊 Feature Importance — What influenced the AI?</div>',
            unsafe_allow_html=True)
        _plot(build_shap_chart(explanation), key='shap_bars')
        st.markdown(build_narrative_html(explanation), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_donut:
        st.markdown(
            '<div class="card"><div class="card-header">🗂 Importance by Category</div>',
            unsafe_allow_html=True)
        _plot(build_group_chart(explanation), key='shap_donut')
        st.markdown(
            '<div style="font-size:11px;color:#94A3B8;line-height:1.6;margin-top:8px">'
            'Each slice shows how much a <b>category</b> of factors influenced the decision. '
            '"Machine Status" = which machines are free/busy. "Job 0" = the most urgent job.'
            '</div>', unsafe_allow_html=True)
        if explainer and not st.session_state.running:
            if st.button('🔍 Explain Current State', key='btn_explain_now',
                         help='Run SHAP analysis on the current factory observation'):
                obs = st.session_state.obs
                if obs and 'machine_0' in obs:
                    with st.spinner('Computing SHAP explanations...'):
                        try:
                            result = explainer.explain(obs['machine_0'])
                            st.session_state.last_explanation = result
                        except Exception as e:
                            st.error(f'SHAP error: {e}')
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── AI Decision Log ─────────────────────────────────────────────
    decision_log = st.session_state.get('decision_log', [])
    if decision_log:
        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
        with st.expander(f"📜 Recent AI Decisions Log ({len(decision_log)} entries)", expanded=False):
            st.markdown(
                '<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:10px">'
                '<div style="font-size:11px;color:#94A3B8;margin-bottom:8px">'
                'A live log of what the AI decided at each time step. '
                'Action 0 = assign most urgent job. Action 6 = wait.</div>',
                unsafe_allow_html=True)
            ACTION_LABELS = {
                0: 'Assign most urgent job',
                1: 'Assign 2nd job',
                2: 'Assign 3rd job',
                3: 'Assign 4th job',
                4: 'Assign 5th job',
                5: 'Assign 6th job',
                6: 'Wait (no assignment)',
            }
            for entry in reversed(decision_log[-5:]):
                s = entry['step']
                acs = entry['actions']
                parts = []
                for mid in range(3):
                    aid = f'machine_{mid}'
                    a = acs.get(aid, 6)
                    col = '#2563EB' if a < 6 else '#94A3B8'
                    parts.append(f'<span style="color:{col}">M{mid+1}: {ACTION_LABELS.get(a, f"Act {a}")}</span>')
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:12px;padding:6px 0;'
                    f'border-bottom:1px solid #F1F5F9;font-size:11px">'
                    f'<span style="color:#64748B;font-family:monospace;min-width:60px">Step {s}</span>'
                    f'{"  ·  ".join(parts)}'
                    f'</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Hypothesis Verification ─────────────────────────────────────
    st.markdown(
        '<div style="height:14px"></div>'
        '<div style="font-size:13px;font-weight:700;color:#1E293B;margin-bottom:10px">'
        '🧪 AI Behaviour Verification — Do the agents think correctly?</div>',
        unsafe_allow_html=True)

    st.markdown(build_hypothesis_html(hypo), unsafe_allow_html=True)

    if not hypo:
        st.info('💡 Run `python run_hypothesis_tests.py` to generate hypothesis results. They will appear here automatically.', icon=None)

    # Feature reference
    with st.expander("📖 Full feature index — what every SHAP feature means"):
        try:
            import pandas as pd
            from xai.shap_explainer import FEATURE_NAMES, FEATURE_GROUPS
            rows = []
            for gname, indices in FEATURE_GROUPS.items():
                for idx in indices:
                    fn = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f'F{idx}'
                    rows.append({'Feature Group': gname, 'Index': idx, 'Raw Name': fn})
            df = pd.DataFrame(rows)
            try:
                st.dataframe(df, width='stretch', hide_index=True, height=280)
            except TypeError:
                st.dataframe(df, use_container_width=True, hide_index=True, height=280)
        except Exception:
            st.info('Feature index unavailable.')


# ─────────────────────────────────────────────────────────────────
#  TAB 3 — FAIRNESS AUDIT
# ─────────────────────────────────────────────────────────────────
def _render_tab_fairness():
    report  = st.session_state.fairness_report
    auditor = st.session_state.auditor

    # ── What is fairness? ──────────────────────────────────────────
    with st.expander("❓ What does 'Fairness' mean in this project?", expanded=False):
        st.markdown("""
**Fairness auditing** checks whether the AI treats all job types equally.

We have 3 job types:
- **Type A** — Short jobs (3–6 steps) 🔵
- **Type B** — Medium jobs (7–12 steps) 🟢
- **Type C** — Long jobs (13–20 steps) 🟠

**What we check (5 fairness metrics):**
1. **Tardiness Spread** — Are all types equally likely to be late?
2. **Disparity Ratio** — Is the worst-treated type at most 2× worse than the best?
3. **Energy Balance** — Does one type use all the energy resources?
4. **Priority Fairness** — Do rush orders get completed faster than normal jobs?
5. **Wait Time Balance** — Do all types wait similar amounts before starting?

**Why Type C often shows BIAS DETECTED:**
Type C jobs take 13–20 steps. If all jobs arrive at the same time, long jobs naturally have to wait longer in queue, giving them less time to meet their deadline. This is a structural challenge — the AI is improving on traditional methods but has not fully solved it yet.

**The "Live Session" section** below shows fairness from your current run — which often looks better than the historical evaluation because the live config uses fewer disruptions.
        """)

    # ── Live Session Fairness ──────────────────────────────────────
    col_reload, _ = st.columns([5, 1])
    with _:
        if st.button('🔄 Reload Report', key='btn_reload_fairness',
                     help='Reload the historical fairness report from logs/'):
            st.session_state.fairness_report = _load_json_safe(
                os.path.join(PROJECT_ROOT, 'logs', 'fairness_report.json'))
            st.rerun()

    st.markdown(
        '<div style="font-size:13px;font-weight:700;color:#1E293B;margin-bottom:8px">'
        '🟢 Live Session — Fairness from Your Current Run</div>',
        unsafe_allow_html=True)

    st.markdown(build_live_fairness_html(auditor), unsafe_allow_html=True)

    # ── Historical Report ──────────────────────────────────────────
    st.markdown(
        '<div style="font-size:13px;font-weight:700;color:#1E293B;margin:16px 0 8px">'
        '📊 Historical Evaluation — 50 Episodes with Training-Level Disruptions</div>',
        unsafe_allow_html=True)

    st.markdown(build_fairness_banner_html(report), unsafe_allow_html=True)

    if not report:
        st.markdown("""
        <div style="background:#F8FAFC;border:1.5px dashed #CBD5E1;border-radius:12px;
        padding:24px;text-align:center;color:#64748B;font-size:13px">
        <div style="font-size:32px;margin-bottom:10px">📊</div>
        <b>No historical report found.</b><br><br>
        Run: <code style="background:#F1F5F9;padding:4px 12px;border-radius:6px">
        python run_fairness_eval.py</code>
        </div>
        """, unsafe_allow_html=True)
        return

    # Show bias explanation if relevant
    if report.get('fairness_status') == 'BIAS_DETECTED':
        st.markdown(build_bias_explanation_html(), unsafe_allow_html=True)

    # Per-type summary cards
    st.markdown(build_per_type_summary_html(report), unsafe_allow_html=True)

    # Charts
    c_tard, c_energy = st.columns([3, 2], gap='medium')

    with c_tard:
        st.markdown(
            '<div class="card"><div class="card-header">📉 Tardiness Rate — Which job type misses deadlines most?</div>',
            unsafe_allow_html=True)
        _plot(build_tardiness_chart(report), key='tard_chart')
        st.markdown('</div>', unsafe_allow_html=True)

    with c_energy:
        st.markdown(
            '<div class="card"><div class="card-header">⚡ Energy Share — How is electricity use distributed?</div>',
            unsafe_allow_html=True)
        _plot(build_energy_chart(report), key='energy_chart')
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="card" style="margin-top:14px">'
        '<div class="card-header">⏳ Wait Time — Which job type waits longest before being processed?</div>',
        unsafe_allow_html=True)
    _plot(build_wait_time_chart(report), key='wait_chart')
    st.markdown('</div>', unsafe_allow_html=True)

    # Metrics table
    st.markdown('<div style="margin-top:14px">', unsafe_allow_html=True)
    st.markdown(build_metrics_html(report), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    _init_session()

    # Load model (cached)
    loading_ph = st.empty()
    with loading_ph:
        if 'resources_loaded' not in st.session_state:
            with st.spinner('⚙️ Loading MAPPO model & SHAP explainer... (30–60 sec first time)'):
                agent, explainer, model_ok = _load_resources()
                st.session_state.resources_loaded = True
        else:
            agent, explainer, model_ok = _load_resources()
    loading_ph.empty()

    _render_sidebar()
    _render_header(st.session_state.state, model_ok)

    tab1, tab2, tab3 = st.tabs([
        '🏭  Factory Floor',
        '🧠  AI Brain',
        '⚖️  Fairness Audit',
    ])

    with tab1:
        _render_tab_schedule()
    with tab2:
        _render_tab_xai(explainer)
    with tab3:
        _render_tab_fairness()

    # Auto-step loop
    if st.session_state.running and not st.session_state.episode_done:
        time.sleep(st.session_state.step_delay)
        _do_step(agent, explainer)
        st.rerun()

    # Episode complete banner
    if st.session_state.episode_done:
        state   = st.session_state.state or {}
        metrics = state.get('metrics', {})
        tard    = metrics.get('tardiness_rate', 0)
        energy  = metrics.get('episode_energy', 0)
        done    = metrics.get('completed', 0)
        total   = metrics.get('total', 6)
        st.success(
            f'✅ **Episode Complete!**  '
            f'Jobs: {done}/{total}  ·  '
            f'Late: {tard:.1%}  ·  '
            f'Energy: {energy:.1f}  —  '
            f'Press ↺ Reset in the sidebar for the next episode.')


if __name__ == '__main__':
    main()