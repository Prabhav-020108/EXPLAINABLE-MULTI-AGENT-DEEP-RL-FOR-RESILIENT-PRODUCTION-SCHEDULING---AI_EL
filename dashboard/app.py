"""
app.py
------
Main Streamlit dashboard for the Explainable MAPPO Factory Scheduler.

Run with:
    streamlit run dashboard/app.py

Three tabs:
    Tab 1 — Live Schedule   : Real-time Gantt chart + machine status
    Tab 2 — AI Decisions    : SHAP feature importance + hypothesis badges
    Tab 3 — Fairness Audit  : Responsible AI monitoring + bias detection
"""

import streamlit as st
import sys
import os
import time
import json
import numpy as np

# ─────────────────────────────────────────────────────────────────
#  PATH SETUP
# ─────────────────────────────────────────────────────────────────

# Add project root to path so all imports resolve correctly
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(DASHBOARD_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dashboard.gantt          import build_gantt, build_machine_status_html
from dashboard.xai_panel      import (
    build_shap_chart, build_group_chart,
    build_hypothesis_html, build_action_card_html, build_narrative_html,
)
from dashboard.fairness_panel import (
    build_fairness_banner_html, build_metrics_html,
    build_tardiness_chart, build_energy_chart,
    build_wait_time_chart, build_per_type_summary_html,
    load_fairness_data,
)

# ─────────────────────────────────────────────────────────────────
#  ENVIRONMENT CONFIGURATION  (must match training)
# ─────────────────────────────────────────────────────────────────

ENV_CONFIG = {
    'n_machines':        3,
    'n_jobs':            6,
    'max_steps':         100,
    'breakdown_rate':    0.003,
    'rush_rate':         0.005,
    'energy_spike_rate': 0.05,
    'reward_weights': {
        'completion': 0.30,
        'tardiness':  0.40,
        'energy':     0.20,
        'idle':       0.10,
    },
    'max_energy_per_step': 9.0,
}

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIGURATION
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
/* ── Google Fonts ────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Global Reset ─────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #FFFFFF !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ── Hide Streamlit branding ────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}

/* ── Sidebar ────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #F8FAFC !important;
    border-right: 1px solid #E2E8F0 !important;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}
[data-testid="stSidebarNav"] { display: none !important; }

/* ── Tabs ───────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #F1F5F9 !important;
    border-radius: 12px !important;
    padding: 5px !important;
    gap: 4px !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    padding: 9px 22px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    color: #64748B !important;
    border: none !important;
    transition: all 0.18s ease !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #1E3A8A !important;
    box-shadow: 0 1px 8px rgba(30,58,138,0.12) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1rem !important;
}

/* ── Buttons ────────────────────────────────────── */
.stButton > button {
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.18s ease !important;
    border: none !important;
    width: 100% !important;
    padding: 9px 16px !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
}

/* ── Metrics ────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #F8FAFC !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
}
[data-testid="stMetric"] label {
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    color: #64748B !important;
}
[data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-weight: 800 !important;
    color: #0F172A !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Slider ─────────────────────────────────────── */
.stSlider [data-testid="stWidgetLabel"] {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #475569 !important;
}

/* ── Selectbox ──────────────────────────────────── */
.stSelectbox [data-testid="stWidgetLabel"] {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #475569 !important;
}

/* ── Plotly chart borders ───────────────────────── */
[data-testid="stPlotlyChart"] {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Divider ────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #E2E8F0 !important;
    margin: 8px 0 !important;
}

/* ── Section headers ────────────────────────────── */
.section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #94A3B8;
    margin: 12px 0 6px 0;
    padding-left: 2px;
}

/* ── LIVE pulse animation ───────────────────────── */
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.85); }
}
.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #10B981;
    border-radius: 50%;
    margin-right: 5px;
    animation: pulse-dot 1.4s ease-in-out infinite;
    vertical-align: middle;
}

/* ── Scrollbar ───────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F1F5F9; border-radius: 3px; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  PLOTLY RENDER HELPER  (handles Streamlit version differences)
# ─────────────────────────────────────────────────────────────────

def _plot(fig, key=None):
    """
    Render a Plotly figure. Handles the use_container_width → width
    deprecation that broke in Streamlit ≥ 1.45 / 2.x.
    """
    cfg = {'displayModeBar': False, 'staticPlot': False}
    kw  = {'key': key} if key else {}
    try:
        # Streamlit 2.x / latest 1.x syntax
        st.plotly_chart(fig, width='stretch', config=cfg, **kw)
    except TypeError:
        # Older Streamlit fallback
        st.plotly_chart(fig, use_container_width=True, config=cfg, **kw)


# ─────────────────────────────────────────────────────────────────
#  CACHED RESOURCE LOADING
# ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_resources():
    """Load MAPPO model + SHAP explainer once. Shared across all reruns."""
    from agents.mappo_agent import MAPPOAgent
    from xai.shap_explainer import SHAPExplainer

    agent    = MAPPOAgent()
    model_ok = False

    for path in ['models/mappo_factory_best.pth',
                 'models/mappo_factory_final.pth']:
        full = os.path.join(PROJECT_ROOT, path)
        if os.path.exists(full):
            agent.load(full)
            model_ok = True
            break

    if not model_ok:
        return agent, None, False

    agent.set_eval_mode()

    try:
        explainer = SHAPExplainer(
            agent.actors[0], ENV_CONFIG,
            n_background=50, verbose=False,
        )
    except Exception:
        explainer = None

    return agent, explainer, model_ok


# ─────────────────────────────────────────────────────────────────
#  SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────

def _init_session():
    """Initialise all session state variables on first load."""
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
        st.session_state.step_delay       = 0.40
        st.session_state.episode_done     = False
        st.session_state.episode_count    = 0
        st.session_state.total_steps      = 0
        st.session_state.last_explanation = None
        st.session_state.enable_shap      = True
        st.session_state.pending_disruption = None
        st.session_state.fairness_report  = _load_json_safe(
            os.path.join(PROJECT_ROOT, 'logs', 'fairness_report.json')
        )
        st.session_state.hypothesis_results = _load_json_safe(
            os.path.join(PROJECT_ROOT, 'logs', 'hypothesis_results.json')
        )
        st.session_state.session_ready    = True


def _load_json_safe(path: str) -> dict:
    """Load JSON file, returning {} if missing or malformed."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────
#  SIMULATION HELPERS
# ─────────────────────────────────────────────────────────────────

def _do_step(agent, explainer):
    """Run one simulation step and update session state."""
    env     = st.session_state.env
    obs     = st.session_state.obs
    auditor = st.session_state.auditor

    # Apply pending disruption before stepping
    if st.session_state.pending_disruption:
        etype, edata = st.session_state.pending_disruption
        _apply_disruption(env, etype, edata)
        st.session_state.pending_disruption = None

    # Agent prediction
    masks   = {
        f'machine_{i}': env.get_action_mask(i)
        for i in range(env.n_machines)
    }
    actions, _, _ = agent.predict(obs, masks)

    # Step environment
    count_before = len(env.completed_jobs)
    next_obs, _, terms, truncs, _ = env.step(actions)

    # Record new job completions for fairness audit
    for job in env.completed_jobs[count_before:]:
        if job.completion_time is not None:
            auditor.record_job_completion(
                job,
                machine_id=(job.assigned_machine if job.assigned_machine is not None else 0),
            )

    # SHAP explanation — only on job completion to keep it fast
    new_completions = env.completed_jobs[count_before:]
    if new_completions and explainer and st.session_state.enable_shap:
        try:
            result = explainer.explain(obs.get('machine_0', np.zeros(38)))
            st.session_state.last_explanation = result
        except Exception:
            pass

    # Update state
    st.session_state.obs         = next_obs
    st.session_state.state       = env.render()
    st.session_state.total_steps += 1

    # Check episode termination
    if terms.get('__all__', False) or truncs.get('__all__', False):
        st.session_state.episode_done  = True
        st.session_state.running       = False
        st.session_state.episode_count += 1


def _apply_disruption(env, evt_type: str, evt_data: dict):
    """Manually apply a disruption event to the environment."""
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
    """Reset environment for a new episode, preserving episode count."""
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


# ─────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────

def _render_header(state: dict, model_ok: bool):
    """Render the top header bar with live metrics."""
    step        = state.get('step', 0) if state else 0
    max_steps   = state.get('max_steps', 100) if state else 100
    metrics     = state.get('metrics', {}) if state else {}
    spike       = state.get('spike_active', False) if state else False
    energy_px   = state.get('energy_price', 1.0) if state else 1.0

    tardiness   = metrics.get('tardiness_rate', 0.0)
    episode_e   = metrics.get('episode_energy', 0.0)
    completed   = metrics.get('completed', 0)
    total_jobs  = metrics.get('total', 6)
    breakdowns  = metrics.get('breakdowns', 0)
    rush        = metrics.get('rush_orders', 0)

    # Status colors
    tard_col  = '#DC2626' if tardiness > 0.15 else ('#D97706' if tardiness > 0.08 else '#059669')
    running   = st.session_state.get('running', False)
    ep_done   = st.session_state.get('episode_done', False)

    live_html = ''
    if running:
        live_html = '<span class="live-dot"></span><span style="color:#059669;font-size:11px;font-weight:700;letter-spacing:0.5px">LIVE</span>'
    elif ep_done:
        live_html = '<span style="color:#D97706;font-size:11px;font-weight:700">✓ COMPLETE</span>'
    else:
        live_html = '<span style="color:#94A3B8;font-size:11px;font-weight:600">PAUSED</span>'

    spike_badge = ''
    if spike:
        spike_badge = (
            '<span style="background:#FFFBEB;border:1.5px solid #F59E0B;'
            'color:#D97706;font-size:10px;font-weight:700;padding:3px 10px;'
            'border-radius:20px;margin-left:8px">⚡ ENERGY SPIKE</span>'
        )

    model_badge = (
        '<span style="background:#ECFDF5;border:1px solid #059669;'
        'color:#065F46;font-size:10px;font-weight:600;padding:2px 8px;'
        'border-radius:20px">✓ Model Loaded</span>'
        if model_ok else
        '<span style="background:#FEF2F2;border:1px solid #DC2626;'
        'color:#991B1B;font-size:10px;font-weight:600;padding:2px 8px;'
        'border-radius:20px">⚠ Model Missing</span>'
    )

    # Progress bar
    pct = int(step / max(max_steps, 1) * 100)
    prog_bar = (
        f'<div style="background:#E2E8F0;border-radius:4px;height:5px;margin:4px 0 0 0;width:160px">'
        f'<div style="width:{pct}%;background:{"#DC2626" if ep_done else "#2563EB"};'
        f'border-radius:4px;height:5px;transition:width 0.3s ease"></div>'
        f'</div>'
    )

    st.markdown(f"""
    <div style="background:white;border-bottom:1.5px solid #E2E8F0;
    padding:10px 20px 10px 20px;display:flex;align-items:center;
    justify-content:space-between;margin-bottom:16px;
    box-shadow:0 1px 8px rgba(0,0,0,0.04)">

      <!-- Left: Brand -->
      <div style="display:flex;align-items:center;gap:12px">
        <div style="background:linear-gradient(135deg,#1E3A8A,#2563EB);
        width:38px;height:38px;border-radius:10px;display:flex;
        align-items:center;justify-content:center;font-size:20px;
        box-shadow:0 2px 8px rgba(37,99,235,0.3)">🏭</div>
        <div>
          <div style="font-size:16px;font-weight:800;color:#0F172A;
          letter-spacing:-0.3px">AI Factory Scheduler</div>
          <div style="font-size:11px;color:#94A3B8;margin-top:1px">
          Explainable MAPPO · Production Scheduling · Industry 4.0</div>
        </div>
        <div style="margin-left:8px">{live_html}{spike_badge}</div>
      </div>

      <!-- Centre: Episode progress -->
      <div style="text-align:center">
        <div style="font-size:11px;color:#64748B;font-weight:600;margin-bottom:2px">
        EPISODE {st.session_state.get('episode_count', 0) + 1}  ·  
        STEP {step} / {max_steps}</div>
        {prog_bar}
      </div>

      <!-- Right: Live KPIs -->
      <div style="display:flex;gap:16px;align-items:center">
        <div style="text-align:center">
          <div style="font-size:18px;font-weight:800;color:{tard_col};
          font-family:'JetBrains Mono',monospace">{tardiness:.0%}</div>
          <div style="font-size:10px;color:#94A3B8;font-weight:600">TARDINESS</div>
        </div>
        <div style="width:1px;height:30px;background:#E2E8F0"></div>
        <div style="text-align:center">
          <div style="font-size:18px;font-weight:800;color:#1E3A8A;
          font-family:'JetBrains Mono',monospace">{episode_e:.1f}</div>
          <div style="font-size:10px;color:#94A3B8;font-weight:600">ENERGY</div>
        </div>
        <div style="width:1px;height:30px;background:#E2E8F0"></div>
        <div style="text-align:center">
          <div style="font-size:18px;font-weight:800;color:#059669;
          font-family:'JetBrains Mono',monospace">{completed}/{total_jobs}</div>
          <div style="font-size:10px;color:#94A3B8;font-weight:600">JOBS DONE</div>
        </div>
        <div style="width:1px;height:30px;background:#E2E8F0"></div>
        <div style="text-align:center">
          <div style="font-size:18px;font-weight:800;color:#7C3AED;
          font-family:'JetBrains Mono',monospace">
          {st.session_state.get('total_steps', 0):,}</div>
          <div style="font-size:10px;color:#94A3B8;font-weight:600">TOTAL STEPS</div>
        </div>
        <div style="margin-left:8px">{model_badge}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────

def _render_sidebar():
    """Render the sidebar with all simulation controls."""
    with st.sidebar:

        # ── Logo area ────────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center;padding:8px 0 16px 0">
          <div style="font-size:28px">🏭</div>
          <div style="font-size:13px;font-weight:800;color:#1E3A8A">
          MAPPO Scheduler</div>
          <div style="font-size:10px;color:#94A3B8;margin-top:2px">
          Multi-Agent Deep RL · XAI</div>
        </div>
        <hr/>
        """, unsafe_allow_html=True)

        # ── Simulation controls ──────────────────────────────────
        st.markdown('<div class="section-label">⚙️ Simulation Controls</div>',
                    unsafe_allow_html=True)

        running   = st.session_state.running
        ep_done   = st.session_state.episode_done

        c1, c2 = st.columns(2)
        with c1:
            if not running and not ep_done:
                if st.button('▶ Run', key='btn_run',
                             help='Start continuous simulation'):
                    st.session_state.running = True
            elif running:
                if st.button('⏸ Pause', key='btn_pause'):
                    st.session_state.running = False
            else:
                st.button('▶ Run', key='btn_run2', disabled=True)

        with c2:
            if st.button('↺ Reset', key='btn_reset',
                         help='Reset to a new episode'):
                _reset_episode()

        # Step-by-step button (only when paused)
        if not running and not ep_done:
            if st.button('→ Step Once', key='btn_step',
                         help='Advance simulation by one step'):
                agent, explainer, model_ok = _load_resources()
                _do_step(agent, explainer)

        # Speed slider
        st.session_state.step_delay = st.slider(
            'Speed  (lower = faster)',
            min_value=0.05, max_value=2.0,
            value=st.session_state.step_delay,
            step=0.05,
            format='%.2fs',
            key='speed_slider',
        )

        # SHAP toggle
        st.session_state.enable_shap = st.toggle(
            '🔍 Live SHAP Explanations',
            value=st.session_state.enable_shap,
            help='Disable for faster simulation speed',
        )

        st.markdown('<hr/>', unsafe_allow_html=True)

        # ── Disruption controls ──────────────────────────────────
        st.markdown('<div class="section-label">⚠️ Trigger Disruptions</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size:11px;color:#94A3B8;margin-bottom:8px;line-height:1.5">
        Click any button below to inject a disruption into the running simulation.
        Agents will respond in real-time.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <style>
        div[data-testid="column"] .stButton > button[kind="secondary"] {
            background: #FEF2F2 !important;
            color: #DC2626 !important;
            border: 1.5px solid #FECACA !important;
        }
        </style>
        """, unsafe_allow_html=True)

        for i in range(3):
            label = f'🔧 Break Machine {i+1}'
            if st.button(label, key=f'btn_break_{i}',
                         help=f'Force Machine {i+1} into 10-step repair'):
                st.session_state.pending_disruption = (
                    'breakdown', {'machine_id': i}
                )
                if not st.session_state.running:
                    agent, explainer, _ = _load_resources()
                    _do_step(agent, explainer)

        if st.button('⚡ Add Rush Order', key='btn_rush',
                     help='Insert a high-priority urgent job into the queue'):
            st.session_state.pending_disruption = ('rush_order', {})
            if not st.session_state.running:
                agent, explainer, _ = _load_resources()
                _do_step(agent, explainer)

        if st.button('💰 Spike Energy Price ×2.5', key='btn_spike',
                     help='Multiply energy costs by 2.5x for 15 steps'):
            st.session_state.pending_disruption = ('energy_spike', {})
            if not st.session_state.running:
                agent, explainer, _ = _load_resources()
                _do_step(agent, explainer)

        st.markdown('<hr/>', unsafe_allow_html=True)

        # ── Episode statistics ───────────────────────────────────
        st.markdown('<div class="section-label">📈 Session Statistics</div>',
                    unsafe_allow_html=True)

        state   = st.session_state.state or {}
        metrics = state.get('metrics', {})

        stat_rows = [
            ('Episodes run',    st.session_state.episode_count),
            ('Total env steps', f"{st.session_state.total_steps:,}"),
            ('Jobs completed',  f"{metrics.get('completed', 0)} / {metrics.get('total', 6)}"),
            ('Breakdowns seen', metrics.get('breakdowns', 0)),
            ('Rush orders',     metrics.get('rush_orders', 0)),
        ]

        for label, val in stat_rows:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:5px 8px;border-radius:6px;margin-bottom:2px;'
                f'background:#F1F5F9">'
                f'<span style="font-size:11px;color:#475569">{label}</span>'
                f'<span style="font-size:11px;font-weight:700;color:#1E293B;'
                f'font-family:monospace">{val}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr/>', unsafe_allow_html=True)

        # ── Info footer ──────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center;padding-top:4px">
          <div style="font-size:10px;color:#CBD5E1;line-height:1.7">
          Explainable MAPPO · SHAP XAI<br>
          Fairness Auditor · Industry 4.0<br>
          <span style="color:#E2E8F0">─────────────────</span><br>
          EL Project · Semester 2
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  TAB 1 — LIVE SCHEDULE
# ─────────────────────────────────────────────────────────────────

def _render_tab_schedule():
    """Render the Live Schedule tab with Gantt chart."""
    state = st.session_state.state or {}
    env   = st.session_state.env

    # Machine status row
    st.markdown(
        build_machine_status_html(state),
        unsafe_allow_html=True,
    )

    # Gantt chart
    st.markdown(
        '<div style="background:#FAFBFC;border:1.5px solid #E2E8F0;'
        'border-radius:14px;padding:16px 12px 8px 12px;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.05)">'
        '<div style="font-size:12px;font-weight:700;color:#1E293B;'
        'margin-bottom:10px;padding-left:4px">📅 Production Schedule — Live Gantt</div>',
        unsafe_allow_html=True,
    )
    gantt_fig = build_gantt(state, env=env)
    _plot(gantt_fig, key='gantt_chart')
    st.markdown('</div>', unsafe_allow_html=True)

    # Metrics row
    st.markdown('<div style="margin-top:14px">', unsafe_allow_html=True)
    metrics  = state.get('metrics', {})
    machines = state.get('machines', [])

    tard     = metrics.get('tardiness_rate', 0.0)
    energy   = metrics.get('episode_energy', 0.0)
    done     = metrics.get('completed', 0)
    total    = metrics.get('total', 6)
    step     = state.get('step', 0)
    e_price  = state.get('energy_price', 1.0)
    spike    = state.get('spike_active', False)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric('🎯 Tardiness Rate',
                  f'{tard:.1%}',
                  delta='On Track ✓' if tard < 0.10 else '⚠ High',
                  delta_color='normal' if tard < 0.10 else 'inverse')
    with c2:
        st.metric('⚡ Episode Energy', f'{energy:.1f}')
    with c3:
        st.metric('✅ Jobs Complete', f'{done} / {total}')
    with c4:
        st.metric('⏱ Current Step', f'{step}')
    with c5:
        price_label = f'×{e_price:.1f}' + (' 🔥' if spike else '')
        st.metric('💰 Energy Price', price_label)

    st.markdown('</div>', unsafe_allow_html=True)

    # Job queue section
    available = state.get('available_jobs', [])
    if available:
        st.markdown('<div style="margin-top:14px">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:12px;font-weight:700;color:#1E293B;'
            'margin-bottom:8px">📋 Current Job Queue (sorted by deadline)</div>',
            unsafe_allow_html=True,
        )

        TYPE_COLORS_HEX = {
            'A': '#EFF6FF', 'B': '#ECFDF5', 'C': '#FFFBEB'
        }
        TYPE_BORDER = {
            'A': '#2563EB', 'B': '#059669', 'C': '#D97706'
        }

        cols = st.columns(min(len(available), 6))
        for idx, (col, job) in enumerate(zip(cols, available)):
            bg     = TYPE_COLORS_HEX.get(job['type'], '#F8FAFC')
            border = TYPE_BORDER.get(job['type'], '#CBD5E1')
            prio   = '🔴 RUSH' if job.get('priority') == 'high' else f"Type {job['type']}"
            with col:
                st.markdown(
                    f'<div style="background:{bg};border:1.5px solid {border}44;'
                    f'border-radius:10px;padding:10px;text-align:center">'
                    f'<div style="font-size:10px;font-weight:700;'
                    f'color:{border};margin-bottom:4px">{prio}</div>'
                    f'<div style="font-size:12px;font-weight:800;color:#0F172A">'
                    f'J{job["id"]}</div>'
                    f'<div style="font-size:10px;color:#64748B;margin-top:3px">'
                    f'⏱ {job["proc"]}s · 🎯 dl:{job["deadline"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  TAB 2 — AI DECISIONS (XAI)
# ─────────────────────────────────────────────────────────────────

def _render_tab_xai(explainer):
    """Render the AI Decisions / SHAP Explainability tab."""
    explanation = st.session_state.last_explanation
    hypo        = st.session_state.hypothesis_results

    # Section: Agent's Last Decision
    st.markdown(
        '<div style="font-size:13px;font-weight:700;color:#1E293B;'
        'margin-bottom:8px">🤖 Latest Agent Decision</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        build_action_card_html(explanation),
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    # Section: SHAP Feature Importance
    col_chart, col_group = st.columns([3, 2], gap='medium')

    with col_chart:
        st.markdown(
            '<div style="background:white;border:1.5px solid #E2E8F0;'
            'border-radius:14px;padding:16px;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.04)">'
            '<div style="font-size:12px;font-weight:700;color:#1E293B;'
            'margin-bottom:10px">📊 Top Feature Contributions (SHAP)</div>',
            unsafe_allow_html=True,
        )
        _plot(build_shap_chart(explanation), key='shap_bar')
        st.markdown('</div>', unsafe_allow_html=True)

        # Narrative below chart
        if explanation:
            st.markdown(
                build_narrative_html(explanation),
                unsafe_allow_html=True,
            )

    with col_group:
        st.markdown(
            '<div style="background:white;border:1.5px solid #E2E8F0;'
            'border-radius:14px;padding:16px;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.04)">'
            '<div style="font-size:12px;font-weight:700;color:#1E293B;'
            'margin-bottom:10px">🗂 Importance by Feature Group</div>',
            unsafe_allow_html=True,
        )
        _plot(build_group_chart(explanation), key='group_donut')
        st.markdown('</div>', unsafe_allow_html=True)

        # Manual explain button
        if explainer and not st.session_state.running:
            if st.button('🔍 Explain Current Observation',
                         key='btn_explain',
                         help='Compute SHAP for current machine_0 observation'):
                obs = st.session_state.obs
                if obs and 'machine_0' in obs:
                    with st.spinner('Computing SHAP explanation...'):
                        try:
                            result = explainer.explain(obs['machine_0'])
                            st.session_state.last_explanation = result
                        except Exception as e:
                            st.error(f'SHAP error: {e}')
                    st.rerun()

    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    # Section: Hypothesis Verification
    st.markdown(
        '<div style="font-size:13px;font-weight:700;color:#1E293B;'
        'margin-bottom:8px">🧪 Hypothesis Verification</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        build_hypothesis_html(hypo),
        unsafe_allow_html=True,
    )

    if not hypo:
        st.info(
            '💡 Run `python run_hypothesis_tests.py` to generate hypothesis '
            'results. They will appear here automatically.',
            icon=None,
        )

    # Feature index reference
    with st.expander('📖 Feature Index Reference (38 dimensions)'):
        import pandas as pd
        from xai.shap_explainer import FEATURE_NAMES, FEATURE_GROUPS

        rows = []
        for name, indices in FEATURE_GROUPS.items():
            for idx in indices:
                fn = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f'F{idx}'
                rows.append({'Group': name, 'Index': idx, 'Feature Name': fn})

        df = pd.DataFrame(rows)
        try:
            st.dataframe(df, width='stretch', hide_index=True, height=280)
        except TypeError:
            st.dataframe(df, use_container_width=True, hide_index=True, height=280)


# ─────────────────────────────────────────────────────────────────
#  TAB 3 — FAIRNESS AUDIT
# ─────────────────────────────────────────────────────────────────

def _render_tab_fairness():
    """Render the Responsible AI / Fairness Audit tab."""
    report = st.session_state.fairness_report

    # Reload button
    col_title, col_btn = st.columns([5, 1])
    with col_title:
        st.markdown(
            '<div style="font-size:13px;font-weight:700;color:#1E293B;'
            'padding-top:6px">📊 Responsible AI Audit — Fairness Report</div>',
            unsafe_allow_html=True,
        )
    with col_btn:
        if st.button('🔄 Reload', key='btn_reload_fairness',
                     help='Reload from logs/fairness_report.json'):
            st.session_state.fairness_report = _load_json_safe(
                os.path.join(PROJECT_ROOT, 'logs', 'fairness_report.json')
            )
            report = st.session_state.fairness_report
            st.rerun()

    # Big status banner
    st.markdown(
        build_fairness_banner_html(report),
        unsafe_allow_html=True,
    )

    if not report:
        st.markdown("""
        <div style="background:#F8FAFC;border:1.5px dashed #E2E8F0;border-radius:12px;
        padding:24px;text-align:center;color:#64748B;font-size:13px">
        <div style="font-size:32px;margin-bottom:10px">📊</div>
        <b>No fairness report found.</b><br><br>
        Run the evaluation script to generate the report:<br><br>
        <code style="background:#F1F5F9;padding:4px 12px;border-radius:6px;
        font-size:13px;color:#1E3A8A">python run_fairness_eval.py</code>
        </div>
        """, unsafe_allow_html=True)
        return

    # Per-type summary cards
    st.markdown(
        build_per_type_summary_html(report),
        unsafe_allow_html=True,
    )

    # Charts row
    col_tard, col_energy = st.columns([3, 2], gap='medium')

    with col_tard:
        st.markdown(
            '<div style="background:white;border:1.5px solid #E2E8F0;'
            'border-radius:14px;padding:16px;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.04)">'
            '<div style="font-size:12px;font-weight:700;color:#1E293B;'
            'margin-bottom:10px">📉 Tardiness Rate by Product Type</div>',
            unsafe_allow_html=True,
        )
        _plot(build_tardiness_chart(report), key='tard_chart')
        st.markdown('</div>', unsafe_allow_html=True)

    with col_energy:
        st.markdown(
            '<div style="background:white;border:1.5px solid #E2E8F0;'
            'border-radius:14px;padding:16px;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.04)">'
            '<div style="font-size:12px;font-weight:700;color:#1E293B;'
            'margin-bottom:10px">⚡ Energy Share by Product Type</div>',
            unsafe_allow_html=True,
        )
        _plot(build_energy_chart(report), key='energy_chart')
        st.markdown('</div>', unsafe_allow_html=True)

    # Wait time chart
    st.markdown(
        '<div style="background:white;border:1.5px solid #E2E8F0;'
        'border-radius:14px;padding:16px;margin-top:14px;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.04)">'
        '<div style="font-size:12px;font-weight:700;color:#1E293B;'
        'margin-bottom:10px">⏳ Average Wait Time by Product Type</div>',
        unsafe_allow_html=True,
    )
    _plot(build_wait_time_chart(report), key='wait_chart')
    st.markdown('</div>', unsafe_allow_html=True)

    # Detailed metrics table
    st.markdown('<div style="margin-top:14px">', unsafe_allow_html=True)
    st.markdown(
        build_metrics_html(report),
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  MAIN APP ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def main():
    # 1. Initialise session state
    _init_session()

    # 2. Load model and explainer (cached — only runs once)
    loading_placeholder = st.empty()
    with loading_placeholder:
        if 'resources_loaded' not in st.session_state:
            with st.spinner('🔄  Loading MAPPO model + SHAP explainer...  (30-60 sec first time)'):
                agent, explainer, model_ok = _load_resources()
                st.session_state.resources_loaded = True
        else:
            agent, explainer, model_ok = _load_resources()

    loading_placeholder.empty()

    # 3. Render sidebar
    _render_sidebar()

    # 4. Render header
    _render_header(st.session_state.state, model_ok)

    # 5. Render main tabs
    tab1, tab2, tab3 = st.tabs([
        '📅  Live Schedule',
        '🔍  AI Decisions',
        '⚖️  Fairness Audit',
    ])

    with tab1:
        _render_tab_schedule()

    with tab2:
        _render_tab_xai(explainer)

    with tab3:
        _render_tab_fairness()

    # 6. Auto-step loop
    if st.session_state.running and not st.session_state.episode_done:
        time.sleep(st.session_state.step_delay)
        _do_step(agent, explainer)
        st.rerun()

    # 7. Episode complete notification
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
            f'Tardiness: {tard:.1%}  ·  '
            f'Energy: {energy:.1f}  ·  '
            f'Press ↺ Reset in the sidebar to start a new episode.',
            icon=None,
        )


if __name__ == '__main__':
    main()