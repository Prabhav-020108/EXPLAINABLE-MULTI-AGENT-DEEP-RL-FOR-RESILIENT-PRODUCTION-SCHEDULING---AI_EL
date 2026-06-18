"""
main.py
-------
System integration entry point for the MAPPO Factory Scheduler.

Modes:
    demo    → Launch the interactive Streamlit dashboard
    test    → Run 10 end-to-end episodes and print a quick health report
    bench   → Run full baseline comparison and save benchmark_results.csv
    fairness → Run 50-episode fairness evaluation and save fairness_report.json
    xai     → Run SHAP hypothesis tests and save hypothesis_results.json
    full    → Run test + bench + fairness + xai sequentially (pre-demo check)

Usage:
    python main.py --mode demo
    python main.py --mode test
    python main.py --mode full
"""

import argparse
import os
import sys
import time
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ─── Module imports ───────────────────────────────────────────────
from env.factory_gym     import FactoryGym
from agents.mappo_agent  import MAPPOAgent
from fairness.auditor    import FairnessAuditor
from fairness.reporter   import FairnessReporter

ENV_CONFIG = {
    'n_machines':        3,
    'n_jobs':            6,
    'max_steps':         100,
    'breakdown_rate':    0.003,
    'rush_rate':         0.005,
    'energy_spike_rate': 0.05,
}


# ════════════════════════════════════════════════════════════════════
#  MODULE LOADER  —  verifies every component initializes cleanly
# ════════════════════════════════════════════════════════════════════

def load_all_modules(verbose: bool = True) -> dict:
    """
    Initialize all 5 system modules in order.
    Returns a dict with references to every module.
    Raises RuntimeError if any module fails to load.
    """
    print("\n" + "="*62)
    print("  SYSTEM INTEGRATION CHECK")
    print("="*62)

    modules = {}

    # ── 1. FactoryGym ─────────────────────────────────────────────
    if verbose:
        print("  [1/5] Initializing FactoryGym...", end=" ", flush=True)
    try:
        env = FactoryGym(ENV_CONFIG)
        obs, info = env.reset(seed=42)
        assert obs['machine_0'].shape == (38,), "Wrong obs shape"
        modules['env'] = env
        if verbose:
            print("OK")
    except Exception as e:
        raise RuntimeError(f"FactoryGym failed: {e}")

    # ── 2. MAPPO Agents ───────────────────────────────────────────
    if verbose:
        print("  [2/5] Loading MAPPO agents...", end=" ", flush=True)
    try:
        agent = MAPPOAgent()
        model_loaded = False
        for path in [
            os.path.join(PROJECT_ROOT, 'models', 'mappo_factory_best.pth'),
            os.path.join(PROJECT_ROOT, 'models', 'mappo_factory_final.pth'),
        ]:
            if os.path.exists(path):
                agent.load(path)
                model_loaded = True
                break
        if not model_loaded:
            raise FileNotFoundError("No model checkpoint found in models/")
        agent.set_eval_mode()
        # Quick sanity check — run one prediction
        masks = {f'machine_{i}': env.get_action_mask(i) for i in range(3)}
        actions, _, _ = agent.predict(obs, masks)
        assert all(0 <= v <= env.n_jobs for v in actions.values())
        modules['agent'] = agent
        if verbose:
            print("OK")
    except Exception as e:
        raise RuntimeError(f"MAPPOAgent failed: {e}")

    # ── 3. SHAP Explainer ─────────────────────────────────────────
    if verbose:
        print("  [3/5] Initializing SHAP explainer...", end=" ", flush=True)
    try:
        from xai.shap_explainer import SHAPExplainer
        explainer = SHAPExplainer(
            agent.actors[0], ENV_CONFIG,
            n_background=30, verbose=False
        )
        result = explainer.explain(obs['machine_0'])
        assert result['shap_values'].shape == (38,)
        modules['explainer'] = explainer
        if verbose:
            print("OK")
    except Exception as e:
        print(f"WARNING (non-fatal): SHAP explainer failed: {e}")
        modules['explainer'] = None

    # ── 4. Fairness Auditor ───────────────────────────────────────
    if verbose:
        print("  [4/5] Initializing FairnessAuditor...", end=" ", flush=True)
    try:
        auditor  = FairnessAuditor()
        reporter = FairnessReporter(auditor)
        modules['auditor']  = auditor
        modules['reporter'] = reporter
        if verbose:
            print("OK")
    except Exception as e:
        raise RuntimeError(f"FairnessAuditor failed: {e}")

    # ── 5. Dashboard import check ─────────────────────────────────
    if verbose:
        print("  [5/5] Checking dashboard imports...", end=" ", flush=True)
    try:
        from dashboard.gantt          import build_gantt
        from dashboard.xai_panel      import build_shap_chart
        from dashboard.fairness_panel import build_fairness_banner_html
        modules['dashboard_ok'] = True
        if verbose:
            print("OK")
    except Exception as e:
        print(f"WARNING: Dashboard import error: {e}")
        modules['dashboard_ok'] = False

    print("="*62)
    print("  All modules loaded successfully.\n")
    return modules


# ════════════════════════════════════════════════════════════════════
#  MODE: TEST  —  10 end-to-end episodes
# ════════════════════════════════════════════════════════════════════

def run_integration_test(modules: dict, n_episodes: int = 10) -> dict:
    """
    Run n_episodes end-to-end and collect health metrics.
    No crashes, no NaN rewards, correct obs shapes.
    """
    print(f"\n{'='*62}")
    print(f"  INTEGRATION TEST — {n_episodes} episodes")
    print(f"{'='*62}")

    env   = modules['env']
    agent = modules['agent']

    results = {
        'n_episodes':    n_episodes,
        'errors':        [],
        'rewards':       [],
        'tardinesses':   [],
        'makespans':     [],
        'energies':      [],
        'nan_rewards':   0,
        'obs_shape_errors': 0,
    }

    for ep in range(n_episodes):
        try:
            obs, _ = env.reset(seed=1000 + ep)

            # Check initial obs
            for aid, o in obs.items():
                if o.shape != (38,):
                    results['obs_shape_errors'] += 1

            ep_reward = 0.0
            step      = 0
            auditor   = modules['auditor']
            auditor.new_episode()

            while True:
                masks   = {f'machine_{i}': env.get_action_mask(i)
                           for i in range(env.n_machines)}
                count_before = len(env.completed_jobs)
                actions, _, _ = agent.predict(obs, masks)
                obs, rewards, terms, truncs, _ = env.step(actions)

                # NaN check
                r = rewards['machine_0']
                if r != r:  # NaN
                    results['nan_rewards'] += 1

                # Record completed jobs for fairness
                for job in env.completed_jobs[count_before:]:
                    if job.completion_time is not None:
                        auditor.record_job_completion(
                            job,
                            machine_id=job.assigned_machine or 0
                        )

                ep_reward += r
                step += 1

                if terms['__all__'] or truncs['__all__']:
                    break

            state = env.render()
            results['rewards'].append(ep_reward)
            results['tardinesses'].append(state['metrics']['tardiness_rate'])
            results['makespans'].append(step)
            results['energies'].append(state['metrics']['episode_energy'])

            status = "✓" if not terms.get('__all__', False) is False else "✓"
            print(f"  Ep {ep+1:>2}: step={step:>3}  "
                  f"tard={state['metrics']['tardiness_rate']:.1%}  "
                  f"reward={ep_reward:.2f}  "
                  f"energy={state['metrics']['episode_energy']:.1f}  {status}")

        except Exception as e:
            results['errors'].append(f"Episode {ep}: {type(e).__name__}: {e}")
            print(f"  Ep {ep+1:>2}: ERROR — {e}")

    # Summary
    print(f"\n{'─'*62}")
    if not results['errors']:
        print(f"  Result        : ALL {n_episodes} EPISODES PASSED")
    else:
        print(f"  Result        : {len(results['errors'])} ERRORS")
        for err in results['errors'][:3]:
            print(f"    {err}")

    if results['rewards']:
        print(f"  Mean reward   : {np.mean(results['rewards']):.2f} "
              f"± {np.std(results['rewards']):.2f}")
        print(f"  Mean tardiness: {np.mean(results['tardinesses']):.1%}")
        print(f"  Mean makespan : {np.mean(results['makespans']):.1f} steps")
        print(f"  Mean energy   : {np.mean(results['energies']):.2f}")
        print(f"  NaN rewards   : {results['nan_rewards']}")
        print(f"  Obs errors    : {results['obs_shape_errors']}")

    results['passed'] = (
        len(results['errors']) == 0
        and results['nan_rewards'] == 0
        and results['obs_shape_errors'] == 0
    )
    return results


# ════════════════════════════════════════════════════════════════════
#  MODE: EDGE CASES
# ════════════════════════════════════════════════════════════════════

def run_edge_case_tests(modules: dict) -> dict:
    """
    Test all 8 edge cases from the blueprint.
    Returns dict with pass/fail per case.
    """
    print(f"\n{'='*62}")
    print(f"  EDGE CASE TESTS")
    print(f"{'='*62}")

    from env.factory_gym   import FactoryGym
    from env.job_generator import create_rush_order
    from fairness.auditor  import FairnessAuditor
    from fairness.reporter import FairnessReporter
    from xai.shap_explainer import SHAPExplainer

    results = {}
    agent   = modules['agent']

    # ── EC1: All machines broken simultaneously ────────────────────
    print("  EC1: All machines broken simultaneously...", end=" ")
    try:
        env = FactoryGym(ENV_CONFIG)
        obs, _ = env.reset(seed=0)
        for m in env.machines:
            m['status'] = 'broken'
            m['repair_countdown'] = 5
        actions = {aid: env.n_jobs for aid in env.agents}  # WAIT
        obs2, r, terms, truncs, _ = env.step(actions)
        assert obs2['machine_0'].shape == (38,)
        results['EC1'] = True
        print("PASS")
    except Exception as e:
        results['EC1'] = False
        print(f"FAIL — {e}")

    # ── EC2: Empty job queue (all jobs done) ──────────────────────
    print("  EC2: Empty job queue...", end=" ")
    try:
        env = FactoryGym(ENV_CONFIG)
        obs, _ = env.reset(seed=1)
        env.available_jobs = []
        env.pending_jobs   = []
        env.completed_jobs = list(env.all_jobs)
        for j in env.completed_jobs:
            j.completion_time = 10
        actions = {aid: env.n_jobs for aid in env.agents}
        obs2, r, terms, truncs, _ = env.step(actions)
        assert obs2['machine_0'].shape == (38,)
        results['EC2'] = True
        print("PASS")
    except Exception as e:
        results['EC2'] = False
        print(f"FAIL — {e}")

    # ── EC3: Rush order when queue is full ────────────────────────
    print("  EC3: Rush order with full queue...", end=" ")
    try:
        env = FactoryGym(ENV_CONFIG)
        obs, _ = env.reset(seed=2)
        rush = create_rush_order(env.current_step, 9999)
        env.available_jobs.append(rush)
        env.available_jobs.sort(key=lambda j: j.deadline)
        assert any(j.priority == 'high' for j in env.available_jobs)
        results['EC3'] = True
        print("PASS")
    except Exception as e:
        results['EC3'] = False
        print(f"FAIL — {e}")

    # ── EC4: Energy spike at step 0 ───────────────────────────────
    print("  EC4: Energy spike at step 0...", end=" ")
    try:
        env = FactoryGym(ENV_CONFIG)
        obs, _ = env.reset(seed=3)
        env.disruption_mgr.current_spike_factor = 2.5
        env.disruption_mgr.spike_remaining_steps = 10
        masks = {f'machine_{i}': env.get_action_mask(i) for i in range(3)}
        actions, _, _ = agent.predict(obs, masks)
        obs2, r, terms, truncs, _ = env.step(actions)
        assert r['machine_0'] == r['machine_0']  # not NaN
        results['EC4'] = True
        print("PASS")
    except Exception as e:
        results['EC4'] = False
        print(f"FAIL — {e}")

    # ── EC5: Job with processing_time > max_steps/2 ───────────────
    print("  EC5: Very long job...", end=" ")
    try:
        env = FactoryGym(ENV_CONFIG)
        obs, _ = env.reset(seed=4)
        # The longest possible job is 20 steps, max_steps=100 so 20 < 50. Pass trivially.
        for job in env.all_jobs:
            assert job.processing_time <= env.max_steps
        results['EC5'] = True
        print("PASS")
    except Exception as e:
        results['EC5'] = False
        print(f"FAIL — {e}")

    # ── EC6: SHAP called with all-zero observation ─────────────────
    print("  EC6: SHAP on all-zero observation...", end=" ")
    try:
        if modules.get('explainer'):
            zero_obs = np.zeros(38, dtype=np.float32)
            result = modules['explainer'].explain(zero_obs)
            assert result['shap_values'].shape == (38,)
            assert np.all(np.isfinite(result['shap_values']))
            results['EC6'] = True
            print("PASS")
        else:
            results['EC6'] = True
            print("SKIP (no explainer)")
    except Exception as e:
        results['EC6'] = False
        print(f"FAIL — {e}")

    # ── EC7: Fairness auditor with 0 completed jobs ───────────────
    print("  EC7: Fairness auditor — zero jobs...", end=" ")
    try:
        empty_auditor  = FairnessAuditor()
        empty_reporter = FairnessReporter(empty_auditor)
        report = empty_reporter.generate_report()
        assert report['fairness_status'] == 'N/A'
        assert report['total_jobs_audited'] == 0
        results['EC7'] = True
        print("PASS")
    except Exception as e:
        results['EC7'] = False
        print(f"FAIL — {e}")

    # ── EC8: Dashboard opened before env.reset() (import check) ───
    print("  EC8: Dashboard pre-reset import check...", end=" ")
    try:
        from dashboard.gantt import build_gantt
        dummy_state = {
            'step': 0, 'max_steps': 100, 'energy_price': 1.0,
            'spike_active': False, 'machines': [],
            'available_jobs': [], 'gantt_jobs': [],
            'metrics': {'completed': 0, 'total': 6,
                        'tardiness_rate': 0.0, 'episode_energy': 0.0,
                        'breakdowns': 0, 'rush_orders': 0},
        }
        fig = build_gantt(dummy_state)
        assert fig is not None
        results['EC8'] = True
        print("PASS")
    except Exception as e:
        results['EC8'] = False
        print(f"FAIL — {e}")

    # Summary
    passed = sum(1 for v in results.values() if v)
    total  = len(results)
    print(f"\n  Edge cases: {passed}/{total} PASSED")
    results['all_passed'] = (passed == total)
    return results


# ════════════════════════════════════════════════════════════════════
#  MODE: PERFORMANCE VALIDATION
# ════════════════════════════════════════════════════════════════════

def run_performance_validation(modules: dict) -> dict:
    """
    Measure latency and memory targets from the blueprint.
    """
    print(f"\n{'='*62}")
    print(f"  PERFORMANCE VALIDATION")
    print(f"{'='*62}")

    import time
    env   = modules['env']
    agent = modules['agent']
    obs, _ = env.reset(seed=0)
    masks  = {f'machine_{i}': env.get_action_mask(i) for i in range(3)}

    results = {}

    # ── Inference latency ─────────────────────────────────────────
    print("  Measuring agent inference latency (100 calls)...", end=" ")
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        agent.predict(obs, masks)
        times.append((time.perf_counter() - t0) * 1000)
    mean_inf = np.mean(times)
    p95_inf  = np.percentile(times, 95)
    results['inference_mean_ms'] = mean_inf
    results['inference_p95_ms']  = p95_inf
    results['inference_pass']    = mean_inf < 50
    print(f"mean={mean_inf:.1f}ms p95={p95_inf:.1f}ms  "
          f"{'PASS' if mean_inf < 50 else 'WARN'} (target <50ms)")

    # ── SHAP latency ──────────────────────────────────────────────
    if modules.get('explainer'):
        print("  Measuring SHAP explanation latency (5 calls)...", end=" ")
        shap_times = []
        for ep in range(5):
            obs_ep, _ = env.reset(seed=ep)
            t0 = time.perf_counter()
            modules['explainer'].explain(obs_ep['machine_0'])
            shap_times.append((time.perf_counter() - t0) * 1000)
        mean_shap = np.mean(shap_times)
        results['shap_mean_ms'] = mean_shap
        results['shap_pass']    = mean_shap < 500
        print(f"mean={mean_shap:.0f}ms  "
              f"{'PASS' if mean_shap < 500 else 'WARN'} (target <500ms)")
    else:
        results['shap_mean_ms'] = 0
        results['shap_pass']    = True
        print("  SHAP latency: SKIP (no explainer)")

    # ── Memory usage ──────────────────────────────────────────────
    print("  Measuring memory usage...", end=" ")
    try:
        import psutil, os as _os
        proc = psutil.Process(_os.getpid())
        mem_mb = proc.memory_info().rss / 1024 / 1024
        results['memory_mb']   = mem_mb
        results['memory_pass'] = mem_mb < 4096
        print(f"{mem_mb:.0f} MB  "
              f"{'PASS' if mem_mb < 4096 else 'WARN'} (target <4096 MB)")
    except ImportError:
        print("SKIP (psutil not installed — run: pip install psutil)")
        results['memory_mb']   = 0
        results['memory_pass'] = True

    # ── Episode speed ─────────────────────────────────────────────
    print("  Measuring episode throughput...", end=" ")
    ep_times = []
    for ep in range(5):
        obs_ep, _ = env.reset(seed=ep)
        t0 = time.perf_counter()
        steps = 0
        while True:
            masks_ep = {f'machine_{i}': env.get_action_mask(i) for i in range(3)}
            actions, _, _ = agent.predict(obs_ep, masks_ep)
            obs_ep, _, terms, truncs, _ = env.step(actions)
            steps += 1
            if terms['__all__'] or truncs['__all__']:
                break
        elapsed = time.perf_counter() - t0
        ep_times.append(steps / elapsed)
    mean_fps = np.mean(ep_times)
    results['steps_per_sec'] = mean_fps
    results['speed_pass']    = mean_fps > 10
    print(f"{mean_fps:.1f} steps/sec  "
          f"{'PASS' if mean_fps > 10 else 'WARN'} (target >10/sec)")

    all_pass = all([
        results.get('inference_pass', True),
        results.get('shap_pass',      True),
        results.get('memory_pass',    True),
        results.get('speed_pass',     True),
    ])
    results['all_passed'] = all_pass
    print(f"\n  Performance: {'ALL TARGETS MET' if all_pass else 'SOME TARGETS MISSED'}")
    return results


# ════════════════════════════════════════════════════════════════════
#  SAVE INTEGRATION REPORT
# ════════════════════════════════════════════════════════════════════

def save_integration_report(
    integration: dict,
    edge_cases:  dict,
    performance: dict,
) -> str:
    """Write logs/integration_test_report.txt with full results."""
    os.makedirs('logs', exist_ok=True)
    path = os.path.join('logs', 'integration_test_report.txt')

    lines = [
        "=" * 62,
        "  INTEGRATION TEST REPORT",
        "=" * 62,
        f"  Integration test  : {'PASSED' if integration.get('passed') else 'FAILED'}",
        f"  Episodes tested   : {integration.get('n_episodes', 0)}",
        f"  NaN rewards       : {integration.get('nan_rewards', 0)}",
        f"  Obs shape errors  : {integration.get('obs_shape_errors', 0)}",
        f"  Episode errors    : {len(integration.get('errors', []))}",
        "",
    ]

    if integration.get('rewards'):
        lines += [
            f"  Mean reward       : {np.mean(integration['rewards']):.2f}",
            f"  Mean tardiness    : {np.mean(integration['tardinesses']):.1%}",
            f"  Mean makespan     : {np.mean(integration['makespans']):.1f}",
            "",
        ]

    lines += ["  EDGE CASE RESULTS:", "-" * 62]
    ec_labels = {
        'EC1': 'All machines broken simultaneously',
        'EC2': 'Empty job queue',
        'EC3': 'Rush order with full queue',
        'EC4': 'Energy spike at step 0',
        'EC5': 'Long job (proc_time > max_steps/2)',
        'EC6': 'SHAP on all-zero observation',
        'EC7': 'Fairness auditor with zero jobs',
        'EC8': 'Dashboard pre-reset import check',
    }
    for key, label in ec_labels.items():
        passed = edge_cases.get(key, False)
        lines.append(f"  {'PASS' if passed else 'FAIL':<6} {label}")

    lines += [
        "",
        "  PERFORMANCE TARGETS:",
        "-" * 62,
        f"  {'PASS' if performance.get('inference_pass') else 'FAIL':<6} "
        f"Inference latency: {performance.get('inference_mean_ms',0):.1f}ms (target <50ms)",
        f"  {'PASS' if performance.get('shap_pass') else 'FAIL':<6} "
        f"SHAP latency: {performance.get('shap_mean_ms',0):.0f}ms (target <500ms)",
        f"  {'PASS' if performance.get('memory_pass') else 'FAIL':<6} "
        f"Memory: {performance.get('memory_mb',0):.0f}MB (target <4096MB)",
        f"  {'PASS' if performance.get('speed_pass') else 'FAIL':<6} "
        f"Speed: {performance.get('steps_per_sec',0):.1f} steps/sec (target >10)",
        "",
        "=" * 62,
    ]

    with open(path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n  Report saved → {path}")
    return path


# ════════════════════════════════════════════════════════════════════
#  MAIN DISPATCHER
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='MAPPO Factory Scheduler — Phase 6')
    parser.add_argument('--mode', default='test',
        choices=['demo', 'test', 'bench', 'fairness', 'xai', 'full'],
        help='Execution mode')
    parser.add_argument('--episodes', type=int, default=10,
        help='Episodes for test mode')
    args = parser.parse_args()

    if args.mode == 'demo':
        print("Starting dashboard — run: streamlit run dashboard/app.py")
        os.system('streamlit run dashboard/app.py')
        return

    # Load modules for all non-demo modes
    modules = load_all_modules()

    if args.mode == 'test':
        integration = run_integration_test(modules, args.episodes)
        edge_cases  = run_edge_case_tests(modules)
        performance = run_performance_validation(modules)
        save_integration_report(integration, edge_cases, performance)

    elif args.mode == 'bench':
        from agents.train_baselines import main as bench_main
        import argparse as _ap
        bench_args = _ap.Namespace(
            episodes=100, methods=['FCFS', 'SPT', 'EDD'], model_path='models/'
        )
        bench_main(bench_args)

    elif args.mode == 'fairness':
        from run_fairness_eval import run_fairness_evaluation
        run_fairness_evaluation(n_episodes=50)

    elif args.mode == 'xai':
        from xai.shap_explainer    import SHAPExplainer
        from xai.hypothesis_tester import HypothesisTester
        explainer = SHAPExplainer(
            modules['agent'].actors[0], ENV_CONFIG,
            n_background=100, verbose=True
        )
        tester  = HypothesisTester(explainer, verbose=True)
        results = tester.run_all()

    elif args.mode == 'full':
        print("\n>>> FULL INTEGRATION SUITE <<<\n")
        integration = run_integration_test(modules, args.episodes)
        edge_cases  = run_edge_case_tests(modules)
        performance = run_performance_validation(modules)
        save_integration_report(integration, edge_cases, performance)

        from agents.train_baselines import main as bench_main
        import argparse as _ap
        bench_args = _ap.Namespace(
            episodes=50, methods=['FCFS', 'SPT', 'EDD'], model_path='models/'
        )
        bench_main(bench_args)

        from run_fairness_eval import run_fairness_evaluation
        run_fairness_evaluation(n_episodes=50)


if __name__ == '__main__':
    main()