"""
reporter.py
-----------
FairnessReporter: computes all five fairness metrics from the
FairnessAuditor log and generates a structured report.

Five Fairness Metrics
─────────────────────
1. Tardiness Std Dev      std(tardy_rate_A, tardy_rate_B, tardy_rate_C)  < 0.15
2. Max Disparity Ratio    max(tardy_rate) / min(tardy_rate)               < 2.00
3. Energy Share Equity    max single-type fraction of total energy        < 0.75
4. Priority Fairness      avg_tardiness_steps(normal) / avg(high)         > 1.00
5. Wait Time Equity       std(avg_wait_A, avg_wait_B, avg_wait_C) steps   < 3.00

Metrics 1 and 2 use tardiness RATE (fraction of jobs that missed deadline)
so that naturally different job-type processing times do not create
artificial "unfairness" in raw tardiness-step comparisons.

Metric 4 uses raw tardiness STEPS (not rate) so that the model's ability to
prioritise urgent rush orders is directly visible: rush orders should
finish with lower absolute tardiness than normal-priority jobs.

Output
──────
    logs/fairness_report.json   — full report dict
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

from fairness.auditor import FairnessAuditor


class FairnessReporter:
    """
    Reads FairnessAuditor data, computes five fairness metrics,
    and generates a FAIR / BIAS_DETECTED report.
    """

    # ── Metric thresholds ─────────────────────────────────────────────
    THRESHOLDS = {
        'tardiness_std_dev':     0.15,   # std of tardy-rate across A/B/C
        'max_disparity_ratio':   2.00,   # max/min tardy-rate ratio
        'energy_share_max':      0.75,   # max single-type energy fraction
        'priority_fairness_min': 1.00,   # normal_tard_steps / high_tard_steps
        'wait_time_std_dev':     3.00,   # std of avg wait steps across A/B/C
    }

    def __init__(self, auditor: FairnessAuditor, max_steps: int = 100):
        self.auditor   = auditor
        self.max_steps = max_steps

    # ──────────────────────────────────────────────────────────────────
    #  INTERNAL HELPERS
    # ──────────────────────────────────────────────────────────────────

    def _tardy_rates(self, df: pd.DataFrame) -> Dict[str, float]:
        """Fraction of late jobs per type (tardiness > 0)."""
        result = {}
        for t in ['A', 'B', 'C']:
            subset = df[df['type'] == t]
            result[t] = (
                float((subset['tardiness'] > 0).mean()) if len(subset) > 0 else 0.0
            )
        return result

    def _avg_tardiness_steps(self, df: pd.DataFrame) -> Dict[str, float]:
        """Average tardiness in raw steps per type."""
        result = {}
        for t in ['A', 'B', 'C']:
            subset = df[df['type'] == t]
            result[t] = float(subset['tardiness'].mean()) if len(subset) > 0 else 0.0
        return result

    def _avg_wait(self, df: pd.DataFrame) -> Dict[str, float]:
        """Average wait time in steps per type."""
        result = {}
        for t in ['A', 'B', 'C']:
            subset = df[df['type'] == t]
            result[t] = float(subset['wait_time'].mean()) if len(subset) > 0 else 0.0
        return result

    # ──────────────────────────────────────────────────────────────────
    #  METRIC 1 — Tardiness Std Dev
    # ──────────────────────────────────────────────────────────────────

    def compute_tardiness_std_dev(self, df: pd.DataFrame) -> float:
        """
        std(tardy_rate_A, tardy_rate_B, tardy_rate_C).
        Uses the fraction-late rate across the three job types.
        Threshold: < 0.15
        """
        rates = self._tardy_rates(df)
        return float(np.std(list(rates.values())))

    # ──────────────────────────────────────────────────────────────────
    #  METRIC 2 — Max Disparity Ratio
    # ──────────────────────────────────────────────────────────────────

    def compute_max_disparity_ratio(self, df: pd.DataFrame) -> float:
        """
        max(tardy_rate) / min(tardy_rate) across A, B, C.
        epsilon guards against divide-by-zero when one type has 0 late jobs.
        Threshold: < 2.00
        """
        rates   = self._tardy_rates(df)
        values  = list(rates.values())
        epsilon = 1e-6
        return float(max(values) / (min(values) + epsilon))

    # ──────────────────────────────────────────────────────────────────
    #  METRIC 3 — Energy Share Equity
    # ──────────────────────────────────────────────────────────────────

    def compute_energy_share(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Fraction of total energy consumed by each type.
        Returns {type: fraction} where fractions sum to ~1.0.
        Threshold: no type > 0.75
        """
        total = df['energy_consumed'].sum()
        if total < 1e-9:
            return {'A': 0.0, 'B': 0.0, 'C': 0.0}
        return {
            t: float(df[df['type'] == t]['energy_consumed'].sum() / total)
            for t in ['A', 'B', 'C']
        }

    # ──────────────────────────────────────────────────────────────────
    #  METRIC 4 — Priority Fairness
    # ──────────────────────────────────────────────────────────────────

    def compute_priority_fairness(self, df: pd.DataFrame) -> float:
        """
        avg_tardiness_steps(normal) / avg_tardiness_steps(high).

        > 1.0 means normal-priority jobs are MORE tardy than high-priority
        ones — which is the desired behaviour (agents favour rush orders).

        Special cases:
            No high-priority jobs  →  1.5  (nominal PASS)
            High-priority avg ≈ 0  →  2.0  (rush orders handled perfectly)
            Both ≈ 0               →  1.5  (trivially fair)

        Threshold: > 1.0
        """
        normal_df = df[df['priority'] == 'normal']
        high_df   = df[df['priority'] == 'high']

        if len(high_df) == 0:
            return 1.5  # no rush orders in this evaluation set

        avg_normal = float(normal_df['tardiness'].mean()) if len(normal_df) > 0 else 0.0
        avg_high   = float(high_df['tardiness'].mean())

        if avg_normal < 0.01 and avg_high < 0.01:
            return 1.5  # both near zero
        if avg_high < 0.01:
            return 2.0  # high-priority never late

        return float(avg_normal / avg_high)

    # ──────────────────────────────────────────────────────────────────
    #  METRIC 5 — Wait Time Equity
    # ──────────────────────────────────────────────────────────────────

    def compute_wait_time_std_dev(self, df: pd.DataFrame) -> float:
        """
        std(avg_wait_A, avg_wait_B, avg_wait_C) in steps.
        Threshold: < 3.00
        """
        avgs = self._avg_wait(df)
        return float(np.std(list(avgs.values())))

    # ──────────────────────────────────────────────────────────────────
    #  REPORT GENERATION
    # ──────────────────────────────────────────────────────────────────

    def generate_report(self) -> Dict[str, Any]:
        """
        Compute all five metrics, check thresholds, and return the
        full report dictionary.

        Returns:
            {
              'fairness_status':    'FAIR' | 'BIAS_DETECTED',
              'metrics':            { metric_name: value, ... },
              'thresholds':         { metric_name: threshold, ... },
              'pass_fail':          { metric_name: bool, ... },
              'per_type_breakdown': { 'A': {...}, 'B': {...}, 'C': {...} },
              'flags':              [ str, ... ],
              'episodes_evaluated': int,
              'total_jobs_audited': int,
            }
        """
        df = self.auditor.get_log_df()

        if len(df) == 0:
            return {
                'fairness_status':    'N/A',
                'metrics':            {},
                'thresholds':         self.THRESHOLDS,
                'pass_fail':          {},
                'per_type_breakdown': {},
                'flags':              ['No job completion data recorded.'],
                'episodes_evaluated': self.auditor.episode_id,
                'total_jobs_audited': 0,
            }

        # ── Compute all five metrics ──────────────────────────────
        tard_std   = self.compute_tardiness_std_dev(df)
        disp_ratio = self.compute_max_disparity_ratio(df)
        energy_sh  = self.compute_energy_share(df)
        pri_fair   = self.compute_priority_fairness(df)
        wait_std   = self.compute_wait_time_std_dev(df)

        max_energy_type  = max(energy_sh, key=energy_sh.get)
        max_energy_value = energy_sh[max_energy_type]

        # ── Check thresholds ─────────────────────────────────────
        passes = {
            'tardiness_std_dev':   tard_std    < self.THRESHOLDS['tardiness_std_dev'],
            'max_disparity_ratio': disp_ratio  < self.THRESHOLDS['max_disparity_ratio'],
            'energy_share_equity': max_energy_value < self.THRESHOLDS['energy_share_max'],
            'priority_fairness':   pri_fair    >= self.THRESHOLDS['priority_fairness_min'],
            'wait_time_equity':    wait_std    < self.THRESHOLDS['wait_time_std_dev'],
        }

        # ── Build human-readable flags for failed metrics ─────────
        flags = []
        tardy_rates = self._tardy_rates(df)

        if not passes['tardiness_std_dev']:
            flags.append(
                f"Tardiness rate std dev {tard_std:.4f} >= threshold "
                f"{self.THRESHOLDS['tardiness_std_dev']} — "
                f"unequal late rates across types: "
                f"A={tardy_rates['A']:.1%}, B={tardy_rates['B']:.1%}, "
                f"C={tardy_rates['C']:.1%}"
            )
        if not passes['max_disparity_ratio']:
            worst = max(tardy_rates, key=tardy_rates.get)
            flags.append(
                f"Disparity ratio {disp_ratio:.2f} >= threshold "
                f"{self.THRESHOLDS['max_disparity_ratio']} — "
                f"Type {worst} ({tardy_rates[worst]:.1%} late) is "
                f"disproportionately disadvantaged"
            )
        if not passes['energy_share_equity']:
            flags.append(
                f"Type {max_energy_type} energy share "
                f"{max_energy_value:.2%} >= threshold "
                f"{self.THRESHOLDS['energy_share_max']:.0%}"
            )
        if not passes['priority_fairness']:
            flags.append(
                f"Priority fairness ratio {pri_fair:.3f} < threshold "
                f"{self.THRESHOLDS['priority_fairness_min']} — "
                f"high-priority jobs are MORE tardy than normal-priority ones"
            )
        if not passes['wait_time_equity']:
            avgs = self._avg_wait(df)
            flags.append(
                f"Wait time std dev {wait_std:.2f} steps >= threshold "
                f"{self.THRESHOLDS['wait_time_std_dev']} steps — "
                f"A={avgs['A']:.1f}, B={avgs['B']:.1f}, C={avgs['C']:.1f}"
            )

        # ── Per-type breakdown ────────────────────────────────────
        tardy_steps = self._avg_tardiness_steps(df)
        wait_steps  = self._avg_wait(df)

        per_type = {
            t: {
                'count':          int(len(df[df['type'] == t])),
                'tardiness_rate': round(tardy_rates[t],  4),
                'avg_tardiness':  round(tardy_steps[t],  3),
                'avg_wait_time':  round(wait_steps[t],   3),
                'energy_share':   round(energy_sh.get(t, 0.0), 4),
            }
            for t in ['A', 'B', 'C']
        }

        report = {
            'fairness_status':    'FAIR' if all(passes.values()) else 'BIAS_DETECTED',
            'metrics': {
                'tardiness_std_dev':   round(tard_std,   4),
                'max_disparity_ratio': round(disp_ratio, 3),
                'energy_share':        {k: round(v, 4) for k, v in energy_sh.items()},
                'priority_fairness':   round(pri_fair,   3),
                'wait_time_std_dev':   round(wait_std,   3),
            },
            'thresholds': {
                'tardiness_std_dev':    self.THRESHOLDS['tardiness_std_dev'],
                'max_disparity_ratio':  self.THRESHOLDS['max_disparity_ratio'],
                'energy_share_max':     self.THRESHOLDS['energy_share_max'],
                'priority_fairness_min':self.THRESHOLDS['priority_fairness_min'],
                'wait_time_std_dev':    self.THRESHOLDS['wait_time_std_dev'],
            },
            'pass_fail':          passes,
            'per_type_breakdown': per_type,
            'flags':              flags,
            'episodes_evaluated': self.auditor.episode_id,
            'total_jobs_audited': int(len(df)),
        }

        return report

    def save_report(
        self,
        path: str = 'logs/fairness_report.json',
    ) -> Dict[str, Any]:
        """
        Generate report, save to JSON, print summary, and return the dict.
        Creates the output directory if it does not exist.
        """
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        report = self.generate_report()

        with open(path, 'w') as f:
            json.dump(report, f, indent=2)

        self._print_summary(report)
        print(f"  Report saved → {path}\n")
        return report

    # ──────────────────────────────────────────────────────────────────
    #  TERMINAL SUMMARY
    # ──────────────────────────────────────────────────────────────────

    def _print_summary(self, report: Dict[str, Any]) -> None:
        status = report['fairness_status']
        m      = report.get('metrics', {})
        pf     = report.get('pass_fail', {})

        print(f"\n{'='*62}")
        print(f"  FAIRNESS AUDIT REPORT  —  {status}")
        print(f"{'='*62}")
        print(f"  Episodes evaluated : {report['episodes_evaluated']}")
        print(f"  Total jobs audited : {report['total_jobs_audited']}")
        print(f"{'-'*62}")

        def row(label, value_str, passed, threshold_str):
            tick = '[PASS]' if passed else '[FAIL]'
            print(f"  {tick:<7} {label:<30} {value_str:<12} (threshold {threshold_str})")

        row('Tardiness std dev (rate)',
            f"{m.get('tardiness_std_dev', 'N/A'):.4f}",
            pf.get('tardiness_std_dev', False),
            f"< {self.THRESHOLDS['tardiness_std_dev']}")

        row('Max disparity ratio',
            f"{m.get('max_disparity_ratio', 'N/A'):.3f}",
            pf.get('max_disparity_ratio', False),
            f"< {self.THRESHOLDS['max_disparity_ratio']}")

        es = m.get('energy_share', {})
        max_es = max(es.values()) if es else 0.0
        row('Max energy share',
            f"{max_es:.2%}",
            pf.get('energy_share_equity', False),
            f"< {self.THRESHOLDS['energy_share_max']:.0%}")

        row('Priority fairness ratio',
            f"{m.get('priority_fairness', 'N/A'):.3f}",
            pf.get('priority_fairness', False),
            f"> {self.THRESHOLDS['priority_fairness_min']}")

        row('Wait time std dev (steps)',
            f"{m.get('wait_time_std_dev', 'N/A'):.3f}",
            pf.get('wait_time_equity', False),
            f"< {self.THRESHOLDS['wait_time_std_dev']}")

        print(f"{'-'*62}")
        print("  Per-type breakdown:")
        for t, data in report.get('per_type_breakdown', {}).items():
            print(
                f"    Type {t}: n={data['count']:>4}  "
                f"late={data['tardiness_rate']:.1%}  "
                f"avg_tard={data['avg_tardiness']:.1f}s  "
                f"avg_wait={data['avg_wait_time']:.1f}s  "
                f"energy={data['energy_share']:.1%}"
            )

        if report.get('flags'):
            print(f"{'-'*62}")
            print("  Flags:")
            for flag in report['flags']:
                print(f"    ! {flag}")

        print(f"{'='*62}")