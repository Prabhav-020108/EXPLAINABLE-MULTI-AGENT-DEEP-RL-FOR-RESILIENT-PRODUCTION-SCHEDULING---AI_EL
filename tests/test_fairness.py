"""
test_fairness.py
----------------
Unit tests for FairnessAuditor and FairnessReporter.

Test classes:
    TestFairnessAuditor    — logging, field names, edge cases
    TestFairnessReporter   — each of the 5 metrics individually
    TestFairScenario       — full pipeline produces FAIR status
    TestBiasInjection      — injected bias correctly flags BIAS_DETECTED
    TestEdgeCases          — empty log, missing types, no high-priority jobs
"""

import pytest
import numpy as np

from fairness.auditor import FairnessAuditor
from fairness.reporter import FairnessReporter
from env.job_generator import Job


# ──────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────

def make_job(job_id=0, job_type='A', arrival=0, proc=5, deadline=15,
             energy=0.2, priority='normal', start=1, completion=10):
    """Create a completed Job for testing."""
    j = Job(
        job_id=job_id, job_type=job_type, arrival_time=arrival,
        processing_time=proc, deadline=deadline,
        energy_cost=energy, priority=priority,
    )
    j.start_time       = start
    j.completion_time  = completion
    j.assigned_machine = 0
    return j


def _make_fair_auditor() -> FairnessAuditor:
    """Auditor with balanced A/B/C records — all on time, similar wait times."""
    auditor = FairnessAuditor()
    auditor.episode_id = 3   # simulate 3 episodes

    # 10 type-A jobs: ~10% late
    for i in range(10):
        late = (i % 10 == 0)  # one job is late
        auditor.log.append({
            'episode': 1, 'job_id': i, 'type': 'A', 'priority': 'normal',
            'arrival': 0, 'deadline': 10, 'processing_time': 4,
            'completion': 11 if late else 9, 
            'tardiness': 1 if late else 0, 
            'wait_time': 1,
            'machine': 0, 'energy_consumed': 0.8,
        })
    # 10 type-B jobs: ~10% late
    for i in range(12, 22):
        late = (i % 10 == 0)  # one job is late
        auditor.log.append({
            'episode': 2, 'job_id': i, 'type': 'B', 'priority': 'normal',
            'arrival': 0, 'deadline': 20, 'processing_time': 9,
            'completion': 21 if late else 18,
            'tardiness': 1 if late else 0,
            'wait_time': 2, 'machine': 1, 'energy_consumed': 4.0,
        })
    # 8 type-C jobs: ~12% late
    for i in range(22, 30):
        late = (i % 8 == 0)  # one job is late
        auditor.log.append({
            'episode': 3, 'job_id': i, 'type': 'C', 'priority': 'normal',
            'arrival': 0, 'deadline': 35, 'processing_time': 16,
            'completion': 36 if late else 32,
            'tardiness': 1 if late else 0,
            'wait_time': 2, 'machine': 2, 'energy_consumed': 12.8,
        })
    # 4 high-priority jobs: tight deadlines, quickly processed → 0 tardiness
    for i in range(30, 34):
        auditor.log.append({
            'episode': 3, 'job_id': i, 'type': 'A', 'priority': 'high',
            'arrival': 10, 'deadline': 16, 'processing_time': 4,
            'completion': 14, 'tardiness': 0, 'wait_time': 0,
            'machine': 0, 'energy_consumed': 0.8,
        })
    return auditor


def _make_biased_auditor() -> FairnessAuditor:
    """
    Auditor where Type C is severely disadvantaged — high tardiness rate.
    Should produce BIAS_DETECTED.
    """
    auditor = FairnessAuditor()
    auditor.episode_id = 1

    # Type A: 5% late (1 out of 20)
    for i in range(20):
        auditor.log.append({
            'episode': 1, 'job_id': i, 'type': 'A', 'priority': 'normal',
            'arrival': 0, 'deadline': 12, 'processing_time': 4,
            'completion': 13 if i == 0 else 10,
            'tardiness': 1 if i == 0 else 0,
            'wait_time': 1, 'machine': 0, 'energy_consumed': 0.8,
        })
    # Type B: 6% late (1 out of 17)
    for i in range(20, 37):
        auditor.log.append({
            'episode': 1, 'job_id': i, 'type': 'B', 'priority': 'normal',
            'arrival': 0, 'deadline': 20, 'processing_time': 9,
            'completion': 22 if i == 20 else 17,
            'tardiness': 2 if i == 20 else 0,
            'wait_time': 2, 'machine': 1, 'energy_consumed': 4.0,
        })
    # Type C: 62% late — INJECTED BIAS
    for i in range(37, 50):
        late = (i < 45)  # 8 out of 13 are late (62%)
        auditor.log.append({
            'episode': 1, 'job_id': i, 'type': 'C', 'priority': 'normal',
            'arrival': 0, 'deadline': 25, 'processing_time': 16,
            'completion': 40 if late else 24,
            'tardiness': 15 if late else 0,
            'wait_time': 10 if late else 1,
            'machine': 2, 'energy_consumed': 12.8,
        })
    return auditor


# ──────────────────────────────────────────────────────────────────────
#  TEST CLASS 1: FairnessAuditor
# ──────────────────────────────────────────────────────────────────────

class TestFairnessAuditor:

    def setup_method(self):
        self.auditor = FairnessAuditor()

    def test_initial_state(self):
        assert self.auditor.total_jobs_logged() == 0
        assert self.auditor.episode_id == 0

    def test_new_episode_increments_counter(self):
        self.auditor.new_episode()
        self.auditor.new_episode()
        assert self.auditor.episode_id == 2

    def test_record_stores_all_required_columns(self):
        self.auditor.new_episode()
        job = make_job(job_id=1, job_type='B', arrival=2, proc=9, deadline=20,
                       energy=0.45, priority='normal', start=3, completion=15)
        self.auditor.record_job_completion(job, machine_id=1)
        df = self.auditor.get_log_df()
        assert list(df.columns) == FairnessAuditor.COLUMNS

    def test_record_computes_tardiness_correctly(self):
        """On-time job: tardiness = 0. Late job: tardiness = completion - deadline."""
        self.auditor.new_episode()

        on_time = make_job(job_id=0, deadline=20, completion=18)
        late    = make_job(job_id=1, deadline=20, completion=25)
        self.auditor.record_job_completion(on_time, 0)
        self.auditor.record_job_completion(late,    0)

        df = self.auditor.get_log_df()
        assert df.loc[0, 'tardiness'] == 0
        assert df.loc[1, 'tardiness'] == 5   # 25 - 20

    def test_record_computes_wait_time_correctly(self):
        self.auditor.new_episode()
        job = make_job(job_id=0, arrival=5, start=8, completion=15, deadline=20)
        self.auditor.record_job_completion(job, 0)
        df = self.auditor.get_log_df()
        assert df.loc[0, 'wait_time'] == 3   # 8 - 5

    def test_record_computes_energy_consumed(self):
        self.auditor.new_episode()
        job = make_job(job_id=0, proc=5, energy=0.4)  # energy_consumed = 0.4*5 = 2.0
        self.auditor.record_job_completion(job, 0)
        df = self.auditor.get_log_df()
        assert abs(df.loc[0, 'energy_consumed'] - 2.0) < 1e-4

    def test_skips_incomplete_jobs(self):
        """record_job_completion silently ignores jobs with completion_time=None."""
        self.auditor.new_episode()
        j = Job(job_id=99, job_type='A', arrival_time=0, processing_time=5,
                deadline=10, energy_cost=0.2)
        # completion_time is None by default
        self.auditor.record_job_completion(j, 0)
        assert self.auditor.total_jobs_logged() == 0

    def test_none_start_time_does_not_crash(self):
        """start_time can be None (after breakdown reset) — wait_time defaults to 0."""
        self.auditor.new_episode()
        j = make_job(job_id=0)
        j.start_time = None          # simulate post-breakdown state
        self.auditor.record_job_completion(j, 0)
        df = self.auditor.get_log_df()
        assert df.loc[0, 'wait_time'] == 0

    def test_compute_metrics_returns_all_types(self):
        self.auditor = _make_fair_auditor()
        metrics = self.auditor.compute_metrics()
        for key in ['A', 'B', 'C', 'priority_normal', 'priority_high']:
            assert key in metrics, f"Missing key: {key}"

    def test_episode_id_tags_records(self):
        self.auditor.new_episode()  # ep 1
        self.auditor.record_job_completion(make_job(0), 0)
        self.auditor.new_episode()  # ep 2
        self.auditor.record_job_completion(make_job(1), 0)
        df = self.auditor.get_log_df()
        assert list(df['episode']) == [1, 2]

    def test_save_to_csv_creates_file(self, tmp_path):
        self.auditor = _make_fair_auditor()
        path = str(tmp_path / 'test_audit.csv')
        self.auditor.save_to_csv(path)
        import os, pandas as pd
        assert os.path.exists(path)
        df = pd.read_csv(path)
        assert len(df) > 0
        assert set(FairnessAuditor.COLUMNS).issubset(set(df.columns))

    def test_reset_clears_log(self):
        self.auditor = _make_fair_auditor()
        self.auditor.reset()
        assert self.auditor.total_jobs_logged() == 0
        assert self.auditor.episode_id == 3  # counter preserved


# ──────────────────────────────────────────────────────────────────────
#  TEST CLASS 2: Individual Metric Computations
# ──────────────────────────────────────────────────────────────────────

class TestMetricComputations:

    def setup_method(self):
        self.auditor  = _make_fair_auditor()
        self.reporter = FairnessReporter(self.auditor)
        self.df       = self.auditor.get_log_df()

    def test_tardiness_std_dev_is_float(self):
        result = self.reporter.compute_tardiness_std_dev(self.df)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_disparity_ratio_ge_one(self):
        """Ratio must always be >= 1.0 by definition (max/min)."""
        result = self.reporter.compute_max_disparity_ratio(self.df)
        assert result >= 1.0

    def test_energy_share_sums_to_one(self):
        shares = self.reporter.compute_energy_share(self.df)
        total  = sum(shares.values())
        assert abs(total - 1.0) < 1e-4, f"Energy shares sum to {total}"

    def test_energy_share_returns_all_types(self):
        shares = self.reporter.compute_energy_share(self.df)
        for t in ['A', 'B', 'C']:
            assert t in shares

    def test_priority_fairness_no_high_priority_returns_pass_value(self):
        """Auditor with only normal-priority jobs must return a passing value."""
        clean_auditor = FairnessAuditor()
        clean_auditor.episode_id = 1
        for i in range(10):
            clean_auditor.log.append({
                'episode':1,'job_id':i,'type':'A','priority':'normal',
                'arrival':0,'deadline':15,'processing_time':4,
                'completion':12,'tardiness':0,'wait_time':1,
                'machine':0,'energy_consumed':0.8,
            })
        rep = FairnessReporter(clean_auditor)
        df  = clean_auditor.get_log_df()
        result = rep.compute_priority_fairness(df)
        assert result >= 1.0, f"Expected >= 1.0, got {result}"

    def test_wait_time_std_is_float(self):
        result = self.reporter.compute_wait_time_std_dev(self.df)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_disparity_ratio_no_div_zero_when_type_has_zero_tardiness(self):
        """If one type has 0% tardiness, ratio should be large but finite."""
        import pandas as pd
        df = pd.DataFrame([
            {'type':'A','tardiness':0,'wait_time':1,'energy_consumed':1.0,
             'priority':'normal','episode':1,'job_id':0,'arrival':0,'deadline':10,
             'processing_time':4,'completion':8,'machine':0},
            {'type':'B','tardiness':3,'wait_time':2,'energy_consumed':4.0,
             'priority':'normal','episode':1,'job_id':1,'arrival':0,'deadline':20,
             'processing_time':9,'completion':23,'machine':1},
            {'type':'C','tardiness':0,'wait_time':1,'energy_consumed':10.0,
             'priority':'normal','episode':1,'job_id':2,'arrival':0,'deadline':30,
             'processing_time':16,'completion':25,'machine':2},
        ])
        auditor          = FairnessAuditor()
        auditor.log      = df.to_dict('records')
        auditor.episode_id = 1
        reporter = FairnessReporter(auditor)
        ratio = reporter.compute_max_disparity_ratio(df)
        assert ratio != float('inf')
        assert ratio == ratio   # not NaN


# ──────────────────────────────────────────────────────────────────────
#  TEST CLASS 3: Full FAIR scenario
# ──────────────────────────────────────────────────────────────────────

class TestFairScenario:

    def setup_method(self):
        self.auditor  = _make_fair_auditor()
        self.reporter = FairnessReporter(self.auditor)

    def test_fair_scenario_produces_fair_status(self):
        report = self.reporter.generate_report()
        assert report['fairness_status'] == 'FAIR', (
            f"Expected FAIR but got {report['fairness_status']}.\n"
            f"Flags: {report['flags']}"
        )

    def test_fair_scenario_has_no_flags(self):
        report = self.reporter.generate_report()
        assert len(report['flags']) == 0, (
            f"Expected no flags, but got: {report['flags']}"
        )

    def test_report_has_all_required_keys(self):
        report = self.reporter.generate_report()
        required = [
            'fairness_status', 'metrics', 'thresholds', 'pass_fail',
            'per_type_breakdown', 'flags', 'episodes_evaluated',
            'total_jobs_audited',
        ]
        for key in required:
            assert key in report, f"Missing report key: {key}"

    def test_all_five_pass_fail_keys_present(self):
        report = self.reporter.generate_report()
        required = [
            'tardiness_std_dev', 'max_disparity_ratio', 'energy_share_equity',
            'priority_fairness', 'wait_time_equity',
        ]
        for key in required:
            assert key in report['pass_fail'], f"Missing pass_fail key: {key}"

    def test_per_type_breakdown_has_all_types(self):
        report = self.reporter.generate_report()
        for t in ['A', 'B', 'C']:
            assert t in report['per_type_breakdown'], f"Missing type: {t}"
            assert report['per_type_breakdown'][t]['count'] > 0

    def test_total_jobs_audited_is_correct(self):
        report = self.reporter.generate_report()
        assert report['total_jobs_audited'] == self.auditor.total_jobs_logged()

    def test_save_report_writes_valid_json(self, tmp_path):
        import json, os
        path   = str(tmp_path / 'test_report.json')
        report = self.reporter.save_report(path)
        assert os.path.exists(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded['fairness_status'] == report['fairness_status']


# ──────────────────────────────────────────────────────────────────────
#  TEST CLASS 4: Bias Injection — must produce BIAS_DETECTED
# ──────────────────────────────────────────────────────────────────────

class TestBiasInjection:
    """
    Verifies the system correctly raises BIAS_DETECTED when one job type
    is artificially disadvantaged. This satisfies Phase 4 exit checklist item 5.

    Injected bias: Type C has 62% tardiness rate vs Type A at 5%.
    The disparity ratio (62/5 ≈ 12.4) far exceeds the 2.0 threshold.
    The tardiness std dev also exceeds 0.15.
    """

    def setup_method(self):
        self.auditor  = _make_biased_auditor()
        self.reporter = FairnessReporter(self.auditor)

    def test_biased_scenario_produces_bias_detected(self):
        report = self.reporter.generate_report()
        assert report['fairness_status'] == 'BIAS_DETECTED', (
            f"Expected BIAS_DETECTED but got {report['fairness_status']}.\n"
            f"Metrics: {report['metrics']}"
        )

    def test_biased_scenario_has_flags(self):
        report = self.reporter.generate_report()
        assert len(report['flags']) > 0, "Expected at least one flag for biased data"

    def test_disparity_ratio_fails_for_biased_data(self):
        report = self.reporter.generate_report()
        ratio  = report['metrics']['max_disparity_ratio']
        assert ratio >= FairnessReporter.THRESHOLDS['max_disparity_ratio'], (
            f"Disparity ratio {ratio} should fail threshold "
            f"{FairnessReporter.THRESHOLDS['max_disparity_ratio']}"
        )

    def test_tardiness_std_dev_fails_for_biased_data(self):
        report   = self.reporter.generate_report()
        std_dev  = report['metrics']['tardiness_std_dev']
        threshold = FairnessReporter.THRESHOLDS['tardiness_std_dev']
        assert std_dev >= threshold, (
            f"Std dev {std_dev} should fail threshold {threshold}"
        )

    def test_type_c_is_identified_as_disadvantaged(self):
        """The flag message should mention Type C."""
        report = self.reporter.generate_report()
        flag_text = ' '.join(report['flags'])
        assert 'C' in flag_text, (
            f"Expected 'C' in flags, got: {flag_text}"
        )

    def test_fair_vs_biased_are_different(self):
        """FAIR auditor and BIASED auditor must give different verdicts."""
        fair_reporter   = FairnessReporter(_make_fair_auditor())
        biased_reporter = FairnessReporter(_make_biased_auditor())
        fair_report     = fair_reporter.generate_report()
        biased_report   = biased_reporter.generate_report()
        assert fair_report['fairness_status']   == 'FAIR'
        assert biased_report['fairness_status'] == 'BIAS_DETECTED'


# ──────────────────────────────────────────────────────────────────────
#  TEST CLASS 5: Edge Cases
# ──────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_auditor_returns_na_status(self):
        auditor  = FairnessAuditor()
        reporter = FairnessReporter(auditor)
        report   = reporter.generate_report()
        assert report['fairness_status'] == 'N/A'
        assert report['total_jobs_audited'] == 0

    def test_empty_auditor_has_flag_message(self):
        auditor  = FairnessAuditor()
        reporter = FairnessReporter(auditor)
        report   = reporter.generate_report()
        assert len(report['flags']) > 0

    def test_only_type_a_jobs(self):
        """Reporter must not crash if only one job type is present."""
        auditor = FairnessAuditor()
        auditor.episode_id = 1
        for i in range(10):
            auditor.log.append({
                'episode':1,'job_id':i,'type':'A','priority':'normal',
                'arrival':0,'deadline':15,'processing_time':4,
                'completion':12,'tardiness':0,'wait_time':1,
                'machine':0,'energy_consumed':0.8,
            })
        reporter = FairnessReporter(auditor)
        report   = reporter.generate_report()
        assert report['fairness_status'] in ('FAIR', 'BIAS_DETECTED')
        assert report['total_jobs_audited'] == 10

    def test_all_jobs_on_time_does_not_crash(self):
        """Zero tardiness for all types: disparity ratio should not divide by zero."""
        auditor = FairnessAuditor()
        auditor.episode_id = 1
        for i, t in enumerate(['A', 'B', 'C'] * 5):
            auditor.log.append({
                'episode':1,'job_id':i,'type':t,'priority':'normal',
                'arrival':0,'deadline':30,'processing_time':4,
                'completion':10,'tardiness':0,'wait_time':1,
                'machine':0,'energy_consumed':1.0,
            })
        reporter = FairnessReporter(auditor)
        report   = reporter.generate_report()
        # disparity ratio should be finite and not NaN
        ratio = report['metrics']['max_disparity_ratio']
        assert ratio == ratio         # not NaN
        assert ratio != float('inf')

    def test_no_high_priority_jobs_passes_priority_metric(self):
        """Priority fairness must pass when no rush orders exist."""
        auditor = FairnessAuditor()
        auditor.episode_id = 1
        for i, t in enumerate(['A', 'B', 'C'] * 4):
            auditor.log.append({
                'episode':1,'job_id':i,'type':t,'priority':'normal',
                'arrival':0,'deadline':20,'processing_time':5,
                'completion':18,'tardiness':0,'wait_time':1,
                'machine':0,'energy_consumed':1.0,
            })
        reporter = FairnessReporter(auditor)
        report   = reporter.generate_report()
        assert report['pass_fail']['priority_fairness'] is True

    def test_integration_with_real_env(self):
        """
        Run 3 real episodes with random actions and verify the auditor
        records jobs correctly — no crashes, correct column types.
        """
        import numpy as np
        from env.factory_gym import FactoryGym

        env     = FactoryGym()
        auditor = FairnessAuditor()

        for ep in range(3):
            obs, _ = env.reset(seed=ep)
            auditor.new_episode()

            for _step in range(50):
                count_before = len(env.completed_jobs)
                actions = {}
                for idx, aid in enumerate(env.agents):
                    mask  = env.get_action_mask(idx)
                    valid = np.where(mask)[0]
                    actions[aid] = int(np.random.choice(valid))
                obs, _, terms, truncs, _ = env.step(actions)
                for job in env.completed_jobs[count_before:]:
                    if job.completion_time is not None:
                        auditor.record_job_completion(
                            job,
                            machine_id=job.assigned_machine or 0,
                        )
                if terms['__all__'] or truncs['__all__']:
                    break

        df = auditor.get_log_df()
        # all required columns exist
        for col in FairnessAuditor.COLUMNS:
            assert col in df.columns, f"Missing column: {col}"
        # type column only has valid values
        assert set(df['type'].unique()).issubset({'A', 'B', 'C'})
        # priority column only has valid values
        assert set(df['priority'].unique()).issubset({'normal', 'high'})
        # no NaN in any critical numeric column
        for col in ['tardiness', 'wait_time', 'energy_consumed']:
            assert not df[col].isnull().any(), f"NaN in column: {col}"