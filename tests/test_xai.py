"""
tests/test_xai.py
-----------------
Unit tests for the XAI module (SHAP explainer and hypothesis tester).

These tests use a SMALL background (10 samples) and FEW perturbations
for speed. Full accuracy requires n_background=100.

Run with:
    pytest tests/test_xai.py -v
"""

import os
import sys
import json
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.mappo_agent    import MAPPOAgent
from xai.shap_explainer    import SHAPExplainer, FEATURE_NAMES, FEATURE_GROUPS
from xai.hypothesis_tester import HypothesisTester
from env.factory_gym       import FactoryGym


# ─────────────────────────────────────────────────────────────────────
#  FIXTURES
# ─────────────────────────────────────────────────────────────────────

ENV_CONFIG = {
    'n_machines': 3, 'n_jobs': 6, 'max_steps': 100,
    'breakdown_rate': 0.003, 'rush_rate': 0.005, 'energy_spike_rate': 0.05,
}

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'mappo_factory_best.pth')


@pytest.fixture(scope='module')
def agent():
    """Load trained agent once for all tests."""
    a = MAPPOAgent()
    a.load(MODEL_PATH)
    a.set_eval_mode()
    return a


@pytest.fixture(scope='module')
def explainer(agent):
    """Build SHAPExplainer once for all tests (small background for speed)."""
    return SHAPExplainer(
        agent.actors[0],
        ENV_CONFIG,
        n_background=15,   # small for test speed
        verbose=False,
    )


@pytest.fixture(scope='module')
def tester(explainer):
    """HypothesisTester built on the test explainer."""
    return HypothesisTester(explainer, verbose=False)


@pytest.fixture(scope='module')
def sample_obs():
    """One real observation from the environment."""
    env = FactoryGym(ENV_CONFIG)
    obs, _ = env.reset(seed=42)
    return obs['machine_0']


# ─────────────────────────────────────────────────────────────────────
#  FEATURE METADATA TESTS
# ─────────────────────────────────────────────────────────────────────

class TestFeatureMetadata:

    def test_feature_names_length(self):
        assert len(FEATURE_NAMES) == 38, \
            f"Expected 38 feature names, got {len(FEATURE_NAMES)}"

    def test_feature_names_unique(self):
        assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES), \
            "Duplicate feature names detected"

    def test_feature_groups_cover_all_indices(self):
        all_indices = []
        for indices in FEATURE_GROUPS.values():
            all_indices.extend(indices)
        all_indices_set = set(all_indices)
        expected = set(range(38))
        missing  = expected - all_indices_set
        assert not missing, f"Indices not covered by any group: {missing}"

    def test_feature_groups_no_overlap(self):
        all_indices = []
        for indices in FEATURE_GROUPS.values():
            all_indices.extend(indices)
        assert len(all_indices) == len(set(all_indices)), \
            "Some indices appear in multiple feature groups"

    def test_machine_status_indices(self):
        """Verify machine status is in first 9 dims."""
        status_indices = FEATURE_GROUPS['Machine Status']
        assert status_indices == list(range(0, 9))

    def test_energy_price_index(self):
        """Energy price must be the last feature."""
        energy_indices = FEATURE_GROUPS['Energy Price']
        assert energy_indices == [37]


# ─────────────────────────────────────────────────────────────────────
#  SHAP EXPLAINER TESTS
# ─────────────────────────────────────────────────────────────────────

class TestSHAPExplainer:

    def test_explainer_initializes(self, explainer):
        assert explainer is not None
        assert explainer.background_np.shape == (15, 38)
        assert explainer.explainer_type in ('DeepExplainer', 'KernelExplainer', 'GradientExplainer (logits)')

    def test_explain_output_structure(self, explainer, sample_obs):
        result = explainer.explain(sample_obs, n_samples=50)
        required_keys = {
            'chosen_action', 'action_label', 'action_prob',
            'action_probs', 'shap_values', 'top_features',
            'groups', 'narrative',
        }
        assert set(result.keys()) >= required_keys, \
            f"Missing keys: {required_keys - set(result.keys())}"

    def test_chosen_action_valid(self, explainer, sample_obs):
        result = explainer.explain(sample_obs, n_samples=50)
        assert result['chosen_action'] in range(7), \
            f"chosen_action out of range: {result['chosen_action']}"

    def test_action_prob_in_range(self, explainer, sample_obs):
        result = explainer.explain(sample_obs, n_samples=50)
        assert 0.0 <= result['action_prob'] <= 1.0

    def test_action_probs_sum_to_one(self, explainer, sample_obs):
        result = explainer.explain(sample_obs, n_samples=50)
        total  = sum(result['action_probs'])
        assert abs(total - 1.0) < 0.01, f"Action probs sum to {total}"

    def test_shap_values_shape(self, explainer, sample_obs):
        result = explainer.explain(sample_obs, n_samples=50)
        assert result['shap_values'].shape == (38,), \
            f"Wrong shape: {result['shap_values'].shape}"

    def test_shap_values_finite(self, explainer, sample_obs):
        result = explainer.explain(sample_obs, n_samples=50)
        assert np.all(np.isfinite(result['shap_values'])), \
            "SHAP values contain NaN or Inf"

    def test_top_features_count(self, explainer, sample_obs):
        result = explainer.explain(sample_obs, n_samples=50)
        assert len(result['top_features']) == 6

    def test_top_features_sum_to_100(self, explainer, sample_obs):
        result = explainer.explain(sample_obs, n_samples=50)
        total  = sum(pct for _, pct in result['top_features'])
        assert total <= 101.0, f"Top feature percentages sum to {total} (expected ≤ 100)"

    def test_groups_keys_correct(self, explainer, sample_obs):
        result = explainer.explain(sample_obs, n_samples=50)
        assert set(result['groups'].keys()) == set(FEATURE_GROUPS.keys())

    def test_groups_sum_to_100(self, explainer, sample_obs):
        result = explainer.explain(sample_obs, n_samples=50)
        total  = sum(result['groups'].values())
        assert abs(total - 100.0) < 2.0, f"Group percentages sum to {total}"

    def test_narrative_is_string(self, explainer, sample_obs):
        result = explainer.explain(sample_obs, n_samples=50)
        assert isinstance(result['narrative'], str)
        assert len(result['narrative']) > 20

    def test_deterministic_same_input(self, explainer, sample_obs):
        """Same input must produce same chosen action."""
        r1 = explainer.explain(sample_obs, n_samples=50)
        r2 = explainer.explain(sample_obs, n_samples=50)
        assert r1['chosen_action'] == r2['chosen_action'], \
            "Non-deterministic action selection"

    def test_different_observations_different_explanations(self, explainer):
        """Different inputs should generally produce different SHAP values."""
        env = FactoryGym(ENV_CONFIG)
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=99)
        r1 = explainer.explain(obs1['machine_0'], n_samples=50)
        r2 = explainer.explain(obs2['machine_0'], n_samples=50)
        # SHAP values should differ (not identical)
        assert not np.allclose(r1['shap_values'], r2['shap_values'], atol=1e-6), \
            "Different observations produced identical SHAP values"

    def test_explain_action_convenience(self, explainer):
        """Test explain_action() helper method."""
        env = FactoryGym(ENV_CONFIG)
        obs_dict, _ = env.reset(seed=5)
        result = explainer.explain_action(obs_dict, agent_idx=0)
        assert result['chosen_action'] in range(7)

    def test_five_different_episodes(self, explainer):
        """Run on 5 episodes — no crashes."""
        env = FactoryGym(ENV_CONFIG)
        for ep in range(5):
            obs, _ = env.reset(seed=ep)
            result = explainer.explain(obs['machine_0'], n_samples=50)
            assert result['shap_values'].shape == (38,)
            assert 0.0 <= result['action_prob'] <= 1.0


# ─────────────────────────────────────────────────────────────────────
#  HYPOTHESIS TESTER TESTS
# ─────────────────────────────────────────────────────────────────────

class TestHypothesisTester:

    def test_h1_returns_correct_keys(self, tester):
        result = tester.test_h1_urgency()
        required = {'hypothesis', 'description', 'result',
                    'job0_group_pct', 'threshold', 'chosen_action'}
        assert set(result.keys()) >= required

    def test_h1_result_is_valid_string(self, tester):
        result = tester.test_h1_urgency()
        assert result['result'] in ('VERIFIED', 'NOT VERIFIED')

    def test_h1_urgency_pct_in_range(self, tester):
        result = tester.test_h1_urgency()
        assert 0.0 <= result['job0_group_pct'] <= 100.0

    def test_h2_returns_correct_keys(self, tester):
        result = tester.test_h2_energy()
        required = {'hypothesis', 'result', 'energy_pct',
                    'threshold'}
        assert set(result.keys()) >= required

    def test_h2_result_is_valid_string(self, tester):
        result = tester.test_h2_energy()
        assert result['result'] in ('VERIFIED', 'NOT VERIFIED')

    def test_h2_energy_pct_in_range(self, tester):
        result = tester.test_h2_energy()
        assert 0.0 <= result['energy_pct'] <= 100.0

    def test_h3_returns_correct_keys(self, tester):
        result = tester.test_h3_slot_ordering()
        required = {'hypothesis', 'result', 'job0_group_pct',
                    'job1_group_pct', 'ratio'}
        assert set(result.keys()) >= required

    def test_h3_result_is_valid_string(self, tester):
        result = tester.test_h3_slot_ordering()
        assert result['result'] in ('VERIFIED', 'NOT VERIFIED')

    def test_run_all_returns_three_hypotheses(self, tester):
        results = tester.run_all()
        assert set(results.keys()) == {'H1', 'H2', 'H3'}

    def test_run_all_saves_json(self, tester):
        tester.run_all()
        log_path = os.path.join(PROJECT_ROOT, 'logs', 'hypothesis_results.json')
        assert os.path.exists(log_path), "hypothesis_results.json not created"
        with open(log_path) as f:
            data = json.load(f)
        assert set(data.keys()) == {'H1', 'H2', 'H3'}

    def test_json_is_valid_and_complete(self, tester):
        tester.run_all()
        log_path = os.path.join(PROJECT_ROOT, 'logs', 'hypothesis_results.json')
        with open(log_path) as f:
            data = json.load(f)
        for h_id, result in data.items():
            assert 'result' in result
            assert result['result'] in ('VERIFIED', 'NOT VERIFIED')
            assert 'description' in result