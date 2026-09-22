import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
import numpy as np
import pandas as pd
from learning import learn, LearnedModel, Rule, MIN_TRADES


@dataclass
class T:
    signal_time: pd.Timestamp
    features: dict
    r_result: float
    status: str


def make(seed, n=400, planted=True):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        f = {"trend_move_pct": rng.uniform(0.5, 6), "retrace_depth_atr": rng.uniform(0.2, 4),
             "risk_pct": rng.uniform(0.3, 2), "h1_body_ratio": rng.uniform(0, 1),
             "volatility_pct": rng.uniform(0.05, 0.5),
             "session": str(rng.choice(["Asia", "Europe", "US", "Late US"])),
             "direction": str(rng.choice(["Long", "Short"]))}
        p = (0.10 if f["retrace_depth_atr"] > 2.7 else 0.32) if planted else 0.25
        win = rng.random() < p
        out.append(T(pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=6 * i),
                     f, 3.0 if win else -1.0, "WIN" if win else "LOSS"))
    return out


class TestFindsRealEffects(unittest.TestCase):
    def test_finds_planted_losing_condition_most_of_the_time(self):
        found = sum(any(r.feature == "retrace_depth_atr" and r.kind == "high"
                        for r in learn(make(s)).rules) for s in range(12))
        self.assertGreaterEqual(found, 7, f"found the real effect in only {found}/12")

    def test_learned_threshold_is_near_the_true_one(self):
        m = learn(make(3))
        rule = next(r for r in m.rules if r.feature == "retrace_depth_atr")
        self.assertAlmostEqual(rule.threshold, 2.7, delta=0.5)

    def test_skipping_learned_setups_improves_unseen_results(self):
        m = learn(make(3))
        self.assertGreater(m.test_avg_after, m.test_avg_before)


class TestDoesNotInventLessonsFromNoise(unittest.TestCase):
    """The crucial property: searching many conditions finds patterns in noise
    by chance. The held-back check must reject almost all of them."""

    def test_rarely_adopts_a_lesson_from_pure_noise(self):
        false = sum(bool(learn(make(s + 500, planted=False)).rules) for s in range(40))
        self.assertLessEqual(false, 6, f"invented lessons from noise in {false}/40 runs")

    def test_noise_summary_says_nothing_changed(self):
        for s in range(500, 540):
            m = learn(make(s, planted=False))
            if not m.rules:
                self.assertIn("nothing", m.summary.lower())
                return
        self.fail("expected at least one noise dataset with no lesson")


class TestSafetyRails(unittest.TestCase):
    def test_too_few_trades_learns_nothing(self):
        m = learn(make(1, n=MIN_TRADES - 1))
        self.assertEqual(m.rules, [])
        self.assertIn("Not enough", m.summary)

    def test_uses_later_trades_as_the_unseen_check(self):
        m = learn(make(1))
        self.assertEqual(m.n_train + m.n_test, m.n_trades)
        self.assertAlmostEqual(m.n_train / m.n_trades, 0.7, delta=0.01)

    def test_at_most_two_lessons(self):
        self.assertLessEqual(len(learn(make(3)).rules), 2)

    def test_every_candidate_explains_its_verdict(self):
        for r in learn(make(3)).candidates:
            self.assertTrue(r.verdict)

    def test_ignores_unresolved_and_featureless_trades(self):
        trades = make(1)
        trades.append(T(pd.Timestamp("2027-01-01", tz="UTC"), {}, 3.0, "WIN"))
        trades.append(T(pd.Timestamp("2027-01-02", tz="UTC"), trades[0].features, 0.0, "EXPIRED"))
        self.assertEqual(learn(trades).n_trades, 400)


class TestApplyingLessons(unittest.TestCase):
    def test_setup_matching_a_lesson_fails(self):
        m = LearnedModel(rules=[Rule("retrace_depth_atr", "high", threshold=2.7,
                                     description="deep retrace")])
        ok, why = m.check({"retrace_depth_atr": 3.1})
        self.assertFalse(ok)
        self.assertEqual(why, ["deep retrace"])

    def test_setup_not_matching_passes(self):
        m = LearnedModel(rules=[Rule("retrace_depth_atr", "high", threshold=2.7)])
        self.assertTrue(m.check({"retrace_depth_atr": 1.0})[0])

    def test_missing_feature_is_not_penalised(self):
        m = LearnedModel(rules=[Rule("retrace_depth_atr", "high", threshold=2.7)])
        self.assertTrue(m.check({})[0])
        self.assertTrue(m.check(None)[0])

    def test_category_rule(self):
        m = LearnedModel(rules=[Rule("session", "category", category="Asia")])
        self.assertFalse(m.check({"session": "Asia"})[0])
        self.assertTrue(m.check({"session": "US"})[0])

    def test_round_trip_through_dict(self):
        m = learn(make(3))
        m2 = LearnedModel.from_dict(m.to_dict())
        f = {"retrace_depth_atr": 3.5}
        self.assertEqual(m.check(f), m2.check(f))
        self.assertEqual(m.summary, m2.summary)


if __name__ == "__main__":
    unittest.main()
