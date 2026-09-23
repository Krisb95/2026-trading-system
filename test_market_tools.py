import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
import numpy as np
import pandas as pd
from market_tools import (rsi, rsi_state, rsi_row, altcoin_season_index, flow_rows,
                           most_crowded, turnover_ratio, FlowRow)


def series(values):
    return pd.Series(values, dtype=float)


class TestRSI(unittest.TestCase):
    def test_straight_uptrend_is_maximal(self):
        self.assertAlmostEqual(rsi(series(range(1, 40))), 100.0, places=4)

    def test_straight_downtrend_is_minimal(self):
        self.assertLess(rsi(series(range(40, 1, -1))), 1.0)

    def test_flat_series_is_neutral(self):
        self.assertAlmostEqual(rsi(series([100] * 40)), 50.0)

    def test_known_range(self):
        rng = np.random.default_rng(1)
        v = rsi(series(100 + np.cumsum(rng.normal(0, 1, 200))))
        self.assertTrue(0 <= v <= 100)

    def test_too_little_data(self):
        self.assertIsNone(rsi(series([1, 2, 3])))
        self.assertIsNone(rsi(None))

    def test_states(self):
        self.assertEqual(rsi_state(80), "Overbought")
        self.assertEqual(rsi_state(20), "Oversold")
        self.assertEqual(rsi_state(50), "Neutral")
        self.assertEqual(rsi_state(None), "—")

    def test_row_across_timeframes(self):
        up = pd.DataFrame({"Close": range(1, 40)})
        row = rsi_row({"1h": up, "4h": pd.DataFrame({"Close": []}), "1d": None})
        self.assertAlmostEqual(row["1h"], 100.0, places=4)
        self.assertIsNone(row["4h"])
        self.assertIsNone(row["1d"])


class TestAltcoinSeason(unittest.TestCase):
    def test_most_alts_beating_btc_is_altcoin_season(self):
        changes = {"BTC": 10.0, **{f"A{i}": 50.0 for i in range(40)}}
        s = altcoin_season_index(changes)
        self.assertEqual(s.index, 100)
        self.assertEqual(s.label, "Altcoin season")

    def test_most_alts_lagging_is_bitcoin_season(self):
        changes = {"BTC": 80.0, **{f"A{i}": 5.0 for i in range(40)}}
        self.assertEqual(altcoin_season_index(changes).label, "Bitcoin season")

    def test_even_split_is_mixed(self):
        changes = {"BTC": 10.0}
        changes.update({f"U{i}": 20.0 for i in range(20)})
        changes.update({f"D{i}": 5.0 for i in range(20)})
        s = altcoin_season_index(changes)
        self.assertEqual(s.index, 50)
        self.assertEqual(s.label, "Neither — mixed")

    def test_respects_the_top_n_sample(self):
        changes = {"BTC": 10.0, **{f"A{i}": 50.0 for i in range(80)}}
        self.assertEqual(altcoin_season_index(changes, top_n=50).sample, 50)

    def test_missing_btc_returns_none(self):
        self.assertIsNone(altcoin_season_index({"ETH": 5.0}))

    def test_empty_returns_none(self):
        self.assertIsNone(altcoin_season_index({}))


@dataclass
class Ctx:
    name: str
    funding_rate: float
    open_interest_usd: float
    day_volume_usd: float


class TestFlow(unittest.TestCase):
    def _rows(self):
        return flow_rows([Ctx("BTC", 0.00001, 1e9, 2e9),
                          Ctx("HOT", 0.0005, 1e8, 9e8),
                          Ctx("COLD", -0.0004, 5e7, 1e8)])

    def test_annualised_funding(self):
        row = self._rows()[1]
        self.assertAlmostEqual(row.funding_annual_pct, 0.0005 * 24 * 365 * 100)

    def test_crowding_flags(self):
        rows = {r.name: r for r in self._rows()}
        self.assertEqual(rows["HOT"].crowded, "Longs crowded")
        self.assertEqual(rows["COLD"].crowded, "Shorts crowded")
        self.assertIsNone(rows["BTC"].crowded)

    def test_most_crowded_both_sides(self):
        longs, shorts = most_crowded(self._rows(), limit=1)
        self.assertEqual(longs[0].name, "HOT")
        self.assertEqual(shorts[0].name, "COLD")

    def test_turnover_ratio(self):
        row = self._rows()[1]
        self.assertAlmostEqual(turnover_ratio(row), 9.0)

    def test_turnover_without_open_interest(self):
        self.assertIsNone(turnover_ratio(FlowRow("X", 0.0, 0.0, 0.0, 1e6, None)))

    def test_handles_missing_fields(self):
        class Bare:
            name = "B"
        rows = flow_rows([Bare()])
        self.assertEqual(rows[0].funding_rate, 0.0)
