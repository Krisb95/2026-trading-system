import unittest
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import storage
import tracking
from scanner import RankedCandidate
from trade_sim import WIN, LOSS, EXPIRED, PENDING


def ranked(label="Bitcoin (BTC)", ticker="BTC-USD", grade="A+", direction="Long",
           entry=100.0, stop=95.0, target=110.0, error=None):
    return RankedCandidate(ticker=ticker, label=label, direction=direction, score=8.0,
                            grade=grade, regime="bullish", price=102.0, stop=stop,
                            target=target, reward_risk=2.0, error=error, entry=entry)


def hourly(rows, start="2026-06-01 01:00"):
    idx = pd.date_range(start, periods=len(rows), freq="1h", tz="UTC")
    return pd.DataFrame({"Open": [r[2] for r in rows], "High": [r[0] for r in rows],
                          "Low": [r[1] for r in rows], "Close": [r[2] for r in rows]},
                         index=idx)


class Base(unittest.TestCase):
    def setUp(self):
        fd, self.db = tempfile.mkstemp(suffix=".db")
        os.close(fd); os.unlink(self.db)
        storage.init_db(self.db)

    def tearDown(self):
        if os.path.exists(self.db):
            os.unlink(self.db)


class TestRecording(Base):
    def test_records_a_plus_and_b(self):
        added = tracking.record_from_ranked(
            [ranked(grade="A+"), ranked(label="E", ticker="ETH-USD", grade="B")],
            db_path=self.db)
        self.assertEqual(added, 2)

    def test_skips_c_grades(self):
        added = tracking.record_from_ranked([ranked(grade="C")], db_path=self.db)
        self.assertEqual(added, 0)

    def test_skips_errored_rows(self):
        added = tracking.record_from_ranked([ranked(error="boom")], db_path=self.db)
        self.assertEqual(added, 0)

    def test_skips_incomplete_plans(self):
        added = tracking.record_from_ranked([ranked(target=None)], db_path=self.db)
        self.assertEqual(added, 0)

    def test_rescanning_does_not_duplicate_open_signal(self):
        tracking.record_from_ranked([ranked()], db_path=self.db)
        tracking.record_from_ranked([ranked()], db_path=self.db)
        self.assertEqual(len(storage.get_signals_df(self.db)), 1)

    def test_new_signal_allowed_after_previous_resolved(self):
        tracking.record_from_ranked([ranked()], db_path=self.db)
        sid = int(storage.get_signals_df(self.db).iloc[0]["id"])
        storage.update_signal(sid, db_path=self.db, status=WIN)
        tracking.record_from_ranked([ranked()], db_path=self.db)
        self.assertEqual(len(storage.get_signals_df(self.db)), 2)

    def test_plan_is_frozen_at_creation(self):
        tracking.record_from_ranked([ranked(entry=100, stop=95, target=110)], db_path=self.db)
        row = storage.get_signals_df(self.db).iloc[0]
        self.assertEqual((row["entry"], row["stop"], row["target"]), (100, 95, 110))


class TestResolution(Base):
    def _one(self):
        tracking.record_from_ranked([ranked()], db_path=self.db)
        sid = int(storage.get_signals_df(self.db).iloc[0]["id"])
        storage.update_signal(sid, db_path=self.db, created_utc="2026-06-01T00:00:00+00:00")
        return sid

    def test_win_resolved(self):
        self._one()
        bars = hourly([(101, 99, 100), (111, 100, 110)])
        counts = tracking.update_all(lambda t: bars, db_path=self.db)
        self.assertEqual(counts["resolved"], 1)
        self.assertEqual(storage.get_signals_df(self.db).iloc[0]["status"], WIN)

    def test_loss_resolved(self):
        self._one()
        bars = hourly([(101, 99, 100), (100, 94, 95)])
        tracking.update_all(lambda t: bars, db_path=self.db)
        self.assertEqual(storage.get_signals_df(self.db).iloc[0]["status"], LOSS)

    def test_bars_before_signal_are_ignored(self):
        """A candle that dipped to entry BEFORE the signal existed cannot fill it."""
        self._one()
        bars = hourly([(101, 90, 95)] + [(108, 104, 106)] * 3, start="2026-05-31 23:00")
        tracking.update_all(lambda t: bars, db_path=self.db)
        self.assertNotEqual(storage.get_signals_df(self.db).iloc[0]["status"], LOSS)

    def test_unfilled_signal_expires(self):
        self._one()
        bars = hourly([(110, 105, 108)] * 80)   # never returns to 100, 72h expiry
        tracking.update_all(lambda t: bars, db_path=self.db)
        self.assertEqual(storage.get_signals_df(self.db).iloc[0]["status"], EXPIRED)

    def test_failed_fetch_is_counted_not_crashed(self):
        self._one()
        counts = tracking.update_all(lambda t: None, db_path=self.db)
        self.assertEqual(counts["failed"], 1)
        self.assertEqual(storage.get_signals_df(self.db).iloc[0]["status"], PENDING)

    def test_resolved_signals_not_rechecked(self):
        self._one()
        tracking.update_all(lambda t: hourly([(101, 99, 100), (111, 100, 110)]),
                             db_path=self.db)
        counts = tracking.update_all(lambda t: hourly([(101, 99, 100)]), db_path=self.db)
        self.assertEqual(counts["checked"], 0)


class TestPersistence(Base):
    def test_export_then_import_restores_signals(self):
        tracking.record_from_ranked([ranked(), ranked(label="E", ticker="ETH-USD")],
                                     db_path=self.db)
        exported = storage.signals_to_csv_bytes(self.db)
        storage.clear_signals(self.db)
        self.assertEqual(len(storage.get_signals_df(self.db)), 0)
        added = storage.import_signals_csv(exported, db_path=self.db)
        self.assertEqual(added, 2)

    def test_import_skips_duplicates(self):
        tracking.record_from_ranked([ranked()], db_path=self.db)
        exported = storage.signals_to_csv_bytes(self.db)
        self.assertEqual(storage.import_signals_csv(exported, db_path=self.db), 0)

    def test_import_rejects_wrong_csv(self):
        with self.assertRaises(ValueError):
            storage.import_signals_csv(b"a,b\n1,2\n", db_path=self.db)


class TestStats(Base):
    def test_stats_by_grade_from_tracked_signals(self):
        tracking.record_from_ranked([ranked()], db_path=self.db)
        sid = int(storage.get_signals_df(self.db).iloc[0]["id"])
        storage.update_signal(sid, db_path=self.db, status=WIN, r_result=2.0)
        stats = {s.grade: s for s in tracking.tracked_stats(self.db)}
        self.assertEqual(stats["A+"].wins, 1)
        self.assertAlmostEqual(stats["A+"].avg_r, 2.0)


if __name__ == "__main__":
    unittest.main()


class TestMinimumRewardRisk(Base):
    def test_setup_below_minimum_is_not_recorded(self):
        low = ranked(); low.reward_risk = 2.5
        self.assertEqual(tracking.record_from_ranked([low], db_path=self.db, min_rr=3.0), 0)

    def test_setup_at_minimum_is_recorded(self):
        ok = ranked(); ok.reward_risk = 3.0
        self.assertEqual(tracking.record_from_ranked([ok], db_path=self.db, min_rr=3.0), 1)

    def test_setup_without_rr_is_not_recorded_under_a_floor(self):
        none_rr = ranked(); none_rr.reward_risk = None
        self.assertEqual(tracking.record_from_ranked([none_rr], db_path=self.db, min_rr=3.0), 0)
