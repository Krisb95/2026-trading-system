import unittest
import sys, os, tempfile, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import storage
import trade_log as tl


class Base(unittest.TestCase):
    def setUp(self):
        fd, self.db = tempfile.mkstemp(suffix=".db")
        os.close(fd); os.unlink(self.db)
        storage.init_db(self.db)

    def tearDown(self):
        if os.path.exists(self.db):
            os.unlink(self.db)

    def take(self, **kw):
        args = dict(ticker="BTC-USD", direction="Long", planned_entry=100.0, stop=97.0,
                    target=109.0, quantity=2.0, features={"retrace_depth_atr": 1.2},
                    strategy_config="TR|x", db_path=self.db)
        args.update(kw)
        return tl.take_trade(**args)

    def row(self, jid):
        df = storage.get_journal_df(self.db)
        return df[df["id"] == jid].iloc[0]


class TestRealizedR(unittest.TestCase):
    def test_long_target_is_plus_three(self):
        self.assertAlmostEqual(tl.realized_r("Long", 100, 97, 109), 3.0)

    def test_long_stop_is_minus_one(self):
        self.assertAlmostEqual(tl.realized_r("Long", 100, 97, 97), -1.0)

    def test_short(self):
        self.assertAlmostEqual(tl.realized_r("Short", 100, 103, 91), 3.0)
        self.assertAlmostEqual(tl.realized_r("Short", 100, 103, 103), -1.0)

    def test_slippage_past_stop_is_worse_than_minus_one(self):
        self.assertLess(tl.realized_r("Long", 100, 97, 96), -1.0)

    def test_zero_risk_is_none(self):
        self.assertIsNone(tl.realized_r("Long", 100, 100, 105))

    def test_status(self):
        self.assertEqual(tl.pl_status_for(2.0), "Win")
        self.assertEqual(tl.pl_status_for(-1.0), "Loss")
        self.assertEqual(tl.pl_status_for(0.01), "Breakeven")


class TestTakeTrade(Base):
    def test_creates_open_journal_entry(self):
        r = self.row(self.take())
        self.assertEqual(r["status"], "Open")
        self.assertEqual(r["source"], tl.SOURCE_SCANNER)

    def test_records_plan_and_snapshot(self):
        r = self.row(self.take())
        self.assertEqual((r["entry"], r["sl"], r["tp"]), (100.0, 97.0, 109.0))
        self.assertEqual(r["initial_stop"], 97.0)
        self.assertEqual(json.loads(r["features"]), {"retrace_depth_atr": 1.2})

    def test_actual_fill_used_as_entry_but_plan_kept(self):
        r = self.row(self.take(actual_entry=100.4))
        self.assertEqual(r["entry"], 100.4)
        self.assertEqual(r["planned_entry"], 100.0)

    def test_money_at_risk_uses_quantity(self):
        r = self.row(self.take())
        self.assertAlmostEqual(r["potential_loss_at_sl"], -6.0)    # 3 x 2 coins
        self.assertAlmostEqual(r["potential_profit_at_tp"], 18.0)

    def test_rejects_stop_on_wrong_side(self):
        with self.assertRaises(ValueError):
            self.take(stop=101.0)

    def test_records_rule_following(self):
        self.assertEqual(self.row(self.take(followed_rules=False))["followed_rules"], 0)


class TestCompleteTrade(Base):
    def test_take_profit_completion(self):
        jid = self.take()
        tl.complete_trade(jid, exit_price=109.0, db_path=self.db)
        r = self.row(jid)
        self.assertEqual(r["status"], "Closed")
        self.assertAlmostEqual(r["realized_r"], 3.0)
        self.assertEqual(r["pl_status"], "Win")
        self.assertAlmostEqual(r["realized_pnl"], 18.0)

    def test_stop_loss_completion(self):
        jid = self.take()
        tl.complete_trade(jid, exit_price=97.0, db_path=self.db)
        self.assertEqual(self.row(jid)["pl_status"], "Loss")

    def test_r_measured_against_original_stop_even_if_moved(self):
        jid = self.take()
        tl.complete_trade(jid, exit_price=103.0, final_stop=103.0, db_path=self.db)
        r = self.row(jid)
        self.assertAlmostEqual(r["realized_r"], 1.0)     # (103-100)/(100-97)
        self.assertEqual(r["updated_sl"], 103.0)

    def test_final_target_recorded_when_changed(self):
        jid = self.take()
        tl.complete_trade(jid, exit_price=112.0, final_target=112.0, db_path=self.db)
        self.assertEqual(self.row(jid)["updated_tp"], 112.0)

    def test_fees_reduce_pnl_not_r(self):
        jid = self.take()
        tl.complete_trade(jid, exit_price=109.0, fees=2.0, db_path=self.db)
        r = self.row(jid)
        self.assertAlmostEqual(r["realized_pnl"], 16.0)
        self.assertAlmostEqual(r["realized_r"], 3.0)

    def test_corrected_entry_on_completion(self):
        jid = self.take()
        tl.complete_trade(jid, exit_price=109.0, actual_entry=101.0, db_path=self.db)
        self.assertAlmostEqual(self.row(jid)["realized_r"], 2.0)   # (109-101)/(101-97)

    def test_missing_entry_raises(self):
        with self.assertRaises(KeyError):
            tl.complete_trade(999, exit_price=1.0, db_path=self.db)

    def test_default_exit_prices(self):
        self.assertEqual(tl.default_exit_price(tl.OUTCOME_TARGET, 97, 109), 109)
        self.assertEqual(tl.default_exit_price(tl.OUTCOME_STOP, 97, 109), 97)
        self.assertEqual(tl.default_exit_price(tl.OUTCOME_MANUAL, 97, 109, 104), 104)


class TestLearningFeed(Base):
    def test_completed_trades_feed_learner(self):
        tl.complete_trade(self.take(), exit_price=109.0, db_path=self.db)
        tl.complete_trade(self.take(), exit_price=97.0, db_path=self.db)
        trades = tl.learning_trades(self.db)
        self.assertEqual([t.status for t in trades], ["WIN", "LOSS"])
        self.assertEqual(trades[0].features, {"retrace_depth_atr": 1.2})

    def test_open_trades_are_not_used(self):
        self.take()
        self.assertEqual(tl.learning_trades(self.db), [])

    def test_rule_breaking_trades_excluded_by_default(self):
        tl.complete_trade(self.take(followed_rules=False), exit_price=109.0, db_path=self.db)
        self.assertEqual(tl.learning_trades(self.db), [])
        self.assertEqual(len(tl.learning_trades(self.db, followed_only=False)), 1)

    def test_filter_by_strategy_settings(self):
        tl.complete_trade(self.take(strategy_config="A"), exit_price=109.0, db_path=self.db)
        tl.complete_trade(self.take(strategy_config="B"), exit_price=97.0, db_path=self.db)
        self.assertEqual(len(tl.learning_trades(self.db, config="A")), 1)

    def test_manual_journal_entries_are_not_learning_data(self):
        jid = storage.add_journal_entry({"asset": "ETH-USD", "direction": "Long",
                                         "entry": 100, "sl": 97}, db_path=self.db)
        storage.close_journal_entry(jid, 50.0, "manual", db_path=self.db)
        self.assertEqual(tl.learning_trades(self.db), [])

    def test_learner_accepts_journal_trades(self):
        from learning import learn
        tl.complete_trade(self.take(), exit_price=109.0, db_path=self.db)
        self.assertEqual(learn(tl.learning_trades(self.db)).n_trades, 1)

    def test_open_scanner_trades_listed(self):
        self.take()
        self.assertEqual(len(tl.open_scanner_trades(self.db)), 1)


class TestOldJournalMigrates(Base):
    def test_journal_without_new_columns_is_migrated(self):
        import sqlite3
        conn = sqlite3.connect(self.db)
        conn.execute("DROP TABLE journal")
        conn.execute("""CREATE TABLE journal (id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT DEFAULT 'Open', asset TEXT, direction TEXT, leverage REAL,
            entry REAL, size_notional_usd REAL, margin REAL, current_pl REAL, tp REAL,
            sl REAL, updated_tp REAL, updated_sl REAL, potential_profit_at_tp REAL,
            potential_loss_at_sl REAL, realized_pnl REAL, pl_status TEXT,
            entry_reason TEXT, score_breakdown TEXT, exit_reason TEXT,
            timestamp_utc TEXT, management_notes TEXT)""")
        conn.execute("INSERT INTO journal (asset, entry) VALUES ('OLD-USD', 5)")
        conn.commit(); conn.close()
        storage.init_db(self.db)
        df = storage.get_journal_df(self.db)
        self.assertEqual(df.iloc[0]["asset"], "OLD-USD")          # old data kept
        self.assertIn("realized_r", df.columns)
        self.row(self.take())                                     # and new trades work


if __name__ == "__main__":
    unittest.main()


class TestJournalSurvivesRedeploy(Base):
    """Free hosting wipes the database on redeploy; export + import must
    restore real trades completely, including what the learner needs."""

    def test_export_wipe_import_restores_learning_data(self):
        tl.complete_trade(self.take(), exit_price=109.0, db_path=self.db)
        tl.complete_trade(self.take(ticker="ETH-USD"), exit_price=97.0, db_path=self.db)
        exported = storage.journal_to_csv_bytes(self.db)

        os.unlink(self.db)                     # simulate the redeploy wipe
        storage.init_db(self.db)
        self.assertEqual(tl.learning_trades(self.db), [])

        added = storage.import_journal_csv(exported, db_path=self.db)
        self.assertEqual(added, 2)
        restored = tl.learning_trades(self.db)
        self.assertEqual(sorted(t.status for t in restored), ["LOSS", "WIN"])
        self.assertEqual(restored[0].features, {"retrace_depth_atr": 1.2})

    def test_importing_twice_does_not_duplicate(self):
        tl.complete_trade(self.take(), exit_price=109.0, db_path=self.db)
        exported = storage.journal_to_csv_bytes(self.db)
        self.assertEqual(storage.import_journal_csv(exported, db_path=self.db), 0)

    def test_open_trades_survive_too(self):
        self.take()
        exported = storage.journal_to_csv_bytes(self.db)
        os.unlink(self.db); storage.init_db(self.db)
        storage.import_journal_csv(exported, db_path=self.db)
        self.assertEqual(len(tl.open_scanner_trades(self.db)), 1)

    def test_rejects_wrong_file(self):
        with self.assertRaises(ValueError):
            storage.import_journal_csv(b"x,y\n1,2\n", db_path=self.db)
