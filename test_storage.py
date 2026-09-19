import unittest
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage import (init_db, add_journal_entry, update_journal_entry,
                      close_journal_entry, delete_journal_entry, get_journal_df,
                      journal_to_csv_bytes, add_position, update_position_stop,
                      delete_position, get_positions, clear_all)


class StorageTestBase(unittest.TestCase):
    def setUp(self):
        fd, self.db = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(self.db)  # let sqlite create it fresh
        init_db(self.db)

    def tearDown(self):
        if os.path.exists(self.db):
            os.unlink(self.db)


class TestJournalPersistence(StorageTestBase):
    def test_add_and_read_back(self):
        eid = add_journal_entry({"asset": "BTC-USD", "direction": "Long", "entry": 50000},
                                 db_path=self.db)
        df = get_journal_df(self.db)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["asset"], "BTC-USD")
        self.assertEqual(df.iloc[0]["id"], eid)

    def test_data_survives_a_new_connection(self):
        add_journal_entry({"asset": "ETH-USD", "entry": 3000}, db_path=self.db)
        # Simulate an app restart: brand new connection to the same file.
        df = get_journal_df(self.db)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["asset"], "ETH-USD")

    def test_timestamp_is_auto_set(self):
        add_journal_entry({"asset": "SOL-USD"}, db_path=self.db)
        df = get_journal_df(self.db)
        self.assertTrue(df.iloc[0]["timestamp_utc"])

    def test_close_sets_win_status(self):
        eid = add_journal_entry({"asset": "BTC-USD"}, db_path=self.db)
        close_journal_entry(eid, 250.0, "Target hit", db_path=self.db)
        row = get_journal_df(self.db).iloc[0]
        self.assertEqual(row["status"], "Closed")
        self.assertEqual(row["pl_status"], "Win")
        self.assertEqual(row["realized_pnl"], 250.0)

    def test_close_sets_loss_status(self):
        eid = add_journal_entry({"asset": "BTC-USD"}, db_path=self.db)
        close_journal_entry(eid, -80.0, "Stopped", db_path=self.db)
        self.assertEqual(get_journal_df(self.db).iloc[0]["pl_status"], "Loss")

    def test_close_sets_breakeven_status(self):
        eid = add_journal_entry({"asset": "BTC-USD"}, db_path=self.db)
        close_journal_entry(eid, 0.0, "Scratched", db_path=self.db)
        self.assertEqual(get_journal_df(self.db).iloc[0]["pl_status"], "Breakeven")

    def test_update_unknown_column_raises(self):
        eid = add_journal_entry({"asset": "BTC-USD"}, db_path=self.db)
        with self.assertRaises(ValueError):
            update_journal_entry(eid, db_path=self.db, not_a_column=1)

    def test_update_missing_row_raises(self):
        with self.assertRaises(KeyError):
            update_journal_entry(9999, db_path=self.db, asset="X")

    def test_delete_removes_row(self):
        eid = add_journal_entry({"asset": "BTC-USD"}, db_path=self.db)
        delete_journal_entry(eid, db_path=self.db)
        self.assertEqual(len(get_journal_df(self.db)), 0)

    def test_csv_export_contains_data(self):
        add_journal_entry({"asset": "SOL-USD", "entry": 150}, db_path=self.db)
        text = journal_to_csv_bytes(self.db).decode()
        self.assertIn("asset", text)
        self.assertIn("SOL-USD", text)

    def test_empty_journal_exports_headers_only(self):
        text = journal_to_csv_bytes(self.db).decode()
        self.assertIn("asset", text)
        self.assertNotIn("BTC-USD", text)


class TestPositionPersistence(StorageTestBase):
    def test_add_and_read_back(self):
        pid = add_position("BTC-USD", "crypto", "Long", 50000, 48000, 0.1, db_path=self.db)
        rows = get_positions(self.db)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["asset"], "BTC-USD")
        self.assertEqual(rows[0]["id"], pid)

    def test_update_stop(self):
        pid = add_position("BTC-USD", "crypto", "Long", 50000, 48000, 0.1, db_path=self.db)
        update_position_stop(pid, 49000, db_path=self.db)
        self.assertEqual(get_positions(self.db)[0]["stop"], 49000)

    def test_update_missing_position_raises(self):
        with self.assertRaises(KeyError):
            update_position_stop(9999, 100, db_path=self.db)

    def test_delete_position(self):
        pid = add_position("BTC-USD", "crypto", "Long", 50000, 48000, 0.1, db_path=self.db)
        delete_position(pid, db_path=self.db)
        self.assertEqual(len(get_positions(self.db)), 0)

    def test_multiple_positions_kept_separate(self):
        add_position("BTC-USD", "crypto", "Long", 50000, 48000, 0.1, db_path=self.db)
        add_position("ETH-USD", "crypto", "Short", 3000, 3200, 1.0, db_path=self.db)
        rows = get_positions(self.db)
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["asset"] for r in rows}, {"BTC-USD", "ETH-USD"})


class TestInitAndReset(StorageTestBase):
    def test_init_is_idempotent(self):
        add_journal_entry({"asset": "BTC-USD"}, db_path=self.db)
        init_db(self.db)   # calling again must not wipe data
        self.assertEqual(len(get_journal_df(self.db)), 1)

    def test_clear_all_wipes_both_tables(self):
        add_journal_entry({"asset": "BTC-USD"}, db_path=self.db)
        add_position("BTC-USD", "crypto", "Long", 50000, 48000, 0.1, db_path=self.db)
        clear_all(self.db)
        self.assertEqual(len(get_journal_df(self.db)), 0)
        self.assertEqual(len(get_positions(self.db)), 0)


if __name__ == "__main__":
    unittest.main()
