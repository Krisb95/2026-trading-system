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


class TestPositionFieldsAndMigration(StorageTestBase):
    def test_new_fields_persist(self):
        pid = add_position("BTC-USD", "crypto", "Long", 80000, 78000, 0.1,
                            target=85000, leverage=3.0, fee_rate=0.0006,
                            slippage_pct=0.0005, entry_reason="4H sweep reclaim",
                            db_path=self.db)
        row = get_positions(self.db)[0]
        self.assertEqual(row["target"], 85000)
        self.assertEqual(row["leverage"], 3.0)
        self.assertAlmostEqual(row["fee_rate"], 0.0006)
        self.assertEqual(row["entry_reason"], "4H sweep reclaim")

    def test_update_position_edits_stop_and_target(self):
        from storage import update_position
        pid = add_position("BTC-USD", "crypto", "Long", 80000, 78000, 0.1,
                            target=85000, db_path=self.db)
        update_position(pid, db_path=self.db, stop=79000, target=90000)
        row = get_positions(self.db)[0]
        self.assertEqual(row["stop"], 79000)
        self.assertEqual(row["target"], 90000)

    def test_update_rejects_non_editable_field(self):
        from storage import update_position
        pid = add_position("BTC-USD", "crypto", "Long", 80000, 78000, 0.1, db_path=self.db)
        with self.assertRaises(ValueError):
            update_position(pid, db_path=self.db, asset="ETH-USD")

    def test_update_missing_position_raises(self):
        from storage import update_position
        with self.assertRaises(KeyError):
            update_position(9999, db_path=self.db, stop=1)

    def test_migration_adds_columns_to_old_schema(self):
        """An existing database created before these columns existed must be
        upgraded in place, not wiped."""
        import sqlite3, os, tempfile
        fd, old_db = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(old_db)
        conn = sqlite3.connect(old_db)
        conn.execute("""CREATE TABLE positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, asset TEXT, asset_class TEXT,
            direction TEXT, entry REAL, stop REAL, quantity REAL,
            contract_multiplier REAL DEFAULT 1.0, opened_utc TEXT)""")
        conn.execute("INSERT INTO positions (asset, entry, stop, quantity) "
                      "VALUES ('BTC-USD', 100, 90, 1)")
        conn.commit()
        conn.close()

        init_db(old_db)                      # should migrate, not destroy
        rows = get_positions(old_db)
        self.assertEqual(len(rows), 1)       # original row survived
        self.assertIn("target", rows[0])     # new column present
        self.assertEqual(rows[0]["asset"], "BTC-USD")
        os.unlink(old_db)

    def test_defaults_when_optional_fields_omitted(self):
        pid = add_position("ETH-USD", "crypto", "Long", 3000, 2900, 1.0, db_path=self.db)
        row = get_positions(self.db)[0]
        self.assertIsNone(row["target"])
        self.assertEqual(row["leverage"], 1.0)


class TestPositionEditing(StorageTestBase):
    """Positions gained target/leverage/notes and full editing."""

    def test_add_with_target_and_leverage(self):
        pid = add_position("BTC-USD", "crypto", "Long", 50000, 48000, 0.1,
                            target=55000, leverage=3.0, entry_reason="4H reclaim",
                            db_path=self.db)
        row = get_positions(self.db)[0]
        self.assertEqual(row["target"], 55000)
        self.assertEqual(row["leverage"], 3.0)
        self.assertEqual(row["entry_reason"], "4H reclaim")

    def test_update_stop_and_target(self):
        from storage import update_position
        pid = add_position("BTC-USD", "crypto", "Long", 50000, 48000, 0.1,
                            target=55000, db_path=self.db)
        update_position(pid, db_path=self.db, stop=49000, target=56000)
        row = get_positions(self.db)[0]
        self.assertEqual(row["stop"], 49000)
        self.assertEqual(row["target"], 56000)

    def test_update_quantity_and_entry_for_scale_in(self):
        from storage import update_position
        pid = add_position("BTC-USD", "crypto", "Long", 50000, 48000, 0.1, db_path=self.db)
        update_position(pid, db_path=self.db, entry=51000, quantity=0.2)
        row = get_positions(self.db)[0]
        self.assertEqual(row["entry"], 51000)
        self.assertAlmostEqual(row["quantity"], 0.2)

    def test_update_rejects_unknown_field(self):
        from storage import update_position
        pid = add_position("BTC-USD", "crypto", "Long", 50000, 48000, 0.1, db_path=self.db)
        with self.assertRaises(ValueError):
            update_position(pid, db_path=self.db, bogus=1)

    def test_update_missing_position_raises(self):
        from storage import update_position
        with self.assertRaises(KeyError):
            update_position(9999, db_path=self.db, stop=1)

    def test_target_may_be_absent(self):
        add_position("BTC-USD", "crypto", "Long", 50000, 48000, 0.1, db_path=self.db)
        self.assertIsNone(get_positions(self.db)[0]["target"])


class TestSchemaMigration(StorageTestBase):
    """An existing database created before target/leverage/notes existed must
    gain those columns rather than crashing with 'no such column'."""

    def test_migration_adds_missing_columns(self):
        import sqlite3, os, tempfile
        fd, old_db = tempfile.mkstemp(suffix=".db")
        os.close(fd); os.unlink(old_db)
        conn = sqlite3.connect(old_db)
        conn.execute("""CREATE TABLE positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, asset TEXT, asset_class TEXT,
            direction TEXT, entry REAL, stop REAL, quantity REAL,
            contract_multiplier REAL DEFAULT 1.0, opened_utc TEXT)""")
        conn.execute("INSERT INTO positions (asset, entry, stop, quantity) "
                      "VALUES ('OLD-USD', 1.0, 0.9, 5)")
        conn.commit(); conn.close()

        init_db(old_db)   # should migrate, not fail
        rows = get_positions(old_db)
        self.assertEqual(len(rows), 1)          # pre-existing row survived
        self.assertEqual(rows[0]["asset"], "OLD-USD")
        self.assertIn("target", rows[0])
        self.assertIn("leverage", rows[0])
        os.unlink(old_db)


class TestKeyValue(StorageTestBase):
    def test_save_and_load_round_trip(self):
        from storage import save_value, load_value
        save_value("evidence", {"a": [1, 2]}, db_path=self.db)
        value, when = load_value("evidence", db_path=self.db)
        self.assertEqual(value, {"a": [1, 2]})
        self.assertIsNotNone(when)

    def test_missing_key(self):
        from storage import load_value
        self.assertEqual(load_value("nope", db_path=self.db), (None, None))

    def test_overwrite(self):
        from storage import save_value, load_value
        save_value("k", 1, db_path=self.db)
        save_value("k", 2, db_path=self.db)
        self.assertEqual(load_value("k", db_path=self.db)[0], 2)
