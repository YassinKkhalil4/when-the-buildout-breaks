"""Offline tests for ai_bust_live.py. Payloads follow the providers' documented layouts:
  - Treasury CSV: header confirmed from a real fetch (Jan 2026 rows); other rows are illustrative.
  - FRED / Stooq / Yahoo / SEC companyfacts: SYNTHETIC payloads in the documented schema (not recorded from the servers)."""
import json, os, tempfile, unittest
os.environ["AI_BUST_DB"] = os.path.join(tempfile.mkdtemp(), "t.db")
import ai_bust_live as L

TREAS = ('Date,"1 Mo","1.5 Month","2 Mo","3 Mo","4 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"\n'
         '01/05/2026,3.71,3.68,3.64,3.64,3.61,3.57,3.47,3.46,3.53,3.71,3.92,4.17,4.79,4.85\n'
         '01/02/2026,3.72,3.71,3.66,3.65,3.62,3.58,3.47,3.47,3.55,3.74,3.95,4.19,4.81,4.86\n')
FRED = "observation_date,DGS10\n2026-09-29,5.20\n2026-09-30,.\n2026-10-01,5.29\n"
STOOQ = "Date,Open,High,Low,Close,Volume\n" + "".join(f"2026-09-{d:02d},1,1,1,{100+d},1\n" for d in range(1, 29))
YAHOO = json.dumps({"chart": {"result": [{"timestamp": [1759276800 + 86400 * i for i in range(10)],
                                          "indicators": {"quote": [{"close": [100, 101, None, 103, 99, 98, 97, 99, 100, 101]}]}}]}})

def fact(tag_vals):
    us = {}
    for tag, ents in tag_vals.items():
        us[tag] = {"units": {"USD": ents}}
    return {"entityName": "TestCo", "cik": 1, "facts": {"us-gaap": us}}

CF = fact({
    "PaymentsToAcquirePropertyPlantAndEquipment": [
        {"start": "2025-01-01", "end": "2025-12-31", "val": 100e9, "form": "10-K", "filed": "2026-02-20"},
        {"start": "2025-01-01", "end": "2025-06-30", "val": 40e9, "form": "10-Q", "filed": "2025-08-01"},
        {"start": "2026-01-01", "end": "2026-06-30", "val": 70e9, "form": "10-Q", "filed": "2026-08-01"}],
    "LongTermDebt": [{"end": "2026-06-30", "val": 35.1e9, "form": "10-Q", "filed": "2026-08-01"},
                     {"end": "2025-12-31", "val": 20e9, "form": "10-K", "filed": "2026-02-20"}],
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths": [{"end": "2026-06-30", "val": 5e9, "form": "10-Q", "filed": "2026-08-01"}],
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo": [{"end": "2026-06-30", "val": 7e9, "form": "10-Q", "filed": "2026-08-01"}],
    "LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive": [{"end": "2026-06-30", "val": 9e9, "form": "10-Q", "filed": "2026-08-01"}],
})

class T(unittest.TestCase):
    def test_treasury(self):
        r = L.parse_treasury_csv(TREAS)
        self.assertEqual(r["date"], "2026-01-05"); self.assertAlmostEqual(r["y3m"], 0.0364); self.assertAlmostEqual(r["y10"], 0.0417)
    def test_treasury_bad(self):
        with self.assertRaises(ValueError): L.parse_treasury_csv("foo,bar\n1,2\n")
    def test_fred_skips_missing(self):
        s = L.parse_fred_csv(FRED); self.assertEqual(len(s), 2); self.assertAlmostEqual(s[-1][1], 0.0529)
    def test_stooq_and_metrics(self):
        m = L.price_metrics(L.parse_stooq_csv(STOOQ), peak_after="2026-09-01")
        self.assertEqual(m["last"], 128.0); self.assertAlmostEqual(m["drawdown"], 0.0); self.assertGreater(m["vol30"], 0)
    def test_drawdown(self):
        s = [("2026-06-30", 100.0), ("2026-07-31", 70.0), ("2026-09-30", 80.0)]
        self.assertAlmostEqual(L.price_metrics(s, peak_after="2026-06-01")["drawdown"], 0.20)
    def test_yahoo_skips_null(self):
        self.assertEqual(len(L.parse_yahoo_chart(YAHOO)), 9)
    def test_companyfacts(self):
        m = L.parse_companyfacts(CF)
        self.assertAlmostEqual(m["capex_ttm"], 130.0)          # 100 + 70 - 40
        self.assertAlmostEqual(m["debt"], 35.1)
        self.assertEqual(m["maturity_ladder"][:2], [5.0, 7.0]); self.assertEqual(m["maturity_ladder"][5], 9.0)
    def test_calibrate_recorded(self):
        snap = json.load(open(os.path.join(L.HERE, "fixtures", "snapshot_recorded_2026-10-05.json")))
        c = L.calibrate(snap)
        self.assertAlmostEqual(c["part1"]["r_free"], 0.044)
        self.assertEqual(c["part1"]["sigma"], (0.322, 0.45))
        self.assertEqual(c["part1"]["sox_drawdown"], (0.12, 0.22))      # reproduces the paper's ranges
        self.assertAlmostEqual(sum(c["model_f"]["nc_maturity_profile"]), 1.0, places=2)
    def test_failed_fetch_falls_back_stale(self):
        def bad(url, headers=None, timeout=20): raise ConnectionError("blocked")
        prev = {"rates": {"y3m": 0.04}, "equities": {}, "companies": {}}
        s = L.build_snapshot(get=bad, manual={}, previous=prev)
        self.assertFalse(s["sources"]["rates"]["ok"]); self.assertTrue(s["sources"]["rates"].get("stale"))
        self.assertEqual(s["rates"]["y3m"], 0.04)
    def test_live_with_fake_server(self):
        def fake(url, headers=None, timeout=20):
            if "treasury.gov" in url: return 200, TREAS
            if "stooq" in url: return 200, STOOQ
            if "companyfacts" in url: return 200, json.dumps(CF)
            if "submissions" in url: return 200, json.dumps({"filings": {"recent": {"form": ["10-K", "10-Q"], "filingDate": ["2026-02-20", "2026-08-01"]}}})
            return 404, ""
        s = L.build_snapshot(get=fake, manual={})
        self.assertTrue(s["sources"]["rates"]["ok"]); self.assertEqual(s["companies"]["CoreWeave"]["latest_10q"], "2026-08-01")
if __name__ == "__main__":
    unittest.main()
