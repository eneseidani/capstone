"""Self-contained tests for the point-in-time spread fallback policy.

Run with:
    python test_spread_fallback.py

The test uses temporary synthetic Yahoo files and does not modify project data.
"""
from pathlib import Path
import importlib.util
import sys
import tempfile
import types

import numpy as np
import pandas as pd


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_dir = root / "input_data"
        yahoo_dir = input_dir / "yahoo"
        yahoo_dir.mkdir(parents=True)
        universe_path = root / "Investable_Universe.csv"

        pd.DataFrame([
            {"Ticker": "TST", "Asset Class": "ETF"},
            {"Ticker": "BTC-USD", "Asset Class": "Crypto"},
            {"Ticker": "DGS3MO", "Asset Class": "Cash Proxy"},
        ]).to_csv(universe_path, index=False)

        dates = pd.bdate_range("2024-01-01", periods=310)
        spreads = np.linspace(0.0010, 0.0030, len(dates))
        # Short gap: one missing ticker observation.
        spreads[280] = np.nan
        # Long gap: ten consecutive missing observations.
        spreads[300:310] = np.nan
        test_data = pd.DataFrame({
            "Date": dates,
            "Open": np.full(len(dates), 10.0),
            "Close": np.full(len(dates), 10.0),
            "High": np.full(len(dates), 10.1),
            "Low": np.full(len(dates), 9.9),
            "Volume": np.full(len(dates), 100_000.0),
            "Spread": spreads,
            "ADV_Dollars": np.full(len(dates), 1_000_000.0),
        })
        test_data.to_csv(yahoo_dir / "TST.csv", index=False)

        btc = test_data.copy()
        btc["Spread"] = np.nan
        btc["ADV_Dollars"] = np.nan
        btc.to_csv(yahoo_dir / "BTC-USD.csv", index=False)

        fake_config = types.ModuleType("config")
        fake_config.INPUT_DATA_DIR = input_dir
        fake_config.UNIVERSE_FILE = universe_path
        sys.modules["config"] = fake_config

        module_path = Path(__file__).with_name("transaction_fee_helper.py")
        spec = importlib.util.spec_from_file_location("transaction_fee_helper_test", module_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)

        # 1. Observed spread is used unchanged.
        observed_date = dates[279]
        observed = helper.load_fee_context(observed_date, tickers=["TST", "DGS3MO"])
        assert np.isclose(observed.at["TST", "Spread"], spreads[279])
        assert not bool(observed.at["TST", "Spread_Fallback"])
        assert observed.at["TST", "Spread_Fallback_Method"] == "observed"

        # 2. One-observation gap carries the last valid historical spread.
        short_date = dates[280]
        short = helper.load_fee_context(short_date, tickers=["TST", "DGS3MO"])
        assert bool(short.at["TST", "Spread_Fallback"])
        assert short.at["TST", "Spread_Fallback_Method"] == "last_valid_historical"
        assert short.at["TST", "Spread_Fallback_Age_Trading_Days"] == 1
        assert np.isclose(short.at["TST", "Spread"], spreads[279])

        # 3. Long gap uses trailing p90; current date and future rows are excluded.
        long_date = dates[309]
        long = helper.load_fee_context(long_date, tickers=["TST", "DGS3MO"])
        trailing_start = max(0, 309 - helper.SPREAD_FALLBACK_LOOKBACK_TRADING_DAYS)
        historical = spreads[trailing_start:309]
        historical = historical[np.isfinite(historical) & (historical >= 0)]
        expected_p90 = float(np.quantile(historical, 0.90))
        assert bool(long.at["TST", "Spread_Fallback"])
        assert long.at["TST", "Spread_Fallback_Method"] == "trailing_252_observation_p90"
        assert long.at["TST", "Spread_Fallback_Age_Trading_Days"] == 10
        assert np.isclose(long.at["TST", "Spread"], expected_p90)
        assert long.at["TST", "Spread_Fallback_History_Count"] == len(historical)

        # 4. Future data cannot alter a historical fallback.
        before_future = float(long.at["TST", "Spread"])
        extended = pd.concat([
            test_data,
            pd.DataFrame([{
                "Date": dates[-1] + pd.offsets.BDay(1),
                "Open": 10.0, "Close": 10.0, "High": 10.1, "Low": 9.9,
                "Volume": 100_000.0, "Spread": 99.0,
                "ADV_Dollars": 1_000_000.0,
            }]),
        ], ignore_index=True)
        extended.to_csv(yahoo_dir / "TST.csv", index=False)
        after_future = helper.load_fee_context(long_date, tickers=["TST", "DGS3MO"])
        assert np.isclose(float(after_future.at["TST", "Spread"]), before_future)

        # 5. A transaction-cost breakdown records the fallback actually used.
        context = long.copy()
        current = pd.Series({"TST": 0.0, "DGS3MO": 1.0})
        target = pd.Series({"TST": 1.0, "DGS3MO": 0.0})
        fee, breakdown = helper.calculate_transaction_fees(
            current, target, 100_000.0, context, return_breakdown=True
        )
        assert fee > 0
        etf_trade = breakdown.loc[breakdown["Ticker"].eq("TST")].iloc[0]
        assert bool(etf_trade["Spread_Fallback"])
        assert etf_trade["Spread_Fallback_Method"] == "trailing_252_observation_p90"
        assert np.isclose(etf_trade["Spread_Used"], expected_p90)

        # 6. Crypto spread is not estimated/fallback-filled; crypto commission remains separate.
        crypto_context = helper.load_fee_context(dates[309], tickers=["BTC-USD", "DGS3MO"])
        assert pd.isna(crypto_context.at["BTC-USD", "Spread"])
        assert crypto_context.at["BTC-USD", "Spread_Fallback_Method"] == "not_applicable_crypto"

        print("PASS: observed spread, short-gap carry, long-gap P90, no-lookahead, audit metadata, and crypto handling.")


if __name__ == "__main__":
    main()
