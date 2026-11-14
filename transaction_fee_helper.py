from pathlib import Path

import numpy as np
import pandas as pd

import config


YAHOO_INPUT_DIR = config.INPUT_DATA_DIR / "yahoo"


# =============================================================================
# IBKR FIXED COMMISSION - U.S. STOCKS / ETFs
# =============================================================================

US_COMMISSION_PER_SHARE = 0.005
US_MIN_COMMISSION = 1.00
US_MAX_COMMISSION_RATE = 0.01


# =============================================================================
# IBKR CRYPTO
# =============================================================================

CRYPTO_MIN_COMMISSION = 1.75
CRYPTO_MAX_COMMISSION_RATE = 0.01

CRYPTO_TIERS = (
    (100_000.0, 0.0018),
    (1_000_000.0, 0.0015),
    (np.inf, 0.0012),
)


# =============================================================================
# MARKET IMPACT
# =============================================================================

MARKET_IMPACT_COEFFICIENT = 0.1


# =============================================================================
# POINT-IN-TIME SPREAD FALLBACK POLICY
# =============================================================================

# A missing current spread may use the latest valid historical spread only
# when that observation is no more than five ticker trading observations old.
SPREAD_SHORT_GAP_MAX_TRADING_DAYS = 5

# For longer gaps, use the 90th percentile of valid, previously observed
# smoothed spreads in the trailing 252 ticker trading observations. This is a
# conservative, ticker-specific estimate and uses no future observations.
SPREAD_FALLBACK_LOOKBACK_TRADING_DAYS = 252
SPREAD_FALLBACK_QUANTILE = 0.90
SPREAD_FALLBACK_MIN_VALID_OBS = 20


def calculate_us_commission(
    trade_value,
    price,
):
    """
    IBKR Fixed commission for one U.S.-listed ETF/ETP order.

    shares:
        trade_value / price

    commission:
        min(
            max(
                0.005 * shares,
                1.00
            ),
            1% of trade value
        )
    """

    trade_value = abs(
        float(trade_value)
    )

    price = float(
        price
    )

    if trade_value == 0:
        return 0.0

    if price <= 0:
        raise ValueError(
            "Price must be positive."
        )

    shares = (
        trade_value
        / price
    )

    commission = min(
        max(
            US_COMMISSION_PER_SHARE
            * shares,
            US_MIN_COMMISSION,
        ),
        US_MAX_COMMISSION_RATE
        * trade_value,
    )

    return commission


def calculate_crypto_commission(
    trade_value,
    monthly_crypto_volume=0.0,
):
    """
    IBKR crypto commission for one order.
    """

    trade_value = abs(
        float(trade_value)
    )

    monthly_crypto_volume = float(
        monthly_crypto_volume
    )

    if trade_value == 0:
        return 0.0

    rate = next(
        rate
        for limit, rate
        in CRYPTO_TIERS
        if monthly_crypto_volume <= limit
    )

    commission = min(
        max(
            rate * trade_value,
            CRYPTO_MIN_COMMISSION,
        ),
        CRYPTO_MAX_COMMISSION_RATE
        * trade_value,
    )

    return commission


def _resolve_spread_asof(data, date, ticker):
    """Return a point-in-time spread and auditable fallback metadata.

    Policy
    ------
    1. Use the calculated S04 Spread when it is finite and non-negative.
    2. If missing/invalid and the most recent valid prior spread is at most
       five ticker trading observations old, carry that value forward.
    3. If the gap is longer, use the 90th percentile of valid S04 Spread
       observations in the preceding 252 ticker observations (excluding the
       current row). Require at least 20 historical observations.
    4. If neither fallback can be estimated, fail closed. Never use future
       observations and never replace missing spread with zero.

    The returned metadata is retained in fee_context and exported by S06.
    """
    asof = pd.Timestamp(date)
    history = data.loc[data["Date"] <= asof].copy().reset_index(drop=True)
    if history.empty or not history["Date"].eq(asof).any():
        raise ValueError(f"{ticker}: no OHLCV row available on {asof.date()}.")

    # Use the final row for the requested date, consistent with load_fee_context.
    current_pos = int(np.flatnonzero(history["Date"].eq(asof).to_numpy())[-1])
    current_raw = pd.to_numeric(
        pd.Series([history.loc[current_pos, "Spread"]]), errors="coerce"
    ).iloc[0]
    current_valid = bool(np.isfinite(current_raw) and current_raw >= 0)

    metadata = {
        "Spread_Original": float(current_raw) if np.isfinite(current_raw) else np.nan,
        "Spread_Fallback": False,
        "Spread_Fallback_Method": "observed" if current_valid else None,
        "Spread_Fallback_Age_Trading_Days": 0,
        "Spread_Reference_Date": asof,
        "Spread_Fallback_Quantile": np.nan,
        "Spread_Fallback_History_Count": 0,
        "Spread_Fallback_History_Start": pd.NaT,
        "Spread_Fallback_History_End": pd.NaT,
    }
    if current_valid:
        return float(current_raw), metadata

    spread_values = pd.to_numeric(history["Spread"], errors="coerce")
    spread_array = spread_values.to_numpy(dtype=float, na_value=np.nan)
    valid_mask = np.isfinite(spread_array) & (spread_array >= 0)
    # Exclude every row on the as-of date, including duplicate-date rows.
    valid_mask &= history["Date"].lt(asof).to_numpy()
    valid_positions = np.flatnonzero(valid_mask)

    if len(valid_positions) == 0:
        raise ValueError(
            f"{ticker}: Spread is missing on {asof.date()} and no valid prior spread exists."
        )

    last_valid_pos = int(valid_positions[-1])
    age = current_pos - last_valid_pos
    last_valid_value = float(spread_values.iloc[last_valid_pos])
    last_valid_date = pd.Timestamp(history.loc[last_valid_pos, "Date"])

    metadata.update({
        "Spread_Fallback": True,
        "Spread_Fallback_Age_Trading_Days": int(age),
        "Spread_Reference_Date": last_valid_date,
    })

    if age <= SPREAD_SHORT_GAP_MAX_TRADING_DAYS:
        metadata["Spread_Fallback_Method"] = "last_valid_historical"
        return last_valid_value, metadata

    # Long-gap fallback: trailing ticker observations strictly before as-of.
    start_pos = max(0, current_pos - SPREAD_FALLBACK_LOOKBACK_TRADING_DAYS)
    trailing = spread_values.iloc[start_pos:current_pos]
    trailing_array = trailing.to_numpy(dtype=float, na_value=np.nan)
    valid_trailing = trailing_array[np.isfinite(trailing_array) & (trailing_array >= 0)]

    if len(valid_trailing) < SPREAD_FALLBACK_MIN_VALID_OBS:
        raise ValueError(
            f"{ticker}: prolonged Spread gap on {asof.date()} ({age} trading observations); "
            f"only {len(valid_trailing)} valid spreads in the trailing "
            f"{SPREAD_FALLBACK_LOOKBACK_TRADING_DAYS} observations; "
            f"at least {SPREAD_FALLBACK_MIN_VALID_OBS} are required."
        )

    conservative_spread = float(np.quantile(valid_trailing, SPREAD_FALLBACK_QUANTILE))
    metadata.update({
        "Spread_Fallback_Method": "trailing_252_observation_p90",
        "Spread_Fallback_Quantile": SPREAD_FALLBACK_QUANTILE,
        "Spread_Fallback_History_Count": int(len(valid_trailing)),
        "Spread_Fallback_History_Start": pd.Timestamp(history.loc[start_pos, "Date"]),
        "Spread_Fallback_History_End": pd.Timestamp(history.loc[current_pos - 1, "Date"]),
    })
    return conservative_spread, metadata


def load_fee_context(
    date,
    tickers=None,
    price_column="Close",
):
    """
    Load the information required for one optimization date.

    This function should be called ONCE per rebalance.

    The optimizer can then call calculate_transaction_fees()
    many times using the same fee_context.

    Parameters
    ----------
    date:
        Current optimization/rebalance date.

    tickers:
        Assets included in the optimization.

        Their supplied order is preserved, which allows
        NumPy weight arrays to be used safely.

    price_column:
        Price used for the IBKR commission calculation.

        Default = Close because Close_t is known when
        optimization takes place after market close.

    Returns
    -------
    DataFrame indexed by Ticker with:

        Asset Class
        Price
        Spread
        ADV_Dollars
        Spread_Original
        Spread_Fallback
        Spread_Fallback_Method
        Spread_Fallback_Age_Trading_Days
        Spread_Reference_Date
        Spread_Fallback_Quantile
        Spread_Fallback_History_Count
        Spread_Fallback_History_Start
        Spread_Fallback_History_End

    Spread fallback is point-in-time and auditable. It does not modify the
    source CSV or the original S04 Spread column.
    """

    date = pd.Timestamp(
        date
    )

    universe = pd.read_csv(
        config.UNIVERSE_FILE
    )

    ticker_order = (
        list(tickers)
        if tickers is not None
        else None
    )

    if ticker_order is not None:

        missing_tickers = (
            set(ticker_order)
            - set(universe["Ticker"])
        )

        if missing_tickers:

            raise ValueError(
                "Tickers missing from universe: "
                f"{sorted(missing_tickers)}"
            )

        universe = universe[
            universe["Ticker"].isin(
                ticker_order
            )
        ]

    rows = []

    for _, asset in universe.iterrows():

        ticker = asset["Ticker"]

        asset_class = str(
            asset["Asset Class"]
        ).strip()

        # ---------------------------------------------------------------------
        # CASH PROXY
        # ---------------------------------------------------------------------

        if (
            asset_class.lower()
            == "cash proxy"
        ):

            rows.append({
                "Ticker": ticker,
                "Asset Class": asset_class,
                "Price": np.nan,
                "Spread": np.nan,
                "ADV_Dollars": np.nan,
                "Spread_Original": np.nan,
                "Spread_Fallback": False,
                "Spread_Fallback_Method": "not_applicable",
                "Spread_Fallback_Age_Trading_Days": 0,
                "Spread_Reference_Date": pd.NaT,
                "Spread_Fallback_Quantile": np.nan,
                "Spread_Fallback_History_Count": 0,
                "Spread_Fallback_History_Start": pd.NaT,
                "Spread_Fallback_History_End": pd.NaT,
            })

            continue

        # ---------------------------------------------------------------------
        # YAHOO ASSET
        # ---------------------------------------------------------------------

        file_path = (
            Path(YAHOO_INPUT_DIR)
            / f"{ticker}.csv"
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"Missing Yahoo file: "
                f"{file_path}"
            )

        data = pd.read_csv(
            file_path,
            parse_dates=["Date"],
        )
        data = (
            data.dropna(subset=["Date"])
            .sort_values("Date")
            .drop_duplicates("Date", keep="last")
            .reset_index(drop=True)
        )

        required_columns = {
            price_column,
            "Spread",
            "ADV_Dollars",
        }

        missing_columns = (
            required_columns
            - set(data.columns)
        )

        if missing_columns:

            raise ValueError(
                f"{ticker}: missing columns "
                f"{sorted(missing_columns)}. "
                "Run S04_add_liquidity_data.py first."
            )

        market_row = data.loc[
            data["Date"].eq(date)
        ]

        if market_row.empty:

            raise ValueError(
                f"{ticker}: no data "
                f"on {date.date()}."
            )

        market_row = (
            market_row.iloc[-1]
        )

        if asset_class.lower() == "crypto":
            # S04 deliberately does not estimate ETF-style spreads or ADV for
            # Bitcoin. The crypto commission model remains unchanged.
            spread_used = np.nan
            spread_meta = {
                "Spread_Original": np.nan,
                "Spread_Fallback": False,
                "Spread_Fallback_Method": "not_applicable_crypto",
                "Spread_Fallback_Age_Trading_Days": 0,
                "Spread_Reference_Date": pd.NaT,
                "Spread_Fallback_Quantile": np.nan,
                "Spread_Fallback_History_Count": 0,
                "Spread_Fallback_History_Start": pd.NaT,
                "Spread_Fallback_History_End": pd.NaT,
            }
        else:
            spread_used, spread_meta = _resolve_spread_asof(
                data,
                date,
                ticker,
            )

        rows.append({
            "Ticker": ticker,
            "Asset Class": asset_class,
            "Price": market_row[price_column],
            "Spread": spread_used,
            "ADV_Dollars": market_row["ADV_Dollars"],
            **spread_meta,
        })

    fee_context = pd.DataFrame(rows).set_index("Ticker")

    if ticker_order is not None:

        fee_context = (
            fee_context.reindex(
                ticker_order
            )
        )

    return fee_context


def _weights_to_series(
    weights,
    fee_context,
    name,
):
    """
    Convert portfolio weights into a ticker-indexed Series.

    Accepted formats:
        dict
        pandas Series
        list
        NumPy array

    Lists/arrays are assumed to follow fee_context.index order.
    """

    if isinstance(
        weights,
        pd.Series,
    ):

        return weights.astype(
            float
        )

    if isinstance(
        weights,
        dict,
    ):

        return pd.Series(
            weights,
            dtype=float,
        )

    values = np.asarray(
        weights,
        dtype=float,
    )

    if (
        values.ndim != 1
        or len(values) != len(fee_context)
    ):

        raise ValueError(
            f"{name} must contain one "
            "weight per fee_context ticker."
        )

    return pd.Series(
        values,
        index=fee_context.index,
        dtype=float,
    )


def calculate_transaction_fees(
    current_weights,
    target_weights,
    portfolio_value,
    fee_context,
    monthly_crypto_volume=0.0,
    include_commission=True,
    return_breakdown=False,
):
    """
    Calculate the transaction cost of ONE candidate portfolio.

    For asset i:

        trade_i =
            portfolio_value
            *
            (
                target_weight_i
                -
                current_weight_i
            )

    ETF/ETP spread cost:

        spread_cost_i =
            0.5
            *
            Spread_i
            *
            |trade_i|

    ETF/ETP market impact:

        impact_cost_i =
            0.1
            *
            trade_i^2
            /
            ADV_i

    DGS3MO:
        zero transaction cost

    BTC-USD:
        IBKR crypto commission only
    """

    current = _weights_to_series(
        current_weights,
        fee_context,
        "current_weights",
    )

    target = _weights_to_series(
        target_weights,
        fee_context,
        "target_weights",
    )

    tickers = current.index.union(
        target.index
    )

    current = current.reindex(
        tickers,
        fill_value=0.0,
    )

    target = target.reindex(
        tickers,
        fill_value=0.0,
    )

    if not np.isclose(
        current.sum(),
        1.0,
        atol=1e-6,
    ):

        raise ValueError(
            "Current weights must sum to 1."
        )

    if not np.isclose(
        target.sum(),
        1.0,
        atol=1e-6,
    ):

        raise ValueError(
            "Target weights must sum to 1."
        )

    portfolio_value = float(
        portfolio_value
    )

    if portfolio_value <= 0:

        raise ValueError(
            "Portfolio value must be positive."
        )

    # Convert changes in portfolio weights into dollar trades.
    trades = (
        portfolio_value
        *
        (
            target
            -
            current
        )
    )

    total_cost = 0.0

    running_crypto_volume = float(
        monthly_crypto_volume
    )

    breakdown_rows = []

    for ticker, signed_trade in trades.items():

        trade_value = abs(
            float(signed_trade)
        )

        if trade_value < 1e-12:
            continue

        if ticker not in fee_context.index:

            raise ValueError(
                f"{ticker} is missing "
                "from fee_context."
            )

        asset_class = str(
            fee_context.at[
                ticker,
                "Asset Class",
            ]
        ).strip().lower()

        spread_cost = 0.0
        market_impact_cost = 0.0
        commission = 0.0
        spread = np.nan

        # ---------------------------------------------------------------------
        # CASH
        # ---------------------------------------------------------------------

        if asset_class == "cash proxy":

            pass

        # ---------------------------------------------------------------------
        # BITCOIN
        # ---------------------------------------------------------------------

        elif asset_class == "crypto":

            if include_commission:

                commission = (
                    calculate_crypto_commission(
                        trade_value=
                            trade_value,

                        monthly_crypto_volume=
                            running_crypto_volume,
                    )
                )

            running_crypto_volume += (
                trade_value
            )

        # ---------------------------------------------------------------------
        # ETFs / ETPs
        # ---------------------------------------------------------------------

        else:

            spread = float(
                fee_context.at[
                    ticker,
                    "Spread",
                ]
            )

            adv = float(
                fee_context.at[
                    ticker,
                    "ADV_Dollars",
                ]
            )

            if (
                not np.isfinite(spread)
                or spread < 0
            ):

                raise ValueError(
                    f"{ticker}: invalid Spread."
                )

            if (
                not np.isfinite(adv)
                or adv <= 0
            ):

                raise ValueError(
                    f"{ticker}: invalid ADV_Dollars."
                )

            # Full spread -> one-sided execution cost = half spread.
            spread_cost = (
                0.5
                * spread
                * trade_value
            )

            # Market impact from the supplied paper.
            market_impact_cost = (
                MARKET_IMPACT_COEFFICIENT
                * trade_value**2
                / adv
            )

            if include_commission:

                price = float(
                    fee_context.at[
                        ticker,
                        "Price",
                    ]
                )

                if (
                    not np.isfinite(price)
                    or price <= 0
                ):

                    raise ValueError(
                        f"{ticker}: invalid Price."
                    )

                commission = (
                    calculate_us_commission(
                        trade_value=
                            trade_value,

                        price=
                            price,
                    )
                )

        asset_cost = (
            spread_cost
            +
            market_impact_cost
            +
            commission
        )

        total_cost += (
            asset_cost
        )

        if return_breakdown:

            breakdown_rows.append(
                {
                    "Ticker":
                        ticker,

                    "Trade_USD":
                        signed_trade,

                    "Spread_Cost":
                        spread_cost,

                    "Market_Impact_Cost":
                        market_impact_cost,

                    "IBKR_Commission":
                        commission,

                    "Spread_Used":
                        spread,

                    "Spread_Original":
                        fee_context.at[ticker, "Spread_Original"]
                        if "Spread_Original" in fee_context.columns else np.nan,

                    "Spread_Fallback":
                        bool(fee_context.at[ticker, "Spread_Fallback"])
                        if "Spread_Fallback" in fee_context.columns else False,

                    "Spread_Fallback_Method":
                        fee_context.at[ticker, "Spread_Fallback_Method"]
                        if "Spread_Fallback_Method" in fee_context.columns else "unknown",

                    "Spread_Fallback_Age_Trading_Days":
                        fee_context.at[ticker, "Spread_Fallback_Age_Trading_Days"]
                        if "Spread_Fallback_Age_Trading_Days" in fee_context.columns else 0,

                    "Spread_Reference_Date":
                        fee_context.at[ticker, "Spread_Reference_Date"]
                        if "Spread_Reference_Date" in fee_context.columns else pd.NaT,

                    "Spread_Fallback_Quantile":
                        fee_context.at[ticker, "Spread_Fallback_Quantile"]
                        if "Spread_Fallback_Quantile" in fee_context.columns else np.nan,

                    "Total_Cost":
                        asset_cost,
                }
            )

    if return_breakdown:

        return (
            total_cost,
            pd.DataFrame(
                breakdown_rows
            ),
        )

    return total_cost


def transaction_cost_rate(
    current_weights,
    target_weights,
    portfolio_value,
    fee_context,
    monthly_crypto_volume=0.0,
    include_commission=True,
):
    """
    Return transaction cost as a fraction of portfolio value.

    This is usually the more useful form inside an optimizer.
    """

    total_cost = (
        calculate_transaction_fees(
            current_weights=
                current_weights,

            target_weights=
                target_weights,

            portfolio_value=
                portfolio_value,

            fee_context=
                fee_context,

            monthly_crypto_volume=
                monthly_crypto_volume,

            include_commission=
                include_commission,
        )
    )

    return (
        total_cost
        / float(portfolio_value)
    )
