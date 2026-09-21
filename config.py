# Import libraries
from pathlib import Path

# Analysis start date
START_DATE = "2008-01-01"

# File paths
UNIVERSE_FILE = Path("Investable_Universe.csv")
RAW_DATA_DIR = Path("raw_data")
INPUT_DATA_DIR = Path("input_data")

# Portfolio rebalancing frequency
REBALANCE_FREQUENCY = 60

# Forward prediction horizon
FORWARD_HORIZON = 60

# Rolling windows
MOMENTUM_WINDOWS = [5, 20, 60, 120, 252]
TREND_WINDOWS = [20, 50, 200]
MEAN_REVERSION_WINDOWS = [20, 60]
VOLATILITY_WINDOWS = [20, 60, 120, 252]
DRAWDOWN_WINDOWS = [120, 252]
VOLUME_WINDOWS = [20, 60]
MACRO_WINDOWS = [20, 60, 252]
CHANGE_WINDOWS = [20, 60]

# FRED API key
fred_api_key="71a6dd410a9a04c82bf76c0ee1981893"

# Fama French files and URLs
FAMA_FRENCH_FILES = {
    "Fama-French_5_Factors_Daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",

    "Momentum_Factor_Daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
}

# FRED features
FRED_SERIES = {
    # Interest rates (daily)
    "INTEREST_RATE_FEATURES": {
        "DFF": "Federal Funds Effective Rate", # published on T+1 afternoon
        "DGS3MO": "3-Month Treasury Yield", # published on T+1 afternoon
        "DGS2": "2-Year Treasury Yield", # published on T+1 afternoon
        "DGS10": "10-Year Treasury Yield", # published on T+1 afternoon
        "DGS30": "30-Year Treasury Yield", # published on T+1 afternoon
        },

    # Yield curve (daily)
    "YIELD_CURVE_FEATURES": {
        "T10Y2Y": "10Y-2Y Treasury Spread", # published on T afternoon
        "T10Y3M": "10Y-3M Treasury Spread", # published on T afternoon
    },

    # Term premium (daily)
    "TERM_PREMIUM_FEATURES": {
        "THREEFYTP2": "2-Year Zero-Coupon Bond Term Premium", # published on T+2 afternoon (weekly on Tuesday)
        "THREEFYTP5": "5-Year Zero-Coupon Bond Term Premium", # published on T+2 afternoon (weekly on Tuesday)
        "THREEFYTP10": "10-Year Zero-Coupon Bond Term Premium", # published on T+2 afternoon (weekly on Tuesday)
    },

    # Inflation (daily)
    "INFLATION_FEATURES": {
        "T5YIFR": "5-Year Forward Inflation Expectation", # published on T afternoon
        "T5YIE": "5-Year Breakeven Inflation Rate ", # published on T afternoon
        "T10YIE": "10-Year Breakeven Inflation Rate ", # published on T afternoon
    },

    # Macro (monthly)
    "MONTHLY_FEATURES": {
        "CPIAUCSL": "CPI", # published in T+1 mid-month
        "UNRATE": "Unemployment Rate", # published in T+1 early-month
        "INDPRO": "Industrial Production", # published in T+1 mid-month
        "MTSDS133FMS": "Federal Surplus or Deficit", # published in T+1 mid-month
        "MTSO133FMS": "Federal Outlays", # published in T+1 mid-month
        "MVGFD027MNFRBDAL": "Market Value of Gross Federal Debt", # published in T+1 mid-month
    },

    # Volatility (daily)
    "VOLATILITY_FEATURES": {
        "VIXCLS": "VIX Index", # published on T+1 morning
        "VXDCLS": "DJIA Volatility Index", # published on T+1 morning
    },

    # Corporate credit (daily)
    "CORP_CREDIT_FEATURES": {
        "BAA10Y": "Moody's Baa Corporate Spread", # published on T+1 afternoon
        "AAA10Y": "Moody's Aaa Corporate Spread", # published on T+1 afternoon
    }
}

# YAHOO features
YAHOO_SERIES = {
    # FX (daily)
    "DX-Y.NYB": "US Dollar Index Futures", # published on T afternoon

    # Commodity (daily)
    "CL=F": "WTI Crude Oil Futures", # published on T afternoon
}

# Conservative publication lags (measured in trading day observations)
INTEREST_RATE_LAG = 1
CORP_CREDIT_LAG = 1
TERM_PREMIUM_LAG = 6
MONTHLY_FEATURE_LAG = 15