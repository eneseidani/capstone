# Import libraries
from pathlib import Path

# File paths
UNIVERSE_FILE = Path("Investable_Universe.csv")
RAW_DATA_DIR = Path("raw_data")
INPUT_DATA_DIR = Path("input_data")

# Analysis start date
start_date="2008-01-01"

# FRED API key
fred_api_key="71a6dd410a9a04c82bf76c0ee1981893"

# Fama French files and URLs
FAMA_FRENCH_FILES = {
    "Fama-French_5_Factors_Daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",

    "Momentum_Factor_Daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
}

# FRED data
FRED_SERIES = {
    # Interest rates
    "DFF": "Federal Funds Effective Rate",
    "DGS3MO": "3-Month Treasury Yield",
    "DGS2": "2-Year Treasury Yield",
    "DGS10": "10-Year Treasury Yield",
    "DGS30": "30-Year Treasury Yield",

    # Yield curve
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "T10Y3M": "10Y-3M Treasury Spread",

    # Term premium
    "THREEFYTP2": "2-Year Zero-Coupon Bond Term Premium",
    "THREEFYTP5": "5-Year Zero-Coupon Bond Term Premium",
    "THREEFYTP10": "10-Year Zero-Coupon Bond Term Premium",

    # Macro
    "CPIAUCSL": "CPI",
    "T5YIFR": "5-Year Forward Inflation Expectation",
    "T5YIE": "5-Year Breakeven Inflation Rate ",
    "T10YIE": "10-Year Breakeven Inflation Rate ",
    "UNRATE": "Unemployment Rate",
    "INDPRO": "Industrial Production",
    "MTSDS133FMS": "Federal Surplus or Deficit",
    "MTSO133FMS": "Federal Outlays",
    "MVGFD027MNFRBDAL": "Market Value of Gross Federal Debt",

    # Volatility
    "VIXCLS": "VIX",

    # Corporate credit
    "BAA10Y": "Moody's Baa Corporate Spread",
    "AAA10Y": "Moody's Aaa Corporate Spread",

    # FX
    "DTWEXBGS": "Nominal Broad Dollar Index",

    # Commodity
    "DCOILWTICO": "WTI Crude Oil",
}