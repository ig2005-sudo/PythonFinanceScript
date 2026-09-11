import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter
import pandas as pd
import numpy as np
import yfinance as yf


# ---------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------

TICKERS = ["UBI.PA", "MC.PA", "SAN.PA", "TTE.PA"]

NAMES = {
    "UBI.PA": "Ubisoft",
    "MC.PA": "LVMH",
    "SAN.PA": "Sanofi",
    "TTE.PA": "TotalEnergies"
}

# Equal-weight portfolio.
# Change these if your real portfolio uses different weights.
WEIGHTS = {
    "UBI.PA": 0.25,
    "MC.PA": 0.25,
    "SAN.PA": 0.25,
    "TTE.PA": 0.25
}

# Assumption used for the Sharpe ratio.
# Replace this with the rate required by your assignment if necessary.
RISK_FREE_RATE = 0.00

TRADING_DAYS = 252


# ---------------------------------------------------------
# 2. STYLE
# ---------------------------------------------------------

BG_COLOR = "#1F2630"
GRAPH_COLOR = "#252E3B"
TEXT_COLOR = "#E0E0E0"
MUTED_TEXT = "#AAB2BD"
GRID_COLOR = "#374151"

COLORS = {
    "UBI.PA": "#FF5252",
    "MC.PA": "#FFB74D",
    "SAN.PA": "#4FC3F7",
    "TTE.PA": "#69F0AE"
}

plt.style.use("dark_background")


# ---------------------------------------------------------
# 3. DOWNLOAD DATA
# ---------------------------------------------------------

print(f"Downloading data for: {', '.join(TICKERS)}")

raw_data = yf.download(
    TICKERS,
    period="1y",
    interval="1d",
    auto_adjust=True,
    progress=False
)

if raw_data.empty:
    raise RuntimeError("No market data was returned by Yahoo Finance.")

prices = raw_data["Close"].copy()

# Put columns in the expected order.
prices = prices.reindex(columns=TICKERS)

# Check that every ticker actually returned data.
missing = [
    ticker
    for ticker in TICKERS
    if ticker not in prices.columns or prices[ticker].dropna().empty
]

if missing:
    raise RuntimeError(
        f"No usable price data returned for: {', '.join(missing)}"
    )

# All four stocks trade in Paris, so using common trading dates is appropriate.
prices = prices.dropna(how="any")

if len(prices) < 2:
    raise RuntimeError("Not enough price observations to calculate returns.")


# ---------------------------------------------------------
# 4. CALCULATE ASSET METRICS
# ---------------------------------------------------------

daily_returns = prices.pct_change().dropna()

# Actual length of the sample in years.
years = (prices.index[-1] - prices.index[0]).days / 365.25

if years <= 0:
    raise RuntimeError("Invalid date range in downloaded data.")

# CAGR / annualized return
annualized_returns = (
    (prices.iloc[-1] / prices.iloc[0]) ** (1 / years)
) - 1

# Annualized volatility
annualized_volatility = (
    daily_returns.std() * np.sqrt(TRADING_DAYS)
)

summary = pd.DataFrame({
    "Name": [NAMES[ticker] for ticker in TICKERS],
    "Return": annualized_returns.reindex(TICKERS),
    "Volatility": annualized_volatility.reindex(TICKERS)
}, index=TICKERS)


# ---------------------------------------------------------
# 5. PORTFOLIO METRICS
# ---------------------------------------------------------

weights = pd.Series(WEIGHTS).reindex(TICKERS)

if not np.isclose(weights.sum(), 1.0):
    raise ValueError("Portfolio weights must add up to 1.")

# Start every stock at 100.
normalized_prices = prices.div(prices.iloc[0]).mul(100)

# Portfolio beginning with the weights specified above.
# This represents a buy-and-hold portfolio: weights are set initially
# and are then allowed to drift as prices change.
portfolio_index = normalized_prices.mul(weights, axis=1).sum(axis=1)

portfolio_daily_returns = portfolio_index.pct_change().dropna()

portfolio_return = (
    (portfolio_index.iloc[-1] / portfolio_index.iloc[0]) ** (1 / years)
) - 1

portfolio_volatility = (
    portfolio_daily_returns.std() * np.sqrt(TRADING_DAYS)
)

if portfolio_volatility > 0:
    sharpe_ratio = (
        portfolio_return - RISK_FREE_RATE
    ) / portfolio_volatility
else:
    sharpe_ratio = np.nan


# ---------------------------------------------------------
# 6. FIGURE LAYOUT
# ---------------------------------------------------------

fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor(BG_COLOR)

gs = gridspec.GridSpec(
    2,
    3,
    width_ratios=[1, 1, 0.82],
    hspace=0.30,
    wspace=0.25
)

ax_scatter = fig.add_subplot(gs[0, :2])
ax_lines = fig.add_subplot(gs[1, :2])
ax_sidebar = fig.add_subplot(gs[:, 2])

for ax in (ax_scatter, ax_lines):
    ax.set_facecolor(GRAPH_COLOR)

    ax.grid(
        True,
        linestyle=":",
        color=GRID_COLOR,
        alpha=0.6
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.spines["left"].set_color(GRID_COLOR)

    ax.tick_params(colors=TEXT_COLOR)

ax_sidebar.set_facecolor(BG_COLOR)
ax_sidebar.axis("off")


# ---------------------------------------------------------
# 7. RISK / RETURN CHART
# ---------------------------------------------------------

for ticker, row in summary.iterrows():

    ax_scatter.scatter(
        row["Volatility"],
        row["Return"],
        color=COLORS[ticker],
        s=130 if ticker == "UBI.PA" else 90,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.6
    )

    ax_scatter.annotate(
        f"{ticker}\n{row['Return']:.1%}",
        (row["Volatility"], row["Return"]),
        xytext=(8, 3),
        textcoords="offset points",
        fontsize=9,
        color=TEXT_COLOR
    )

ax_scatter.axhline(
    0,
    color=MUTED_TEXT,
    linewidth=1,
    linestyle="--"
)

ax_scatter.set_title(
    "Risk and Return",
    fontsize=14,
    fontweight="bold",
    color=TEXT_COLOR,
    loc="left"
)

ax_scatter.set_xlabel(
    "Annualized Volatility",
    fontsize=10,
    color=TEXT_COLOR
)

ax_scatter.set_ylabel(
    "Annualized Return",
    fontsize=10,
    color=TEXT_COLOR
)

ax_scatter.xaxis.set_major_formatter(PercentFormatter(1))
ax_scatter.yaxis.set_major_formatter(PercentFormatter(1))

# Give the highest-volatility observation some breathing room.
max_vol = summary["Volatility"].max()
ax_scatter.set_xlim(0, max_vol * 1.15)


# ---------------------------------------------------------
# 8. HISTORICAL PERFORMANCE
# ---------------------------------------------------------

for ticker in TICKERS:

    linewidth = 1.8 if ticker == "UBI.PA" else 1.3

    ax_lines.plot(
        normalized_prices.index,
        normalized_prices[ticker],
        label=ticker,
        color=COLORS[ticker],
        linewidth=linewidth
    )

ax_lines.axhline(
    100,
    color=MUTED_TEXT,
    linestyle="--",
    linewidth=1
)

ax_lines.set_title(
    "Price Performance — Base 100",
    fontsize=14,
    fontweight="bold",
    color=TEXT_COLOR,
    loc="left"
)

ax_lines.set_ylabel(
    "Indexed Value",
    fontsize=10,
    color=TEXT_COLOR
)

legend = ax_lines.legend(
    loc="upper left",
    ncol=4,
    fontsize=9,
    frameon=False
)

for text in legend.get_texts():
    text.set_color(TEXT_COLOR)


# ---------------------------------------------------------
# 9. SIDEBAR
# ---------------------------------------------------------

x = 0.05
y = 0.96

ax_sidebar.text(
    x,
    y,
    "PORTFOLIO SUMMARY",
    transform=ax_sidebar.transAxes,
    fontsize=17,
    fontweight="bold",
    color=TEXT_COLOR
)

y -= 0.07

ax_sidebar.text(
    x,
    y,
    f"Annualized return     {portfolio_return:>8.1%}",
    transform=ax_sidebar.transAxes,
    fontsize=11,
    family="monospace",
    color=TEXT_COLOR
)

y -= 0.045

ax_sidebar.text(
    x,
    y,
    f"Annualized volatility {portfolio_volatility:>8.1%}",
    transform=ax_sidebar.transAxes,
    fontsize=11,
    family="monospace",
    color=TEXT_COLOR
)

y -= 0.045

ax_sidebar.text(
    x,
    y,
    f"Sharpe ratio          {sharpe_ratio:>8.2f}",
    transform=ax_sidebar.transAxes,
    fontsize=11,
    family="monospace",
    color=TEXT_COLOR
)

y -= 0.035

ax_sidebar.text(
    x,
    y,
    f"Risk-free rate: {RISK_FREE_RATE:.1%}",
    transform=ax_sidebar.transAxes,
    fontsize=9,
    color=MUTED_TEXT
)


# ---------------------------------------------------------
# INDIVIDUAL STOCKS
# ---------------------------------------------------------

y -= 0.09

ax_sidebar.text(
    x,
    y,
    "INDIVIDUAL STOCKS",
    transform=ax_sidebar.transAxes,
    fontsize=12,
    fontweight="bold",
    color=TEXT_COLOR
)

y -= 0.05

ax_sidebar.text(
    x,
    y,
    f"{'Ticker':<9}{'Return':>9}{'Vol.':>9}",
    transform=ax_sidebar.transAxes,
    fontsize=10,
    family="monospace",
    color=MUTED_TEXT
)

y -= 0.035

for ticker, row in summary.iterrows():

    ax_sidebar.text(
        x,
        y,
        f"{ticker:<9}{row['Return']:>8.1%}{row['Volatility']:>9.1%}",
        transform=ax_sidebar.transAxes,
        fontsize=10,
        family="monospace",
        color=COLORS[ticker]
    )

    y -= 0.045


# ---------------------------------------------------------
# 10. SIMPLE OBSERVATIONS
# ---------------------------------------------------------

best = summary["Return"].idxmax()
worst = summary["Return"].idxmin()
riskiest = summary["Volatility"].idxmax()

y -= 0.06

ax_sidebar.text(
    x,
    y,
    "OBSERVATIONS",
    transform=ax_sidebar.transAxes,
    fontsize=12,
    fontweight="bold",
    color=TEXT_COLOR
)

y -= 0.055

observations = [
    (
        "Best return",
        f"{best} ({summary.loc[best, 'Return']:.1%})"
    ),
    (
        "Worst return",
        f"{worst} ({summary.loc[worst, 'Return']:.1%})"
    ),
    (
        "Highest volatility",
        f"{riskiest} ({summary.loc[riskiest, 'Volatility']:.1%})"
    )
]

for label, value in observations:

    ax_sidebar.text(
        x,
        y,
        label,
        transform=ax_sidebar.transAxes,
        fontsize=9,
        color=MUTED_TEXT
    )

    ax_sidebar.text(
        x,
        y - 0.025,
        value,
        transform=ax_sidebar.transAxes,
        fontsize=10,
        color=TEXT_COLOR
    )

    y -= 0.075


# ---------------------------------------------------------
# 11. FINAL DISPLAY
# ---------------------------------------------------------

fig.suptitle(
    "One-Year Portfolio Overview",
    x=0.07,
    y=0.98,
    ha="left",
    fontsize=19,
    fontweight="bold",
    color=TEXT_COLOR
)

plt.show()
