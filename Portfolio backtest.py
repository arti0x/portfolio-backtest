##############
#To use this backtest firstly look for tickers on yahoo finance.
#Once you got them you can start using the backtest.
##############


#Importing all necessary libraries
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter import font as tkfont
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib as mpl
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import skew as stat_skew, kurtosis as stat_kurtosis

TRADING_DAYS = 252 

# Two palettes (Dark/Light). Colors can be changed by the user, during run time we can change the theme
THEMES = {
    'Dark': {
        'bg': '#0c111a',
        'panel': '#121826',
        'chart_bg': '#0f1723',
        'grid': '#2a3442',
        'text': '#0df',  # Will override with proper text after style init (placeholder)
        'text': '#e6e8ee',
        'muted': '#9aa6b2',
        'accent': '#4f46e5',
        'accent_hover': '#4338ca',
        'btn_fg': '#ffffff',
        'pos': '#22c55e',
        'neg': '#ef4444',
        'line_port': '#10b981',
        'line_bench': '#f59e0b',
        'drawdown_fill': '#e11d48',
        'selection_bg': '#1f2937',
        'selection_fg': '#e6e8ee',
    },
    'Light': {
        'bg': '#f5f7fb',
        'panel': '#ffffff',
        'chart_bg': '#ffffff',
        'grid': '#d2d6dc',
        'text': '#111827',
        'muted': '#6b7280',
        'accent': '#2563eb',
        'accent_hover': '#1e40af',
        'btn_fg': '#ffffff',
        'pos': '#16a34a',
        'neg': '#dc2626',
        'line_port': '#2563eb',
        'line_bench': '#f59e0b',
        'drawdown_fill': '#dc2626',
        'selection_bg': '#e5e7eb',
        'selection_fg': '#111827',
    }
}

# Default
COLORS = THEMES['Dark'].copy()

# Metric documentation and explanation for double-click info
METRIC_DOCS = {
    'Total Return': "Total Return = (Final Value / Initial Capital) - 1.\n\nAggregate gain or loss over the period.",
    'CAGR': "CAGR = (1 + Total Return)^(252 / Ndays) - 1.\n\nCAGR stands for Compound Annual Growth Rate.\nThe steady annual growth rate that would lead to the same final value.",
    'Volatility (ann)': "Annualized Volatility = stdev(daily returns) * sqrt(252).\n\nDispersion of returns.\nIs a statistical measure of the returns for a given security or market index over time.\nIt is often measured from either the standard deviation or variance between those returns. In most cases, the higher the volatility, the riskier the security.",
    'Sharpe (ann)': "Sharpe = (Annualized Return - Risk-Free Rate) / Annualized Volatility.\n\nExcess return per unit of risk.\nThe Sharpe ratio shows whether a portfolio's excess returns are attributable to smart investment decisions or luck and risk.\nA measure of an investment's risk-adjusted performance, calculated by comparing its return to that of a risk-free asset",
    'Sortino (ann)': "Sortino = (Annualized Return - Risk-Free Rate) / Downside Volatility.\n\nA risk-adjusted measure of portfolio performance that only considers the standard deviation of the downside risk.\nThe Sortino ratio can help investors and analysts evaluate an investment's return for a degree of bad risk.\nPenalizes downside only.",
    'Max Drawdown': "Max Drawdown = min(Equity / Rolling Max - 1).\n\nWorst peak-to-trough decline.",
    'Calmar': "Calmar = CAGR / |Max Drawdown|.\n\nIt is a function of the fund's average compounded annual rate of return versus its maximum drawdown. The higher the Calmar ratio, the better it performed on a risk-adjusted basis during the given time frame, which is mostly commonly set at 36 months.\nGrowth relative to drawdowns.",
    'Daily VaR 95%': "Historical VaR(95%) = 5th percentile of daily returns.\n\nLoss threshold exceeded 5% of the time.",
    'Daily CVaR 95%': "Historical CVaR(95%) = average of daily returns at or below the 95% VaR (expected shortfall).",
    'Daily VaR 99%': "Historical VaR(99%) = 1st percentile of daily returns.",
    'Daily CVaR 99%': "Historical CVaR(99%) = average of daily returns at or below the 99% VaR.",
    'Hit Ratio': "Hit Ratio = fraction of days with positive returns.",
    'Best Day': "Best Day = maximum daily return.",
    'Worst Day': "Worst Day = minimum daily return.",
    'Skew': "Skewness of daily returns.\n\nSkewness is a measure of symmetry, or more precisely, the lack of symmetry. A distribution, or data set, is symmetric if it looks the same to the left and right of the center point.\nPositive skew: fatter right tail; negative: fatter left tail.",
    'Kurtosis (excess)': "Excess kurtosis of daily returns relative to normal distribution.\nKurtosis is a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution. That is, data sets with high kurtosis tend to have heavy tails, or outliers. Data sets with low kurtosis tend to have light tails, or lack of outliers. A uniform distribution would be the extreme case.",
    'Beta vs Benchmark': "Beta = Cov(Rp,Rb)/Var(Rb).\n\nRp and Rb are the portfolio returns and the benchmark returns\nSensitivity to the benchmark.",
    'Alpha (ann) vs Benchmark': "Alpha = Annualized Return - [Rf + Beta*(Annualized Benchmark Return - Rf)].\n\nAlpha iss a term used in investing to describe an investment strategy's ability to beat the market.\nIs often referred to as excess return or the abnormal rate of return in relation to the benchmark.",
    'Correlation vs Benchmark': "Correlation of daily returns with the benchmark.",
    'R^2 vs Benchmark': "R-squared = Correlation^2.\n\nVariance explained by the benchmark.",
    'Initial Capital': "Starting portfolio value used for the backtest.",
    'Final Value': "Ending portfolio value over the backtest horizon.",
}

def parse_float_list(s):
    if not s.strip():
        return None
    try:
        vals = [float(x.strip()) for x in s.split(',') if x.strip() != ""]
        return vals
    except Exception:
        return None

def parse_ticker_list(s):
    return [t.strip().upper() for t in s.split(',') if t.strip()]

def annualize_return(daily_mean):
    return daily_mean * TRADING_DAYS

def annualize_vol(daily_std):
    return daily_std * np.sqrt(TRADING_DAYS)

def compute_drawdown(value_series: pd.Series):
    roll_max = value_series.cummax()
    dd = value_series / roll_max - 1.0
    max_dd = dd.min()
    max_dd_duration = (dd < 0).astype(int).groupby((dd >= 0).astype(int).cumsum()).cumcount().max()
    return dd, max_dd, max_dd_duration

def monthly_return_table(value_series: pd.Series):
    m = value_series.resample('M').last().pct_change().dropna()
    df = m.to_frame('Return')
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    df['Month'] = df['Month'].map(month_names)
    pivot = df.pivot(index='Year', columns='Month', values='Return')
    pivot = pivot.reindex(columns=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    return pivot
# Starting values for backtest
class PortfolioBacktester:
    def __init__(self, tickers, weights=None, start=None, end=None, initial_capital=10000.0,
                 rebalance='Monthly', rebalance_n=None, benchmark=None, risk_free=0.02,
                 tx_cost_bps=0.0, slippage_bps=0.0):
        self.tickers = tickers
        self.weights = None if weights is None else np.array(weights, dtype=float)
        self.start = start
        self.end = end
        self.initial_capital = float(initial_capital)
        self.rebalance = rebalance  # 'None', 'Monthly', 'Quarterly', 'Yearly', 'Every N days'
        self.rebalance_n = int(rebalance_n) if rebalance_n not in (None, "", "0") else None
        self.benchmark = benchmark
        self.risk_free = float(risk_free)
        self.tx_cost_rate = (float(tx_cost_bps) + float(slippage_bps)) / 10000.0
        
        # Weights or autoadjust to equal
        if self.weights is not None:
            if len(self.weights) != len(self.tickers):
                raise ValueError("Weights count must match number of tickers.")
            total = self.weights.sum()
            if total <= 0:
                raise ValueError("Weights must sum to > 0.")
            self.weights = self.weights / total  # normalize even if not exactly 1
        else:
            self.weights = np.ones(len(self.tickers)) / len(self.tickers)
    # Downloading historical data
    def _download_prices(self, tickers):
        data = yf.download(tickers, start=self.start, end=self.end, progress=False, auto_adjust=False)
        try:
            prices = data['Adj Close']
        except Exception:
            prices = data['Close']
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        prices = prices.sort_index().ffill().dropna(how='any')
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices.index = pd.to_datetime(prices.index)
        return prices
    
    def _rebalance_flags(self, index, freq): #rebalancing
        n = len(index)
        flags = np.zeros(n, dtype=bool)
        if n == 0:
            return flags
        flags[0] = True
        s = pd.Series(index=index, data=index)
        if not isinstance(index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)
            s[:] = pd.to_datetime(s)
        if freq is None or freq == 'None':
            return flags
        if freq == 'Monthly':
            period = s.dt.to_period('M')
            changes = period != period.shift(1)
            return changes.fillna(True).values
        elif freq == 'Quarterly':
            period = s.dt.to_period('Q')
            changes = period != period.shift(1)
            return changes.fillna(True).values
        elif freq == 'Yearly':
            period = s.dt.to_period('Y')
            changes = period != period.shift(1)
            return changes.fillna(True).values
        elif freq == 'Every N days' and self.rebalance_n and self.rebalance_n > 0:
            idx = np.arange(n)
            flags[:] = (idx % self.rebalance_n == 0)
            return flags
        else:
            return flags

    def _compute_portfolio_path(self, prices: pd.DataFrame):
        idx = prices.index
        cols = list(prices.columns)
        W = self.weights.copy()
        rebal_flags = self._rebalance_flags(idx, self.rebalance)
        shares = pd.DataFrame(index=idx, columns=cols, dtype=float)
        tx_costs = pd.Series(0.0, index=idx, name='Tx Costs')
        turnover = pd.Series(0.0, index=idx, name='Turnover')

        # Initial allocation including costs
        initial_cost = self.initial_capital * self.tx_cost_rate
        capital_net = max(0.0, self.initial_capital - initial_cost)
        shares.iloc[0, :] = (capital_net * W) / prices.iloc[0, :].values
        tx_costs.iloc[0] = initial_cost
        turnover.iloc[0] = 1.0  # all capital used

        for i in range(1, len(idx)):
            prev_val = float((shares.iloc[i-1, :].values * prices.iloc[i, :].values).sum())
            if rebal_flags[i]:
                # Desired shares before cost
                target_shares_pre = (prev_val * W) / prices.iloc[i, :].values
                # Turnover value in $
                tv = np.abs(target_shares_pre - shares.iloc[i-1, :].values) * prices.iloc[i, :].values
                tv_sum = float(tv.sum())
                cost = tv_sum * self.tx_cost_rate
                net_val = max(0.0, prev_val - cost)
                shares.iloc[i, :] = (net_val * W) / prices.iloc[i, :].values
                tx_costs.iloc[i] = cost
                turnover.iloc[i] = tv_sum / prev_val if prev_val > 0 else 0.0
            else:
                shares.iloc[i, :] = shares.iloc[i-1, :].values
                tx_costs.iloc[i] = 0.0
                turnover.iloc[i] = 0.0

        # portfolio
        value = (shares * prices).sum(axis=1)
        weights_over_time = (shares * prices).div(value, axis=0)
        return value, shares, weights_over_time, tx_costs, turnover

    def run(self):
        prices = self._download_prices(self.tickers)
        if prices.empty:
            raise ValueError("No price data returned. Check tickers and dates.")
        port_value, shares, weights_ot, tx_costs, turnover = self._compute_portfolio_path(prices)
        port_rets = port_value.pct_change().dropna()

        #Benchmark data
        bench_prices = bench_rets = None
        if self.benchmark and self.benchmark.strip():
            bench_prices = self._download_prices([self.benchmark]).iloc[:, 0]
            bench_prices = bench_prices.reindex(port_value.index).dropna()
            common_idx = port_value.index.intersection(bench_prices.index)
            port_value = port_value.reindex(common_idx)
            port_rets = port_value.pct_change().dropna()
            bench_prices = bench_prices.reindex(common_idx)
            bench_rets = bench_prices.pct_change().dropna()
            tx_costs = tx_costs.reindex(common_idx).fillna(0.0)
            turnover = turnover.reindex(common_idx).fillna(0.0)

        metrics = self._compute_metrics(port_value, port_rets, bench_rets)
        contrib = self._compute_contributions(prices, shares, port_value.iloc[0], port_value.iloc[-1])
        asset_rets = prices.pct_change().dropna()
        corr = asset_rets.corr() if asset_rets.shape[1] > 1 else None
        monthly_tbl = monthly_return_table(port_value)

        result = {
            'prices': prices,
            'portfolio_value': port_value,
            'portfolio_returns': port_rets,
            'weights_over_time': weights_ot,
            'metrics': metrics,
            'contributions': contrib,
            'correlation': corr,
            'monthly_table': monthly_tbl,
            'benchmark_prices': bench_prices,
            'benchmark_returns': bench_rets,
            'tx_costs': tx_costs,
            'turnover': turnover,
        }
        return result

#Metrics
    def _compute_metrics(self, value: pd.Series, returns: pd.Series, bench_rets: pd.Series | None):
        rf = float(self.risk_free)
        total_return = value.iloc[-1] / value.iloc[0] - 1.0
        n_days = max(1, len(value) - 1)
        cagr = (1.0 + total_return) ** (TRADING_DAYS / n_days) - 1.0

        mu_daily = returns.mean()
        sd_daily = returns.std(ddof=0)
        mu_ann = annualize_return(mu_daily)
        vol_ann = annualize_vol(sd_daily)
        sharpe = (mu_ann - rf) / vol_ann if vol_ann != 0 else np.nan

        downside = returns[returns < 0]
        ds_std_daily = downside.std(ddof=0)
        ds_std_ann = annualize_vol(ds_std_daily)
        sortino = (mu_ann - rf) / ds_std_ann if ds_std_ann != 0 else np.nan

        _, max_dd, _ = compute_drawdown(value)
        calmar = (cagr / abs(max_dd)) if max_dd != 0 else np.nan

        if len(returns) > 0:
            var95 = np.percentile(returns, 5)
            cvar95 = returns[returns <= var95].mean() if (returns <= var95).any() else np.nan
            var99 = np.percentile(returns, 1)
            cvar99 = returns[returns <= var99].mean() if (returns <= var99).any() else np.nan
        else:
            var95 = cvar95 = var99 = cvar99 = np.nan

        hit_ratio = (returns > 0).mean() if len(returns) > 0 else np.nan
        best_day = returns.max() if len(returns) > 0 else np.nan
        worst_day = returns.min() if len(returns) > 0 else np.nan

        skewv = stat_skew(returns, bias=False) if len(returns) > 2 else np.nan
        kurtv = stat_kurtosis(returns, fisher=True, bias=False) if len(returns) > 3 else np.nan

        beta = alpha_ann = r2 = corr = np.nan
        if bench_rets is not None and len(bench_rets) > 10:
            df = pd.DataFrame({'p': returns, 'b': bench_rets}).dropna()
            if len(df) > 10:
                cov = np.cov(df['p'], df['b'])[0, 1]
                var_b = np.var(df['b'])
                beta = cov / var_b if var_b != 0 else np.nan
                mu_b_ann = annualize_return(df['b'].mean())
                corr = df['p'].corr(df['b'])
                r2 = corr ** 2 if pd.notna(corr) else np.nan
                alpha_ann = mu_ann - (rf + beta * (mu_b_ann - rf)) if pd.notna(beta) else np.nan

        metrics = {
            'Start': value.index[0].strftime('%Y-%m-%d'),
            'End': value.index[-1].strftime('%Y-%m-%d'),
            'Initial Capital': value.iloc[0],
            'Final Value': value.iloc[-1],
            'Total Return': total_return,
            'CAGR': cagr,
            'Volatility (ann)': vol_ann,
            'Sharpe (ann)': sharpe,
            'Sortino (ann)': sortino,
            'Max Drawdown': max_dd,
            'Calmar': calmar,
            'Daily VaR 95%': var95,
            'Daily CVaR 95%': cvar95,
            'Daily VaR 99%': var99,
            'Daily CVaR 99%': cvar99,
            'Hit Ratio': hit_ratio,
            'Best Day': best_day,
            'Worst Day': worst_day,
            'Skew': skewv,
            'Kurtosis (excess)': kurtv,
        }
        if bench_rets is not None:
            metrics.update({
                'Beta vs Benchmark': beta,
                'Alpha (ann) vs Benchmark': alpha_ann,
                'Correlation vs Benchmark': corr,
                'R^2 vs Benchmark': r2,
            })
        return metrics

    def _compute_contributions(self, prices: pd.DataFrame, shares: pd.DataFrame, v0: float, vend: float):
        start_values = shares.iloc[0] * prices.iloc[0]
        end_values = shares.iloc[-1] * prices.iloc[-1]
        contrib = (end_values - start_values) / v0
        contrib.name = 'Contribution'
        contrib = contrib.sort_values(ascending=False)
        return contrib

class BacktestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Portfolio Backtester")
        self.root.geometry("1300x980")
        self.last_result = None
        self.last_mc = None  # store MC results for export

        # Theme + font
        self.font_family = self._choose_font_family()
        self.theme_name = 'Light'
        self._apply_theme(self.font_family, self.theme_name)
        self._apply_mpl_theme(self.font_family)

        self._build_ui()
 # Font and theme
    def _choose_font_family(self):
        families = set(tkfont.families())
        if "Aptos" in families:
            return "Aptos"
        elif "Segoe UI" in families:
            return "Segoe UI"
        elif "Arial" in families:
            return "Arial"
        return "TkDefaultFont"

    def _apply_theme(self, font_family, theme_name='Dark'):
        global COLORS
        COLORS = THEMES[theme_name].copy()
        self.root.configure(bg=COLORS['bg'])

        style = ttk.Style(self.root)
        style.theme_use('clam')

        default_font = (font_family, 10)
        heading_font = (font_family, 10, 'bold')
        small_font = (font_family, 9)
        #Button and ecc style
        style.configure("TFrame", background=COLORS['bg'])
        style.configure("TLabelframe", background=COLORS['panel'], foreground=COLORS['text'])
        style.configure("TLabelframe.Label", background=COLORS['panel'], foreground=COLORS['text'], font=heading_font)
        style.configure("TLabel", background=COLORS['panel'], foreground=COLORS['text'], font=default_font)
        style.configure("Status.TLabel", background=COLORS['bg'], foreground=COLORS['muted'], font=small_font)

        style.configure("TEntry", fieldbackground=COLORS['panel'], background=COLORS['panel'],
                        foreground=COLORS['text'], insertcolor=COLORS['text'])
        style.configure("TCombobox", fieldbackground=COLORS['panel'], background=COLORS['panel'],
                        foreground=COLORS['text'])
        style.map("TCombobox",
                  fieldbackground=[('readonly', COLORS['panel'])],
                  foreground=[('readonly', COLORS['text'])])

        style.configure("TCheckbutton", background=COLORS['panel'], foreground=COLORS['text'])

        style.configure("TButton", font=default_font, background=COLORS['panel'], foreground=COLORS['text'])
        style.map("TButton",
                  background=[('active', COLORS['panel'])],
                  foreground=[('disabled', COLORS['muted'])])

        style.configure("Accent.TButton", background=COLORS['accent'], foreground=COLORS['btn_fg'])
        style.map("Accent.TButton",
                  background=[('active', COLORS['accent_hover']), ('!disabled', COLORS['accent'])],
                  foreground=[('!disabled', COLORS['btn_fg'])])

        style.configure("TNotebook", background=COLORS['bg'], borderwidth=0)
        style.configure("TNotebook.Tab", background=COLORS['panel'], foreground=COLORS['muted'], padding=[12, 6])
        style.map("TNotebook.Tab",
                  background=[('selected', COLORS['bg'])],
                  foreground=[('selected', COLORS['text'])])

        style.configure("Treeview",
                        background=COLORS['panel'],
                        fieldbackground=COLORS['panel'],
                        foreground=COLORS['text'],
                        rowheight=26,
                        font=default_font)
        style.configure("Treeview.Heading",
                        background=COLORS['panel'],
                        foreground=COLORS['text'],
                        font=heading_font)
        style.map("Treeview",
                  background=[('selected', COLORS['selection_bg'])],
                  foreground=[('selected', COLORS['selection_fg'])])

        # Seaborn style
        sns.set_theme(context='notebook', style='darkgrid' if self.theme_name == 'Dark' else 'whitegrid')

        self._fonts = {'default': default_font, 'heading': heading_font, 'small': small_font}

    def _apply_mpl_theme(self, font_family):
        mpl.rcParams.update({
            'font.family': font_family,
            'figure.facecolor': COLORS['panel'],
            'axes.facecolor': COLORS['chart_bg'],
            'axes.edgecolor': COLORS['grid'],
            'axes.labelcolor': COLORS['text'],
            'axes.titlecolor': COLORS['text'],
            'xtick.color': COLORS['text'],
            'ytick.color': COLORS['text'],
            'grid.color': COLORS['grid'],
            'grid.alpha': 0.3,
            'axes.grid': True,
            'savefig.facecolor': COLORS['panel'],
            'text.color': COLORS['text'],
        })

    def _build_ui(self):
        ctrl = ttk.LabelFrame(self.root, text="Inputs")
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        ctrl.configure(style="TLabelframe")

        # Inputs at start
        self.var_tickers = tk.StringVar(value="AAPL,MSFT,GOOGL")
        self.var_weights = tk.StringVar(value="")
        self.var_start = tk.StringVar(value="2018-01-01")
        self.var_end = tk.StringVar(value=datetime.today().strftime('%Y-%m-%d'))
        self.var_capital = tk.StringVar(value="100000")
        self.var_rebalance = tk.StringVar(value="Monthly")
        self.var_rebalance_n = tk.StringVar(value="")
        self.var_benchmark = tk.StringVar(value="^GSPC") #This is the ticker of S&P500 index, the user can change to whatever index wants. Look on yahoo finance
        self.var_rf = tk.StringVar(value="0.02")
        self.var_logscale = tk.BooleanVar(value=False)
        self.var_theme = tk.StringVar(value=self.theme_name)
        self.var_fee_bps = tk.StringVar(value="0")       # transaction costs in bps of turnover
        self.var_slip_bps = tk.StringVar(value="0")      # slippage in bps of turnover

        r = 0
        ttk.Label(ctrl, text="Tickers (comma-separated):").grid(row=r, column=0, sticky="e", padx=6, pady=5)
        ttk.Entry(ctrl, textvariable=self.var_tickers, width=40).grid(row=r, column=1, sticky="w", padx=6, pady=5)

        ttk.Label(ctrl, text="Weights (optional, comma, sum≈1):").grid(row=r, column=2, sticky="e", padx=6, pady=5)
        ttk.Entry(ctrl, textvariable=self.var_weights, width=28).grid(row=r, column=3, sticky="w", padx=6, pady=5)

        r += 1
        ttk.Label(ctrl, text="Start (YYYY-MM-DD):").grid(row=r, column=0, sticky="e", padx=6, pady=5)
        ttk.Entry(ctrl, textvariable=self.var_start, width=20).grid(row=r, column=1, sticky="w", padx=6, pady=5)

        ttk.Label(ctrl, text="End (YYYY-MM-DD):").grid(row=r, column=2, sticky="e", padx=6, pady=5)
        ttk.Entry(ctrl, textvariable=self.var_end, width=20).grid(row=r, column=3, sticky="w", padx=6, pady=5)

        r += 1
        ttk.Label(ctrl, text="Initial Capital:").grid(row=r, column=0, sticky="e", padx=6, pady=5)
        ttk.Entry(ctrl, textvariable=self.var_capital, width=20).grid(row=r, column=1, sticky="w", padx=6, pady=5)

        ttk.Label(ctrl, text="Rebalance:").grid(row=r, column=2, sticky="e", padx=6, pady=5)
        cmb = ttk.Combobox(ctrl, textvariable=self.var_rebalance,
                           values=["None","Monthly","Quarterly","Yearly","Every N days"],
                           width=18, state="readonly")
        cmb.grid(row=r, column=3, sticky="w", padx=6, pady=5)
        cmb.bind("<<ComboboxSelected>>", self._on_rebalance_change)

        r += 1
        ttk.Label(ctrl, text="N (if Every N days):").grid(row=r, column=0, sticky="e", padx=6, pady=5)
        self.entry_reb_n = ttk.Entry(ctrl, textvariable=self.var_rebalance_n, width=20, state='disabled')
        self.entry_reb_n.grid(row=r, column=1, sticky="w", padx=6, pady=5)

        ttk.Label(ctrl, text="Benchmark (optional):").grid(row=r, column=2, sticky="e", padx=6, pady=5)
        ttk.Entry(ctrl, textvariable=self.var_benchmark, width=20).grid(row=r, column=3, sticky="w", padx=6, pady=5)

        r += 1
        ttk.Label(ctrl, text="Risk-free (annual, e.g., 0.02):").grid(row=r, column=0, sticky="e", padx=6, pady=5)
        ttk.Entry(ctrl, textvariable=self.var_rf, width=20).grid(row=r, column=1, sticky="w", padx=6, pady=5)

        ttk.Label(ctrl, text="Fees bps (turnover):").grid(row=r, column=2, sticky="e", padx=6, pady=5)
        ttk.Entry(ctrl, textvariable=self.var_fee_bps, width=20).grid(row=r, column=3, sticky="w", padx=6, pady=5)

        r += 1
        ttk.Checkbutton(ctrl, text="Log scale equity", variable=self.var_logscale).grid(row=r, column=1, sticky="w", padx=6, pady=5)

        ttk.Label(ctrl, text="Slippage bps:").grid(row=r, column=2, sticky="e", padx=6, pady=5)
        ttk.Entry(ctrl, textvariable=self.var_slip_bps, width=20).grid(row=r, column=3, sticky="w", padx=6, pady=5)

        r += 1
        ttk.Label(ctrl, text="Theme:").grid(row=r, column=0, sticky="e", padx=6, pady=5)
        theme_cmb = ttk.Combobox(ctrl, textvariable=self.var_theme, values=["Dark","Light"], width=18, state="readonly")
        theme_cmb.grid(row=r, column=1, sticky="w", padx=6, pady=5)
        theme_cmb.bind("<<ComboboxSelected>>", self.on_theme_change)

        ttk.Button(ctrl, text="Run Backtest", command=self.run_backtest, style="Accent.TButton").grid(row=r, column=3, sticky="e", padx=6, pady=5)
        self.btn_export = ttk.Button(ctrl, text="Export to Excel", command=self.export_to_excel, state='disabled', style="Accent.TButton")
        self.btn_export.grid(row=r, column=2, sticky="e", padx=6, pady=5)

        # Status
        self.status = ttk.Label(self.root, text="Ready for backtest. For tickers look on Yahoo Finance.", anchor="w", style="Status.TLabel")
        self.status.pack(side=tk.TOP, fill=tk.X, padx=10)

        # Tabs
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Summary tab
        self.tab_summary = ttk.Frame(self.nb, style="TFrame")
        self.nb.add(self.tab_summary, text="Summary")

        self.metrics_tree = ttk.Treeview(self.tab_summary, columns=("Metric","Value"), show='headings', height=18)
        self.metrics_tree.heading("Metric", text="Metric")
        self.metrics_tree.heading("Value", text="Value")
        self.metrics_tree.column("Metric", width=320, anchor='w')
        self.metrics_tree.column("Value", width=180, anchor='e')
        self.metrics_tree.tag_configure('pos', foreground=COLORS['pos'])
        self.metrics_tree.tag_configure('neg', foreground=COLORS['neg'])
        self.metrics_tree.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        self.metrics_tree.bind("<Double-1>", self.on_metric_double_click)

        self.summary_text = tk.Text(self.tab_summary, height=18, width=80,
                                    font=self._fonts['default'],
                                    bg=COLORS['panel'], fg=COLORS['text'],
                                    insertbackground=COLORS['text'], borderwidth=0, highlightthickness=0)
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.contrib_tree = ttk.Treeview(self.tab_summary, columns=("Asset","Contribution"), show='headings', height=18)
        self.contrib_tree.heading("Asset", text="Asset")
        self.contrib_tree.heading("Contribution", text="Contribution to Total Return")
        self.contrib_tree.column("Asset", width=160, anchor='w')
        self.contrib_tree.column("Contribution", width=220, anchor='e')
        self.contrib_tree.tag_configure('pos', foreground=COLORS['pos'])
        self.contrib_tree.tag_configure('neg', foreground=COLORS['neg'])
        self.contrib_tree.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        # Equity & Drawdown
        self.tab_charts = ttk.Frame(self.nb, style="TFrame")
        self.nb.add(self.tab_charts, text="Equity & Drawdown")
        self.canvas_charts = None
        self.toolbar_charts = None

        # Rolling Risk
        self.tab_rolling = ttk.Frame(self.nb, style="TFrame")
        self.nb.add(self.tab_rolling, text="Rolling Risk")
        self.canvas_rolling = None
        self.toolbar_rolling = None

        # Correlations
        self.tab_corr = ttk.Frame(self.nb, style="TFrame")
        self.nb.add(self.tab_corr, text="Correlations")
        self.canvas_corr = None
        self.toolbar_corr = None

        # Monthly Returns
        self.tab_monthly = ttk.Frame(self.nb, style="TFrame")
        self.nb.add(self.tab_monthly, text="Monthly Returns")
        self.canvas_monthly = None
        self.toolbar_monthly = None

        # Monte Carlo
        self.tab_mc = ttk.Frame(self.nb, style="TFrame")
        self.nb.add(self.tab_mc, text="Monte Carlo")
        self._build_mc_tab()

    def _on_rebalance_change(self, event=None):
        mode = self.var_rebalance.get()
        if mode == "Every N days":
            self.entry_reb_n.configure(state='normal')
        else:
            self.entry_reb_n.configure(state='disabled')
            self.var_rebalance_n.set("")

    # Monte Carlo
    def _build_mc_tab(self):
        frm = ttk.LabelFrame(self.tab_mc, text="Simulation Settings")
        frm.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.var_mc_paths = tk.StringVar(value="1000")
        self.var_mc_years = tk.StringVar(value="1.0")
        self.var_mc_model = tk.StringVar(value="GBM (Normal)")
        self.var_mc_seed = tk.StringVar(value="")
        self.var_mc_show_paths = tk.StringVar(value="50")
        self.var_mc_start_value = tk.StringVar(value="")  # default to last PV after backtest

        r = 0
        ttk.Label(frm, text="Model:").grid(row=r, column=0, sticky="e", padx=6, pady=5)
        ttk.Combobox(frm, textvariable=self.var_mc_model, values=["GBM (Normal)", "Bootstrap"], state="readonly", width=18)\
            .grid(row=r, column=1, sticky="w", padx=6, pady=5)

        ttk.Label(frm, text="# Paths:").grid(row=r, column=2, sticky="e", padx=6, pady=5)
        ttk.Entry(frm, textvariable=self.var_mc_paths, width=10).grid(row=r, column=3, sticky="w", padx=6, pady=5)

        r += 1
        ttk.Label(frm, text="Horizon (years):").grid(row=r, column=0, sticky="e", padx=6, pady=5)
        ttk.Entry(frm, textvariable=self.var_mc_years, width=10).grid(row=r, column=1, sticky="w", padx=6, pady=5)

        ttk.Label(frm, text="Show paths:").grid(row=r, column=2, sticky="e", padx=6, pady=5)
        ttk.Entry(frm, textvariable=self.var_mc_show_paths, width=10).grid(row=r, column=3, sticky="w", padx=6, pady=5)

        r += 1
        ttk.Label(frm, text="Seed (optional):").grid(row=r, column=0, sticky="e", padx=6, pady=5)
        ttk.Entry(frm, textvariable=self.var_mc_seed, width=10).grid(row=r, column=1, sticky="w", padx=6, pady=5)

        ttk.Label(frm, text="Start Value (blank = last):").grid(row=r, column=2, sticky="e", padx=6, pady=5)
        ttk.Entry(frm, textvariable=self.var_mc_start_value, width=14).grid(row=r, column=3, sticky="w", padx=6, pady=5)

        ttk.Button(frm, text="Run Simulation", style="Accent.TButton", command=self.run_mc_simulation)\
            .grid(row=r, column=4, sticky="w", padx=12, pady=5)

        self.mc_summary = ttk.Label(self.tab_mc, text="Run a backtest, then simulate.", style="Status.TLabel")
        self.mc_summary.pack(side=tk.TOP, anchor='w', padx=12)

        self.canvas_mc = None
        self.toolbar_mc = None

    def on_theme_change(self, event=None):
        self.theme_name = self.var_theme.get()
        # Re-apply themes and redraw
        self._apply_theme(self.font_family, self.theme_name)
        self._apply_mpl_theme(self.font_family)

        # Update background colors of custom widgets
        try:
            self.summary_text.configure(bg=COLORS['panel'], fg=COLORS['text'], insertbackground=COLORS['text'])
        except Exception:
            pass

        # Redraw charts if we have results
        if self.last_result:
            self._draw_charts(self.last_result)
            self._draw_rolling(self.last_result)
            self._draw_corr(self.last_result)
            self._draw_monthly(self.last_result)
        if self.last_mc:
            self.run_mc_simulation()  # re-render MC figure with new theme

    def run_backtest(self):
        try:
            tickers = parse_ticker_list(self.var_tickers.get())
            if not tickers:
                raise ValueError("Enter at least one ticker.")
            weights = parse_float_list(self.var_weights.get())
            if weights is not None and len(weights) != len(tickers):
                raise ValueError("Weights count must match number of tickers.")
            start = self.var_start.get().strip()
            end = self.var_end.get().strip()
            _ = datetime.strptime(start, "%Y-%m-%d")
            _ = datetime.strptime(end, "%Y-%m-%d")
            capital = float(self.var_capital.get().strip())
            rebalance = self.var_rebalance.get()
            rebalance_n = self.var_rebalance_n.get()
            benchmark = self.var_benchmark.get().strip()
            rf = float(self.var_rf.get().strip())
            fee_bps = float(self.var_fee_bps.get().strip() or 0)
            slip_bps = float(self.var_slip_bps.get().strip() or 0)

            self.status.config(text="Running backtest...")
            self.root.update_idletasks()

            bt = PortfolioBacktester(
                tickers=tickers,
                weights=weights,
                start=start,
                end=end,
                initial_capital=capital,
                rebalance=rebalance,
                rebalance_n=rebalance_n,
                benchmark=benchmark if benchmark else None,
                risk_free=rf,
                tx_cost_bps=fee_bps,
                slippage_bps=slip_bps
            )
            result = bt.run()
            self.last_result = result
            self.last_mc = None  # clear MC after new backtest

            # Default MC start value to last PV
            try:
                self.var_mc_start_value.set(f"{result['portfolio_value'].iloc[-1]:.2f}")
            except Exception:
                pass

            self._populate_summary(result)
            self._draw_charts(result)
            self._draw_rolling(result)
            self._draw_corr(result)
            self._draw_monthly(result)
            self.status.config(text="Done. Double click on metrics to get information about them. Better in full screen.")
            self.btn_export.config(state='normal')

        except Exception as e:
            self.status.config(text=f"Error: {e}")
            messagebox.showerror("Error", str(e))

    # Print all the results of the metrics
    def _populate_summary(self, result):
        for i in self.metrics_tree.get_children():
            self.metrics_tree.delete(i)
        for i in self.contrib_tree.get_children():
            self.contrib_tree.delete(i)
        self.summary_text.delete("1.0", tk.END)

        metrics = result['metrics']
        order = [
            'Start', 'End',
            'Initial Capital', 'Final Value',
            'Total Return', 'CAGR',
            'Volatility (ann)', 'Sharpe (ann)', 'Sortino (ann)',
            'Max Drawdown', 'Calmar',
            'Daily VaR 95%', 'Daily CVaR 95%', 'Daily VaR 99%', 'Daily CVaR 99%',
            'Hit Ratio', 'Best Day', 'Worst Day',
            'Skew', 'Kurtosis (excess)',
            'Beta vs Benchmark', 'Alpha (ann) vs Benchmark',
            'Correlation vs Benchmark', 'R^2 vs Benchmark'
        ]

        def fmt_value(key, val):
            if not isinstance(val, (int, float)) or pd.isna(val):
                return str(val)
            if key in ('Initial Capital', 'Final Value'):
                return f"${val:,.2f}"
            pct_keys = {'Total Return', 'CAGR', 'Volatility (ann)', 'Max Drawdown',
                        'Daily VaR 95%', 'Daily CVaR 95%', 'Daily VaR 99%', 'Daily CVaR 99%',
                        'Hit Ratio', 'Best Day', 'Worst Day', 'Alpha (ann) vs Benchmark'}
            if key in pct_keys:
                return f"{val*100:.2f}%"
            if key in ('Sharpe (ann)', 'Sortino (ann)', 'Calmar', 'Beta vs Benchmark',
                       'Correlation vs Benchmark', 'R^2 vs Benchmark', 'Skew', 'Kurtosis (excess)'):
                return f"{val:.2f}"
            return f"{val:.4f}"

        for k in order:
            if k in metrics:
                v = metrics[k]
                display = fmt_value(k, v)
                tags = ()
                if isinstance(v, (int, float)) and not pd.isna(v):
                    if k in ('Total Return', 'CAGR', 'Alpha (ann) vs Benchmark', 'Best Day'):
                        tags = ('pos',) if v > 0 else ('neg',)
                    elif k in ('Max Drawdown', 'Worst Day', 'Daily VaR 95%', 'Daily CVaR 95%', 'Daily VaR 99%', 'Daily CVaR 99%'):
                        tags = ('neg',) if v < 0 else ()
                self.metrics_tree.insert('', tk.END, values=(k, display), tags=tags)

        #Contribution of the single stocks
        contrib = result['contributions']
        for asset, val in contrib.items():
            tag = 'pos' if val >= 0 else 'neg'
            self.contrib_tree.insert('', tk.END, values=(asset, f"{val*100:.2f}%"), tags=(tag,))

        interp = self._interpret(metrics)
        self.summary_text.insert(tk.END, interp)

    def _interpret(self, m):
        def pct(x):
            return f"{x*100:.2f}%" if pd.notna(x) else "n/a"
        def num(x):
            return f"{x:,.2f}" if pd.notna(x) else "n/a"
        lines = []
        try:
            init_v = float(m.get('Initial Capital', np.nan))
            end_v = float(m.get('Final Value', np.nan))
            lines.append(f"Growth: ${init_v:,.0f} → ${end_v:,.0f} ({pct(m.get('Total Return'))}) from {m.get('Start')} to {m.get('End')}.")
        except Exception:
            pass
        lines.append(f"CAGR: {pct(m.get('CAGR'))} | Volatility (ann): {pct(m.get('Volatility (ann)'))}")
        lines.append(f"Sharpe: {num(m.get('Sharpe (ann)'))} | Sortino: {num(m.get('Sortino (ann)'))} | Calmar: {num(m.get('Calmar'))}")
        lines.append(f"Max drawdown: {pct(m.get('Max Drawdown'))}")
        lines.append(f"Tail risk (daily): VaR95 {pct(m.get('Daily VaR 95%'))}, CVaR95 {pct(m.get('Daily CVaR 95%'))}")
        if 'Beta vs Benchmark' in m and pd.notna(m.get('Beta vs Benchmark')):
            lines.append(f"Benchmark: Beta {num(m.get('Beta vs Benchmark'))}, Alpha {pct(m.get('Alpha (ann) vs Benchmark'))}, Corr {num(m.get('Correlation vs Benchmark'))}, R² {num(m.get('R^2 vs Benchmark'))}")
        lines.append(f"Hit ratio: {pct(m.get('Hit Ratio'))} | Best day: {pct(m.get('Best Day'))} | Worst day: {pct(m.get('Worst Day'))}")
        lines.append(f"Shape: Skew {num(m.get('Skew'))}, Excess kurtosis {num(m.get('Kurtosis (excess)'))}")
        lines.append("\nNotes: Uses adjusted close (dividends included), fractional shares, optional fees/slippage, and scheduled rebalancing.")
        return "\n".join(lines)

    def _destroy_canvas(self, canvas, toolbar):
        if toolbar is not None:
            toolbar.destroy()
        if canvas is not None:
            canvas.get_tk_widget().destroy()

    def _style_axes(self, ax):
        ax.set_facecolor(COLORS['chart_bg'])
        for spine in ax.spines.values():
            spine.set_color(COLORS['grid'])
        ax.tick_params(colors=COLORS['text'])
        ax.yaxis.label.set_color(COLORS['text'])
        ax.xaxis.label.set_color(COLORS['text'])

    #COmparison bewteen index and portfolio
    def _draw_charts(self, result):
        self._destroy_canvas(self.canvas_charts, self.toolbar_charts)

        fig = Figure(figsize=(10.5, 7.2), dpi=100)
        gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1.0])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.34)

        pv = result['portfolio_value']
        bench = result['benchmark_prices']
        logscale = self.var_logscale.get()

        self._style_axes(ax1)
        self._style_axes(ax2)

        ax1.plot(pv.index, pv.values, label='Portfolio', color=COLORS['line_port'], linewidth=1.9)
        if bench is not None:
            bench_norm = bench / bench.iloc[0] * pv.iloc[0]
            ax1.plot(bench_norm.index, bench_norm.values, label='Benchmark (scaled)', color=COLORS['line_bench'], alpha=0.95)
        ax1.set_title("Equity Curve", color=COLORS['text'])
        ax1.set_ylabel("Value ($)")
        ax1.legend(loc='upper left', facecolor=COLORS['panel'], edgecolor=COLORS['grid'])
        if logscale:
            ax1.set_yscale('log')

        dd, max_dd, _ = compute_drawdown(pv)
        ax2.fill_between(dd.index, dd.values, 0, color=COLORS['drawdown_fill'], alpha=0.35)
        ax2.set_title(f"Drawdown (Min: {max_dd*100:.2f}%)", color=COLORS['text'])
        ax2.set_ylabel("Drawdown")
        ax2.yaxis.set_major_formatter(lambda x, pos: f"{x*100:.0f}%")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        fig.set_facecolor(COLORS['panel'])

        self.canvas_charts = FigureCanvasTkAgg(fig, master=self.tab_charts)
        self.canvas_charts.draw()
        self.canvas_charts.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_charts = NavigationToolbar2Tk(self.canvas_charts, self.tab_charts)
        self.toolbar_charts.update()

    # Rolling risk
    def _draw_rolling(self, result):
        self._destroy_canvas(self.canvas_rolling, self.toolbar_rolling)

        fig = Figure(figsize=(10.5, 6.6), dpi=100)
        gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.36)

        self._style_axes(ax1)
        self._style_axes(ax2)

        r = result['portfolio_returns']
        rf = float(self.var_rf.get())
        if len(r) > 20:
            win1 = 63
            roll_vol = r.rolling(win1).std() * np.sqrt(TRADING_DAYS)
            ax1.plot(roll_vol.index, roll_vol.values, color='#a78bfa')
            ax1.set_title(f"Rolling Volatility (ann, {win1}d window)", color=COLORS['text'])

            win2 = 252
            roll_mu_ann = r.rolling(win2).mean() * TRADING_DAYS
            roll_sd_ann = r.rolling(win2).std() * np.sqrt(TRADING_DAYS)
            roll_sharpe = (roll_mu_ann - rf) / roll_sd_ann
            ax2.plot(roll_sharpe.index, roll_sharpe.values, color='#34d399')
            ax2.set_title(f"Rolling Sharpe ({win2}d window)", color=COLORS['text'])
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        else:
            ax1.text(0.5, 0.5, "Not enough data for rolling stats", ha='center', va='center', transform=ax1.transAxes, color=COLORS['muted'])

        fig.set_facecolor(COLORS['panel'])

        self.canvas_rolling = FigureCanvasTkAgg(fig, master=self.tab_rolling)
        self.canvas_rolling.draw()
        self.canvas_rolling.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_rolling = NavigationToolbar2Tk(self.canvas_rolling, self.tab_rolling)
        self.toolbar_rolling.update()

    # Correlation between stocks
    def _draw_corr(self, result):
        self._destroy_canvas(self.canvas_corr, self.toolbar_corr)

        corr = result['correlation']
        fig = Figure(figsize=(7.8, 6.8), dpi=100)
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.92, bottom=0.10)

        if corr is not None and corr.shape[0] > 1:
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax, fmt=".2f", square=True,
                        cbar_kws={'shrink': 0.8})
            ax.set_title("Asset Return Correlations", color=COLORS['text'])
        else:
            ax.text(0.5, 0.5, "Correlation heatmap requires 2+ assets", ha='center', va='center',
                    transform=ax.transAxes, color=COLORS['muted'])

        ax.set_facecolor(COLORS['chart_bg'])
        for spine in ax.spines.values():
            spine.set_color(COLORS['grid'])
        fig.set_facecolor(COLORS['panel'])

        self.canvas_corr = FigureCanvasTkAgg(fig, master=self.tab_corr)
        self.canvas_corr.draw()
        self.canvas_corr.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_corr = NavigationToolbar2Tk(self.canvas_corr, self.tab_corr)
        self.toolbar_corr.update()

    #Montly returns
    def _draw_monthly(self, result):
        self._destroy_canvas(self.canvas_monthly, self.toolbar_monthly)

        tbl = result['monthly_table']
        fig = Figure(figsize=(8.8, 6.8), dpi=100)
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.92, bottom=0.10)

        if tbl is not None and not tbl.empty:
            sns.heatmap(tbl * 100.0, annot=True, fmt=".1f", cmap='RdYlGn', center=0, ax=ax, cbar_kws={'label': '%', 'shrink': 0.8})
            ax.set_title("Monthly Returns (%)", color=COLORS['text'])
        else:
            ax.text(0.5, 0.5, "Insufficient data for monthly returns", ha='center', va='center', transform=ax.transAxes, color=COLORS['muted'])

        ax.set_facecolor(COLORS['chart_bg'])
        for spine in ax.spines.values():
            spine.set_color(COLORS['grid'])
        fig.set_facecolor(COLORS['panel'])

        self.canvas_monthly = FigureCanvasTkAgg(fig, master=self.tab_monthly)
        self.canvas_monthly.draw()
        self.canvas_monthly.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_monthly = NavigationToolbar2Tk(self.canvas_monthly, self.tab_monthly)
        self.toolbar_monthly.update()

    # Metrics info, user must double-click
    def on_metric_double_click(self, event):
        item_id = self.metrics_tree.identify_row(event.y)
        if not item_id:
            return
        values = self.metrics_tree.item(item_id, 'values')
        if not values or len(values) < 1:
            return
        metric = values[0]
        doc = METRIC_DOCS.get(metric, "No information available for this metric.")
        self._show_metric_info(metric, doc)

    def _show_metric_info(self, title, text):
        top = tk.Toplevel(self.root)
        top.title(f"{title} — Info")
        top.configure(bg=COLORS['panel'])
        top.geometry("600x380")
        top.transient(self.root)
        top.grab_set()

        lbl_title = tk.Label(top, text=title, bg=COLORS['panel'], fg=COLORS['text'], font=(self.font_family, 11, 'bold'))
        lbl_title.pack(anchor='w', padx=12, pady=(10, 6))

        txt = tk.Text(top, wrap='word', bg=COLORS['panel'], fg=COLORS['text'],
                      insertbackground=COLORS['text'], font=(self.font_family, 10), borderwidth=0, highlightthickness=0)
        txt.insert('1.0', text)
        txt.config(state='disabled')
        txt.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        btn = ttk.Button(top, text="Close", command=top.destroy, style="Accent.TButton")
        btn.pack(pady=10)

    # Monte Carlo simulation
    def run_mc_simulation(self):
        if not self.last_result:
            messagebox.showinfo("Monte Carlo Simulation", "Run a backtest first.")
            return

        try:
            n_paths = max(1, int(float(self.var_mc_paths.get())))
            years = float(self.var_mc_years.get())
            display_paths = max(1, int(float(self.var_mc_show_paths.get())))
            model = self.var_mc_model.get()
            seed_str = self.var_mc_seed.get().strip()
            start_val_str = self.var_mc_start_value.get().strip()
            start_val = float(start_val_str) if start_val_str else float(self.last_result['portfolio_value'].iloc[-1])
            if years <= 0:
                raise ValueError("Horizon (years) must be > 0.")
            if n_paths > 50000: #user can change the amount of paths to generate
                raise ValueError(f"Please use {n_paths} paths or fewer.")
        except Exception as e:
            messagebox.showerror("Monte Carlo", f"Invalid settings: {e}")
            return

        r = self.last_result['portfolio_returns']
        if r is None or len(r) < 5:
            messagebox.showerror("Monte Carlo", "Insufficient return history. Run a longer backtest.")
            return

        # Daily log-returns
        log_r = np.log1p(r.dropna().values)
        mu_l = float(np.mean(log_r))
        sigma_l = float(np.std(log_r, ddof=0))
        T = int(round(years * TRADING_DAYS))
        if T <= 1:
            T = 2

        # RNG
        try:
            rng = np.random.default_rng(int(seed_str)) if seed_str else np.random.default_rng()
        except Exception:
            rng = np.random.default_rng()

        # Generate log-return increments
        if model.startswith("Bootstrap"):
            increments = rng.choice(log_r, size=(T, n_paths), replace=True)
        else:
            increments = rng.normal(loc=mu_l, scale=sigma_l, size=(T, n_paths))

        # Price paths
        log_paths = np.vstack([np.zeros((1, n_paths)), np.cumsum(increments, axis=0)])
        paths = start_val * np.exp(log_paths)  # shape: (T+1, n_paths)

        # Percentile bands and end returns
        q05 = np.percentile(paths, 5, axis=1)
        q50 = np.percentile(paths, 50, axis=1)
        q95 = np.percentile(paths, 95, axis=1)
        end_vals = paths[-1, :]
        end_rets = end_vals / start_val - 1.0

        exp_return = float(np.mean(end_rets))
        med_return = float(np.median(end_rets))
        var95 = float(np.percentile(end_rets, 5))
        cvar95 = float(end_rets[end_rets <= var95].mean()) if np.any(end_rets <= var95) else np.nan
        prob_loss = float(np.mean(end_rets < 0.0))

        # Store for export
        self.last_mc = {
            'model': model,
            'n_paths': n_paths,
            'years': years,
            'start_value': start_val,
            'q05': q05,
            'q50': q50,
            'q95': q95,
            'end_rets': end_rets,
            'exp_return': exp_return,
            'med_return': med_return,
            'var95': var95,
            'cvar95': cvar95,
            'prob_loss': prob_loss,
            'T': T,
        }

        # Summary
        self.mc_summary.config(
            text=f"Ending return distribution — E[R]: {exp_return*100:.2f}% | Median: {med_return*100:.2f}% | "
                 f"VaR95: {var95*100:.2f}% | CVaR95: {cvar95*100:.2f}% | P(Loss): {prob_loss*100:.1f}%"
        )

        # Draw
        self._destroy_canvas(self.canvas_mc, self.toolbar_mc)
        fig = Figure(figsize=(11.0, 6.8), dpi=100)
        gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.25)
        ax_paths = fig.add_subplot(gs[0, 0])
        ax_hist = fig.add_subplot(gs[0, 1])

        self._style_axes(ax_paths)
        self._style_axes(ax_hist)
        fig.set_facecolor(COLORS['panel'])

        # Plot subset of paths
        show_n = min(display_paths, n_paths)
        idx = np.arange(n_paths)
        rng.shuffle(idx)
        chosen = idx[:show_n]
        for j in chosen:
            ax_paths.plot(paths[:, j], color='#1dd1a1' if self.theme_name == 'Dark' else '#10b981', alpha=0.24, linewidth=1.0)

        # Percentile band + median
        x = np.arange(T + 1)
        ax_paths.fill_between(x, q05, q95, color=COLORS['line_port'], alpha=0.16, label='5–95% band')
        ax_paths.plot(x, q50, color=COLORS['line_port'], linewidth=2.0, label='Median')
        ax_paths.set_title(f"Monte Carlo Paths ({model}, {n_paths} paths, {years}y)")
        ax_paths.set_ylabel("Portfolio Value ($)")
        ax_paths.legend(loc='upper left', facecolor=COLORS['panel'], edgecolor=COLORS['grid'])

        # Ending returns distribution
        ax_hist.hist(end_rets, bins=40, color=COLORS['line_bench'], alpha=0.85, edgecolor=COLORS['grid'])
        try:
            sns.kdeplot(x=end_rets, ax=ax_hist, color=COLORS['pos'], lw=1.6)
        except Exception:
            pass
        ax_hist.axvline(0, color=COLORS['neg'], linestyle='--', alpha=0.8)
        ax_hist.set_title("Ending Returns Distribution")
        ax_hist.set_xlabel("Return")
        ax_hist.set_ylabel("Frequency")
        ax_hist.xaxis.set_major_formatter(lambda x, pos: f"{x*100:.0f}%")

        self.canvas_mc = FigureCanvasTkAgg(fig, master=self.tab_mc)
        self.canvas_mc.draw()
        self.canvas_mc.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_mc = NavigationToolbar2Tk(self.canvas_mc, self.tab_mc)
        self.toolbar_mc.update()

    # Command to export on excel the portfolio history, main metrics and MC simulation
    def export_to_excel(self):
        if not self.last_result:
            messagebox.showinfo("Export to Excel", "Run a backtest first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save Excel File",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile="backtest_export.xlsx"
        )
        if not path:
            return

        try:
            res = self.last_result
            pv = res['portfolio_value']
            pr = res['portfolio_returns']
            dd, _, _ = compute_drawdown(pv)

            hist_df = pd.concat([
                pv.rename('Portfolio Value'),
                pr.reindex(pv.index).rename('Portfolio Return'),
                dd.rename('Drawdown')
            ], axis=1)

            # Benchmark
            if res.get('benchmark_prices') is not None:
                hist_df = hist_df.join(res['benchmark_prices'].rename('Benchmark Price'))
                if res.get('benchmark_returns') is not None:
                    hist_df = hist_df.join(res['benchmark_returns'].rename('Benchmark Return'))

            # Weights, turnover, and tx costs
            weights_ot = res.get('weights_over_time')
            if weights_ot is not None and not weights_ot.empty:
                hist_df = hist_df.join(weights_ot.rename(columns=lambda c: f"Weight: {c}"))
            if res.get('turnover') is not None:
                hist_df = hist_df.join(res['turnover'])
            if res.get('tx_costs') is not None:
                hist_df = hist_df.join(res['tx_costs'])

            hist_df = hist_df.copy()
            hist_df.index.name = 'Date'
            hist_df.reset_index(inplace=True)

            metrics = res['metrics']
            metrics_order = [
                'Start', 'End',
                'Initial Capital', 'Final Value',
                'Total Return', 'CAGR',
                'Volatility (ann)', 'Sharpe (ann)', 'Sortino (ann)',
                'Max Drawdown', 'Calmar',
                'Daily VaR 95%', 'Daily CVaR 95%', 'Daily VaR 99%', 'Daily CVaR 99%',
                'Hit Ratio', 'Best Day', 'Worst Day',
                'Skew', 'Kurtosis (excess)',
                'Beta vs Benchmark', 'Alpha (ann) vs Benchmark',
                'Correlation vs Benchmark', 'R^2 vs Benchmark'
            ]
            metrics_items = [(k, metrics[k]) for k in metrics_order if k in metrics]

            with pd.ExcelWriter(path, engine='xlsxwriter', datetime_format='yyyy-mm-dd', date_format='yyyy-mm-dd') as writer:
                # Sheet 1: Portfolios history
                hist_df.to_excel(writer, sheet_name='Portfolio_History', index=False)
                wb = writer.book
                ws_hist = writer.sheets['Portfolio_History']

                fmt_date = wb.add_format({'num_format': 'yyyy-mm-dd'})
                fmt_usd = wb.add_format({'num_format': '$#,##0.00'})
                fmt_pct = wb.add_format({'num_format': '0.00%'})
                fmt_hdr = wb.add_format({'bold': True, 'bg_color': '#F0F0F0'})
                fmt_num = wb.add_format({'num_format': '#,##0.00'})

                ws_hist.set_row(0, 20, fmt_hdr)
                col_map = {name: idx for idx, name in enumerate(hist_df.columns)}
                ws_hist.set_column(col_map['Date'], col_map['Date'], 12, fmt_date)
                if 'Portfolio Value' in col_map:
                    ws_hist.set_column(col_map['Portfolio Value'], col_map['Portfolio Value'], 16, fmt_usd)
                if 'Benchmark Price' in col_map:
                    ws_hist.set_column(col_map['Benchmark Price'], col_map['Benchmark Price'], 16, fmt_usd)
                for cname in hist_df.columns:
                    if cname in ('Portfolio Return', 'Drawdown', 'Benchmark Return') or cname.startswith('Weight:') or cname == 'Turnover':
                        ws_hist.set_column(col_map[cname], col_map[cname], 14, fmt_pct)
                    elif cname == 'Tx Costs':
                        ws_hist.set_column(col_map[cname], col_map[cname], 14, fmt_usd)
                    elif cname not in ('Date', 'Portfolio Value', 'Benchmark Price'):
                        ws_hist.set_column(col_map[cname], col_map[cname], 14, fmt_num)
                ws_hist.freeze_panes(1, 1)
                ws_hist.autofilter(0, 0, len(hist_df), len(hist_df.columns) - 1)

                # Sheet 2: Metrics
                ws_met = wb.add_worksheet('Metrics')
                ws_met.write(0, 0, "Metric", fmt_hdr)
                ws_met.write(0, 1, "Value", fmt_hdr)
                fmt_plain4 = wb.add_format({'num_format': '0.0000'})
                fmt_pos = wb.add_format({'num_format': '0.00', 'font_color': '#2ca02c'})
                fmt_neg = wb.add_format({'num_format': '0.00', 'font_color': '#d62728'})
                fmt_pct_pos = wb.add_format({'num_format': '0.00%', 'font_color': '#2ca02c'})
                fmt_pct_neg = wb.add_format({'num_format': '0.00%', 'font_color': '#d62728'})
                cur_keys = {'Initial Capital', 'Final Value'}
                pct_keys = {'Total Return', 'CAGR', 'Volatility (ann)', 'Max Drawdown',
                            'Daily VaR 95%', 'Daily CVaR 95%', 'Daily VaR 99%', 'Daily CVaR 99%',
                            'Hit Ratio', 'Best Day', 'Worst Day', 'Alpha (ann) vs Benchmark'}
                for i, (k, v) in enumerate(metrics_items, start=1):
                    ws_met.write_string(i, 0, k)
                    if isinstance(v, (int, float)) and not pd.isna(v):
                        if k in cur_keys:
                            ws_met.write_number(i, 1, float(v), fmt_usd)
                        elif k in pct_keys:
                            ws_met.write_number(i, 1, float(v), fmt_pct_pos if v >= 0 else fmt_pct_neg)
                        elif k in ('Sharpe (ann)', 'Sortino (ann)', 'Calmar', 'Beta vs Benchmark',
                                   'Correlation vs Benchmark', 'R^2 vs Benchmark', 'Skew', 'Kurtosis (excess)'):
                            ws_met.write_number(i, 1, float(v), fmt_pos if v >= 0 else fmt_neg)
                        else:
                            ws_met.write_number(i, 1, float(v), fmt_plain4)
                    else:
                        ws_met.write_string(i, 1, str(v))
                ws_met.set_column(0, 0, 32)
                ws_met.set_column(1, 1, 22)
                ws_met.freeze_panes(1, 0)

                # Sheet 3: Monte Carlo, it appears only if the user generates it
                if self.last_mc:
                    ws_mc = wb.add_worksheet('Monte_Carlo')
                    ws_mc.set_row(0, 20, fmt_hdr)
                    # Bands table
                    ws_mc.write(0, 0, "Percentile Bands (Values)")
                    ws_mc.write(1, 0, "Day", fmt_hdr)
                    ws_mc.write(1, 1, "P05", fmt_hdr)
                    ws_mc.write(1, 2, "P50", fmt_hdr)
                    ws_mc.write(1, 3, "P95", fmt_hdr)
                    T = self.last_mc['T']
                    q05 = self.last_mc['q05']
                    q50 = self.last_mc['q50']
                    q95 = self.last_mc['q95']
                    for i in range(T + 1):
                        ws_mc.write_number(i + 2, 0, i)
                        ws_mc.write_number(i + 2, 1, float(q05[i]), fmt_usd)
                        ws_mc.write_number(i + 2, 2, float(q50[i]), fmt_usd)
                        ws_mc.write_number(i + 2, 3, float(q95[i]), fmt_usd)
                    ws_mc.set_column(0, 0, 8)
                    ws_mc.set_column(1, 3, 16, fmt_usd)

                    # Ending returns table (below with a blank row)
                    start_row = T + 4
                    ws_mc.write(start_row - 1, 0, "Ending Returns")
                    ws_mc.write(start_row, 0, "Return", fmt_hdr)
                    end_rets = self.last_mc['end_rets']
                    fmt_pct = wb.add_format({'num_format': '0.00%'})
                    for i, v in enumerate(end_rets, start=start_row + 1):
                        ws_mc.write_number(i, 0, float(v), fmt_pct)
                    ws_mc.set_column(0, 0, 14, fmt_pct)

                    # Summary (to the right)
                    col = 5
                    ws_mc.write(0, col, "Monte Carlo Summary", fmt_hdr)
                    ws_mc.write(1, col, "Model")
                    ws_mc.write(1, col+1, str(self.last_mc['model']))
                    ws_mc.write(2, col, "# Paths")
                    ws_mc.write(2, col+1, int(self.last_mc['n_paths']))
                    ws_mc.write(3, col, "Years")
                    ws_mc.write(3, col+1, float(self.last_mc['years']))
                    ws_mc.write(4, col, "Start Value")
                    ws_mc.write_number(4, col+1, float(self.last_mc['start_value']), fmt_usd)
                    ws_mc.write(5, col, "E[Ending Return]")
                    ws_mc.write_number(5, col+1, float(self.last_mc['exp_return']), fmt_pct)
                    ws_mc.write(6, col, "Median Return")
                    ws_mc.write_number(6, col+1, float(self.last_mc['med_return']), fmt_pct)
                    ws_mc.write(7, col, "VaR95")
                    ws_mc.write_number(7, col+1, float(self.last_mc['var95']), fmt_pct)
                    ws_mc.write(8, col, "CVaR95")
                    ws_mc.write_number(8, col+1, float(self.last_mc['cvar95']), fmt_pct)
                    ws_mc.write(9, col, "P(Loss)")
                    ws_mc.write_number(9, col+1, float(self.last_mc['prob_loss']), fmt_pct)

            messagebox.showinfo("Export to Excel", f"Exported to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export to Excel", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = BacktestApp(root)

    root.mainloop()

