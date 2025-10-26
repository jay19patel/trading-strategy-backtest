# Crypto Backtesting Platform

A powerful Flask-based web application for backtesting cryptocurrency trading strategies with realistic margin trading simulation.

## Features

- 🚀 **Realistic Margin Trading**: Simulate leverage trading with up to 125x leverage
- 📊 **Advanced Analytics**: Comprehensive performance metrics and statistics
- 📈 **Visual Charts**: Balance growth curves and trade entry/exit visualizations
- 💾 **CSV Export**: Download backtest results for further analysis
- ⚙️ **Fully Customizable**: Adjust all trading parameters via UI
- 🎯 **Risk Management**: Configurable stop loss, take profit, and risk:reward ratios
- 📉 **Multiple Timeframes**: Support for various candle intervals and data periods

## Installation

### Using UV (Recommended)

1. Install UV if you haven't already:
```bash
pip install uv
```

2. Clone or download this project

3. Install dependencies:
```bash
uv sync
```

4. Run the application:
```bash
uv run app.py
```

### Traditional Python

1. Install dependencies:
```bash
pip install flask yfinance pandas-ta matplotlib numpy plotly tabulate
```

2. Run the application:
```bash
python app.py
```

## Usage

1. Open your browser and navigate to `http://localhost:5000`
2. Fill in the trading parameters:
   - **Symbol**: Enter the trading pair (e.g., BTCUSDT, ETHUSDT)
   - **Strategy Name**: Name your strategy
   - **Initial Balance**: Starting capital in USDT
   - **Leverage**: Trading leverage (1-125x)
   - **Risk:Reward**: Risk to reward ratio (e.g., 1:2)
   - **Stop Loss**: Maximum loss percentage
   - **Margin per Trade**: Percentage of balance to use per trade
   - **Max Holding Time**: Maximum hours to hold a position
   - **Period & Interval**: Data time period and candle intervals

3. Click "Run Backtest" to start the simulation
4. View comprehensive results including:
   - Performance statistics
   - Balance growth chart
   - Trade visualization
   - Complete trade history
   - Download CSV export

## Project Structure

```
Backtesting/
├── app.py                 # Main Flask application
├── trader.py              # CryptoMarginTrader class
├── templates/             # HTML templates
│   ├── index.html        # Main form
│   └── results.html      # Results page
├── static/               # Static files
│   └── css/
│       └── style.css     # Styling
├── results/              # Backtest results (CSV files)
└── README.md            # This file
```

## Trading Parameters Explained

- **Symbol**: Trading pair identifier (e.g., BTCUSDT, ETHUSDT)
- **Leverage**: Multiplier for position size (increases both profit and risk)
- **Risk:Reward**: Ratio defining how much to risk vs potential reward
- **Stop Loss**: Maximum loss percentage before position closes
- **Margin per Trade**: Percentage of available balance used per trade
- **Max Holding Time**: Time limit for holding positions
- **Trading Fee**: Platform trading fee percentage

## How It Works

1. Data is downloaded from Yahoo Finance using the specified symbol, period, and interval
2. Technical indicators are calculated (EMAs, candlestick patterns)
3. Buy/sell signals are generated based on EMA crossovers
4. The trader simulates opening and closing positions with:
   - Position sizing based on margin and leverage
   - Stop loss and take profit calculations
   - Trading fee deduction
   - Time-based exits
   - Liquidation checks
5. Results are displayed with analytics, charts, and downloadable CSV

## Technologies Used

- **Flask**: Web framework
- **yfinance**: Market data fetching
- **pandas-ta**: Technical analysis
- **matplotlib**: Chart generation
- **numpy**: Numerical computations

## License

MIT License - Feel free to use and modify as needed.

## Support

For issues or questions, please open an issue on the repository.

