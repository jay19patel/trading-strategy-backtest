# Quick Start Guide

## Running the Application

### Method 1: Using UV (Recommended)
```bash
uv run app.py
```

### Method 2: Using the shell script
```bash
./run.sh
```

### Method 3: Direct Python
```bash
python app.py
```

## Accessing the UI

Once the server starts, open your browser and go to:
```
http://localhost:5000
```

## How to Use

### Step 1: Enter Trading Parameters

Fill in the form with your trading parameters:

**Basic Info:**
- **Symbol**: Enter a valid symbol (e.g., BTC-USD, ETH-USD, BTCUSDT)
- **Strategy Name**: Give your strategy a name

**Trading Parameters:**
- **Initial Balance**: Starting capital (e.g., 1000 USDT)
- **Leverage**: Trading leverage (1-125x)
- **Period**: Data period (1d, 2d, 5d, 1mo, 3mo)
- **Interval**: Candle interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)

**Risk Management:**
- **Risk:Reward**: Risk to reward ratio (e.g., "1:2" or "1:3")
- **Stop Loss (%)**: Maximum loss percentage (e.g., 2.0%)
- **Margin per Trade (%)**: Percentage of balance per trade (e.g., 30%)
- **Trading Fee (%)**: Platform fee (e.g., 0.1%)

**Time Settings:**
- **Max Holding Time (Hours)**: Maximum time to hold a position

### Step 2: Click "Run Backtest"

Wait for the backtest to complete. This may take 10-30 seconds depending on the amount of data.

### Step 3: Review Results

You'll see:
- **Summary Cards**: Quick stats (Final Balance, Return %, Win Rate, Total Trades)
- **Performance Statistics**: Detailed metrics
- **Balance Growth Chart**: Visual account balance over time
- **Trade Visualization**: Entry/exit points on a chart
- **Trade History Table**: Complete list of all trades

### Step 4: Download Results

Click "Download CSV" to save the results to your computer.

## Example Parameters

**Conservative Strategy:**
- Initial Balance: 1000
- Leverage: 10x
- Risk:Reward: 1:3
- Stop Loss: 3%
- Margin per Trade: 20%
- Max Hold: 24 hours

**Aggressive Strategy:**
- Initial Balance: 1000
- Leverage: 50x
- Risk:Reward: 1:2
- Stop Loss: 2%
- Margin per Trade: 40%
- Max Hold: 6 hours

## Available Cryptocurrency Symbols

Popular symbols that work:
- **BTC-USD**: Bitcoin
- **ETH-USD**: Ethereum
- **BNB-USD**: Binance Coin
- **ADA-USD**: Cardano
- **SOL-USD**: Solana
- **DOGE-USD**: Dogecoin

For crypto pairs (USDT), use the exchange format like:
- BTCUSDT (for Binance)
- ETHUSDT (for Binance)

Note: Not all symbols are available on Yahoo Finance. Use standard USD pairs for best compatibility.

## Troubleshooting

**No Data Found:**
- Try a different symbol format (e.g., BTC-USD instead of BTCUSDT)
- Try a longer period (5d instead of 1d)
- Try a different interval (1h instead of 15m)

**No Trades Executed:**
- Lower the stop loss percentage
- Reduce the margin per trade percentage
- Increase the data period to get more signals
- Try a more volatile symbol

**Application Won't Start:**
- Make sure all dependencies are installed: `uv sync`
- Check that Python 3.13+ is installed
- Check the console for error messages

## Understanding the Results

**Win Rate**: Percentage of profitable trades
**Max Drawdown**: Largest peak-to-trough decline
**Sharpe Ratio**: Risk-adjusted return (if shown)
**Total Fees**: Sum of all trading fees paid
**Avg Holding Time**: Average time before exiting

Green values indicate profit, red values indicate loss.

