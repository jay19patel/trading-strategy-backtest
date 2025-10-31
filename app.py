from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas_ta as ta
import os
import json
import base64
from io import BytesIO
from trader import CryptoMarginTrader

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'results'

# Ensure results directory exists
os.makedirs('results', exist_ok=True)
os.makedirs('saved_strategies', exist_ok=True)
os.makedirs('saved_backtests', exist_ok=True)

# File paths
STRATEGIES_FILE = 'saved_strategies/strategies.json'
BACKTESTS_FILE = 'saved_backtests/backtests.json'


def download_and_process_data(symbol, period, interval):
    """Download and process market data"""
    # Download data
    df = yf.download(symbol, period=period, interval=interval)

    # Remove Ticker row (if present) by resetting columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df["DateTime"] = df.index
    # Convert index to Asia/Kolkata timezone if timezone-aware
    if df.index.tz is not None:
        df.index = df.index.tz_convert('Asia/Kolkata')
    df['Date'] = df.index.date
    df['Time'] = df.index.strftime('%I:%M %p')  # AM/PM format

    # Calculate fixed EMAs (9 and 15) for strategy
    ema_list = [9, 15]
    for ema_length in ema_list:
        ema_name = f'{ema_length}EMA'
        df[ema_name] = ta.ema(df['Close'], length=ema_length)

    # No RSI needed for fixed strategy

    df['Candle'] = df.apply(lambda row: 'Green' if row['Close'] >= row['Open'] else 'Red', axis=1)

    # Calculate body and shadows
    Body = abs(df['Close'] - df['Open'])
    Upper_Shadow = df['High'] - df[['Close', 'Open']].max(axis=1)
    Lower_Shadow = df[['Close', 'Open']].min(axis=1) - df['Low']
    Total_Range = df['High'] - df['Low']

    # Calculate body percentage
    df['Body'] = (Body / Total_Range) * 100

    # Calculate shadow %
    df['Upper_Shadow'] = (Upper_Shadow / Total_Range) * 100
    df['Lower_Shadow'] = (Lower_Shadow / Total_Range) * 100

    # Rolling average of previous 5 candles
    SEMA = 5
    df['Avg_Upper_Shadow'] = df['Upper_Shadow'].rolling(window=SEMA, min_periods=1).mean()
    df['Avg_Lower_Shadow'] = df['Lower_Shadow'].rolling(window=SEMA, min_periods=1).mean()
    df["ALUS"] = df['Avg_Lower_Shadow'] / df['Avg_Upper_Shadow']

    return df


def apply_fixed_strategy_conditions(df):
    """Apply fixed strategy conditions"""
    df['Action'] = None
    
    # Ensure required columns exist
    if '9EMA' not in df.columns or '15EMA' not in df.columns:
        return df
    
    # Create shifted columns to avoid NaN issues
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_9EMA'] = df['9EMA'].shift(1)
    
    # LONG Entry Conditions (All must be true):
    # 1. 15 EMA > 9 EMA
    # 2. Previous High < Current High
    # 3. Current High > 9 EMA
    
    long_condition_1 = df['15EMA'] > df['9EMA']
    long_condition_2 = df['Prev_High'] < df['High']
    long_condition_3 = df['High'] > df['9EMA']
    
    long_entry = long_condition_1 & long_condition_2 & long_condition_3
    
    # SHORT Entry Conditions (All must be true):
    # 1. Previous 9 EMA > 15 EMA
    # 2. Current Close < Previous Close
    # 3. Current Close < 9 EMA
    
    short_condition_1 = df['Prev_9EMA'] > df['15EMA']
    short_condition_2 = df['Close'] < df['Prev_Close']
    short_condition_3 = df['Close'] < df['9EMA']
    
    short_entry = short_condition_1 & short_condition_2 & short_condition_3
    
    # Set actions (fillna to avoid issues)
    df.loc[long_entry.fillna(False), 'Action'] = 'buy'
    df.loc[short_entry.fillna(False), 'Action'] = 'sell'
    
    return df


def parse_and_apply_conditions(df, long_conditions, short_conditions, long_logic='AND', short_logic='AND'):
    """Parse conditions and apply them to dataframe with AND/OR logic"""
    def get_series(df, field, shift=0, period=None):
        """Get series with optional shift for previous candles"""
        if field in ['Close', 'High', 'Low']:
            series = df[field]
        elif field == 'EMA':
            series = df[f'{period}EMA']
        elif field == 'RSI':
            rsi_column = f'RSI{period}'
            if rsi_column in df.columns:
                series = df[rsi_column]
            else:
                series = df.get('RSI14', pd.Series())
        
        # Apply shift for previous candles
        if shift > 0:
            series = series.shift(shift)
        return series
    
    def apply_condition_list(df, conditions, logic='AND'):
        """Apply multiple conditions with AND/OR logic"""
        if not conditions or len(conditions) == 0:
            return pd.Series([False] * len(df))
        
        results = []
        
        for condition in conditions:
            field = condition.get('field')
            shift = int(condition.get('shift', 0))
            operator = condition.get('operator')
            compare_to = condition.get('compare_to', 'value')
            value = condition.get('value')
            
            # Get left side series
            if field == 'EMA':
                ema_period = condition.get('ema_period', '15')
                left_series = get_series(df, 'EMA', shift, ema_period)
            elif field == 'RSI':
                rsi_period = condition.get('rsi_period', '14')
                left_series = get_series(df, 'RSI', shift, rsi_period)
            else:
                left_series = get_series(df, field, shift)
            
            # Get right side series
            if compare_to == 'value':
                if operator == 'gt':
                    result = left_series > float(value)
                elif operator == 'gte':
                    result = left_series >= float(value)
                elif operator == 'lt':
                    result = left_series < float(value)
                elif operator == 'lte':
                    result = left_series <= float(value)
                elif operator == 'eq':
                    result = left_series == float(value)
                else:
                    result = pd.Series([False] * len(df))
            elif compare_to == 'Close':
                right_series = get_series(df, 'Close', 0)
                if operator == 'gt':
                    result = left_series > right_series
                elif operator == 'gte':
                    result = left_series >= right_series
                elif operator == 'lt':
                    result = left_series < right_series
                elif operator == 'lte':
                    result = left_series <= right_series
                elif operator == 'eq':
                    result = left_series == right_series
                else:
                    result = pd.Series([False] * len(df))
            elif compare_to == 'EMA':
                compare_ema_period = condition.get('compare_ema_period', '15')
                right_series = get_series(df, 'EMA', 0, compare_ema_period)
                if operator == 'gt':
                    result = left_series > right_series
                elif operator == 'gte':
                    result = left_series >= right_series
                elif operator == 'lt':
                    result = left_series < right_series
                elif operator == 'lte':
                    result = left_series <= right_series
                elif operator == 'eq':
                    result = left_series == right_series
                else:
                    result = pd.Series([False] * len(df))
            elif compare_to == 'RSI':
                compare_rsi_period = condition.get('compare_rsi_period', '14')
                right_series = get_series(df, 'RSI', 0, compare_rsi_period)
                if operator == 'gt':
                    result = left_series > right_series
                elif operator == 'gte':
                    result = left_series >= right_series
                elif operator == 'lt':
                    result = left_series < right_series
                elif operator == 'lte':
                    result = left_series <= right_series
                elif operator == 'eq':
                    result = left_series == right_series
                else:
                    result = pd.Series([False] * len(df))
            else:
                continue
            
            results.append(result)
        
        # Combine conditions based on logic
        if not results:
            return pd.Series([False] * len(df))
        
        if logic == 'AND':
            combined = results[0]
            for r in results[1:]:
                combined = combined & r
        else:  # OR
            combined = results[0]
            for r in results[1:]:
                combined = combined | r
        
        return combined
    
    # Apply LONG conditions
    long_mask = apply_condition_list(df, long_conditions, long_logic)
    
    # Apply SHORT conditions
    short_mask = apply_condition_list(df, short_conditions, short_logic)
    
    # Create Action column
    df['Action'] = None
    df.loc[long_mask, 'Action'] = 'buy'
    df.loc[short_mask, 'Action'] = 'sell'
    
    return df


def create_balance_curve_image(trader):
    """Create balance curve plot with annotations"""
    closed_trades = trader.trade_book[trader.trade_book['Status'] == 'CLOSED'].copy()

    if len(closed_trades) == 0:
        return None

    closed_trades['ExitTime'] = pd.to_datetime(closed_trades['ExitTime'])
    closed_trades = closed_trades.sort_values('ExitTime')
    closed_trades['CumPnL'] = closed_trades['RealizedPnL'].cumsum()
    closed_trades['Balance'] = trader.initial_balance + closed_trades['CumPnL']
    closed_trades['PercentageChange'] = ((closed_trades['Balance'] - trader.initial_balance) / trader.initial_balance) * 100

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(closed_trades['ExitTime'], closed_trades['Balance'],
           linewidth=2, color='#2E86AB', label='Account Balance', zorder=5)
    ax.axhline(y=trader.initial_balance, color='#A23B72', linestyle='--',
              alpha=0.7, label=f'Initial Balance ({trader.initial_balance} USDT)', zorder=3)

    ax.set_title(f'{trader.strategy_name} - Balance Growth Over Time\n{trader.symbol_name} | {trader.leverage}x Leverage',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Balance (USDT)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, zorder=0)

    # Add profit/loss coloring with annotations
    for i in range(len(closed_trades)):
        row = closed_trades.iloc[i]
        color = '#00B894' if row['RealizedPnL'] > 0 else '#E17055'

        # Plot point
        ax.scatter(row['ExitTime'], row['Balance'],
                  color=color, alpha=0.7, s=50, zorder=10,
                  edgecolors='white', linewidth=1.5)

        # Create annotation text: amount (percentage)
        annotation_text = f"${row['Balance']:.2f} ({row['PercentageChange']:+.2f}%)"

        # Alternate annotation position (above/below) to avoid overlap
        y_offset = 15 if i % 2 == 0 else -15

        ax.annotate(annotation_text,
                   xy=(row['ExitTime'], row['Balance']),
                   xytext=(0, y_offset),
                   textcoords='offset points',
                   fontsize=7,
                   fontweight='600',
                   color=color,
                   ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor=color, alpha=0.85, linewidth=1),
                   arrowprops=dict(arrowstyle='->', color=color, lw=1, alpha=0.6),
                   zorder=15)

    plt.tight_layout()

    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return image_base64


def create_trade_visualization_image(trader, df=None):
    """Create trade entry/exit visualization with candlestick background"""
    trades = trader.trade_book[trader.trade_book['Status'] == 'CLOSED'].copy()

    if len(trades) == 0:
        return None

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot candlesticks in background if DataFrame is provided
    if df is not None and len(df) > 0:
        # Filter df to time range of trades
        trades['EntryTime'] = pd.to_datetime(trades['EntryTime'])
        trades['ExitTime'] = pd.to_datetime(trades['ExitTime'])
        min_time = trades['EntryTime'].min()
        max_time = trades['ExitTime'].max()

        # Ensure df has DateTime column
        if 'DateTime' in df.columns:
            df_filtered = df[(df['DateTime'] >= min_time) & (df['DateTime'] <= max_time)].copy()

            # Plot candlesticks with low opacity
            for idx, row in df_filtered.iterrows():
                color = '#28a745' if row['Close'] >= row['Open'] else '#dc3545'

                # Candlestick body
                body_height = abs(row['Close'] - row['Open'])
                body_bottom = min(row['Open'], row['Close'])
                ax.bar(row['DateTime'], body_height, bottom=body_bottom,
                      width=pd.Timedelta(minutes=1), color=color, alpha=0.8,
                      edgecolor=color, linewidth=0.5)

                # Upper wick
                ax.plot([row['DateTime'], row['DateTime']],
                       [max(row['Open'], row['Close']), row['High']],
                       color=color, linewidth=0.8, alpha=0.8)

                # Lower wick
                ax.plot([row['DateTime'], row['DateTime']],
                       [row['Low'], min(row['Open'], row['Close'])],
                       color=color, linewidth=0.8, alpha=0.8)

    # Plot trade lines on top
    for i, (idx, row) in enumerate(trades.iterrows()):
        color = 'green' if row['RealizedPnL'] > 0 else 'red'
        entry_time = pd.to_datetime(row['EntryTime'])
        exit_time = pd.to_datetime(row['ExitTime'])

        ax.plot(
            [entry_time, exit_time],
            [row['EntryPrice'], row['ExitPrice']],
            marker='o',
            color=color,
            linewidth=2.5,
            markersize=10,
            alpha=0.9,
            zorder=10
        )

        # Annotate profit/loss
        ax.text(exit_time, row['ExitPrice'],
               f"{row['RealizedPnL']:+.2f}",
               fontsize=8,
               color=color,
               fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
               zorder=11)

    ax.set_title('Trade Entry & Exit Visualization with Candlestick Background', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.grid(True, alpha=0.3, zorder=0)
    fig.autofmt_xdate()
    plt.tight_layout()

    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return image_base64


@app.route('/')
def index():
    """Home page with backtest form"""
    return render_template('index.html')


@app.route('/backtest', methods=['POST'])
def run_backtest():
    """Process backtest request"""
    try:
        # Get form parameters
        symbol_input = request.form.get('symbol', 'BTC-USD').upper()
        
        # Convert symbol format for yfinance
        # If it's like BTCUSDT, convert to BTC-USD
        if len(symbol_input) >= 6 and symbol_input[-4:] == 'USDT':
            symbol = symbol_input[:-4] + '-USD'
        else:
            symbol = symbol_input
        initial_balance = float(request.form.get('initial_balance', 1000))
        max_hold_hours = float(request.form.get('max_hold_hours', 12))
        risk_reward = request.form.get('risk_reward', '1:2')
        stop_loss = float(request.form.get('stop_loss', 2.0))
        leverage = int(request.form.get('leverage', 50))
        trading_fee = float(request.form.get('trading_fee', 1.0))
        margin_per_trade = float(request.form.get('margin_per_trade', 30))
        period = request.form.get('period', '2d')
        interval = request.form.get('interval', '15m')
        strategy_name = request.form.get('strategy_name', 'Scalping Strategy')
        
        # Convert max hold hours to minutes
        max_hold_minutes = max_hold_hours * 60

        # Download and process data
        df = download_and_process_data(symbol, period, interval)
        
        # Apply fixed strategy conditions
        df = apply_fixed_strategy_conditions(df)

        if df.empty:
            return jsonify({'error': 'No data available for this symbol'}), 400

        # Create trader and run backtest
        trader = CryptoMarginTrader(
            initial_balance=initial_balance,
            maxHoldMinutes=max_hold_minutes,
            RiskToReward=risk_reward,
            symbol_name=symbol,
            strategy_name=strategy_name,
            stoploss_percentage=stop_loss,
            leverage=leverage,
            trading_fee=trading_fee
        )

        trader.backtest(df, margin_per_trade=margin_per_trade)

        # Get stats
        stats = trader.get_performance_stats()

        if not stats:
            return jsonify({'error': 'No trades executed'}), 400

        # Create images
        balance_curve = create_balance_curve_image(trader)
        trade_viz = create_trade_visualization_image(trader, df)

        # Do not save anything on run; only generate a timestamp for display
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Convert trades to dict for JSON
        trades_df = trader.trade_book.copy()
        trades_df['EntryTime'] = trades_df['EntryTime'].astype(str)
        trades_df['ExitTime'] = trades_df['ExitTime'].astype(str)
        trades_df = trades_df.replace({np.nan: None})
        trades_data = trades_df.to_dict('records')

        # Combine all results (no csv filename)
        results = {
            'stats': stats,
            'balance_curve': balance_curve,
            'trade_visualization': trade_viz,
            'trades': trades_data,
            'timestamp': timestamp
        }

        return render_template('results.html', **results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history')
def view_history():
    """Display list of all historical backtests (CSV and saved)"""
    try:
        # Get all CSV files in results directory
        files = []
        if os.path.exists('results'):
            for filename in os.listdir('results'):
                if filename.startswith('backtest_') and filename.endswith('.csv'):
                    # Parse filename: backtest_SYMBOL_TIMESTAMP.csv
                    # Remove 'backtest_' prefix and '.csv' suffix
                    clean_name = filename.replace('backtest_', '').replace('.csv', '')
                    # Split from the right to handle symbols with dashes (e.g., BTC-USD, SHIB-USD)
                    parts = clean_name.rsplit('_', 2)
                    if len(parts) == 3:
                        symbol = parts[0]
                        date_str = parts[1]
                        time_str = parts[2]
                        
                        # Format date and time
                        date_obj = datetime.strptime(date_str + '_' + time_str, '%Y%m%d_%H%M%S')
                        
                        # Read CSV to get stats
                        try:
                            df = pd.read_csv(f'results/{filename}')
                            closed_trades = df[df['Status'] == 'CLOSED']
                            
                            if len(closed_trades) > 0:
                                total_trades = len(closed_trades)
                                winning_trades = len(closed_trades[closed_trades['RealizedPnL'] > 0])
                                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                                total_pnl = closed_trades['RealizedPnL'].sum()
                                
                                files.append({
                                    'filename': filename,
                                    'symbol': symbol,
                                    'date': date_obj.strftime('%Y-%m-%d'),
                                    'time': date_obj.strftime('%H:%M:%S'),
                                    'total_trades': total_trades,
                                    'win_rate': round(win_rate, 2),
                                    'total_pnl': round(total_pnl, 2),
                                    'type': 'csv'
                                })
                        except:
                            pass
        
        # Get saved backtests
        saved_backtests = []
        if os.path.exists(BACKTESTS_FILE):
            try:
                with open(BACKTESTS_FILE, 'r') as f:
                    backtests = json.load(f)
                
                for bt in backtests:
                    settings = bt.get('trading_settings', {})
                    stats = bt.get('stats', {})
                    saved_at = bt.get('saved_at', '')
                    
                    try:
                        if saved_at:
                            date_obj = datetime.fromisoformat(saved_at.replace('Z', '+00:00'))
                            if date_obj.tzinfo:
                                date_obj = date_obj.replace(tzinfo=None)
                    except:
                        date_obj = datetime.now()
                    
                    saved_backtests.append({
                        'filename': '',  # No filename for saved backtests
                        'backtest_id': bt.get('id'),
                        'symbol': settings.get('symbol', 'N/A'),
                        'date': date_obj.strftime('%Y-%m-%d') if isinstance(date_obj, datetime) else saved_at[:10] if saved_at else '',
                        'time': date_obj.strftime('%H:%M:%S') if isinstance(date_obj, datetime) else saved_at[11:19] if saved_at and len(saved_at) > 11 else '',
                        'total_trades': stats.get('total_trades', 0),
                        'win_rate': round(stats.get('win_rate', 0), 2),
                        'total_pnl': round(stats.get('total_pnl', 0), 2),
                        'type': 'saved'
                    })
            except:
                pass
        
        # Combine and sort by date and time (most recent first)
        all_backtests = files + saved_backtests
        all_backtests.sort(key=lambda x: (x['date'], x['time']), reverse=True)
        
        return render_template('history.html', backtests=all_backtests)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_dataframe_info', methods=['POST'])
def get_dataframe_info():
    """Get DataFrame columns and last 3 rows preview"""
    try:
        symbol_input = request.json.get('symbol', 'BTC-USD').upper()

        # Convert symbol format
        if len(symbol_input) >= 6 and symbol_input[-4:] == 'USDT':
            symbol = symbol_input[:-4] + '-USD'
        else:
            symbol = symbol_input

        period = request.json.get('period', '5d')
        interval = request.json.get('interval', '15m')

        # Download and process data
        df = download_and_process_data(symbol, period, interval)

        if df.empty:
            return jsonify({'error': 'No data available'}), 400

        # Get column names
        columns = df.columns.tolist()

        # Get last 3 rows
        last_3_rows = df.tail(3).copy()

        # Convert datetime columns to string
        for col in last_3_rows.columns:
            if pd.api.types.is_datetime64_any_dtype(last_3_rows[col]):
                last_3_rows[col] = last_3_rows[col].astype(str)

        # Replace NaN with None for JSON serialization
        last_3_rows = last_3_rows.replace({np.nan: None})

        # Convert to dict
        preview_data = last_3_rows.to_dict('records')

        return jsonify({
            'columns': columns,
            'preview': preview_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/save_custom_strategy', methods=['POST'])
def save_custom_strategy():
    """Save custom strategy with trading settings"""
    try:
        strategy_code = request.json.get('strategy_code', '')
        strategy_name = request.json.get('strategy_name', 'Unnamed Strategy')
        trading_settings = request.json.get('trading_settings', {})

        if not strategy_code:
            return jsonify({'error': 'No strategy code provided'}), 400

        # Load existing strategies
        strategies = []
        if os.path.exists(STRATEGIES_FILE):
            try:
                with open(STRATEGIES_FILE, 'r') as f:
                    strategies = json.load(f)
            except:
                strategies = []

        # Create new strategy entry
        strategy_entry = {
            'id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'name': strategy_name,
            'code': strategy_code,
            'settings': trading_settings,
            'created_at': datetime.now().isoformat()
        }

        # Add or update strategy
        strategies.append(strategy_entry)

        # Save to JSON file
        with open(STRATEGIES_FILE, 'w') as f:
            json.dump(strategies, f, indent=2)

        # Also save to txt file for backward compatibility
        strategy_file = 'custom_strategy.txt'
        with open(strategy_file, 'w') as f:
            f.write(strategy_code)

        return jsonify({'success': True, 'message': 'Strategy saved successfully', 'strategy_id': strategy_entry['id']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/load_custom_strategy', methods=['GET'])
def load_custom_strategy():
    """Load saved custom strategy code - returns list of strategies"""
    try:
        # Load from JSON file
        strategies = []
        if os.path.exists(STRATEGIES_FILE):
            try:
                with open(STRATEGIES_FILE, 'r') as f:
                    strategies = json.load(f)
            except:
                strategies = []

        # If no JSON strategies, try to load from txt file for backward compatibility
        if not strategies:
            strategy_file = 'custom_strategy.txt'
            if os.path.exists(strategy_file):
                with open(strategy_file, 'r') as f:
                    strategy_code = f.read()
                return jsonify({
                    'success': True,
                    'strategy_code': strategy_code,
                    'strategies': [],
                    'legacy': True
                })

        return jsonify({'success': True, 'strategies': strategies})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/load_strategy/<strategy_id>', methods=['GET'])
def load_strategy_by_id(strategy_id):
    """Load a specific strategy by ID"""
    try:
        if not os.path.exists(STRATEGIES_FILE):
            return jsonify({'error': 'No strategies found'}), 404

        with open(STRATEGIES_FILE, 'r') as f:
            strategies = json.load(f)

        strategy = next((s for s in strategies if s['id'] == strategy_id), None)
        if not strategy:
            return jsonify({'error': 'Strategy not found'}), 404

        return jsonify({
            'success': True,
            'strategy_code': strategy['code'],
            'settings': strategy.get('settings', {}),
            'name': strategy.get('name', 'Unnamed Strategy')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/backtest_custom', methods=['POST'])
def run_backtest_custom():
    """Run backtest with custom strategy"""
    try:
        # Get form parameters
        symbol_input = request.form.get('symbol', 'BTC-USD').upper()

        # Convert symbol format
        if len(symbol_input) >= 6 and symbol_input[-4:] == 'USDT':
            symbol = symbol_input[:-4] + '-USD'
        else:
            symbol = symbol_input

        initial_balance = float(request.form.get('initial_balance', 1000))
        max_hold_hours = float(request.form.get('max_hold_hours', 12))
        risk_reward = request.form.get('risk_reward', '1:2')
        stop_loss = float(request.form.get('stop_loss', 2.0))
        leverage = int(request.form.get('leverage', 50))
        trading_fee = float(request.form.get('trading_fee', 1.0))
        margin_per_trade = float(request.form.get('margin_per_trade', 30))
        period = request.form.get('period', '2d')
        interval = request.form.get('interval', '15m')
        strategy_name = request.form.get('strategy_name', 'Custom Strategy')
        strategy_code = request.form.get('strategy_code', '')

        max_hold_minutes = max_hold_hours * 60

        # Download and process data
        df = download_and_process_data(symbol, period, interval)

        if df.empty:
            return jsonify({'error': 'No data available'}), 400

        # Apply custom strategy if provided, otherwise use fixed
        if strategy_code:
            try:
                # Create a safe namespace for executing strategy
                namespace = {
                    'df': df,
                    'pd': pd,
                    'np': np,
                    'ta': ta
                }

                # Execute the custom strategy code
                exec(strategy_code, namespace)
                df = namespace['df']

                # Validate that Action column exists
                if 'Action' not in df.columns:
                    return jsonify({'error': 'Strategy code must create an "Action" column in the DataFrame'}), 400
            except Exception as e:
                return jsonify({'error': f'Strategy execution error: {str(e)}'}), 400
        else:
            # Use fixed strategy
            df = apply_fixed_strategy_conditions(df)

        # Create trader and run backtest
        trader = CryptoMarginTrader(
            initial_balance=initial_balance,
            maxHoldMinutes=max_hold_minutes,
            RiskToReward=risk_reward,
            symbol_name=symbol,
            strategy_name=strategy_name,
            stoploss_percentage=stop_loss,
            leverage=leverage,
            trading_fee=trading_fee
        )

        trader.backtest(df, margin_per_trade=margin_per_trade)

        # Get stats
        stats = trader.get_performance_stats()

        if not stats:
            return jsonify({'error': 'No trades executed'}), 400

        # Create images
        balance_curve = create_balance_curve_image(trader)
        trade_viz = create_trade_visualization_image(trader, df)

        # Do not save anything on run; only generate a timestamp for display
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Convert trades to dict
        trades_df = trader.trade_book.copy()
        trades_df['EntryTime'] = trades_df['EntryTime'].astype(str)
        trades_df['ExitTime'] = trades_df['ExitTime'].astype(str)
        trades_df = trades_df.replace({np.nan: None})
        trades_data = trades_df.to_dict('records')

        # Prepare trading settings
        trading_settings = {
            'symbol': symbol,
            'initial_balance': initial_balance,
            'max_hold_hours': max_hold_hours,
            'risk_reward': risk_reward,
            'stop_loss': stop_loss,
            'leverage': leverage,
            'trading_fee': trading_fee,
            'margin_per_trade': margin_per_trade,
            'period': period,
            'interval': interval,
            'strategy_name': strategy_name
        }

        results = {
            'stats': stats,
            'balance_curve': balance_curve,
            'trade_visualization': trade_viz,
            'trades': trades_data,
            'timestamp': timestamp,
            'strategy_code': strategy_code,
            'trading_settings': trading_settings,
            'backtest_id': timestamp  # Use timestamp as unique ID
        }

        return render_template('results.html', **results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/save_backtest', methods=['POST'])
def save_backtest():
    """Save backtest results with settings, code, results, and pandas data"""
    try:
        data = request.json
        
        # Get all required data
        strategy_code = data.get('strategy_code', '')
        trading_settings = data.get('trading_settings', {})
        trades = data.get('trades', [])
        stats = data.get('stats', {})
        balance_curve = data.get('balance_curve', '')
        trade_visualization = data.get('trade_visualization', '')
        backtest_id = data.get('backtest_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        # Load existing backtests
        backtests = []
        if os.path.exists(BACKTESTS_FILE):
            try:
                with open(BACKTESTS_FILE, 'r') as f:
                    backtests = json.load(f)
            except:
                backtests = []
        
        # Create backtest entry
        backtest_entry = {
            'id': backtest_id,
            'strategy_code': strategy_code,
            'trading_settings': trading_settings,
            'trades': trades,
            'stats': stats,
            'balance_curve': balance_curve,
            'trade_visualization': trade_visualization,
            'saved_at': datetime.now().isoformat()
        }
        
        # Check if backtest with same ID exists and update it
        existing_index = next((i for i, b in enumerate(backtests) if b.get('id') == backtest_id), None)
        if existing_index is not None:
            backtests[existing_index] = backtest_entry
        else:
            backtests.append(backtest_entry)
        
        # Save to JSON file
        with open(BACKTESTS_FILE, 'w') as f:
            json.dump(backtests, f, indent=2)
        
        return jsonify({'success': True, 'message': 'Backtest saved successfully', 'backtest_id': backtest_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_saved_backtests', methods=['GET'])
def get_saved_backtests():
    """Get list of all saved backtests"""
    try:
        backtests = []
        if os.path.exists(BACKTESTS_FILE):
            try:
                with open(BACKTESTS_FILE, 'r') as f:
                    backtests = json.load(f)
            except:
                backtests = []
        
        # Return only summary info for list view
        summaries = []
        for bt in backtests:
            settings = bt.get('trading_settings', {})
            stats = bt.get('stats', {})
            summaries.append({
                'id': bt.get('id'),
                'symbol': settings.get('symbol', 'N/A'),
                'strategy_name': settings.get('strategy_name', 'N/A'),
                'date': bt.get('saved_at', '')[:10] if bt.get('saved_at') else '',
                'time': bt.get('saved_at', '')[11:19] if bt.get('saved_at') else '',
                'total_trades': stats.get('total_trades', 0),
                'win_rate': stats.get('win_rate', 0),
                'total_pnl': stats.get('total_pnl', 0)
            })
        
        return jsonify({'success': True, 'backtests': summaries})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/view_saved_backtest/<backtest_id>')
def view_saved_backtest(backtest_id):
    """View a saved backtest with full details"""
    try:
        if not os.path.exists(BACKTESTS_FILE):
            return jsonify({'error': 'No saved backtests found'}), 404
        
        with open(BACKTESTS_FILE, 'r') as f:
            backtests = json.load(f)
        
        backtest = next((b for b in backtests if b.get('id') == backtest_id), None)
        if not backtest:
            return jsonify({'error': 'Backtest not found'}), 404
        
        # Render results template with saved data
        return render_template('results.html',
                            stats=backtest.get('stats', {}),
                            balance_curve=backtest.get('balance_curve', ''),
                            trade_visualization=backtest.get('trade_visualization', ''),
                            trades=backtest.get('trades', []),
                            csv_filename='',  # No CSV for saved backtests
                            timestamp=backtest.get('saved_at', ''),
                            strategy_code=backtest.get('strategy_code', ''),
                            trading_settings=backtest.get('trading_settings', {}),
                            backtest_id=backtest_id,
                            is_saved=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/view_backtest/<filename>')
def view_backtest(filename):
    """View a specific backtest from history"""
    try:
        # Load CSV file
        csv_path = f'results/{filename}'
        if not os.path.exists(csv_path):
            return jsonify({'error': 'Backtest not found'}), 404
        
        df = pd.read_csv(csv_path)
        closed_trades = df[df['Status'] == 'CLOSED'].copy()
        
        if len(closed_trades) == 0:
            return jsonify({'error': 'No completed trades in this backtest'}), 400
        
        # Parse filename to get metadata
        clean_name = filename.replace('backtest_', '').replace('.csv', '')
        parts = clean_name.rsplit('_', 2)
        symbol = parts[0]
        timestamp = parts[1] + '_' + parts[2]
        
        # Extract metadata from first trade
        first_trade = closed_trades.iloc[0]
        symbol_name = first_trade['Symbol']
        
        # Calculate stats
        total_trades = len(closed_trades)
        winning_trades = len(closed_trades[closed_trades['RealizedPnL'] > 0])
        losing_trades = len(closed_trades[closed_trades['RealizedPnL'] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = closed_trades['RealizedPnL'].sum()
        avg_win = closed_trades[closed_trades['RealizedPnL'] > 0]['RealizedPnL'].mean() if winning_trades > 0 else 0
        avg_loss = closed_trades[closed_trades['RealizedPnL'] <= 0]['RealizedPnL'].mean() if losing_trades > 0 else 0
        max_win = closed_trades['RealizedPnL'].max()
        max_loss = closed_trades['RealizedPnL'].min()
        
        # Get balance from trade history
        initial_balance = closed_trades['AvailableBalance'].iloc[0] - closed_trades['RealizedPnL'].iloc[0]
        final_balance = closed_trades['AvailableBalance'].iloc[-1]
        total_return_pct = ((final_balance - initial_balance) / initial_balance) * 100 if initial_balance > 0 else 0
        
        # Drawdown calculation
        closed_trades_sorted = closed_trades.sort_values('ExitTime')
        closed_trades_sorted['CumPnL'] = closed_trades_sorted['RealizedPnL'].cumsum()
        closed_trades_sorted['Balance'] = initial_balance + closed_trades_sorted['CumPnL']
        closed_trades_sorted['Peak'] = closed_trades_sorted['Balance'].cummax()
        closed_trades_sorted['Drawdown'] = ((closed_trades_sorted['Balance'] - closed_trades_sorted['Peak']) / closed_trades_sorted['Peak']) * 100
        max_drawdown = closed_trades_sorted['Drawdown'].min()
        
        # Create balance curve image
        fig, ax = plt.subplots(figsize=(14, 8))
        closed_trades_sorted['ExitTime'] = pd.to_datetime(closed_trades_sorted['ExitTime'])
        closed_trades_sorted = closed_trades_sorted.sort_values('ExitTime')
        
        ax.plot(closed_trades_sorted['ExitTime'], closed_trades_sorted['Balance'],
               linewidth=2, color='#2E86AB', label='Account Balance')
        ax.axhline(y=initial_balance, color='#A23B72', linestyle='--',
                  alpha=0.7, label=f'Initial Balance ({initial_balance:.2f} USDT)')
        
        ax.set_title(f'Historical Backtest - Balance Growth Over Time\n{symbol_name}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Balance (USDT)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        for i in range(len(closed_trades_sorted)):
            color = '#00B894' if closed_trades_sorted.iloc[i]['RealizedPnL'] > 0 else '#E17055'
            ax.scatter(closed_trades_sorted.iloc[i]['ExitTime'], closed_trades_sorted.iloc[i]['Balance'],
                      color=color, alpha=0.6, s=30)
        
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        balance_curve = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Create trade visualization
        fig, ax = plt.subplots(figsize=(14, 6))
        for i, (idx, row) in enumerate(closed_trades_sorted.iterrows()):
            color = 'green' if row['RealizedPnL'] > 0 else 'red'
            entry_time = pd.to_datetime(row['EntryTime'])
            exit_time = pd.to_datetime(row['ExitTime'])
            
            ax.plot([entry_time, exit_time], [row['EntryPrice'], row['ExitPrice']],
                   marker='o', color=color, linewidth=2, markersize=8, alpha=0.7)
            
            ax.text(exit_time, row['ExitPrice'], f"{row['RealizedPnL']:+.2f}",
                   fontsize=8, color=color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.set_title('Trade Entry & Exit Visualization', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        trade_viz = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Get additional stats
        long_trades = closed_trades[closed_trades['Side'] == 'LONG']
        short_trades = closed_trades[closed_trades['Side'] == 'SHORT']
        long_wins = len(long_trades[long_trades['RealizedPnL'] > 0])
        short_wins = len(short_trades[short_trades['RealizedPnL'] > 0])
        
        avg_holding_minutes = closed_trades['HoldingMinutes'].mean()
        total_fees = closed_trades['TradingFee'].sum()
        
        if len(closed_trades) > 1:
            first_trade_time = pd.to_datetime(closed_trades['EntryTime'].iloc[0])
            last_trade_time = pd.to_datetime(closed_trades['ExitTime'].iloc[-1])
            trading_days = (last_trade_time - first_trade_time).days + 1
        else:
            trading_days = 1
        
        leverage = closed_trades['Leverage'].iloc[0]
        risk_reward = "1:2"  # Default, could be extracted from CSV if stored
        stop_loss_pct = "2%"  # Default
        
        stats = {
            'symbol': symbol_name,
            'strategy': 'Historical Backtest',
            'initial_balance': round(initial_balance, 2),
            'final_balance': round(final_balance, 2),
            'total_return_pct': round(total_return_pct, 2),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'max_win': round(max_win, 2),
            'max_loss': round(max_loss, 2),
            'max_drawdown': round(max_drawdown, 2),
            'long_trades': f"{long_wins}/{len(long_trades)}" if len(long_trades) > 0 else "0/0",
            'short_trades': f"{short_wins}/{len(short_trades)}" if len(short_trades) > 0 else "0/0",
            'avg_holding_minutes': round(avg_holding_minutes, 2),
            'trading_days': trading_days,
            'total_fees': round(total_fees, 4),
            'leverage': leverage,
            'risk_reward': risk_reward,
            'stop_loss_pct': stop_loss_pct
        }
        
        # Convert trades to dict for JSON
        trades_df = df.copy()
        trades_df['EntryTime'] = trades_df['EntryTime'].astype(str)
        trades_df['ExitTime'] = trades_df['ExitTime'].astype(str)
        trades_df = trades_df.replace({np.nan: None})
        trades_data = trades_df.to_dict('records')
        
        return render_template('results.html',
                            stats=stats,
                            balance_curve=balance_curve,
                            trade_visualization=trade_viz,
                            trades=trades_data,
                            csv_filename=filename,
                            timestamp=timestamp)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8082)

