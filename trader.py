import pandas as pd
import numpy as np
from datetime import datetime
import ta


class CryptoMarginTrader:
    def __init__(self, initial_balance, maxHoldMinutes, RiskToReward, symbol_name, strategy_name,
                 stoploss_percentage=5.0, leverage=20, trading_fee=0.1):
        """
        Real Crypto Margin Trading Simulator

        Parameters:
        - initial_balance: Starting capital in USDT
        - maxHoldMinutes: Maximum holding time in minutes
        - RiskToReward: Risk to reward ratio (e.g., "1:3")
        - symbol_name: Trading pair (e.g., "ETHUSDT")
        - strategy_name: Strategy name
        - stoploss_percentage: Stop loss percentage (default 5%)
        - leverage: Trading leverage (default 20x)
        - trading_fee: Trading fee percentage (default 0.1%)
        """
        self.initial_balance = initial_balance
        self.available_balance = initial_balance  # Available for trading
        self.total_balance = initial_balance      # Total account balance
        self.margin_balance = 0                   # Balance locked in margins
        self.unrealized_pnl = 0                   # Current open position P&L

        self.RiskToReward = RiskToReward
        self.symbol_name = symbol_name
        self.strategy_name = strategy_name
        self.maxHoldMinutes = maxHoldMinutes
        self.stoploss_percentage = stoploss_percentage / 100
        self.leverage = leverage
        self.trading_fee = trading_fee / 100

        # Active position tracking
        self.active_position = None
        self.position_details = {}

        self.trade_book = pd.DataFrame(columns=[
            'Symbol', 'Side', 'Status', 'PositionSize', 'EntryPrice', 'ExitPrice',
            'EntryTime', 'ExitTime', 'HoldingMinutes', 'StopLoss', 'TakeProfit',
            'Leverage', 'MarginUsed', 'TradingFee', 'RealizedPnL', 'PnL%',
            'AvailableBalance', 'TotalBalance', 'ExitReason', 'ALUS',
            'Avg_Lower_Shadow', 'Avg_Upper_Shadow', '9EMA', '15EMA'
        ])

    def calculate_position_size(self, price, margin_to_use):
        """Calculate position size based on margin and leverage"""
        # Position Size = (Margin × Leverage) / Price
        position_value = margin_to_use * self.leverage
        position_size = position_value / price
        return position_size

    def calculate_margin_required(self, position_size, price):
        """Calculate margin required for a position"""
        # Margin Required = (Position Size × Price) / Leverage
        position_value = position_size * price
        return position_value / self.leverage

    def calculate_trading_fee(self, position_size, price):
        """Calculate trading fee - 0.1% of position value"""
        position_value = position_size * price
        return position_value * self.trading_fee

    def calculate_unrealized_pnl(self, current_price):
        """Calculate unrealized P&L for active position"""
        if not self.active_position:
            return 0

        details = self.position_details
        if details['side'] == 'LONG':
            pnl = (current_price - details['entry_price']) * details['position_size']
        else:  # SHORT
            pnl = (details['entry_price'] - current_price) * details['position_size']

        return pnl - details['total_fees']  # Subtract fees

    def update_account_balance(self):
        """Update total balance including unrealized P&L"""
        self.total_balance = self.available_balance + self.margin_balance + self.unrealized_pnl

    def open_position(self, side, entry_price, entry_time, margin_percentage=50, row=None):
        """Open a new position"""
        if self.active_position:
            return False, "Position already active"

        # Calculate margin to use (percentage of available balance)
        margin_to_use = self.available_balance * (margin_percentage / 100)

        if margin_to_use <= 0:
            return False, "Insufficient balance"

        # Calculate position size
        position_size = self.calculate_position_size(entry_price, margin_to_use)

        # Calculate actual margin required
        margin_required = self.calculate_margin_required(position_size, entry_price)

        # Calculate fees (entry fee)
        entry_fee = self.calculate_trading_fee(position_size, entry_price)

        # Check if we have enough balance for margin + fees
        total_required = margin_required + entry_fee
        if total_required > self.available_balance:
            return False, "Insufficient balance for margin and fees"

        # Parse Risk:Reward ratio
        risk, reward = map(int, self.RiskToReward.split(":"))

        # Calculate Stop Loss and Take Profit
        if side == 'LONG':
            stop_loss = entry_price * (1 - self.stoploss_percentage)
            risk_amount = entry_price - stop_loss
            take_profit = entry_price + (risk_amount * reward)
        else:  # SHORT
            stop_loss = entry_price * (1 + self.stoploss_percentage)
            risk_amount = stop_loss - entry_price
            take_profit = entry_price - (risk_amount * reward)

        # Update balances
        self.available_balance -= total_required
        self.margin_balance += margin_required

        # Store position details
        self.active_position = True
        self.position_details = {
            'side': side,
            'position_size': position_size,
            'entry_price': entry_price,
            'entry_time': entry_time,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'margin_used': margin_required,
            'entry_fee': entry_fee,
            'total_fees': entry_fee,  # Will add exit fee later
            'leverage': self.leverage
        }

        # Get additional data from row
        alus = row.get('ALUS', None) if row is not None else None
        avg_lower = row.get('Avg_Lower_Shadow', None) if row is not None else None
        avg_upper = row.get('Avg_Upper_Shadow', None) if row is not None else None
        ema9 = row.get('9EMA', None) if row is not None else None
        ema15 = row.get('15EMA', None) if row is not None else None

        # Add to trade book
        new_trade = pd.DataFrame({
            'Symbol': [self.symbol_name],
            'Side': [side],
            'Status': ['OPEN'],
            'PositionSize': [round(position_size, 6)],
            'EntryPrice': [entry_price],
            'ExitPrice': [None],
            'EntryTime': [entry_time],
            'ExitTime': [None],
            'HoldingMinutes': [0],
            'StopLoss': [round(stop_loss, 2)],
            'TakeProfit': [round(take_profit, 2)],
            'Leverage': [f"{self.leverage}x"],
            'MarginUsed': [round(margin_required, 2)],
            'TradingFee': [round(entry_fee, 4)],
            'RealizedPnL': [0],
            'PnL%': [0],
            'AvailableBalance': [round(self.available_balance, 2)],
            'TotalBalance': [round(self.total_balance, 2)],
            'ExitReason': [None],
            'ALUS': [alus],
            'Avg_Lower_Shadow': [avg_lower],
            'Avg_Upper_Shadow': [avg_upper],
            '9EMA': [ema9],
            '15EMA': [ema15]
        })

        self.trade_book = pd.concat([self.trade_book, new_trade], ignore_index=True)
        return True, "Position opened successfully"

    def close_position(self, exit_price, exit_time, exit_reason="Manual"):
        """Close active position"""
        if not self.active_position:
            return False, "No active position"

        details = self.position_details

        # Calculate exit fee
        exit_fee = self.calculate_trading_fee(details['position_size'], exit_price)
        total_fees = details['total_fees'] + exit_fee

        # Calculate realized P&L
        if details['side'] == 'LONG':
            price_pnl = (exit_price - details['entry_price']) * details['position_size']
        else:  # SHORT
            price_pnl = (details['entry_price'] - exit_price) * details['position_size']

        realized_pnl = price_pnl - total_fees
        pnl_percentage = (realized_pnl / details['margin_used']) * 100

        # Calculate holding time
        holding_minutes = abs((pd.to_datetime(exit_time) - pd.to_datetime(details['entry_time'])).total_seconds() / 60)

        # Update balances
        self.available_balance += details['margin_used'] + realized_pnl  # Return margin + P&L
        self.margin_balance -= details['margin_used']
        self.unrealized_pnl = 0

        # Update trade book
        last_index = len(self.trade_book) - 1
        self.trade_book.at[last_index, 'Status'] = 'CLOSED'
        self.trade_book.at[last_index, 'ExitPrice'] = exit_price
        self.trade_book.at[last_index, 'ExitTime'] = exit_time
        self.trade_book.at[last_index, 'HoldingMinutes'] = round(holding_minutes, 2)
        self.trade_book.at[last_index, 'TradingFee'] = round(total_fees, 4)
        self.trade_book.at[last_index, 'RealizedPnL'] = round(realized_pnl, 2)
        self.trade_book.at[last_index, 'PnL%'] = round(pnl_percentage, 2)
        self.trade_book.at[last_index, 'AvailableBalance'] = round(self.available_balance, 2)
        self.trade_book.at[last_index, 'TotalBalance'] = round(self.available_balance, 2)  # No open positions
        self.trade_book.at[last_index, 'ExitReason'] = exit_reason

        # Clear position
        self.active_position = None
        self.position_details = {}

        return True, f"Position closed. P&L: {realized_pnl:.2f} USDT"

    def check_liquidation(self, current_price):
        """Check if position should be liquidated"""
        if not self.active_position:
            return False

        details = self.position_details

        # Calculate current P&L
        if details['side'] == 'LONG':
            pnl = (current_price - details['entry_price']) * details['position_size']
        else:
            pnl = (details['entry_price'] - current_price) * details['position_size']

        pnl -= details['total_fees']  # Subtract fees

        # Liquidation occurs when loss approaches margin (with safety buffer)
        liquidation_threshold = -details['margin_used'] * 0.9  # 90% of margin

        return pnl <= liquidation_threshold

    def backtest(self, df, margin_per_trade=50):
        """
        Run backtest on historical data

        Parameters:
        - df: DataFrame with OHLC data and signals
        - margin_per_trade: Percentage of available balance to use per trade
        """
        required_columns = ["Close", "Open", "High", "Low", "DateTime", "Action"]
        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"Error: Column '{column}' is missing in the DataFrame.")

        allowed_signals = {"buy", "sell", None, ""}
        if not df['Action'].isin(allowed_signals).all():
            invalid_signals = df[~df['Action'].isin(allowed_signals)]['Action'].unique()
            raise ValueError(f"Error: Invalid signals found: {invalid_signals}. Use 'buy', 'sell', or None.")

        for index, row in df.iterrows():
            current_price = float(row["Close"])
            current_time = row["DateTime"]

            # Update unrealized P&L for active position
            if self.active_position:
                self.unrealized_pnl = self.calculate_unrealized_pnl(current_price)
                self.update_account_balance()

                # Check for liquidation
                if self.check_liquidation(current_price):
                    self.close_position(current_price, current_time, "Liquidation")
                    continue

                # Check exit conditions
                details = self.position_details
                exit_triggered = False
                exit_reason = ""

                # Check Stop Loss and Take Profit
                if details['side'] == 'LONG':
                    if current_price <= details['stop_loss']:
                        exit_triggered = True
                        exit_reason = "Stop Loss"
                    elif current_price >= details['take_profit']:
                        exit_triggered = True
                        exit_reason = "Take Profit"
                else:  # SHORT
                    if current_price >= details['stop_loss']:
                        exit_triggered = True
                        exit_reason = "Stop Loss"
                    elif current_price <= details['take_profit']:
                        exit_triggered = True
                        exit_reason = "Take Profit"

                # Check time-based exit
                holding_time = abs((pd.to_datetime(current_time) - pd.to_datetime(details['entry_time'])).total_seconds() / 60)
                if holding_time >= self.maxHoldMinutes:
                    exit_triggered = True
                    exit_reason = "Time Exit"

                if exit_triggered:
                    self.close_position(current_price, current_time, exit_reason)

            # Check for new entry signals
            elif row["Action"] in ["buy", "sell"] and self.available_balance > 0:
                side = "LONG" if row["Action"] == "buy" else "SHORT"
                success, message = self.open_position(side, current_price, current_time, margin_per_trade, row)

                if not success:
                    pass  # Silently continue if position cannot be opened

        # Close any remaining open position
        if self.active_position and len(df) > 0:
            last_row = df.iloc[-1]
            self.close_position(float(last_row["Close"]), last_row["DateTime"], "End of Data")

    def get_performance_stats(self):
        """Generate comprehensive performance statistics"""
        closed_trades = self.trade_book[self.trade_book['Status'] == 'CLOSED'].copy()

        if len(closed_trades) == 0:
            return {}

        # Basic stats
        total_trades = len(closed_trades)
        winning_trades = len(closed_trades[closed_trades['RealizedPnL'] > 0])
        losing_trades = len(closed_trades[closed_trades['RealizedPnL'] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        # P&L stats
        total_pnl = closed_trades['RealizedPnL'].sum()
        avg_win = closed_trades[closed_trades['RealizedPnL'] > 0]['RealizedPnL'].mean() if winning_trades > 0 else 0
        avg_loss = closed_trades[closed_trades['RealizedPnL'] <= 0]['RealizedPnL'].mean() if losing_trades > 0 else 0
        max_win = closed_trades['RealizedPnL'].max()
        max_loss = closed_trades['RealizedPnL'].min()

        # Returns
        total_return_pct = ((self.available_balance - self.initial_balance) / self.initial_balance) * 100

        # Drawdown calculation
        closed_trades_sorted = closed_trades.sort_values('ExitTime')
        closed_trades_sorted['CumPnL'] = closed_trades_sorted['RealizedPnL'].cumsum()
        closed_trades_sorted['Balance'] = self.initial_balance + closed_trades_sorted['CumPnL']
        closed_trades_sorted['Peak'] = closed_trades_sorted['Balance'].cummax()
        closed_trades_sorted['Drawdown'] = ((closed_trades_sorted['Balance'] - closed_trades_sorted['Peak']) / closed_trades_sorted['Peak']) * 100
        max_drawdown = closed_trades_sorted['Drawdown'].min()

        # Side analysis
        long_trades = closed_trades[closed_trades['Side'] == 'LONG']
        short_trades = closed_trades[closed_trades['Side'] == 'SHORT']
        long_wins = len(long_trades[long_trades['RealizedPnL'] > 0])
        short_wins = len(short_trades[short_trades['RealizedPnL'] > 0])

        # Trading frequency
        if len(closed_trades) > 1:
            first_trade = pd.to_datetime(closed_trades['EntryTime'].iloc[0])
            last_trade = pd.to_datetime(closed_trades['ExitTime'].iloc[-1])
            trading_days = (last_trade - first_trade).days + 1
        else:
            trading_days = 1

        avg_holding_time = closed_trades['HoldingMinutes'].mean()
        total_fees = closed_trades['TradingFee'].sum()

        stats = {
            'symbol': self.symbol_name,
            'strategy': self.strategy_name,
            'initial_balance': self.initial_balance,
            'final_balance': round(self.available_balance, 2),
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
            'avg_holding_minutes': round(avg_holding_time, 2),
            'trading_days': trading_days,
            'total_fees': round(total_fees, 4),
            'leverage': f"{self.leverage}x",
            'risk_reward': self.RiskToReward,
            'stop_loss_pct': f"{self.stoploss_percentage*100}%"
        }

        return stats

