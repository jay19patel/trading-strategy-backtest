df['Action'] = None

# Ensure required columns exist
if '9EMA' not in df.columns or '15EMA' not in df.columns:
    pass
else:
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