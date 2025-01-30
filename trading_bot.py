import pandas as pd
import numpy as np
import os
import time
import schedule
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from dotenv import load_dotenv
from bybit_demo_session import BybitDemoSession  # Import your session class here
from datetime import datetime, timedelta

logging.basicConfig(filename="trading_bot.log", level=logging.INFO, format="%(asctime)s - %(message)s")


class MovingAverageCrossoverMLBot:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("BYBIT_API_KEY")
        self.api_secret = os.getenv("BYBIT_API_SECRET")
        if not self.api_key or not self.api_secret:
            raise ValueError("API keys not found. Please set BYBIT_API_KEY and BYBIT_API_SECRET in your .env file.")

        self.symbol = os.getenv("TRADING_SYMBOL", "BTCUSDT")
        self.data_fetcher = BybitDemoSession(self.api_key, self.api_secret)
        self.sl_model = RandomForestRegressor()
        self.tp_model = RandomForestRegressor()
        self.trained = False

    def preprocess_data(self, historical_data, short_window=10, long_window=50):
        df = pd.DataFrame(historical_data)
        df.columns = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
        
        # Convert numeric columns to float
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        # Calculate SMAs
        df['SMA_Short'] = df['close'].rolling(window=short_window).mean()
        df['SMA_Long'] = df['close'].rolling(window=long_window).mean()

        # Volatility: High-Low difference
        df['Volatility'] = df['high'] - df['low']

        # Average True Range (ATR)
        df['TrueRange'] = df[['high', 'low', 'close']].apply(
            lambda row: max(row['high'] - row['low'], 
                            abs(row['high'] - row['close']), 
                            abs(row['low'] - row['close'])), axis=1)
        df['ATR'] = df['TrueRange'].rolling(window=14).mean()

        # Signal: SMA crossover
        df['Signal'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1, 0)

        # Next_Signal for prediction
        df['Next_Signal'] = df['Signal'].shift(-1)

        # SL and TP based on price movement (target variables)
        df['SL_Price'] = df['close'] - df['ATR'] * 1.5
        df['TP_Price'] = df['close'] + df['ATR'] * 2

        df.dropna(inplace=True)
        return df


    def train_model(self, processed_data):
        X = processed_data[['SMA_Short', 'SMA_Long', 'Volatility', 'ATR']]
        y_sl = processed_data['SL_Price']
        y_tp = processed_data['TP_Price']

        X_train, X_test, y_sl_train, y_sl_test, y_tp_train, y_tp_test = train_test_split(
            X, y_sl, y_tp, test_size=0.2, random_state=42)

        self.sl_model.fit(X_train, y_sl_train)
        self.tp_model.fit(X_train, y_tp_train)

        sl_score = self.sl_model.score(X_test, y_sl_test)
        tp_score = self.tp_model.score(X_test, y_tp_test)
        logging.info(f"SL model score: {sl_score}, TP model score: {tp_score}")
        self.trained = True

    def predict_sl_tp(self, current_data):
        if not self.trained:
            raise ValueError("Models are not trained yet!")

        # Extract features as a DataFrame with valid column names
        features = current_data[['SMA_Short', 'SMA_Long', 'Volatility', 'ATR']].iloc[-1].to_frame().T
        sl_prediction = self.sl_model.predict(features)
        tp_prediction = self.tp_model.predict(features)

        return sl_prediction[0], tp_prediction[0]


    def place_order(self, signal, current_price, quantity, processed_data):
        try:
            side = "Buy" if signal == "Buy" else "Sell"
            sl, tp = self.predict_sl_tp(processed_data)
            print(f"Predicted SL: {sl}, TP: {tp}")  # Debugging log

            # Place market order
            order_result = self.data_fetcher.place_order(
                symbol=self.symbol,
                side=side,
                qty=quantity,
                current_price=current_price,
                leverage=20,
                stop_loss=sl,
                take_profit=tp
            )

            if order_result:
                logging.info(f"Order placed: {order_result}")
                print(f"Order placed successfully: {order_result}")
            else:
                logging.warning("Order failed to execute.")
                print("Order failed to execute.")
        except Exception as e:
            logging.error(f"Error placing order: {e}")
            print(f"Error placing order: {e}")


    def run_strategy(self):
        try:
            print("Running strategy...")
            logging.info("Running strategy...")

            # Check for open positions
            open_positions = self.data_fetcher.get_open_positions(self.symbol)
            print("Open Positions:", open_positions)  # Debugging output
            if open_positions:
                logging.info("Open positions exist. Skipping new trade.")
                print("Open positions exist. Skipping new trade.")
                return
            
            last_closed_time = self.data_fetcher.get_last_closed_position_time(self.symbol)
            if last_closed_time:
                time_since_last_trade = datetime.utcnow() - last_closed_time
                if time_since_last_trade < timedelta(hours=3):
                    logging.info(f"Last trade closed {time_since_last_trade} ago. Skipping trade.")
                    print(f"Last trade closed {time_since_last_trade} ago. Waiting for 3 hours.")
                    return

            print("Fetching historical data...")
            historical_data = self.data_fetcher.get_historical_data(self.symbol, interval="60", limit=200)
            if not historical_data:
                logging.error("No historical data fetched.")
                print("No historical data fetched. Check your API or internet connection.")
                return

            print("Processing data...")
            processed_data = self.preprocess_data(historical_data)
            if not self.trained:
                print("Training model...")
                self.train_model(processed_data)

            print("Predicting signal...")
            signal = "Buy" if processed_data['Signal'].iloc[-1] == 1 else "Sell"
            print(f"Predicted Signal: {signal}")

            print("Fetching current price...")
            current_price = self.data_fetcher.get_real_time_price(self.symbol)
            print(f"Current Price: {current_price}")

            quantity = float(os.getenv("TRADE_QUANTITY", 0.2))
            print(f"Placing order with quantity: {quantity}")
            self.place_order(signal, current_price, quantity, processed_data)
        except Exception as e:
            logging.error(f"Error in run_strategy: {e}")
            print(f"Error in run_strategy: {e}")


    def run(self):
        self.run_strategy()
        schedule.every(1).minutes.do(self.run_strategy)
        while True:
            schedule.run_pending()
            time.sleep(1)


if __name__ == "__main__":
    print("Starting the bot...")
    bot = MovingAverageCrossoverMLBot()
    bot.run()

