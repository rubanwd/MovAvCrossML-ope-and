import pandas as pd
import numpy as np
import os
import time
import schedule
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
from bybit_demo_session import BybitDemoSession  # Import your session class here

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
        self.model = RandomForestClassifier()
        self.trained = False

    def preprocess_data(self, historical_data, short_window=10, long_window=50):
        df = pd.DataFrame(historical_data)
        df.columns = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
        df['close'] = df['close'].astype(float)

        df['SMA_Short'] = df['close'].rolling(window=short_window).mean()
        df['SMA_Long'] = df['close'].rolling(window=long_window).mean()
        df['Signal'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1, 0)
        df['Next_Signal'] = df['Signal'].shift(-1)

        df.dropna(inplace=True)
        return df

    def train_model(self, processed_data):
        X = processed_data[['SMA_Short', 'SMA_Long']]
        y = processed_data['Next_Signal']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        logging.info(f"Model trained with accuracy: {accuracy}")
        self.trained = True

    def predict_signal(self, current_data):
        if not self.trained:
            raise ValueError("Model is not trained yet!")
        
        features = current_data[['SMA_Short', 'SMA_Long']].iloc[-1].values.reshape(1, -2)
        prediction = self.model.predict(features)
        return "Buy" if prediction[0] == 1 else "Sell"

    def place_order(self, signal, current_price, quantity):
        try:
            side = "Buy" if signal == "Buy" else "Sell"
            order_result = self.data_fetcher.place_order(
                symbol=self.symbol,
                side=side,
                qty=quantity,
                current_price=current_price,
                leverage=20
            )
            if order_result:
                logging.info(f"Order placed: {order_result}")
                return order_result
            else:
                logging.warning("Order failed to execute.")
        except Exception as e:
            logging.error(f"Error placing order: {e}")

    def monitor_trades(self):
        open_positions = self.data_fetcher.get_open_positions(self.symbol)
        if not open_positions:
            logging.info("No open trades to monitor.")
            return

        for position in open_positions:
            stop_loss = float(position['stopLoss'])
            take_profit = float(position['takeProfit'])
            current_price = self.data_fetcher.get_real_time_price(self.symbol)

            if current_price <= stop_loss or current_price >= take_profit:
                self.close_order(position)

    def close_order(self, position):
        try:
            order_id = position['orderId']
            self.data_fetcher.cancel_order(order_id, self.symbol)
            logging.info(f"Order {order_id} closed.")
            self.log_trade(position, "Closed")
        except Exception as e:
            logging.error(f"Error closing order: {e}")

    def log_trade(self, trade_details, status):
        try:
            trade_log = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": self.symbol,
                "side": trade_details.get('side'),
                "quantity": trade_details.get('size'),
                "price": trade_details.get('entryPrice'),
                "status": status
            }
            df = pd.DataFrame([trade_log])
            if not os.path.exists("trade_log.csv"):
                df.to_csv("trade_log.csv", index=False)
            else:
                df.to_csv("trade_log.csv", mode="a", header=False, index=False)
        except Exception as e:
            logging.error(f"Error logging trade: {e}")

    def sleep_with_details(self, seconds):
        for remaining in range(seconds, 0, -1):
            print(f"Sleeping for {remaining} seconds...", end="\r")
            time.sleep(1)
        print("Waking up...")

    def run_strategy(self, interval=60):
        try:
            print("Checking for open positions...")
            open_positions = self.data_fetcher.get_open_positions(self.symbol)
            if open_positions:
                logging.info("Open positions exist. Skipping new trade.")
                print("Open positions exist. Skipping new trade.")
                return  # Skip this iteration if positions are already open
            
            print("Fetching historical data...")
            historical_data = self.data_fetcher.get_historical_data(self.symbol, interval="60", limit=200)
            if not historical_data:
                logging.error("No historical data fetched.")
                return

            print("Processing data...")
            processed_data = self.preprocess_data(historical_data)

            if not self.trained:
                print("Training model...")
                self.train_model(processed_data)

            print("Predicting signal...")
            signal = self.predict_signal(processed_data)
            logging.info(f"Predicted Signal: {signal}")
            print(f"Predicted Signal: {signal}")

            print("Fetching current price...")
            current_price = self.data_fetcher.get_real_time_price(self.symbol)
            print(f"Current Price: {current_price}")

            quantity = float(os.getenv("TRADE_QUANTITY", 0.2))
            self.place_order(signal, current_price, quantity)

            print("Monitoring trades...")
            self.monitor_trades()

            print("Sleeping for 60 seconds...")
            self.sleep_with_details(60)
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
    bot = MovingAverageCrossoverMLBot()
    bot.run()
