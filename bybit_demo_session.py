import requests
import time
import hashlib
import hmac
from datetime import datetime, timedelta


class BybitDemoSession:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api-demo.bybit.com"

    def _generate_signature(self, params):
        param_str = '&'.join([f'{k}={params[k]}' for k in sorted(params)])
        return hmac.new(self.api_secret.encode('utf-8'), param_str.encode('utf-8'), hashlib.sha256).hexdigest()

    def _get_timestamp(self):
        return str(int(time.time() * 1000))

    def send_request(self, method, endpoint, params=None):
        if params is None:
            params = {}

        params['api_key'] = self.api_key
        params['timestamp'] = self._get_timestamp()
        params['sign'] = self._generate_signature(params)

        url = f"{self.base_url}{endpoint}"
        if method == "GET":
            response = requests.get(url, params=params)
        elif method == "POST":
            response = requests.post(url, json=params)
        else:
            raise ValueError("Unsupported HTTP method")
        return response.json()

    def get_historical_data(self, symbol, interval, limit):
        endpoint = "/v5/market/kline"
        params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
        return self.send_request("GET", endpoint, params)['result']['list']

    def place_order(self, symbol, side, qty, current_price, leverage, stop_loss=None, take_profit=None):
        try:
            # Set leverage before placing an order
            self.set_leverage(symbol, leverage=leverage)

            endpoint = "/v5/order/create"
            position_mode = "one_way"  # Set this to "hedge" if you are in hedge mode

            # Determine position index based on the trade mode
            if position_mode == "hedge":
                position_idx = 1 if side.lower() == 'buy' else 2
            else:  # one_way
                position_idx = 0

            # Adjust price based on the side of the order
            if side.lower() == 'buy':
                price = current_price * 0.9997  # Slightly below market price for Buy
                if stop_loss and stop_loss >= price:
                    print("Stop-loss is higher than or equal to the limit price for a Buy order. Adjusting stop-loss...")
                    stop_loss = price * 0.995  # Ensure stop-loss is slightly below the limit price
            else:  # side.lower() == 'sell'
                price = current_price * 1.0003  # Slightly above market price for Sell
                if stop_loss and stop_loss <= price:
                    print("Stop-loss is lower than or equal to the limit price for a Sell order. Adjusting stop-loss...")
                    stop_loss = price * 1.005  # Ensure stop-loss is slightly above the limit price

            # Build order parameters for a Market order
            order_params = {
                "category": "linear",
                "symbol": symbol,
                "side": side.capitalize(),  # Capitalize side to match API expectations
                "orderType": "Market",  # Market order
                "qty": str(qty),  # Convert quantity to string
                "positionIdx": position_idx,  # Use the positionIdx determined above
            }

            # Add stop loss and take profit parameters if provided
            if stop_loss:
                order_params["stopLoss"] = str(round(stop_loss, 2))  # Round to 2 decimal places
            if take_profit:
                order_params["takeProfit"] = str(round(take_profit, 2))  # Round to 2 decimal places

            # Debugging: Print order parameters
            print(f"Order parameters: {order_params}")

            # Send the order request
            response = self.send_request("POST", endpoint, order_params)

            # Check if the API response indicates success
            if response['retCode'] != 0:
                raise Exception(f"API Error: {response['retMsg']}")

            return response['result']
        except Exception as e:
            print(f"Error placing order: {e}")
            return None



    def set_leverage(self, symbol, leverage):
        endpoint = "/v5/position/set-leverage"
        params = {"category": "linear", "symbol": symbol, "buyLeverage": leverage, "sellLeverage": leverage}
        return self.send_request("POST", endpoint, params)

    def get_open_positions(self, symbol):
        try:
            endpoint = "/v5/position/list"
            params = {"category": "linear", "symbol": symbol}
            response = self.send_request("GET", endpoint, params)

            if response['retCode'] != 0:
                raise Exception(f"API Error: {response['retMsg']}")

            positions = response['result']['list']
            print("API Response for Open Positions:", positions)  # Debugging log

            # Filter for active positions: size > 0 and side not empty
            active_positions = [pos for pos in positions if float(pos['size']) > 0 and pos['side']]
            return active_positions
        except Exception as e:
            print(f"Error fetching open positions: {e}")
            return None



    def get_real_time_price(self, symbol):
        endpoint = "/v5/market/tickers"
        params = {"category": "linear", "symbol": symbol}
        return float(self.send_request("GET", endpoint, params)['result']['list'][0]['lastPrice'])
    
    def get_last_closed_position_time(self, symbol):
        """Fetches the last closed position time."""
        try:
            endpoint = "/v5/position/closed-pnl"
            params = {"category": "linear", "symbol": symbol, "limit": 1}
            response = self.send_request("GET", endpoint, params)

            if response['retCode'] != 0:
                raise Exception(f"API Error: {response['retMsg']}")

            closed_positions = response['result']['list']
            if not closed_positions:
                return None  # No closed positions found

            last_closed_time = int(closed_positions[0]['updatedTime'])  # Timestamp in milliseconds
            return datetime.utcfromtimestamp(last_closed_time / 1000)  # Convert to datetime
        except Exception as e:
            print(f"Error fetching last closed position time: {e}")
            return None
