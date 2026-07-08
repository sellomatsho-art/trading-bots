from flask import Flask
import requests
import threading
import time
from datetime import datetime

app = Flask(__name__)
trades = []

def bot_loop():
    while True:
        try:
            r = requests.get("https://gamma-api.polymarket.com/markets", params={"active": "true", "limit": 50}, timeout=10)
            markets = r.json()
            for m in markets:
                if "BTC" in m.get("question", "").upper():
                    price = float(m["orderBook"]["bids"][0][0]) if m.get("orderBook", {}).get("bids") else 0.5
                    trades.append({"time": datetime.now().isoformat()[:19], "price": price, "bot": "BTC"})
                    break
            time.sleep(60)
        except:
            time.sleep(15)

@app.route("/")
def home():
    return f"<h1>Polymarket Bot</h1><p>Status: Running</p><p>Trades: {len(trades)}</p>"

threading.Thread(target=bot_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
