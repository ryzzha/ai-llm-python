import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

coins = {
    "1": ("Bitcoin", "bitcoin"),
    "2": ("Ethereum", "ethereum"),
    "3": ("Solana", "solana"),
    "4": ("Toncoin", "the-open-network"),
    "5": ("Dogecoin", "dogecoin"),
}

print("Оберіть криптовалюту:")
for k, v in coins.items():
    print(f"{k}. {v[0]}")

choice = input("Ваш вибір: ")

if choice not in coins:
    print("❌ Невірний вибір")
    exit()

coin_name, coin_id = coins[choice]
print(f"\n📈 Ви обрали: {coin_name}\n")

url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
params = {"vs_currency": "usd", "days": "90"}
response = requests.get(url, params=params)
data = response.json()

# 2. transform to DataFrame
prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
df = pd.merge(prices, volumes, on="timestamp")
df["date"] = pd.to_datetime(df["timestamp"], unit='ms')
df = df[["date", "price", "volume"]]

# 3. features
df["price_change"] = df["price"].pct_change()  # % changes
df["target"] = df["price_change"].shift(-1) > 0  # will grow tommorow?
df.dropna(inplace=True)

# 4. model learning
X = df[["price_change", "volume"]]
y = df["target"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. estimation
y_pred = model.predict(X_test)
print("Точність моделі:", accuracy_score(y_test, y_pred))

# 6. зredictions for the last day
last_row = df.iloc[-1:][["price_change", "volume"]]
pred = model.predict(last_row)
print("💸 Прогноз на завтра: 🔺 зросте" if pred[0] else "📉 впаде")
