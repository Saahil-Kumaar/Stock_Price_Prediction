# Stock_Price_Prediction
# S\&P 500 Movement Prediction using Machine Learning

This project uses historical S\&P 500 index data to build a machine learning model that predicts whether the index will go up or down the next day. It utilizes technical indicators and past trends with a Random Forest Classifier.

## 📈 Objective

The goal is to:

* Use historical S\&P 500 data to train a binary classifier.
* Predict if the market will go **up (1)** or **down (0)** the next day.
* Evaluate the performance using precision and compare it to baseline probabilities.
* Improve performance using rolling average ratios and trend features.

## 🛠️ Tools & Libraries

* `yfinance`: For downloading historical stock market data.
* `pandas`, `numpy`: Data manipulation and analysis.
* `matplotlib`: For plotting results.
* `scikit-learn`: For building and evaluating the machine learning model.

## 🚀 Workflow

### 1. Data Collection

Download the full historical data for the S\&P 500 index (`^GSPC`) using `yfinance`.

```python
sp500 = yf.Ticker("^GSPC").history(period="max")
```

### 2. Data Cleaning

* Remove irrelevant columns (`Dividends`, `Stock Splits`)
* Filter out rows where volume is 0
* Keep data from 1990 onwards

### 3. Feature Engineering

* Create a `Tomorrow` column as the shifted value of the closing price
* Create a binary `Target` column indicating if the next day’s closing price is higher

```python
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
```

### 4. Model Training & Testing

Train a `RandomForestClassifier` with selected predictors:

```python
predictors = ["Close", "Volume", "Open", "High", "Low"]
```

Evaluate precision on the last 100 data points.

### 5. Backtesting Strategy

Create a backtesting function to simulate how the model would perform over time:

```python
def backtest(data, model, predictors, start=2500, step=250):
    ...
```

### 6. Enhanced Feature Engineering

Add rolling average ratios and trends for various horizons (`[2, 5, 60, 250, 1000]`):

```python
sp500["Close_Ratio_60"] = sp500["Close"] / sp500.rolling(60).mean()["Close"]
sp500["Trend_60"] = sp500.shift(1).rolling(60).sum()["Target"]
```

### 7. Improved Model & Thresholding

Use a probability-based prediction with a custom threshold of `0.6`:

```python
preds = model.predict_proba(test[predictors])[:,1]
preds[preds >= .6] = 1
```

## 📊 Evaluation Metrics

* **Precision Score**: Accuracy of positive predictions
* **Prediction Distribution**: How often the model predicts up/down
* **Baseline Probability**: Frequency of actual up/down movements

## 📉 Results

* Precision score is used as the main metric to measure prediction effectiveness.
* Rolling feature engineering helps improve model precision.

## 📎 Future Improvements

* Add more technical indicators (RSI, MACD, Bollinger Bands).
* Incorporate macroeconomic indicators or news sentiment.
* Experiment with different ML models (Gradient Boosting, XGBoost).

## 🧠 Author

[Saahil Kumaar](https://github.com/Saahil-Kumaar)

---

Let me know if you'd like this tailored for a GitHub repository with badges, or formatted with LaTeX/math blocks for a report.
