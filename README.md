# Cryptocurrency Prediction Using Machine Learning

Introduction

This project aims to develop a machine learning-based cryptocurrency prediction model using Long Short-Term Memory (LSTM) networks. The model is designed to forecast future price trends of cryptocurrencies based on historical price, volume, and other influential factors such as market sentiment and news data.

Motivation

The cryptocurrency market is highly volatile and difficult to predict using traditional statistical models. Machine learning, specifically deep learning models like LSTM, can analyze patterns in large datasets and provide more accurate price forecasts. This project contributes to the growing research on AI-driven financial predictions by leveraging machine learning techniques to model crypto price movements.

Features

Dataset: Historical cryptocurrency data obtained from cryptocompare.com

Model: LSTM (Long Short-Term Memory) network

Libraries Used: TensorFlow, Keras, Pandas, NumPy, Matplotlib, Scikit-learn

Key Predictions: Future price trends, market fluctuations, trading opportunities

Expected Accuracy: ~76%

Research Questions

Can machine learning models accurately forecast cryptocurrency price trends?

How do LSTM models compare to traditional machine learning techniques?

What factors (technical indicators, sentiment analysis, etc.) influence crypto price movements?

Can ML models identify trading opportunities and improve risk management for investors?

Methodology

Data Collection: Acquiring price, volume, and sentiment data from cryptocompare.com

Data Preprocessing: Cleaning and normalizing data for ML models

Feature Engineering: Selecting relevant features such as moving averages, RSI, MACD, and social media sentiment

Model Selection: Comparing traditional ML models (Random Forest, SVM) with deep learning models (LSTM)

Training & Evaluation: Training the LSTM model and fine-tuning hyperparameters

Prediction & Visualization: Forecasting crypto prices and visualizing trends

Installation

Clone the repository and install dependencies:

# Clone the repo
git clone https://github.com/your-username/crypto-price-prediction.git
cd crypto-price-prediction

# Install dependencies
pip install -r requirements.txt

Usage

Run the Jupyter notebook or Python script to train the model and make predictions:

# Running the model
python train_model.py

Results & Findings

The LSTM model captures temporal dependencies and patterns in cryptocurrency price data.

The model achieves an expected accuracy of 76% on the test dataset.

Trends and market patterns can be identified to assist in decision-making for traders and investors.

Future Improvements

Enhancing model accuracy with additional features such as social media sentiment analysis

Integrating real-time data streaming for live predictions

Testing alternative deep learning architectures like GRU or Transformer models

Contributors

Suraj Dharmik

License

This project is licensed under the MIT License - see the LICENSE file for details.
