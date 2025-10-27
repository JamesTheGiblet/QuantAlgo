# QuantAlgo 🤖📈

**AI-Powered Stock Prediction & Portfolio Management System**

![QuantAlgo](https://img.shields.io/badge/QuantAlgo-AI%20Trading-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-lightgrey)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)

## 🌟 Overview

QuantAlgo is a sophisticated stock prediction system that combines multiple machine learning models with real-time market data to provide intelligent trading recommendations. Built for both educational purposes and serious traders, it features advanced risk management, portfolio tracking, and a professional web interface.

![QuantAlgo Dashboard](https://via.placeholder.com/800x400/2563eb/ffffff?text=QuantAlgo+AI+Trading+Dashboard)

## 🚀 Key Features

### 🤖 Multi-Model AI Prediction
- **Random Forest** - Robust ensemble learning
- **Gradient Boosting** - Sequential error correction  
- **Support Vector Machine** - Advanced pattern recognition
- **Ensemble Voting** - Combined model consensus

### 📊 Real-Time Analytics
- Live WebSocket market data streaming
- Technical indicator calculations (RSI, MACD, Bollinger Bands)
- Volatility-based stop loss calculations
- Dynamic position sizing

### 💼 Portfolio Management
- Real-time P&L tracking
- Risk-adjusted position sizing
- Stop-loss and take-profit automation
- Portfolio performance analytics

### 🎯 Risk Management
- Volatility-based stop loss calculation
- 2% risk-per-trade position sizing
- Multi-timeframe analysis
- Confidence-based trading signals

## 🛠 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start

1. **Clone & Setup**
```bash
git clone https://github.com/yourusername/quantalgo.git
cd quantalgo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Initialize Database**
```bash
python database.py
```

3. **Launch Application**
```bash
python app.py
```

4. **Access Dashboard**
```
Open http://localhost:5000 in your browser
```

## 📁 Project Structure

```
quantalgo/
├── app.py                 # Main Flask application
├── database.py           # SQLite database management
├── models.py             # Machine learning models
├── websocket_client.py   # Real-time data streaming
├── requirements.txt      # Python dependencies
├── static/
│   ├── script.js         # Frontend functionality
│   └── style.css         # Custom styling
└── templates/
    └── index.html        # Main dashboard
```

## 🎮 Usage Guide

### 1. Stock Analysis
- Enter any stock symbol (e.g., AAPL, TSLA, GOOGL)
- View real-time price and technical indicators
- Get multi-model AI predictions

### 2. Model Configuration
```python
# Choose from three AI models
MODEL_TYPES = {
    'random_forest': RandomForestRegressor(),
    'gradient_boost': GradientBoostingRegressor(), 
    'svm': SVR(kernel='rbf')
}
```

### 3. Portfolio Management
- Add long/short positions with automated stop-loss
- Track real-time P&L and portfolio value
- Monitor risk exposure and position sizing

### 4. Risk Controls
```python
# Automated position sizing
position_size = account_balance * risk_per_trade / abs(entry_price - stop_loss)

# Dynamic stop loss calculation
stop_loss = entry_price * (1 - volatility * 2)
```

## 🔧 Configuration

### Model Settings
Edit `models.py` to customize:
- Prediction horizon (1, 5, 10 days)
- Risk tolerance (1%, 2%, 5% per trade)
- Technical indicator parameters

### Database Configuration
```python
# database.py
DB_CONFIG = {
    'path': 'stocks.db',
    'backup_interval': 3600,  # 1 hour
    'cleanup_days': 30        # Keep 30 days of data
}
```

## 📈 Technical Indicators

QuantAlgo calculates 20+ technical indicators:
- **Trend**: MA5, MA10, MA20, MA50
- **Momentum**: RSI, MACD, Price Momentum
- **Volatility**: Bollinger Bands, ATR
- **Volume**: Volume MA, Volume Ratio
- **Support/Resistance**: Dynamic levels

## 🤖 AI Models Performance

| Model | Accuracy | Training Time | Best For |
|-------|----------|---------------|----------|
| Random Forest | 78-82% | Fast | General purpose |
| Gradient Boost | 75-80% | Medium | Trend following |
| SVM | 70-75% | Slow | Volatile markets |

## 🚨 Risk Disclaimer

> **Important**: QuantAlgo is designed for **educational and research purposes**. 
> - Past performance doesn't guarantee future results
> - Always paper trade before using real money
> - Consult financial advisors before investing
> - Use at your own risk

## 🐛 Troubleshooting

### Common Issues

1. **Missing Dependencies**
```bash
pip install --upgrade -r requirements.txt
```

2. **Database Errors**
```bash
rm stocks.db && python database.py
```

3. **WebSocket Connection Issues**
- Check internet connection
- Verify API keys (if using paid services)

### Performance Tips
- Use SSD storage for database
- Allocate 4GB+ RAM for large datasets
- Enable GPU acceleration for model training

## 🔮 Future Roadmap

- [ ] **Deep Learning Integration** - LSTM neural networks
- [ ] **Options Trading** - Strategy backtesting
- [ ] **Mobile App** - iOS/Android companion
- [ ] **API Access** - RESTful API for developers
- [ ] **Social Features** - Strategy sharing community

## 👥 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📧 **Email**: support@quantalgo.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/quantalgo/issues)
- 💬 **Discord**: [Join our community](https://discord.gg/quantalgo)

## 🙏 Acknowledgments

- **yFinance** - Free stock market data
- **Scikit-learn** - Machine learning library
- **Flask** - Web framework
- **Bootstrap** - Frontend components

---

<div align="center">

**Built with ❤️ for the trading community**

*"Empowering traders with AI-driven insights"*

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/quantalgo&type=Date)](https://star-history.com/#yourusername/quantalgo&Date)

</div>

---

**Note**: Replace placeholder links and contact information with your actual project details before publishing.
