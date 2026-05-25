# llm prediction market arbitrage

synthetic benchmark testing whether LLM agents can do cross-platform prediction market arbitrage (kalshi + polymarket).

## running

```bash
pip install -r requirements.txt
python generate_figures.py
```

## files

- simulate_markets.py -- synthetic kalshi tickers, polymarket events, order book simulation
- matching.py -- hybrid NLP matching pipeline (regex + sbert + rapidfuzz + mock llm)
- fees.py -- kalshi and polymarket fee models
- orderbook.py -- order book representation, arbitrage detection algorithms
- agent.py -- rule-based baseline vs mock llm agent
- backtest.py -- end-to-end 90-day paper trading simulation
- generate_figures.py -- makes all the paper figures
- fetch_live_data.py -- fetches live market data from kalshi and polymarket APIs

**note:** the pipeline has been validated on synthetic data. live-market evaluation and LLM agent experiments are in progress.
