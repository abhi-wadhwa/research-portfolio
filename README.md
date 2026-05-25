# Research Portfolio

Abhi Wadhwa, University of Southern California

---

## Projects

### [F24 — Measure Theory and Fine Properties of Functions](f24-measure-theoretic-pricing/)

Study notes based on Evans & Gariepy's *Measure Theory and Fine Properties of Functions* (1992). Covers all six chapters: outer measures and Carathéodory's criterion, Hausdorff measure and dimension, the area and coarea formulas, Sobolev spaces, BV functions and sets of finite perimeter, and differentiability/approximation by C¹ functions. Written as lecture notes in a self-study format.

### [F25 — On the Resolution of the Spielman-Teng Conjecture](f25-spielman-teng/)

Expository notes on the Sah-Sahasrabudhe-Sawhney proof of the Spielman-Teng conjecture (arXiv:2405.20308). Walks through the four-step proof architecture — geometric reduction, truncation, Gaussian replacement via Lindeberg exchange, and rescaling — and verifies the bound P(σₙ(M) ≤ εn⁻¹ᐟ²) ≤ (1+o(1))ε + e⁻ᴼ⁽ⁿ⁾ computationally with Monte Carlo simulations.

### [S25 — From Classical BSDEs to Deep Solvers](s25-bsde-pricing/)

A study of backward stochastic differential equations and their applications to nonlinear pricing PDEs. Covers classical BSDE existence and uniqueness, the deep BSDE method of Han-Jentzen-E for high-dimensional PDEs, credit valuation adjustment via BSDEs, and g-expectations as dynamic risk measures. Includes Python implementations of SDE/BSDE solvers and a deep BSDE neural network.

### [S26 — Deep Learning for High-Frequency Trading Signal Detection](s26-dl-hft/)

Theory-heavy treatment of ML architectures for limit order book signal detection. Derives Rademacher complexity bounds for RNNs, temporal convolutional networks, and Transformers applied to HFT data. Simulates a Hawkes-process limit order book, trains models on synthetic microstructure features, and analyzes online learning regret bounds for non-stationary environments.

### [S26 — LLM Agents for Cross-Platform Prediction Market Arbitrage](s26-prediction-arb/)

Develops a hybrid NLP matching pipeline (regex + Sentence-BERT + rapidfuzz) for cross-platform contract resolution between Kalshi and Polymarket, implements fee-adjusted arbitrage detection, and designs an LLM agent evaluation framework. Pipeline validated on synthetic data; live-market evaluation in progress.
