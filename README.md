
# Repository Description

* This repository provides an FX options pricing library based on the Vanna–Volga methodology, calibrated directly to FX smile quotes. The framework transforms market delta quotes into strikes, reconstructing a consistent volatility smile, and producing market-consistent prices for vanilla and first generation exotic FX options.

* More specifically, the library reconstructs the FX volatility smile from sparse market inputs and applies Vanna–Volga adjustments to Black–Scholes prices. Building on the calibrated smile, the framework is extended to exotic payoff foundations, pricing digital options via finite differences on Vanna–Volga adjusted vanilla prices.



# ⚙️ Key Features
* **Advanced Numerical Greeks**: The system computes second-order sensitivities Vanna, and Volga using adaptive finite differences and Richardson extrapolation to ensure numerical stability even near market bound

* **Performance Optimization (Caching)**: To avoid redundant calculations, the pricer caches Vanna-Volga weights and the pillar 3x3 Greek matrix per market slice, significantly speeding up the pricing of multiple strikes

* **Configurable FX Conventions**: The engine is built to handle different market conventions, specifically allowing the user to switch between Spot Premium Excluded and Forward Premium Excluded delta logic

* **Modular Architecture**: The library uses a decoupled design where pricing methodology, numerical infrastructure, and market conventions are independent modules that can be replaced or extended

* **Self-Contained Frameworks**: The library is written from scratch with zero external dependencies, implementing built-in mathematical utilities and numerical approximations (e.g. Abramowitz-Stegun for the Normal CDF)



# 🔍 Key Objectives

* Reconstruct FX volatility smiles from market quotes

* Apply Vanna–Volga adjustments to obtain smile-consistent vanilla prices

* Build a foundation for exotic option pricing using Vanna-Volga method

* Design a modular, dependency-free numerical architecture for future extension



# 📌 Key Takeaways

* FX options are quoted in delta terms rather than strike terms

* Vanna–Volga method extends Black–Scholes pricing to incorporate market volatility smile effects

* Digital options can be priced via finite differences on smile-consistent vanilla prices

$$\text{Digital}(K) = - \frac{\partial C(K)}{\partial K} \approx \frac{C(K-\epsilon)-C(K+\epsilon)}{2\epsilon}$$



# ⚠️ Challenges

* **Market conventions complexity**: FX options rely on multiple delta conventions (spot/forward, premium included/excluded), and incorrect assumptions lead to incorrect strikes and pricing.

* **Finite-difference sensitivity**: Greeks and digital prices depend on step-size choices and require adaptive bumping for numerical stability.

* **Performance optimization**: Repeated evaluations across multiple strikes and maturities require caching and efficient numerical routines to prevent redundant computations.







