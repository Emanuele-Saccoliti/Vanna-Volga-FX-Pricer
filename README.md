
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

<br>

# Project Layout

```text
src/main/java/fxvv/
├── Main.java
├── bs/
│   ├── GKBlackScholes.java
│   └── GreeksFD.java
├── conventions/
│   └── DeltaConvention.java
├── market/
│   ├── MarketSlice.java
│   ├── MarketSliceBuilder.java
│   └── SmileQuote.java
├── numerics/
│   ├── LinearSolver.java
│   ├── NormalDist.java
│   ├── RootFinder.java
│   └── impl/
│       ├── AbramowitzStegunNormal.java
│       ├── BisectionRootFinder.java
│       └── GaussianElimination3.java
└── pricer/
    ├── SmilePricer.java
    └── VannaVolgaPricer.java
```

<br>

# Numerical Design
- `NormalDist` abstracts the normal distribution implementation
- `RootFinder` abstracts root solving
- `LinearSolver` abstracts the 3-by-3 solve
- `SmilePricer` abstracts the pricing methodology
- `DeltaConvention` isolates FX market convention logic

This makes the code easier to extend. For example, a different normal CDF, another root finder, or another smile pricer could be injected without changing the market data classes.


# Caching System
A cache stores results that have already been computed. If the same result is needed again, the pricer can reuse it instead of recomputing it. `VannaVolgaPricer` caches two things:

1. the ATM/RR/BF Greek matrix for each `MarketSlice`
2. the solved pillar weights for target strikes already priced on that slice


## Slice Cache
The pricer stores slice caches in:

```java
private final Map<MarketSlice, SliceCache> sliceCaches =
        Collections.synchronizedMap(new WeakHashMap<>());
```

The key is a `MarketSlice`, and the value is a `SliceCache`. 

The use of `WeakHashMap` means the cache doesn't force old market slices to remain in memory forever. If the caller no longer holds a reference to a `MarketSlice`, the corresponding cache entry can be garbage-collected.

Each slice cache stores:

```java
double[][] atmRrBfGreekMatrix;
Map<Long, double[]> weightsByStrike;
```

The matrix depends only on:

```text
S, rd, rf, T, sigmaATM, sigma25P, sigma25C, K_ATM, K_25P, K_25C
```

It n't depend on the target strike. This makes it reusable across many strikes for the same maturity and market data.



## Strike Weight Cache
For each market slice, the pricer also caches solved pillar weights by target strike.

The strike key is:
$$\text{key}(K) = \text{round}(K \times 10^8) $$

In Java:

```java
private long strikeKey(double K) {
    return Math.round(K * 1e8);
}
```

The cached value is:
$$
(w_{25P}, w_{\text{ATM,pillar}}, w_{25C})
$$

This avoids recomputing target Greeks and solving the 3-by-3 system when the same strike is priced again.


## LRU Eviction
LRU means **Least Recently Used**. The idea is to keep recently used strikes in memory and discard old strikes that are less likely to be requested again.

The strike cache is implemented as an access-order `LinkedHashMap`:

```java
new LinkedHashMap<Long, double[]>(128, 0.75f, true)
```

The final `true` means entries are ordered by access, not insertion. Recently used strikes move toward the end of the map.

The cache is capped by:

```java
MAX_STRIKE_CACHE_PER_SLICE = 2048
```

When the cache grows beyond that limit, Java removes the least recently used entry:

```java
protected boolean removeEldestEntry(Map.Entry<Long, double[]> eldest) {
    return size() > MAX_STRIKE_CACHE_PER_SLICE;
}
```


# Adaptive Finite Differences
`GreeksFD.java` computes Vega, Vanna, and Volga by finite differences.

# Adaptive Finite Differences
Finite differences approximate derivatives by perturbing an input and observing how the output changes.

For a first derivative:
$$f'(x) \approx\frac{f(x+h)-f(x-h)}{2h}$$

For a second derivative:
$$f''(x) \approx \frac{f(x+h)-2f(x)+f(x-h)}{h^2}$$


The central numerical problem is choosing a good bump size $h$.
* if $h$ is too large, the derivative is too coarse because the price is measured over a wide interval. This is called **truncation error**.
* if $h$ is too small, the two prices being subtracted become almost identical: $f(x+h) \approx f(x-h)$

The subtraction can then lose numerical precision. This is called **floating-point cancellation** or **round-off error**.

The script therefore starts from **scale-aware bumps**. This means the initial bump is proportional to the variable being bumped, but it also has a minimum floor so it never becomes numerically meaningless.

For volatility Greeks, such as Vega and Volga, the bumped variable is $\sigma$:
$$h_\sigma = \max(5 \times 10^{-6}, 0.005|\sigma|)$$

The term $0.005|\sigma|$ means the initial volatility bump is 0.5% of the current
volatility level. The term $5 \times 10^{-6}$ is a minimum floor.

<br>

For the spot bump used in Vanna, the bumped variable is $S$:
$$h_S = \max(10^{-6}, 10^{-4}|S|)$$

The term $10^{-4}|S|$ means the initial spot bump is 0.01% of the current spot level. The term $10^{-6}$ is a minimum floor.

After choosing the initial bump, the script refines it by repeatedly halving it:
$$h_{n+1} = \frac{h_n}{2}$$

So the sequence of tested bumps is:
$$h_0,\quad \frac{h_0}{2},\quad \frac{h_0}{4},\quad \frac{h_0}{8},\quad \ldots$$


The loop doesn't refine forever. The maximum number of refinements is:

```java
MAX_REFINEMENTS = 5
```

This means the script can test at most:

$$
h_0,\quad
\frac{h_0}{2},\quad
\frac{h_0}{4},\quad
\frac{h_0}{8},\quad
\frac{h_0}{16},\quad
\frac{h_0}{32}
$$

The value `5` is a practical numerical compromise. It gives the derivative enough chances to stabilize, but it avoids making $h$ so small that round-off error dominates or the calculation becomes unnecessarily expensive.


# Stability Test
The adaptive loops stop when consecutive Richardson-improved estimates are close enough:

$$
|D_n - D_{n-1}| \le \epsilon_{\text{abs}} +\epsilon_{\text{rel}} \max(1, |D_n|, |D_{n-1}|)
$$

The constants are:

```java
ABS_TOL = 1e-10
REL_TOL = 5e-4
```

# Boundary Protection
The finite-difference code avoids invalid bumps through lower bounds:

```java
MIN_SIGMA = 1e-8
MIN_SPOT = 1e-12
MIN_STEP = 1e-8
```

If a symmetric scheme would cross the lower bound, the code uses a one-sided scheme.

For the first derivative near a lower bound:
$$f'(x) \approx \frac{-3f(x)+4f(x+h)-f(x+2h)}{2h}$$

For the second derivative near a lower bound:
$$ f''(x) \approx \frac{f(x)-2f(x+h)+f(x+2h)}{h^2} $$



# Richardson Extrapolation
Richardson extrapolation improves a finite-difference estimate by combining two estimates computed with different step sizes.

Assume:
$$D(h)=D^\* + C h^p + O(h^{p+1})$$

where:
- \(D(h)\) is the derivative estimate using step \(h\)
- \(D^\*\) is the true derivative
- \(p\) is the convergence order

With half the step:
$$D(h/2)=D^\* + C\left(\frac{h}{2}\right)^p + O(h^{p+1})$$

Eliminating the leading error term gives:
$$D_{\text{rich}} = \frac{2^pD(h/2)-D(h)}{2^p-1}$$

For central first derivatives $p = 2$$, so:
$$D_{\text{rich}}=\frac{4D(h/2)-D(h)}{3}$$

For central second derivatives, the order is also $p = 2$.

Near a lower boundary, the Java code may use a one-sided second-derivative scheme to avoid invalid values below the lower bound. In that case the order is $p = 1$, so:
$$D_{\text{rich}} =2D(h/2)-D(h)$$


The Java implementation chooses the order with:

```java
private static int secondDerivativeOrder(double x, double h, double lowerBound) {
    return (x - h > lowerBound) ? 2 : 1;
}
```


# Digital Options
Digital options are priced by finite differences on Vanna-Volga adjusted vanilla prices.

For a digital call:
$$\text{DigitalCall}(K) = -\frac{\partial C(K)}{\partial K} \approx \frac{C(K-\epsilon)-C(K+\epsilon)}{2\epsilon}$$

For a digital put:
$$\text{DigitalPut}(K)=\frac{\partial P(K)}{\partial K} \approx \frac{P(K+\epsilon)-P(K-\epsilon)}{2\epsilon}$$

The finite-difference approximation is:
$$\text{DigitalPut}(K) \approx \frac{P(K+\epsilon)-P(K-\epsilon)}{2\epsilon}$$

The strike bump is:
$$\epsilon = \max(10^{-6}, 10^{-4}K)$$

Because the vanilla prices are already Vanna-Volga adjusted, the digital prices inherit the smile correction.


# End-To-End Flow
The demo follows this pipeline:
1. choose market inputs and smile quotes
2. parse the delta convention
3. instantiate numerical components
4. build a `MarketSlice`
5. convert delta quotes into strikes
6. build or retrieve the cached pillar Greek matrix
7. compute target-strike Greeks with adaptive finite differences and Richardson
   extrapolation
8. solve the ATM/RR/BF Vanna-Volga system
9. convert ATM/RR/BF weights into pillar weights
10. cache strike-level weights
11. price vanilla calls and puts
12. price digital calls and puts by finite differences on VV vanilla prices







