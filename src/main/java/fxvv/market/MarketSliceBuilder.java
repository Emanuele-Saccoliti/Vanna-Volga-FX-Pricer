package main.java.fxvv.market;

import java.util.function.DoubleUnaryOperator;
import main.java.fxvv.bs.GKBlackScholes;
import main.java.fxvv.conventions.DeltaConvention;
import main.java.fxvv.numerics.RootFinder;

public class MarketSliceBuilder {

    private static final int BRACKET_SCAN_STEPS = 240;

    private final GKBlackScholes bs;
    private final RootFinder rootFinder;
    private final DeltaConvention deltaConvention;

    public MarketSliceBuilder(GKBlackScholes bs, RootFinder rootFinder, DeltaConvention deltaConvention) {
        this.bs = bs;
        this.rootFinder = rootFinder;
        this.deltaConvention = deltaConvention;
    }

    // Backward-compatible constructor; build() will fail fast unless BS is injected.
    public MarketSliceBuilder(RootFinder rootFinder, DeltaConvention deltaConvention) {
        this(null, rootFinder, deltaConvention);
    }

    public MarketSlice build(double S, double rd, double rf, SmileQuote q) {
        if (bs == null) {
            throw new IllegalStateException("MarketSliceBuilder requires GKBlackScholes injection to compute deltas.");
        }

        double T = q.T;

        double sigmaATM = q.sigmaATM;
        double sigma25P = q.sigma25P();
        double sigma25C = q.sigma25C();

        double F = GKBlackScholes.forward(S, rd, rf, T);
        double K_ATM = F; // ATM-forward proxy

        double K_25C = strikeFromDelta(S, rd, rf, T, sigma25C, true, 0.25);
        double K_25P = strikeFromDelta(S, rd, rf, T, sigma25P, false, -0.25);

        return new MarketSlice(S, rd, rf, T, sigmaATM, sigma25P, sigma25C, K_ATM, K_25P, K_25C);
    }

    private double strikeFromDelta(double S, double rd, double rf, double T, double sigma,
                                   boolean isCall, double targetDelta) {
        double F = GKBlackScholes.forward(S, rd, rf, T);

        DoubleUnaryOperator f = (K) ->
                bs.deltaInstance(deltaConvention, isCall, S, K, T, rd, rf, sigma) - targetDelta;

        double[] scanRange = scanRange(F, isCall);
        double[] bracket = findBracketByScan(f, scanRange[0], scanRange[1], targetDelta);
        return rootFinder.solve(f, bracket[0], bracket[1]);
    }

    private double[] scanRange(double F, boolean isCall) {
        double minK = 0.05 * F;
        double maxK = 20.0 * F;

        if (deltaConvention == DeltaConvention.SPOT_PREM_INCLUDED) {
            // Premium-included delta can be non-monotonic over (0, +inf); use OTM side only.
            if (isCall) {
                return new double[]{Math.max(F, minK), maxK};
            }
            return new double[]{minK, Math.min(F, maxK)};
        }

        return new double[]{minK, maxK};
    }

    private double[] findBracketByScan(DoubleUnaryOperator f, double lo, double hi, double targetDelta) {
        double prevK = lo;
        double prevV = f.applyAsDouble(prevK);

        if (Double.isNaN(prevV)) {
            throw new IllegalArgumentException("Delta inversion produced NaN at K=" + prevK);
        }

        for (int i = 1; i <= BRACKET_SCAN_STEPS; i++) {
            double t = (double) i / (double) BRACKET_SCAN_STEPS;
            double k = lo * Math.pow(hi / lo, t);
            double v = f.applyAsDouble(k);

            if (Double.isNaN(v)) {
                continue;
            }
            if (prevV == 0.0) {
                return new double[]{prevK, prevK};
            }
            if (prevV * v <= 0.0) {
                return new double[]{prevK, k};
            }

            prevK = k;
            prevV = v;
        }

        throw new IllegalArgumentException(
                "Could not bracket delta->strike root for convention=" + deltaConvention
                        + ", targetDelta=" + targetDelta
                        + ", strikeRange=[" + lo + ", " + hi + "]"
        );
    }
}
