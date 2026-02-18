package main.java.fxvv.pricer;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.WeakHashMap;
import main.java.fxvv.bs.GKBlackScholes;
import main.java.fxvv.bs.GreeksFD;
import main.java.fxvv.market.MarketSlice;
import main.java.fxvv.numerics.LinearSolver;

public class VannaVolgaPricer implements SmilePricer {

    private static final int MAX_STRIKE_CACHE_PER_SLICE = 2048;

    private final GKBlackScholes bs;
    private final LinearSolver solver;

    // Weak keys avoid retaining MarketSlice instances once callers drop them.
    private final Map<MarketSlice, SliceCache> sliceCaches =
            Collections.synchronizedMap(new WeakHashMap<>());

    public VannaVolgaPricer(GKBlackScholes bs, LinearSolver solver) {
        this.bs = bs;
        this.solver = solver;
    }

    public double priceVanilla(MarketSlice slice, boolean isCall, double K) {
        double T = slice.T;

        // Base at ATM vol
        double base = bs.price(isCall, slice.S, K, T, slice.rd, slice.rf, slice.sigmaATM);

        // Solve in ATM/RR/BF basis, then convert to pillar weights [25P, ATM, 25C].
        double[] w = vvWeightsAtATM(slice, K);

        // Correction computed at target strike using pillar vols
        double p25P = bs.price(isCall, slice.S, K, T, slice.rd, slice.rf, slice.sigma25P);
        double pATM = bs.price(isCall, slice.S, K, T, slice.rd, slice.rf, slice.sigmaATM);
        double p25C = bs.price(isCall, slice.S, K, T, slice.rd, slice.rf, slice.sigma25C);

        return base + w[0] * (p25P - pATM) + w[2] * (p25C - pATM);
    }


    public double priceDigitalCall(MarketSlice slice, double K) {
        double eps = strikeEps(K);
        double cDn = priceVanilla(slice, true, K - eps);
        double cUp = priceVanilla(slice, true, K + eps);
        return (cDn - cUp) / (2.0 * eps); // approximately -dC/dK
    }


    public double priceDigitalPut(MarketSlice slice, double K) {
        double eps = strikeEps(K);
        double pDn = priceVanilla(slice, false, K - eps);
        double pUp = priceVanilla(slice, false, K + eps);
        return (pUp - pDn) / (2.0 * eps); // approximately dP/dK
    }

    private double strikeEps(double K) {
        return Math.max(1e-6, K * 1e-4);
    }

    // Returns pillar weights ordered as [25P, ATM, 25C].
    private double[] vvWeightsAtATM(MarketSlice slice, double KTarget) {
        SliceCache sliceCache = getOrBuildSliceCache(slice);
        long key = strikeKey(KTarget);

        synchronized (sliceCache.weightsByStrike) {
            double[] cached = sliceCache.weightsByStrike.get(key);
            if (cached != null) return cached;
        }

        double[] targetGreeks = greekVector(slice, KTarget, slice.sigmaATM);

        // Solve in ATM/RR/BF basis.
        double[] wAtmRrBf = solver.solve(sliceCache.atmRrBfGreekMatrix, targetGreeks);
        // Convert to pillar basis [25P, ATM, 25C] used by pricing adjustment.
        double[] wPillars = rrBfToPillar(wAtmRrBf);

        synchronized (sliceCache.weightsByStrike) {
            sliceCache.weightsByStrike.put(key, wPillars);
        }
        return wPillars;
    }

    private SliceCache getOrBuildSliceCache(MarketSlice slice) {
        synchronized (sliceCaches) {
            SliceCache cached = sliceCaches.get(slice);
            if (cached != null) return cached;

            SliceCache built = buildSliceCache(slice);
            sliceCaches.put(slice, built);
            return built;
        }
    }

    private SliceCache buildSliceCache(MarketSlice slice) {
        double sigma = slice.sigmaATM;
        double[] greek25P = greekVector(slice, slice.K_25P, sigma);
        double[] greekATM = greekVector(slice, slice.K_ATM, sigma);
        double[] greek25C = greekVector(slice, slice.K_25C, sigma);

        double[] greekRR = new double[]{
                greek25C[0] - greek25P[0],
                greek25C[1] - greek25P[1],
                greek25C[2] - greek25P[2]
        };

        double[] greekBF = new double[]{
                0.5 * (greek25C[0] + greek25P[0]) - greekATM[0],
                0.5 * (greek25C[1] + greek25P[1]) - greekATM[1],
                0.5 * (greek25C[2] + greek25P[2]) - greekATM[2]
        };

        // Columns are [ATM, RR, BF].
        double[][] matrix = new double[][]{
                {greekATM[0], greekRR[0], greekBF[0]},
                {greekATM[1], greekRR[1], greekBF[1]},
                {greekATM[2], greekRR[2], greekBF[2]}
        };

        return new SliceCache(matrix, new LinkedHashMap<Long, double[]>(128, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<Long, double[]> eldest) {
                return size() > MAX_STRIKE_CACHE_PER_SLICE;
            }
        });
    }

    private static double[] rrBfToPillar(double[] wAtmRrBf) {
        double wATM = wAtmRrBf[0];
        double wRR = wAtmRrBf[1];
        double wBF = wAtmRrBf[2];

        double w25P = -wRR + 0.5 * wBF;
        double wAtmPillar = wATM - wBF;
        double w25C = wRR + 0.5 * wBF;

        return new double[]{w25P, wAtmPillar, w25C};
    }

    private double[] greekVector(MarketSlice slice, double K, double sigma) {
        boolean isCallForGreeks = true;
        double vega = GreeksFD.vegaFD(bs, isCallForGreeks, slice.S, K, slice.T, slice.rd, slice.rf, sigma);
        double vanna = GreeksFD.vannaFD(bs, isCallForGreeks, slice.S, K, slice.T, slice.rd, slice.rf, sigma);
        double volga = GreeksFD.volgaFD(bs, isCallForGreeks, slice.S, K, slice.T, slice.rd, slice.rf, sigma);
        return new double[]{vega, vanna, volga};
    }

    private long strikeKey(double K) {
        return Math.round(K * 1e8);
    }

    private static final class SliceCache {
        private final double[][] atmRrBfGreekMatrix;
        private final Map<Long, double[]> weightsByStrike;

        private SliceCache(double[][] atmRrBfGreekMatrix, Map<Long, double[]> weightsByStrike) {
            this.atmRrBfGreekMatrix = atmRrBfGreekMatrix;
            this.weightsByStrike = weightsByStrike;
        }
    }
}
