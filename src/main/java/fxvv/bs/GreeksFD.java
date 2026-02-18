package main.java.fxvv.bs;

import java.util.function.DoubleUnaryOperator;

public class GreeksFD {

    private static final double MIN_SIGMA = 1e-8;
    private static final double MIN_SPOT = 1e-12;
    private static final double MIN_STEP = 1e-8;
    private static final int MAX_REFINEMENTS = 5;
    private static final double REL_TOL = 5e-4;
    private static final double ABS_TOL = 1e-10;

    private GreeksFD() {}

    public static double vegaFD(GKBlackScholes bs, boolean isCall, double S, double K, double T, double rd, double rf, double sigma) {
        double safeSigma = Math.max(MIN_SIGMA, sigma);
        DoubleUnaryOperator priceBySigma = (x) -> bs.price(isCall, S, K, T, rd, rf, x);
        double h0 = Math.max(5e-6, Math.abs(safeSigma) * 5e-3);
        return firstDerivativeAdaptive(priceBySigma, safeSigma, h0, MIN_SIGMA);
    }

    public static double volgaFD(GKBlackScholes bs, boolean isCall, double S, double K, double T, double rd, double rf, double sigma) {
        double safeSigma = Math.max(MIN_SIGMA, sigma);
        DoubleUnaryOperator priceBySigma = (x) -> bs.price(isCall, S, K, T, rd, rf, x);
        double h0 = Math.max(5e-6, Math.abs(safeSigma) * 5e-3);
        return secondDerivativeAdaptive(priceBySigma, safeSigma, h0, MIN_SIGMA);
    }

    public static double vannaFD(GKBlackScholes bs, boolean isCall, double S, double K, double T, double rd, double rf, double sigma) {
        double safeSpot = Math.max(MIN_SPOT, S);
        DoubleUnaryOperator vegaBySpot = (x) -> vegaFD(bs, isCall, x, K, T, rd, rf, sigma);
        double hS = Math.max(1e-6, Math.abs(safeSpot) * 1e-4);
        return firstDerivativeAdaptive(vegaBySpot, safeSpot, hS, MIN_SPOT);
    }

    private static double firstDerivativeAdaptive(DoubleUnaryOperator f, double x, double h0, double lowerBound) {
        double h = boundedStep(x, h0, lowerBound);
        double prev = firstDerivative(f, x, h, lowerBound);

        for (int i = 0; i < MAX_REFINEMENTS; i++) {
            double hNext = boundedStep(x, h * 0.5, lowerBound);
            if (hNext >= h) break;
            double curr = firstDerivative(f, x, hNext, lowerBound);
            if (isStable(prev, curr)) return curr;
            prev = curr;
            h = hNext;
        }
        return prev;
    }

    private static double secondDerivativeAdaptive(DoubleUnaryOperator f, double x, double h0, double lowerBound) {
        double h = boundedStep(x, h0, lowerBound);
        double prev = secondDerivative(f, x, h, lowerBound);

        for (int i = 0; i < MAX_REFINEMENTS; i++) {
            double hNext = boundedStep(x, h * 0.5, lowerBound);
            if (hNext >= h) break;
            double curr = secondDerivative(f, x, hNext, lowerBound);
            if (isStable(prev, curr)) return curr;
            prev = curr;
            h = hNext;
        }
        return prev;
    }

    private static double firstDerivative(DoubleUnaryOperator f, double x, double h, double lowerBound) {
        if (x - h > lowerBound) {
            return (f.applyAsDouble(x + h) - f.applyAsDouble(x - h)) / (2.0 * h);
        }
        return (-3.0 * f.applyAsDouble(x)
                + 4.0 * f.applyAsDouble(x + h)
                - f.applyAsDouble(x + 2.0 * h)) / (2.0 * h);
    }

    private static double secondDerivative(DoubleUnaryOperator f, double x, double h, double lowerBound) {
        if (x - h > lowerBound) {
            return (f.applyAsDouble(x + h) - 2.0 * f.applyAsDouble(x) + f.applyAsDouble(x - h)) / (h * h);
        }
        return (f.applyAsDouble(x) - 2.0 * f.applyAsDouble(x + h) + f.applyAsDouble(x + 2.0 * h)) / (h * h);
    }

    private static double boundedStep(double x, double h, double lowerBound) {
        double candidate = Math.max(MIN_STEP, h);
        if (x > lowerBound) {
            double maxSymmetricStep = 0.5 * (x - lowerBound);
            if (maxSymmetricStep > 0.0) {
                candidate = Math.min(candidate, maxSymmetricStep);
            }
        }
        return Math.max(MIN_STEP, candidate);
    }

    private static boolean isStable(double a, double b) {
        double scale = Math.max(1.0, Math.max(Math.abs(a), Math.abs(b)));
        return Math.abs(a - b) <= ABS_TOL + REL_TOL * scale;
    }
}
