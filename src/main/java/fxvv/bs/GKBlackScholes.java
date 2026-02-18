package main.java.fxvv.bs;

import main.java.fxvv.conventions.DeltaConvention;
import main.java.fxvv.numerics.NormalDist;

public class GKBlackScholes {

    private final NormalDist N;

    public GKBlackScholes(NormalDist normal) {
        this.N = normal;
    }

    public static double forward(double S, double rd, double rf, double T) {
        return S * Math.exp((rd - rf) * T);
    }

    public static double df(double r, double T) {
        return Math.exp(-r * T);
    }

    public double price(boolean isCall, double S, double K, double T, double rd, double rf, double sigma) {
        if (T <= 0.0) return isCall ? Math.max(S - K, 0.0) : Math.max(K - S, 0.0);

        double F = forward(S, rd, rf, T);
        double DFd = df(rd, T);

        double vs = sigma * Math.sqrt(T);
        double d1 = (Math.log(F / K) + 0.5 * sigma * sigma * T) / vs;
        double d2 = d1 - vs;

        if (isCall) return DFd * (F * N.cdf(d1) - K * N.cdf(d2));
        return DFd * (K * N.cdf(-d2) - F * N.cdf(-d1));
    }

    public static double delta(DeltaConvention conv, boolean isCall,
                               double S, double K, double T, double rd, double rf, double sigma) {
        throw new UnsupportedOperationException("Use GKBlackScholes.deltaInstance(...) with an injected NormalDist.");
    }

    public double deltaInstance(DeltaConvention conv, boolean isCall,
                                double S, double K, double T, double rd, double rf, double sigma) {
        if (T <= 0.0) {
            if (isCall) return (S > K) ? 1.0 : 0.0;
            return (S < K) ? -1.0 : 0.0;
        }

        double F = forward(S, rd, rf, T);
        double vs = sigma * Math.sqrt(T);
        double d1 = (Math.log(F / K) + 0.5 * sigma * sigma * T) / vs;
        double d2 = d1 - vs;

        double DFf = df(rf, T);
        double DFd = df(rd, T);

        switch (conv) {
            case SPOT_PREM_EXCLUDED:
                // Call: DFf * N(d1), Put: -DFf * N(-d1)
                return isCall ? DFf * N.cdf(d1) : -DFf * N.cdf(-d1);

            case FWD_PREM_EXCLUDED:
                // Forward delta (common EM): Call N(d1), Put -N(-d1)
                return isCall ? N.cdf(d1) : -N.cdf(-d1);

            case SPOT_PREM_INCLUDED:
                // Premium-adjusted spot delta:
                // Call: DFd * (K/S) * N(d2), Put: -DFd * (K/S) * N(-d2)
                double premiumAdjustedScale = DFd * (K / S);
                return isCall ? premiumAdjustedScale * N.cdf(d2)
                              : -premiumAdjustedScale * N.cdf(-d2);

            default:
                throw new IllegalArgumentException("Unknown delta convention.");
        }
    }
}
