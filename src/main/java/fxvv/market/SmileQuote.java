package main.java.fxvv.market;

public class SmileQuote {
    public final double T;
    public final double sigmaATM;
    public final double rr25;
    public final double bf25;

    public SmileQuote(double T, double sigmaATM, double rr25, double bf25) {
        this.T = T;
        this.sigmaATM = sigmaATM;
        this.rr25 = rr25;
        this.bf25 = bf25;
    }

    public double sigma25P() {
        return clamp(sigmaATM + bf25 - 0.5 * rr25);
    }

    public double sigma25C() {
        return clamp(sigmaATM + bf25 + 0.5 * rr25);
    }

    private static double clamp(double v) {
        return Math.max(1e-6, Math.min(3.0, v));
    }
}
