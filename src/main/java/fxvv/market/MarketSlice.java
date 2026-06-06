package main.java.fxvv.market;

public class MarketSlice {
    public final double S;
    public final double rd;
    public final double rf;
    public final double T;

    public final double sigmaATM;
    public final double sigma25P;
    public final double sigma25C;

    public final double K_ATM;
    public final double K_25P;
    public final double K_25C;

    public MarketSlice(double S, double rd, double rf, double T,
                       double sigmaATM, double sigma25P, double sigma25C,
                       double K_ATM, double K_25P, double K_25C) {
        this.S = S; this.rd = rd; this.rf = rf; this.T = T;
        this.sigmaATM = sigmaATM; this.sigma25P = sigma25P; this.sigma25C = sigma25C;
        this.K_ATM = K_ATM; this.K_25P = K_25P; this.K_25C = K_25C;
    }
}
