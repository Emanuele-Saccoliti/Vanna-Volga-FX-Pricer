package main.java.fxvv.numerics.impl;

import main.java.fxvv.numerics.NormalDist;

public class AbramowitzStegunNormal implements NormalDist {


    public double pdf(double x) {
        return Math.exp(-0.5 * x * x) / Math.sqrt(2.0 * Math.PI);
    }


    public double cdf(double x) {
        // Abramowitz-Stegun approximation
        double a1 = 0.319381530;
        double a2 = -0.356563782;
        double a3 = 1.781477937;
        double a4 = -1.821255978;
        double a5 = 1.330274429;

        double L = Math.abs(x);
        double k = 1.0 / (1.0 + 0.2316419 * L);

        double poly = a1*k + a2*Math.pow(k,2) + a3*Math.pow(k,3) + a4*Math.pow(k,4) + a5*Math.pow(k,5);
        double w = 1.0 - pdf(L) * poly;

        return (x < 0.0) ? (1.0 - w) : w;
    }
}
