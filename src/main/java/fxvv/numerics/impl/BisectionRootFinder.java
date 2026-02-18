package main.java.fxvv.numerics.impl;

import java.util.function.DoubleUnaryOperator;
import main.java.fxvv.numerics.RootFinder;

public class BisectionRootFinder implements RootFinder {

    private final int maxIter;
    private final double tol;

    public BisectionRootFinder() {
        this(100, 1e-12);
    }

    public BisectionRootFinder(int maxIter, double tol) {
        this.maxIter = maxIter;
        this.tol = tol;
    }

    public double solve(DoubleUnaryOperator f, double lo, double hi) {
        double flo = f.applyAsDouble(lo);
        double fhi = f.applyAsDouble(hi);

        if (Double.isNaN(flo) || Double.isNaN(fhi)) {
            throw new IllegalArgumentException("RootFinder: f(lo) or f(hi) is NaN.");
        }
        if (flo * fhi > 0.0) {
            throw new IllegalArgumentException("RootFinder: root not bracketed.");
        }

        double a = lo, b = hi, fa = flo, fb = fhi;

        for (int i = 0; i < maxIter; i++) {
            double m = 0.5 * (a + b);
            double fm = f.applyAsDouble(m);

            if (Math.abs(fm) < tol || (b - a) < tol) {
                return m;
            }

            if (fa * fm <= 0.0) {
                b = m; fb = fm;
            } else {
                a = m; fa = fm;
            }
        }
        return 0.5 * (a + b);
    }
}

