package main.java.fxvv.numerics;

import java.util.function.DoubleUnaryOperator;

public interface RootFinder {
    double solve(DoubleUnaryOperator f, double lo, double hi);
}
