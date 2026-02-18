package main.java.fxvv.numerics.impl;

import main.java.fxvv.numerics.LinearSolver;

public class GaussianElimination3 implements LinearSolver {

    public double[] solve(double[][] A, double[] b) {
        if (A.length != 3 || A[0].length != 3 || b.length != 3) {
            throw new IllegalArgumentException("LinearSolver3 expects 3x3 matrix and length-3 vector.");
        }

        // Copy to avoid side effects
        double[][] M = new double[3][3];
        double[] y = new double[3];
        for (int i=0;i<3;i++){
            System.arraycopy(A[i], 0, M[i], 0, 3);
            y[i] = b[i];
        }

     
        for (int col = 0; col < 3; col++) {
            int piv = col;
            double best = Math.abs(M[col][col]);
            for (int r = col + 1; r < 3; r++) {
                double v = Math.abs(M[r][col]);
                if (v > best) { best = v; piv = r; }
            }
            if (best < 1e-14) {
                throw new ArithmeticException("LinearSolver3: singular/ill-conditioned matrix.");
            }
            if (piv != col) {
                double[] tmp = M[col]; M[col] = M[piv]; M[piv] = tmp;
                double ty = y[col]; y[col] = y[piv]; y[piv] = ty;
            }

            for (int r = col + 1; r < 3; r++) {
                double f = M[r][col] / M[col][col];
                for (int c = col; c < 3; c++) {
                    M[r][c] -= f * M[col][c];
                }
                y[r] -= f * y[col];
            }
        }

        
        double[] x = new double[3];
        for (int i = 2; i >= 0; i--) {
            double s = y[i];
            for (int j = i + 1; j < 3; j++) s -= M[i][j] * x[j];
            x[i] = s / M[i][i];
        }
        return x;
    }
}
