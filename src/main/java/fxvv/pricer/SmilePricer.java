package main.java.fxvv.pricer;

import main.java.fxvv.market.MarketSlice;

public interface SmilePricer {
    double priceVanilla(MarketSlice slice, boolean isCall, double K);
    double priceDigitalCall(MarketSlice slice, double K);
    double priceDigitalPut(MarketSlice slice, double K);
}
