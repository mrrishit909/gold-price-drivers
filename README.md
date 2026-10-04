# What moves the price of gold?

The monthly price of gold since 1960 and four things said to move it: real interest rates, the dollar, inflation expectations and market fear. Five questions:
- how has gold done in the long run, in dollars and after inflation;
- how much of its monthly moves do the four drivers explain;
- whether that held up after 2022, when gold took off;
- whether the drivers, or anything else, forecast next month's return;
- why last month's return seems to.

The questions follow GeeksforGeeks' "Gold Price Analysis and Forecasting Over Time" (a gold price time series and its forecast), answered with rolling and expanding tests that only use the past. The page with the results and the charts is the build log:

Build log: https://mrrishit909.github.io/projects/gold-price-drivers/

## Data

The World Bank's Commodity Markets "Pink Sheet", monthly prices in nominal US dollars since 1960 (the Gold column, $ per troy ounce, updated 2 October 2026; CC BY 4.0). The World Bank notes that gold is the "spot average of daily rates, from June 2025; previously (UK), 99.5% fine, London afternoon fixing, average of daily rates", so the series changes definition in June 2025. And five series from FRED (Federal Reserve Bank of St. Louis): DFII10 (the 10-year Treasury inflation-indexed, or real, yield) and T10YIE (10-year breakeven inflation), from the Board of Governors; DTWEXBGS (the broad nominal dollar index, from January 2006), Board of Governors; VIXCLS (the CBOE volatility index); and CPIAUCSL (US consumer prices, BLS). The files are about 1.1 MB and are fetched by `download.py` (checksums in `results/pull.json`), not committed. FRED series are extended daily, so a later download has more recent days.

```
./venv/bin/pip install pandas numpy scipy openpyxl matplotlib certifi      # in a venv
./venv/bin/python download.py    # data/*
./venv/bin/python analyze.py     # results/*
./venv/bin/python check.py       # about a second
./venv/bin/python charts.py
```

## Steps

1. **Download** the World Bank workbook and the five FRED series, with their checksums.
2. **Audit.** Months and gaps in each series. The consumer price index for October 2025 was never published (the federal shutdown), so that one month has no real price and is left out of the real-price results.
3. **Long run.** The nominal price, the real price in August 2026 dollars, the 1980 real peak, the real low, the month the real price regained the peak, and the compound growth rate of each decade.
4. **Monthly drivers.** The daily FRED series are averaged over each month, to match gold's monthly average. Gold's monthly return is 100 times the change in its log. Four drivers: the monthly change of the real yield (percentage points), of the log of the dollar index (times 100), of breakeven inflation (points) and of the VIX (points), from February 2006 to August 2026 (247 months). OLS with Newey-West standard errors, for the whole period and for before and from 2022 (a Chow test of whether the two periods share coefficients), and for 60-month rolling windows.
5. **Forecasts.** Next month's return, month by month from February 2014 (151 months): the expanding average, an AR(1) on last month's return, last year's average return, last month's four drivers, and, as a ceiling that no forecaster could use, the same month's drivers. Each model is refitted on all earlier months, and scored by out-of-sample R-squared against the expanding average and a Diebold-Mariano test.
6. **Averaging.** The lag-1 autocorrelation of the returns, against the exact value that averaging a daily random walk over 21 days gives.
7. **Check.** `check.py` rebuilds everything in plain Python (csv, math and loops: no pandas, numpy or scipy; openpyxl only reads the spreadsheet): the monthly means, the long-run table, every regression with its Newey-West errors and the Chow test (the p-value from the incomplete beta function), the rolling regressions, all 151 x 5 expanding-window forecasts, the Diebold-Mariano tests and the autocorrelations: 1,156 numbers, to a relative tolerance of 1e-8.
8. **Charts.**

## Results

- **The long run.** Gold went from $35 an ounce in January 1960 to a peak of $5,020 in February 2026 and $4,411 in August 2026. In August 2026 dollars it peaked at $2,892 in January 1980, fell to $492 in April 2001, and did not regain the 1980 level until February 2025, 45 years later: 97% of the months from 1971 to 2026 were below it. Per year, after inflation: 1970s +21%, 1980s -9%, 1990s -6%, 2000s +12%, 2010s +1%, 2020s +13% (the 2020s to August 2026).
- **Four drivers explain about a third of the monthly moves.** Over 247 months, the real yield, the dollar, breakeven inflation and the VIX explain 35% of the variation of gold's monthly return. A one-point rise in the real yield goes with a -8.2% return, a 1% rise in the dollar with -0.92%, a one-point rise in breakeven inflation with +5.4% and a one-point rise in the VIX with +0.25%; each is more than two standard errors from zero. (A "return" here is the change of a monthly average, so every coefficient is a same-month association, not a forecast.)
- **Whether it held up after 2022.** Before 2022 the model explained 38% and after it 47%, but the coefficients moved: the real yield from -10.4 to -3.6, the dollar from -0.76 to -1.78, breakeven inflation from +4.4 to +12.2, and the baseline drift from +0.5% to +2.0% a month. A Chow test rejects equal coefficients (F = 4.57 on 5 and 237, p = 0.0005). Applied to 2022 to 2026 with the actual drivers, the pre-2022 model has an out-of-sample R-squared of 0.18 against the pre-2022 mean return; the refit gets 0.47.
- **Real yields alone explain less, and share their work with the dollar.** On its own, the real yield explains 28% of the monthly variation before 2022 (slope -12.0) and 19% after it (slope -8.5), 31% less. With the other three held fixed the real-yield coefficient falls by two-thirds, as the real yield and the dollar became more correlated (correlation of their monthly changes 0.34 before 2022, 0.58 since), so part of the real-yield effect shows up in the dollar.
- **Nothing in the drivers forecasts next month's return.** From February 2014, over 151 months, a regression on last month's four drivers has an out-of-sample R-squared of -1.5% against the expanding average (Diebold-Mariano p = 0.69), and last year's average return 1.0% (p = 0.80); 50% of the signs are right, against 53% of months with a positive return. The same month's drivers, which no forecaster has, would reach 23.5% and 72% of the signs: explaining is not predicting.
- **Last month's return seems to predict, and averaging is enough to explain it.** An AR(1) on last month's return has an out-of-sample R-squared of 8.6% (p = 0.003). The monthly price is an average of daily prices, and the changes of averages of a random walk are correlated at lag one by construction: for a random walk in the price, with 21 days a month, the exact value is 0.249 (Working, 1960). The observed lag-1 autocorrelation is 0.20 (1975 to 2026) and 0.19 (2006 to 2026), close to what averaging alone gives.
- **Gold is volatile.** The monthly return has a standard deviation of 3.9% since 2006, 13.5% a year.

## Not done, and caveats

- Gold is a monthly average of daily prices, and so are the drivers here: same-month associations are smoothed, and the averaging creates the apparent AR(1) forecastability described above. Month-end prices, which the World Bank file does not have, would not.
- The World Bank changed the gold series' definition in June 2025 (from the London afternoon fixing to a spot average), which falls inside the last 15 months of the sample; its effect was not tested.
- Real yields, the dollar, breakeven inflation and the VIX are not independent of each other or of the news that moves gold (central-bank buying, geopolitics, flows into exchange-traded funds); none of those was used, and the regressions describe associations, not causes.
- Only linear, constant-coefficient models were fitted, and one break date (2022) was chosen after seeing the price; the rolling regressions show the change is not a single step.
- The out-of-sample test starts in 2014 with eight years of data, and tests one horizon, one month, in one market; other horizons and a volatility forecast were not tried.
- One month (October 2025) has no real price, because the CPI was not published; real-price statistics skip it.
- The World Bank licenses its data CC BY 4.0; FRED series are public data from the sources named. Nothing from the data is committed, only summaries.
