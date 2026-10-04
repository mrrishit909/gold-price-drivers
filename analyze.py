"""Steps 2-4: audit the series, then ask what moves the price of gold, whether that held up, and whether it predicts anything.

    ./venv/bin/python analyze.py
        -> results/audit.csv  longrun.csv  decades.csv  regress.csv  chow.csv  oos_post.csv  rolling.csv  forecast.csv  acf.csv  summary.csv

Data: the World Bank's monthly gold price ($ per troy ounce, an average of daily rates), 1960 to September 2026, and FRED's 10-year real yield (DFII10), 10-year breakeven inflation (T10YIE), broad dollar index (DTWEXBGS), VIX (VIXCLS) and consumer prices (CPIAUCSL). Daily series are averaged
over each calendar month, to match the gold price, which is itself a monthly average; the analysis stops at August 2026, the last month for which every series is complete. The monthly return of gold is 100 times the change in its log. The four drivers are the monthly changes of the real yield (percentage
points), of the log of the dollar index (times 100), of breakeven inflation (percentage points) and of the VIX (points), from February 2006, when the dollar index starts. The regressions are OLS with Newey-West (Bartlett, 3 lags, n/(n-k)) standard errors. The forecast test refits every model on all months
before the one being forecast, from February 2014, using only drivers and returns known the month before (the "oracle" row uses the month's own drivers, which no forecaster would have).
"""
import csv
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
D, R = HERE / "data", HERE / "results"
LAST = "2026-08"
DRIVERS = ["d_real_yield", "d_dollar", "d_breakeven", "d_vix"]
PRE_END, FC_START, LAGS, WIN = "2021-12", "2014-02", 3, 60


def gold():
    ws = openpyxl.load_workbook(D / "CMO-Historical-Data-Monthly.xlsx", data_only=True)["Monthly Prices"]; rows = list(ws.iter_rows(values_only=True)); col = [i for i, c in enumerate(rows[4]) if c == "Gold"][0]
    return pd.Series({pd.Period(r[0].replace("M", "-"), "M"): float(r[col]) for r in rows[6:] if r[0] and isinstance(r[col], (int, float))})


def fred(sid):
    d = pd.read_csv(D / f"{sid}.csv"); d.columns = ["date", "v"]; d["v"] = pd.to_numeric(d["v"], errors="coerce"); d = d.dropna(); d["m"] = pd.to_datetime(d["date"]).dt.to_period("M"); return d.groupby("m")["v"].mean(), len(d)


def ols(X, y):
    Xa = np.c_[np.ones(len(y)), X]; inv = np.linalg.inv(Xa.T @ Xa); b = inv @ Xa.T @ y; e = y - Xa @ b; return b, inv, e, Xa


def hac(Xa, e, inv, lags=LAGS):
    n, k = Xa.shape; u = Xa * e[:, None]; S = u.T @ u
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1); G = u[l:].T @ u[:-l]; S += w * (G + G.T)
    return np.sqrt(np.diag(inv @ S @ inv * n / (n - k)))


def dm(d, lags=LAGS):
    n = len(d); e = d - d.mean(); g = lambda l: float((e[l:] * e[:n - l]).sum() / n); var = (g(0) + 2 * sum((1 - l / (lags + 1)) * g(l) for l in range(1, lags + 1))) / n; t = d.mean() / np.sqrt(var); return float(t), float(2 * stats.norm.sf(abs(t)))


def acf(x, k):
    m = x.mean(); return float(((x[:-k] - m) * (x[k:] - m)).sum() / ((x - m) ** 2).sum())


def main():
    R.mkdir(exist_ok=True)
    g = gold(); series, counts = {}, {}
    for sid in ("DFII10", "T10YIE", "DTWEXBGS", "VIXCLS", "CPIAUCSL"):
        series[sid], counts[sid] = fred(sid)
    idx = pd.period_range("1960-01", LAST, freq="M"); cpi = series["CPIAUCSL"].reindex(idx); gl = g.reindex(idx); real = gl * cpi.iloc[-1] / cpi; rd = real.dropna()                 # the October 2025 CPI was never published, so one month has no real price
    audit = [("gold months", len(g)), ("first gold month", int(str(g.index[0]).replace("-", ""))), ("last gold month", int(str(g.index[-1]).replace("-", ""))), ("gold months with no price", int(g.reindex(pd.period_range(g.index[0], g.index[-1], freq="M")).isna().sum())), ("analysis ends", int(LAST.replace("-", ""))), ("months without a consumer price index, 1960 to the end", int(cpi.isna().sum())), ("first month without one", int(str(cpi[cpi.isna()].index[0]).replace("-", "")) if cpi.isna().any() else 0)]
    for sid in ("DFII10", "T10YIE", "DTWEXBGS", "VIXCLS", "CPIAUCSL"): audit += [(f"{sid} observations", counts[sid]), (f"{sid} first month", int(str(series[sid].index[0]).replace("-", ""))), (f"{sid} last month", int(str(series[sid].index[-1]).replace("-", "")))]
    pd.DataFrame(audit, columns=["check", "value"]).to_csv(R / "audit.csv", index=False)
    # long run, nominal and in August 2026 dollars
    npk = gl.idxmax(); r80 = rd[rd.index <= pd.Period("1999-12", "M")]; r80pk = r80.idxmax(); after = rd[(rd.index > r80pk) & (rd >= rd[r80pk])]; rtr = rd[(rd.index > r80pk) & (rd.index <= pd.Period("2010-12", "M"))].idxmin()
    lr = {"nominal price, 1960-01": float(gl.iloc[0]), "nominal price, latest month": float(gl.iloc[-1]), "nominal peak": float(gl[npk]), "nominal peak month": int(str(npk).replace("-", "")), "real peak before 2000 (August 2026 dollars)": float(real[r80pk]), "real peak month before 2000": int(str(r80pk).replace("-", "")),
          "real low after that peak (August 2026 dollars)": float(real[rtr]), "real low month": int(str(rtr).replace("-", "")), "real price, latest month": float(rd.iloc[-1]), "first month the real price regained that peak": int(str(after.index[0]).replace("-", "")) if len(after) else 0, "months below the real peak": int((after.index[0] - r80pk).n) if len(after) else 0,
          "share of months in 1971-2026 with a real price below the real 1980 peak": float((rd[rd.index >= pd.Period("1971-01", "M")] < rd[r80pk]).mean())}
    pd.DataFrame(list(lr.items()), columns=["key", "value"]).to_csv(R / "longrun.csv", index=False)
    dec = []
    for y0 in range(1970, 2030, 10):
        a, b = pd.Period(f"{y0}-01", "M"), min(pd.Period(f"{y0 + 9}-12", "M"), idx[-1]); yrs = (b - a).n / 12
        dec.append({"decade": f"{y0}s", "months": (b - a).n + 1, "nominal_start": float(gl[a]), "nominal_end": float(gl[b]), "nominal_cagr": float((gl[b] / gl[a]) ** (1 / yrs) - 1), "real_cagr": float((real[b] / real[a]) ** (1 / yrs) - 1)})
    pd.DataFrame(dec).to_csv(R / "decades.csv", index=False)
    # monthly drivers
    m = pd.DataFrame({"gold": gl, "ry": series["DFII10"].reindex(idx), "dollar": series["DTWEXBGS"].reindex(idx), "be": series["T10YIE"].reindex(idx), "vix": series["VIXCLS"].reindex(idx)})
    df = pd.DataFrame({"gold_return": 100 * np.log(m["gold"]).diff(), "d_real_yield": m["ry"].diff(), "d_dollar": 100 * np.log(m["dollar"]).diff(), "d_breakeven": m["be"].diff(), "d_vix": m["vix"].diff()}).loc["2006-02":]; assert df.notna().all().all()
    n = len(df); tr = df.loc[:PRE_END]; po = df.loc["2022-01":]; reg = []
    def fit(name, d):
        b, inv, e, Xa = ols(d[DRIVERS].to_numpy(), d["gold_return"].to_numpy()); se = hac(Xa, e, inv); r2 = 1 - float(e @ e) / float(((d["gold_return"] - d["gold_return"].mean()) ** 2).sum())
        for j, t in enumerate(["intercept"] + DRIVERS): reg.append({"sample": name, "term": t, "coefficient": float(b[j]), "se": float(se[j]), "t": float(b[j] / se[j]), "r2": r2, "n": len(d)})
        return b, e
    bf, ef = fit("2006-02 to 2026-08", df); bp, ep = fit("2006-02 to 2021-12", tr); bq, eq = fit("2022-01 to 2026-08", po)
    pd.DataFrame(reg).to_csv(R / "regress.csv", index=False)
    simple = []
    for name, d in (("2006-02 to 2026-08", df), ("2006-02 to 2021-12", tr), ("2022-01 to 2026-08", po)):
        b, inv, e, Xa = ols(d[["d_real_yield"]].to_numpy(), d["gold_return"].to_numpy()); se = hac(Xa, e, inv); simple.append({"sample": name, "slope": float(b[1]), "se": float(se[1]), "t": float(b[1] / se[1]), "r2": 1 - float(e @ e) / float(((d["gold_return"] - d["gold_return"].mean()) ** 2).sum()), "n": len(d)})
    pd.DataFrame(simple).to_csv(R / "simple.csv", index=False)
    k = len(DRIVERS) + 1; rss_r, rss_u = float(ef @ ef), float(ep @ ep + eq @ eq); F = ((rss_r - rss_u) / k) / (rss_u / (n - 2 * k)); pd.DataFrame([{"f_statistic": F, "df1": k, "df2": n - 2 * k, "p_value": float(stats.f.sf(F, k, n - 2 * k)), "rss_pooled": rss_r, "rss_split": rss_u}]).to_csv(R / "chow.csv", index=False)
    pred = np.c_[np.ones(len(po)), po[DRIVERS].to_numpy()] @ bp; y = po["gold_return"].to_numpy(); sse = float(((y - pred) ** 2).sum()); sst0 = float(((y - tr["gold_return"].mean()) ** 2).sum()); sst1 = float(((y - y.mean()) ** 2).sum())
    pd.DataFrame([{"months": len(po), "r2_oos_vs_pre_mean": 1 - sse / sst0, "r2_in_post_sample_refit": float(1 - float(eq @ eq) / sst1), "rmse_pre_fit_on_post": float(np.sqrt(sse / len(po))), "rmse_post_sd": float(np.sqrt(sst1 / len(po))), "pre_mean_return": float(tr["gold_return"].mean()), "post_mean_return": float(y.mean())}]).to_csv(R / "oos_post.csv", index=False)
    rol = []
    for i in range(WIN, n + 1):
        w = df.iloc[i - WIN:i]; b, inv, e, Xa = ols(w[DRIVERS].to_numpy(), w["gold_return"].to_numpy()); b1, _, e1, _ = ols(w[["d_real_yield"]].to_numpy(), w["gold_return"].to_numpy()); ss = float(((w["gold_return"] - w["gold_return"].mean()) ** 2).sum())
        rol.append({"end_month": str(w.index[-1]), "coef_real_yield": float(b[1]), "coef_dollar": float(b[2]), "r2": 1 - float(e @ e) / ss, "slope_real_yield_alone": float(b1[1]), "r2_real_yield_alone": 1 - float(e1 @ e1) / ss})
    pd.DataFrame(rol).to_csv(R / "rolling.csv", index=False)
    # forecasting next month's return
    yv = df["gold_return"].to_numpy(); X = df[DRIVERS].to_numpy(); mom = df["gold_return"].rolling(12).mean().to_numpy(); start = list(df.index.astype(str)).index(FC_START); preds = {k: [] for k in ("mean", "AR(1)", "momentum", "drivers, lagged", "drivers, same month (oracle)")}; act = []
    for t in range(start, n):
        act.append(yv[t]); preds["mean"].append(yv[1:t].mean() if False else yv[:t].mean())
        def lagfit(Xl, yy):
            b, _, _, _ = ols(Xl, yy); return b
        s = np.arange(13, t)                                  # training months, for models that need 12 lagged months
        b = lagfit(yv[s - 1][:, None], yv[s]); preds["AR(1)"].append(b[0] + b[1] * yv[t - 1])
        b = lagfit(mom[s - 1][:, None], yv[s]); preds["momentum"].append(b[0] + b[1] * mom[t - 1])
        b = lagfit(X[s - 1], yv[s]); preds["drivers, lagged"].append(b[0] + X[t - 1] @ b[1:])
        b = lagfit(X[s], yv[s]); preds["drivers, same month (oracle)"].append(b[0] + X[t] @ b[1:])
    act = np.array(act); em = act - np.array(preds["mean"]); fc = []
    for name, p in preds.items():
        p = np.array(p); e = act - p; row = {"model": name, "months": len(act), "r2_oos": 1 - float((e ** 2).sum()) / float((em ** 2).sum()), "rmse": float(np.sqrt((e ** 2).mean())), "hit_rate": float(((p > 0) == (act > 0)).mean())}
        if name != "mean": t_, p_ = dm(em ** 2 - e ** 2); row.update({"dm_t": t_, "dm_p": p_})
        fc.append(row)
    pd.DataFrame(fc).to_csv(R / "forecast.csv", index=False)
    full = np.log(g[g.index >= pd.Period("1975-01", "M")]).diff().dropna().to_numpy() * 100; recent = df["gold_return"].to_numpy()
    n21 = 21; art = (n21 ** 2 - 1) / (4 * n21 ** 2 + 2)          # lag-1 autocorrelation of the changes of monthly averages of a daily random walk with 21 days a month (Working, 1960): sum (n-i)i / sum ((n-i)^2 + i^2)
    pd.DataFrame([{"sample": s, "lag": l, "autocorrelation": acf(x, l)} for s, x in (("1975-02 to 2026-09", full), ("2006-02 to 2026-08", recent)) for l in (1, 2, 3)] + [{"sample": "averaging alone: daily random walk, 21 days a month", "lag": 1, "autocorrelation": art}]).to_csv(R / "acf.csv", index=False)
    summ = {"months_in_regression": n, "months_before_2022": len(tr), "months_from_2022": len(po), "forecast_months": len(act), "share_positive_return_forecast_sample": float((act > 0).mean()), "monthly_sd_2006_2026": float(df["gold_return"].std(ddof=1)), "annualised_sd_2006_2026": float(df["gold_return"].std(ddof=1) * np.sqrt(12)), **{f"sd_{c}": float(df[c].std(ddof=1)) for c in DRIVERS}, "corr_real_yield_dollar_before_2022": float(tr["d_real_yield"].corr(tr["d_dollar"])), "corr_real_yield_dollar_from_2022": float(po["d_real_yield"].corr(po["d_dollar"])), "corr_real_yield_breakeven_before_2022": float(tr["d_real_yield"].corr(tr["d_breakeven"])), "corr_real_yield_breakeven_from_2022": float(po["d_real_yield"].corr(po["d_breakeven"]))}
    pd.DataFrame(list(summ.items()), columns=["key", "value"]).to_csv(R / "summary.csv", index=False)
    for f in ("audit", "longrun", "decades", "regress", "simple", "chow", "oos_post", "forecast", "acf", "summary"):
        print("==", f); print(pd.read_csv(R / f"{f}.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
