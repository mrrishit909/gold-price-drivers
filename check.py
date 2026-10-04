"""Step 5: the project's check. Recompute with different code, in plain Python straight from the downloaded files (csv, math and loops: no pandas, numpy or scipy; openpyxl only reads the World Bank spreadsheet), and compare with results/:

    ./venv/bin/python check.py

the audit; the monthly means of the daily FRED series; the long-run table (the nominal and real peaks, the real low, the month the real price regained its 1980 peak, the decades); the four-driver regression for the whole period and for each part, with Newey-West standard errors; the Chow F statistic and
its p-value (the regularised incomplete beta function, by continued fraction); the pre-2022 model scored on 2022 to 2026; the 60-month rolling regressions; every expanding-window forecast of next month's return (5 models, 151 months, 755 refits) with R-squared against the expanding mean, Diebold-Mariano tests and hit
rates; the autocorrelations, and the autocorrelation that averaging alone produces. The normal equations are solved by Gauss-Jordan elimination. Comparisons are at a relative tolerance of 1e-8.
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

import openpyxl

HERE = Path(__file__).parent
D, R = HERE / "data", HERE / "results"
LAST = (2026, 8)
DRIVERS = ["d_real_yield", "d_dollar", "d_breakeven", "d_vix"]
LAGS, WIN = 3, 60
n_ok = 0


def same(label, got, want, rel=1e-8):
    global n_ok
    n_ok += 1; want = float(want)
    assert abs(got - want) <= rel * abs(want) + 1e-10, f"{label}: recount {got} != results {want}"


def table(name):
    return list(csv.DictReader(open(R / f"{name}.csv")))


def mean(v):
    return sum(v) / len(v)


def mi(y, m):
    return y * 12 + m - 1


def ym(i):
    return divmod(i, 12)[0], divmod(i, 12)[1] + 1


def inverse(a):
    n = len(a); m = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c])); m[c], m[p] = m[p], m[c]; d = m[c][c]; m[c] = [x / d for x in m[c]]
        for r in range(n):
            if r != c:
                f = m[r][c]; m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [row[n:] for row in m]


def ols(X, y):
    n = len(y); Xa = [[1.0] + list(r) for r in X]; k = len(Xa[0]); xtx = [[sum(Xa[i][a] * Xa[i][b] for i in range(n)) for b in range(k)] for a in range(k)]; inv = inverse(xtx)
    xty = [sum(Xa[i][a] * y[i] for i in range(n)) for a in range(k)]; b = [sum(inv[a][c] * xty[c] for c in range(k)) for a in range(k)]; e = [y[i] - sum(Xa[i][a] * b[a] for a in range(k)) for i in range(n)]
    return b, inv, e, Xa


def hac(Xa, e, inv):
    n, k = len(Xa), len(Xa[0]); u = [[Xa[i][a] * e[i] for a in range(k)] for i in range(n)]; S = [[sum(u[i][a] * u[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    for l in range(1, LAGS + 1):
        w = 1 - l / (LAGS + 1)
        for a in range(k):
            for b in range(k):
                g = sum(u[i][a] * u[i - l][b] for i in range(l, n)) + sum(u[i][b] * u[i - l][a] for i in range(l, n)); S[a][b] += w * g
    V = [[sum(inv[a][p] * S[p][q] * inv[q][b] for p in range(k) for q in range(k)) * n / (n - k) for b in range(k)] for a in range(k)]
    return [math.sqrt(V[a][a]) for a in range(k)]


def rss(e):
    return sum(v * v for v in e)


def tss(y):
    m = mean(y); return sum((v - m) ** 2 for v in y)


def betacf(a, b, x):
    tiny = 1e-300; qab, qap, qam = a + b, a + 1, a - 1; c = 1.0; d = 1 - qab * x / qap; d = tiny if abs(d) < tiny else d; d = 1 / d; h = d
    for m in range(1, 500):
        m2 = 2 * m; aa = m * (b - m) * x / ((qam + m2) * (a + m2)); d = 1 + aa * d; d = tiny if abs(d) < tiny else d; c = 1 + aa / c; c = tiny if abs(c) < tiny else c; d = 1 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)); d = 1 + aa * d; d = tiny if abs(d) < tiny else d; c = 1 + aa / c; c = tiny if abs(c) < tiny else c; d = 1 / d; de = d * c; h *= de
        if abs(de - 1) < 1e-15:
            break
    return h


def betainc(a, b, x):
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    return bt * betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * betacf(b, a, 1 - x) / b


def dm(d):
    n = len(d); m = mean(d); e = [v - m for v in d]; g = lambda l: sum(e[i] * e[i - l] for i in range(l, n)) / n
    var = (g(0) + 2 * sum((1 - l / (LAGS + 1)) * g(l) for l in range(1, LAGS + 1))) / n; t = m / math.sqrt(var); return t, math.erfc(abs(t) / math.sqrt(2))


def acf(x, k):
    m = mean(x); return sum((x[i] - m) * (x[i + k] - m) for i in range(len(x) - k)) / sum((v - m) ** 2 for v in x)


def main():
    ws = openpyxl.load_workbook(D / "CMO-Historical-Data-Monthly.xlsx", data_only=True)["Monthly Prices"]; rows = list(ws.iter_rows(values_only=True)); col = [i for i, c in enumerate(rows[4]) if c == "Gold"][0]
    gold = {mi(int(r[0][:4]), int(r[0][5:])): float(r[col]) for r in rows[6:] if r[0] and isinstance(r[col], (int, float))}
    fred, counts = {}, {}
    for sid in ("DFII10", "T10YIE", "DTWEXBGS", "VIXCLS", "CPIAUCSL"):
        acc = defaultdict(list); cnt = 0
        for r in list(csv.reader(open(D / f"{sid}.csv")))[1:]:
            if r[1] not in ("", "."):
                acc[mi(int(r[0][:4]), int(r[0][5:7]))].append(float(r[1])); cnt += 1
        fred[sid] = {k: mean(v) for k, v in acc.items()}; counts[sid] = cnt
    g0, g1 = min(gold), max(gold); ymd = lambda i: ym(i)[0] * 100 + ym(i)[1]; aud = {r["check"]: float(r["value"]) for r in table("audit")}
    got = {"gold months": len(gold), "first gold month": ymd(g0), "last gold month": ymd(g1), "gold months with no price": sum(1 for i in range(g0, g1 + 1) if i not in gold), "analysis ends": LAST[0] * 100 + LAST[1]}
    for sid in fred:
        got[f"{sid} observations"] = counts[sid]; got[f"{sid} first month"] = ymd(min(fred[sid])); got[f"{sid} last month"] = ymd(max(fred[sid]))
    assert set(got) | {"months without a consumer price index, 1960 to the end", "first month without one"} == set(aud)
    for k, v in got.items():
        same("audit " + k, v, aud[k])
    first, last = mi(1960, 1), mi(*LAST); allidx = list(range(first, last + 1)); cpi = fred["CPIAUCSL"]; nocpi = [i for i in allidx if i not in cpi]; idx = [i for i in allidx if i in cpi]; real = {i: gold[i] * cpi[last] / cpi[i] for i in idx}
    same("months without cpi", len(nocpi), aud["months without a consumer price index, 1960 to the end"]); same("first month without cpi", ymd(nocpi[0]) if nocpi else 0, aud["first month without one"])
    npk = max(allidx, key=lambda i: (gold[i], -i)); r80 = [i for i in idx if i <= mi(1999, 12)]; r80pk = max(r80, key=lambda i: (real[i], -i)); aft = [i for i in idx if i > r80pk and real[i] >= real[r80pk]]; rtr = min([i for i in idx if r80pk < i <= mi(2010, 12)], key=lambda i: (real[i], i))
    lr = {"nominal price, 1960-01": gold[first], "nominal price, latest month": gold[last], "nominal peak": gold[npk], "nominal peak month": ymd(npk), "real peak before 2000 (August 2026 dollars)": real[r80pk], "real peak month before 2000": ymd(r80pk), "real low after that peak (August 2026 dollars)": real[rtr], "real low month": ymd(rtr),
          "real price, latest month": real[last], "first month the real price regained that peak": ymd(aft[0]), "months below the real peak": aft[0] - r80pk, "share of months in 1971-2026 with a real price below the real 1980 peak": mean([1.0 if real[i] < real[r80pk] else 0.0 for i in idx if i >= mi(1971, 1)])}
    t = {r["key"]: float(r["value"]) for r in table("longrun")}; assert set(lr) == set(t)
    for k, v in lr.items():
        same("longrun " + k, v, t[k])
    for r in table("decades"):
        y0 = int(r["decade"][:4]); a, b = mi(y0, 1), min(mi(y0 + 9, 12), last); yrs = (b - a) / 12
        same("months", b - a + 1, r["months"]); same("start", gold[a], r["nominal_start"]); same("end", gold[b], r["nominal_end"]); same("nominal cagr", (gold[b] / gold[a]) ** (1 / yrs) - 1, r["nominal_cagr"]); same("real cagr", (real[b] / real[a]) ** (1 / yrs) - 1, r["real_cagr"])
    ms = list(range(mi(2006, 2), last + 1)); ret, X = [], []
    for i in ms:
        ret.append(100 * (math.log(gold[i]) - math.log(gold[i - 1]))); X.append([fred["DFII10"][i] - fred["DFII10"][i - 1], 100 * (math.log(fred["DTWEXBGS"][i]) - math.log(fred["DTWEXBGS"][i - 1])), fred["T10YIE"][i] - fred["T10YIE"][i - 1], fred["VIXCLS"][i] - fred["VIXCLS"][i - 1]])
    n = len(ms); cut = ms.index(mi(2022, 1)); samples = {"2006-02 to 2026-08": (0, n), "2006-02 to 2021-12": (0, cut), "2022-01 to 2026-08": (cut, n)}; fits = {}
    for name, (a, b) in samples.items():
        beta, inv, e, Xa = ols(X[a:b], ret[a:b]); se = hac(Xa, e, inv); r2 = 1 - rss(e) / tss(ret[a:b]); fits[name] = (beta, e)
        for j, term in enumerate(["intercept"] + DRIVERS):
            row = [r for r in table("regress") if r["sample"] == name and r["term"] == term][0]; same(f"{name} {term}", beta[j], row["coefficient"]); same("se", se[j], row["se"]); same("t", beta[j] / se[j], row["t"]); same("r2", r2, row["r2"]); same("n", b - a, row["n"])
    for name, (a, b) in samples.items():
        b1, inv1, e1, Xa1 = ols([[x[0]] for x in X[a:b]], ret[a:b]); se1 = hac(Xa1, e1, inv1); row = [r for r in table("simple") if r["sample"] == name][0]; same("simple slope", b1[1], row["slope"]); same("simple se", se1[1], row["se"]); same("simple t", b1[1] / se1[1], row["t"]); same("simple r2", 1 - rss(e1) / tss(ret[a:b]), row["r2"]); same("simple n", b - a, row["n"])
    k = 5; rr = rss(fits["2006-02 to 2026-08"][1]); ru = rss(fits["2006-02 to 2021-12"][1]) + rss(fits["2022-01 to 2026-08"][1]); F = ((rr - ru) / k) / (ru / (n - 2 * k)); d2 = n - 2 * k; p = betainc(d2 / 2, k / 2, d2 / (d2 + k * F)); c = table("chow")[0]
    same("F", F, c["f_statistic"]); same("df2", d2, c["df2"]); same("p", p, c["p_value"], 1e-6); same("rss pooled", rr, c["rss_pooled"]); same("rss split", ru, c["rss_split"])
    bp = fits["2006-02 to 2021-12"][0]; yp = ret[cut:]; pred = [bp[0] + sum(bp[1 + j] * x[j] for j in range(4)) for x in X[cut:]]; sse = sum((a - b) ** 2 for a, b in zip(yp, pred)); pm = mean(ret[:cut]); o = table("oos_post")[0]
    same("oos months", len(yp), o["months"]); same("oos r2", 1 - sse / sum((v - pm) ** 2 for v in yp), o["r2_oos_vs_pre_mean"]); same("refit r2", 1 - rss(fits["2022-01 to 2026-08"][1]) / tss(yp), o["r2_in_post_sample_refit"]); same("rmse", math.sqrt(sse / len(yp)), o["rmse_pre_fit_on_post"]); same("sd", math.sqrt(tss(yp) / len(yp)), o["rmse_post_sd"]); same("pre mean", pm, o["pre_mean_return"]); same("post mean", mean(yp), o["post_mean_return"])
    rolls = table("rolling"); assert len(rolls) == n - WIN + 1
    for j, r in enumerate(rolls):
        i = WIN + j; beta, inv, e, Xa = ols(X[i - WIN:i], ret[i - WIN:i]); yy, mm = ym(ms[i - 1]); assert r["end_month"] == f"{yy}-{mm:02d}"; same("rolling real yield", beta[1], r["coef_real_yield"]); same("rolling dollar", beta[2], r["coef_dollar"]); same("rolling r2", 1 - rss(e) / tss(ret[i - WIN:i]), r["r2"])
        b1, _, e1, _ = ols([[x[0]] for x in X[i - WIN:i]], ret[i - WIN:i]); same("rolling slope alone", b1[1], r["slope_real_yield_alone"]); same("rolling r2 alone", 1 - rss(e1) / tss(ret[i - WIN:i]), r["r2_real_yield_alone"])
    mom = [None] * n
    for i in range(11, n):
        mom[i] = mean(ret[i - 11:i + 1])
    start = ms.index(mi(2014, 2)); preds = {k: [] for k in ("mean", "AR(1)", "momentum", "drivers, lagged", "drivers, same month (oracle)")}; act = []
    for t in range(start, n):
        act.append(ret[t]); preds["mean"].append(mean(ret[:t])); s = list(range(13, t))
        b, _, _, _ = ols([[ret[i - 1]] for i in s], [ret[i] for i in s]); preds["AR(1)"].append(b[0] + b[1] * ret[t - 1])
        b, _, _, _ = ols([[mom[i - 1]] for i in s], [ret[i] for i in s]); preds["momentum"].append(b[0] + b[1] * mom[t - 1])
        b, _, _, _ = ols([X[i - 1] for i in s], [ret[i] for i in s]); preds["drivers, lagged"].append(b[0] + sum(b[1 + j] * X[t - 1][j] for j in range(4)))
        b, _, _, _ = ols([X[i] for i in s], [ret[i] for i in s]); preds["drivers, same month (oracle)"].append(b[0] + sum(b[1 + j] * X[t][j] for j in range(4)))
    em = [a - p for a, p in zip(act, preds["mean"])]; fc = table("forecast"); assert len(fc) == 5
    for r in fc:
        pp = preds[r["model"]]; e = [a - p for a, p in zip(act, pp)]; same("months", len(act), r["months"]); same(f"r2 oos {r['model']}", 1 - rss(e) / rss(em), r["r2_oos"]); same("rmse", math.sqrt(rss(e) / len(e)), r["rmse"]); same("hit rate", mean([1.0 if (p > 0) == (a > 0) else 0.0 for p, a in zip(pp, act)]), r["hit_rate"])
        if r["model"] != "mean":
            tt, pv = dm([a * a - b * b for a, b in zip(em, e)]); same("dm t", tt, r["dm_t"]); same("dm p", pv, r["dm_p"], 1e-6)
    full = [100 * (math.log(gold[i]) - math.log(gold[i - 1])) for i in range(mi(1975, 2), g1 + 1)]; n21 = 21
    for r in table("acf"):
        if r["sample"].startswith("averaging"):
            same("artefact", sum((n21 - i) * i for i in range(n21)) / sum((n21 - i) ** 2 + i ** 2 for i in range(n21)), r["autocorrelation"])
        else:
            same(f"acf {r['sample']} {r['lag']}", acf(full if r["sample"].startswith("1975") else ret, int(r["lag"])), r["autocorrelation"])
    sm = {r["key"]: float(r["value"]) for r in table("summary")}
    same("months in regression", n, sm["months_in_regression"]); same("pre", cut, sm["months_before_2022"]); same("post", n - cut, sm["months_from_2022"]); same("forecast months", len(act), sm["forecast_months"]); same("share positive", mean([1.0 if a > 0 else 0.0 for a in act]), sm["share_positive_return_forecast_sample"])
    sd = math.sqrt(tss(ret) / (n - 1)); same("sd", sd, sm["monthly_sd_2006_2026"]); same("annual sd", sd * math.sqrt(12), sm["annualised_sd_2006_2026"])
    for j, name in enumerate(DRIVERS):
        same("sd " + name, math.sqrt(tss([x[j] for x in X]) / (n - 1)), sm["sd_" + name])
    def corr(u, v):
        mu, mv = mean(u), mean(v); return sum((p - mu) * (q - mv) for p, q in zip(u, v)) / math.sqrt(tss(u) * tss(v))
    for nm, lo, hi in (("before_2022", 0, cut), ("from_2022", cut, n)):
        same("corr dollar " + nm, corr([x[0] for x in X[lo:hi]], [x[1] for x in X[lo:hi]]), sm["corr_real_yield_dollar_" + nm]); same("corr breakeven " + nm, corr([x[0] for x in X[lo:hi]], [x[2] for x in X[lo:hi]]), sm["corr_real_yield_breakeven_" + nm])
    print(f"check.py: {n_ok} recomputed numbers agree with results/; everything is written out in plain Python from the downloaded files")


if __name__ == "__main__":
    main()
