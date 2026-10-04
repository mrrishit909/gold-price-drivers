"""Step 1: the monthly price of gold since 1960 and the macro series that are said to move it.

    ./venv/bin/python download.py   -> data/CMO-Historical-Data-Monthly.xlsx  DFII10.csv  T10YIE.csv  DTWEXBGS.csv  VIXCLS.csv  CPIAUCSL.csv, results/pull.json

World Bank Commodity Markets, "Pink Sheet" monthly prices in nominal US dollars, 1960 to present (sheet "Monthly Prices", column "Gold", $/troy oz; updated 2 October 2026), https://www.worldbank.org/en/research/commodity-markets, licence CC BY 4.0 as the World Bank states for its data. Its note: gold is "spot average of daily
rates, from June 2025; previously (UK), 99.5% fine, London afternoon fixing, average of daily rates": the series changes definition in June 2025. FRED (Federal Reserve Bank of St. Louis, https://fred.stlouisfed.org/series/<id>), public series with their sources: DFII10, the 10-year Treasury inflation-indexed
(real) yield, and T10YIE, the 10-year breakeven inflation rate (Board of Governors of the Federal Reserve System, from US Treasury data); DTWEXBGS, the nominal broad US dollar index (Board of Governors, from January 2006); VIXCLS, the CBOE volatility index (Chicago Board Options Exchange);
CPIAUCSL, the consumer price index for all urban consumers, seasonally adjusted (US Bureau of Labor Statistics). The files are fetched, never committed; the checksums are in results/pull.json. FRED series are revised and extended daily, so a later download has more recent days.
"""
import hashlib
import json
import ssl
import time
import urllib.request
from datetime import date
from pathlib import Path

import certifi

HERE = Path(__file__).parent
FILES = {"CMO-Historical-Data-Monthly.xlsx": "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx"}
FILES.update({f"{s}.csv": f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}" for s in ("DFII10", "T10YIE", "DTWEXBGS", "VIXCLS", "CPIAUCSL")})
CTX = ssl.create_default_context(cafile=certifi.where())


def main():
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    info = {"pulled": date.today().isoformat(), "files": {}}
    for name, url in FILES.items():
        for attempt in range(4):
            try:
                raw = urllib.request.urlopen(url, context=CTX, timeout=60).read(); break          # FRED stalls any request with a custom User-Agent header, so the default one is used
            except OSError:
                if attempt == 3: raise
                time.sleep(5)
        (HERE / "data" / name).write_bytes(raw)
        info["files"][name] = {"url": url, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    (HERE / "results" / "pull.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: v["bytes"] for k, v in info["files"].items()}))


if __name__ == "__main__":
    main()
