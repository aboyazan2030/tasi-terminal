"""
fetch_data.py — مشروع مستقل تمامًا (Tasi Terminal)
لا يشارك أي كود أو ملف مع أي مشروع آخر.

يجلب بيانات قطاع البنوك السعودية عبر yfinance، ويحسب:
- RSI14
- نسبة الحجم غير الاعتيادي (Volume Ratio)
- درجة "التجميع الذكي" (Accumulation Score) من OBV + الحجم + القرب من الدعم
- الدعم والمقاومة (آخر 20 يوم)
- المسافة عن أعلى/أدنى 52 أسبوع
- تصنيف نصي للحالة

يُشغَّل عبر GitHub Actions (راجع .github/workflows/update_data.yml)
ويكتب النتيجة إلى data/banks_sa.json
"""

import json
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

# قطاع تجريبي أول (المرحلة 1 من خطة التطوير) — البنوك السعودية الرئيسية
SECTOR_NAME = "البنوك السعودية"
TICKERS = {
    "1120.SR": "مصرف الراجحي",
    "1180.SR": "البنك الأهلي السعودي",
    "1010.SR": "بنك الرياض",
    "1050.SR": "البنك السعودي الفرنسي",
    "1060.SR": "بنك الجزيرة",
    "1150.SR": "مصرف الإنماء",
    "1080.SR": "البنك السعودي للاستثمار",
    "1030.SR": "البنك العربي الوطني",
}

OUTPUT_PATH = "data/banks_sa.json"


def compute_rsi(closes: pd.Series, period: int = 14) -> float:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0


def compute_obv_slope(closes: pd.Series, volumes: pd.Series, lookback: int = 10) -> float:
    direction = np.sign(closes.diff().fillna(0))
    obv = (direction * volumes).cumsum()
    recent = obv.tail(lookback)
    if len(recent) < 2:
        return 0.0
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent.values, 1)[0]
    norm = slope / (volumes.tail(lookback).mean() + 1e-9)
    return float(np.clip(norm * 100, -100, 100))


def classify(accumulation_score: float, rsi: float) -> str:
    if accumulation_score >= 65 and rsi < 70:
        return "تجميع نشط"
    if accumulation_score <= 35 and rsi > 30:
        return "تصريف"
    if rsi >= 70:
        return "تشبع شرائي"
    if rsi <= 30:
        return "تشبع بيعي"
    return "محايد"


def analyze_ticker(symbol: str, name: str) -> dict | None:
    try:
        hist = yf.Ticker(symbol).history(period="1y", interval="1d")
    except Exception:
        return None

    if hist is None or len(hist) < 30:
        return None

    closes = hist["Close"]
    volumes = hist["Volume"]

    price = float(closes.iloc[-1])
    prev_price = float(closes.iloc[-2])
    change_pct = ((price - prev_price) / prev_price) * 100

    rsi14 = compute_rsi(closes)

    avg_vol_20 = float(volumes.tail(20).mean())
    today_vol = float(volumes.iloc[-1])
    volume_ratio = (today_vol / avg_vol_20) if avg_vol_20 > 0 else 1.0

    obv_slope = compute_obv_slope(closes, volumes)

    support = float(closes.tail(20).min())
    resistance = float(closes.tail(20).max())

    week52_low = float(closes.min())
    week52_high = float(closes.max())
    dist_from_low_pct = ((price - week52_low) / week52_low) * 100

    proximity_to_support = max(0.0, 1 - ((price - support) / (resistance - support + 1e-9)))
    accumulation_score = float(np.clip(
        (obv_slope * 0.5) + (min(volume_ratio, 3) / 3 * 30) + (proximity_to_support * 20) + 50,
        0, 100
    ))

    classification = classify(accumulation_score, rsi14)

    atr = float((hist["High"] - hist["Low"]).tail(14).mean())
    stop_loss = round(support - atr * 0.5, 2)
    target1 = round(price + atr * 1.5, 2)
    target2 = round(price + atr * 3, 2)
    target3 = round(resistance, 2)

    return {
        "code": symbol.replace(".SR", ""),
        "name": name,
        "price": round(price, 2),
        "change_pct": round(change_pct, 2),
        "rsi14": round(rsi14, 1),
        "volume_ratio": round(volume_ratio, 2),
        "accumulation_score": round(accumulation_score, 1),
        "classification": classification,
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "week52_low": round(week52_low, 2),
        "week52_high": round(week52_high, 2),
        "dist_from_low_pct": round(dist_from_low_pct, 1),
        "stop_loss": stop_loss,
        "target1": target1,
        "target2": target2,
        "target3": target3,
    }


def main():
    results = []
    for symbol, name in TICKERS.items():
        row = analyze_ticker(symbol, name)
        if row:
            results.append(row)

    sector_score = round(np.mean([r["accumulation_score"] for r in results]), 1) if results else 0.0

    output = {
        "sector": SECTOR_NAME,
        "sector_score": sector_score,
        "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "stocks": sorted(results, key=lambda r: r["accumulation_score"], reverse=True),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"تم كتابة {len(results)} سهم إلى {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
