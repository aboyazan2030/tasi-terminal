"""
fetch_data.py — Tasi Terminal (مشروع مستقل بالكامل)
المرحلة 2: كل القطاعات السعودية الرئيسية + السوق الأمريكي، في ملف واحد.

يكتب النتيجة الموحّدة إلى data/data.json
"""

import json
import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

SECTORS = {
    "sa": {
        "label": "السوق السعودي",
        "currency": "ر.س",
        "sectors": {
            "البنوك": {
                "1120.SR": "مصرف الراجحي", "1180.SR": "البنك الأهلي السعودي",
                "1010.SR": "بنك الرياض", "1050.SR": "البنك السعودي الفرنسي",
                "1020.SR": "بنك الجزيرة", "1150.SR": "مصرف الإنماء",
                "1080.SR": "البنك السعودي للاستثمار", "1030.SR": "البنك العربي الوطني",
                "1140.SR": "بنك البلاد",
            },
            "التعدين": {
                "1211.SR": "معادن",
            },
            "الطاقة": {
                "2222.SR": "أرامكو السعودية",
            },
            "البتروكيماويات": {
                "2010.SR": "سابك", "2290.SR": "ينساب",
                "2330.SR": "المتقدمة", "2350.SR": "كيان السعودية", "2020.SR": "سافكو",
            },
            "الاتصالات": {
                "7010.SR": "الاتصالات السعودية", "7020.SR": "موبايلي", "7030.SR": "زين السعودية",
            },
            "التأمين": {
                "8010.SR": "التعاونية", "8210.SR": "بوبا العربية", "8230.SR": "الراجحي تكافل",
            },
            "الأغذية": {
                "2050.SR": "صافولا", "2280.SR": "المراعي", "6001.SR": "حلواني إخوان",
            },
            "التجزئة": {
                "4190.SR": "جرير", "4001.SR": "العثيم", "4161.SR": "بن داود",
            },
            "الرعاية الصحية": {
                "4013.SR": "د. سليمان الحبيب", "4002.SR": "المواساة", "4004.SR": "دله للصحة",
            },
            "العقار": {
                "4020.SR": "إعمار الاقتصادية", "4300.SR": "دار الأركان", "4150.SR": "طيبة",
            },
            "المرافق": {
                "5110.SR": "السعودية للكهرباء", "2082.SR": "أكوا باور",
            },
            "النقل": {
                "4030.SR": "البحري", "4200.SR": "الدريس",
            },
            "الإعلام": {
                "4210.SR": "SRMG", "4072.SR": "MBC",
            },
            "الصناعة": {
                "1303.SR": "الصناعات الكهربائية",
            },
            "الفنادق والسياحة": {
                "1810.SR": "سيرا",
            },
            "البرمجيات والتقنية": {
                "7203.SR": "علم",
            },
            "مواد البناء": {
                "1302.SR": "بوان",
            },
            "الزراعة": {
                "4061.SR": "أنعام القابضة",
            },
            "الخدمات المالية": {
                "4084.SR": "دراية المالية",
            },
            "السلع الاستهلاكية المعمرة": {
                "2120.SR": "المتطورة",
            },
            "الخدمات التجارية": {
                "4110.SR": "باتك",
            },
            "الأدوية": {
                "2070.SR": "الصناعات الدوائية",
            },
            "الاستثمار المتعدد": {
                "4280.SR": "المملكة القابضة",
            },
            "الأسمنت": {
                "3030.SR": "أسمنت السعودية", "3040.SR": "أسمنت القصيم", "3050.SR": "أسمنت الجنوب",
                "3060.SR": "أسمنت ينبع",
            },
        },
    },
    "us": {
        "label": "السوق الأمريكي",
        "currency": "$",
        "sectors": {
            "التقنية": {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia"},
            "الخدمات المالية": {"JPM": "JPMorgan", "BAC": "Bank of America", "GS": "Goldman Sachs"},
            "الرعاية الصحية": {"JNJ": "Johnson & Johnson", "UNH": "UnitedHealth", "PFE": "Pfizer"},
            "السلع الكمالية": {"AMZN": "Amazon", "TSLA": "Tesla", "HD": "Home Depot"},
            "السلع الأساسية": {"PG": "Procter & Gamble", "KO": "Coca-Cola", "WMT": "Walmart"},
            "الطاقة": {"XOM": "Exxon Mobil", "CVX": "Chevron"},
            "الصناعة": {"BA": "Boeing", "CAT": "Caterpillar", "GE": "GE"},
            "المواد الأساسية": {"LIN": "Linde", "FCX": "Freeport-McMoRan"},
            "المرافق": {"NEE": "NextEra Energy", "DUK": "Duke Energy"},
            "العقار": {"PLD": "Prologis", "AMT": "American Tower"},
            "الاتصالات والإعلام": {"META": "Meta", "GOOGL": "Alphabet", "NFLX": "Netflix"},
        },
    },
}

OUTPUT_PATH = "data/data.json"


def compute_rsi(closes, period=14):
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0


def compute_obv_slope(closes, volumes, lookback=10):
    direction = np.sign(closes.diff().fillna(0))
    obv = (direction * volumes).cumsum()
    recent = obv.tail(lookback)
    if len(recent) < 2:
        return 0.0
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent.values, 1)[0]
    norm = slope / (volumes.tail(lookback).mean() + 1e-9)
    return float(np.clip(norm * 100, -100, 100))


def classify(accumulation_score, rsi):
    if accumulation_score >= 65 and rsi < 70:
        return "تجميع نشط"
    if accumulation_score <= 35 and rsi > 30:
        return "تصريف"
    if rsi >= 70:
        return "تشبع شرائي"
    if rsi <= 30:
        return "تشبع بيعي"
    return "محايد"


def analyze_ticker(symbol, name):
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
        "stop_loss": stop_loss,
        "target1": target1,
        "target2": target2,
        "target3": target3,
    }


def main():
    output = {"updated_at": datetime.datetime.utcnow().isoformat() + "Z", "markets": {}}

    for market_key, market in SECTORS.items():
        sectors_out = []
        for sector_name, tickers in market["sectors"].items():
            stocks = []
            for symbol, name in tickers.items():
                row = analyze_ticker(symbol, name)
                if row:
                    stocks.append(row)
            if stocks:
                sector_score = round(float(np.mean([s["accumulation_score"] for s in stocks])), 1)
                sectors_out.append({
                    "name": sector_name,
                    "score": sector_score,
                    "stocks": sorted(stocks, key=lambda s: s["accumulation_score"], reverse=True),
                })
        sectors_out.sort(key=lambda s: s["score"], reverse=True)
        output["markets"][market_key] = {
            "label": market["label"],
            "currency": market["currency"],
            "sectors": sectors_out,
        }

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    total = sum(len(s["stocks"]) for m in output["markets"].values() for s in m["sectors"])
    print(f"تم تحليل {total} سهم عبر {len(output['markets'])} سوق، وكتابتها إلى {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
