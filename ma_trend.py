# -*- coding: utf-8 -*-
"""
科创50ETF (588000) MA趋势渐进出清 v4 动量版
递进仓位 + 布林带因子 + MA动量加速
"""

import numpy as np
import pandas as pd
import json, os, sys
from datetime import datetime
import baostock as bs
import warnings; warnings.filterwarnings("ignore")

CAPITAL = 100000.0; COMM = 0.0001; SLIP = 0.0002
INIT_CAP = 0.60; CONFIRM_DAYS = 1
TRADE_THRESHOLD = 0.10
VOL_MA = 20; VOL_MIN = 0.8
VOLA_S = 5; VOLA_L = 20; VOLA_MAX = 1.5

DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(DIR, "state.json")
LOG_FILE = os.path.join(DIR, "trade_log.csv")
CHART_FILE = os.path.join(DIR, "performance.png")


def fetch(start="2024-06-01", end="2026-05-30"):
    bs.login(); rs = bs.query_history_k_data_plus("sh.588000","date,close,volume",
        start_date=start, end_date=end, frequency="d", adjustflag="2")
    d = []
    while rs.next():
        d.append(rs.get_row_data())
    bs.logout()
    df = pd.DataFrame(d, columns=["date","close","volume"]).astype({"close":float,"volume":float})
    df["date"] = pd.to_datetime(df["date"]); return df


def target_base(m5v, m10v, m20v):
    """均线基础仓位"""
    if m5v > m20v and m10v > m20v:
        return 0.90 if m5v > m10v else 0.70
    if m5v > m20v:
        return 0.40
    return 0.0


def bb_factor(p, m20v, bu, bl, bf, bh):
    """布林带乘数"""
    if p >= bu: return 0.80
    if p <= bl: return 1.15 if (bf or bh) else 0.85
    if p <= m20v: return 0.90
    return 1.0


def mom_factor(m5v, m20v, bf, bh):
    """MA动量加速加仓"""
    if not (bf or bh): return 0.0
    dev = (m5v - m20v) / max(m20v, 1e-8)
    if dev > 0.08: return 0.15
    if dev > 0.03: return 0.10
    if dev > 0.01: return 0.05
    return 0.0


def calc_target(p, m5v, m10v, m20v, bu, bl, bf, bh, bw_val):
    """综合仓位"""
    base = target_base(m5v, m10v, m20v)
    bbm = bb_factor(p, m20v, bu, bl, bf, bh)
    mom = mom_factor(m5v, m20v, bf, bh)
    t = base * bbm + mom
    t = max(0, min(1.0, t))
    if bw_val < 0.12: t = min(t, 0.50)
    return t, base, bbm, mom


def load():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f: return json.load(f)
    return None

def save(s):
    with open(STATE_FILE,"w") as f: json.dump(s,f,ensure_ascii=False,indent=2)

def log_ap(action,price,shares,amount,ca,ha,signal=""):
    cols = ["datetime","action","price","shares","amount","cash_after","holdings_after","signal"]
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=cols).to_csv(LOG_FILE,index=False)
    df = pd.read_csv(LOG_FILE)
    row = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M"),action,round(price,3),
        shares,round(amount,2),round(ca,2),ha,signal]],columns=cols)
    pd.concat([df,row],ignore_index=True).to_csv(LOG_FILE,index=False)


# ============================ 回测 ============================
def backtest(closes, volumes=None):
    n = len(closes)
    m5 = pd.Series(closes).rolling(5).mean().values
    m10 = pd.Series(closes).rolling(10).mean().values
    m20 = pd.Series(closes).rolling(20).mean().values
    s = pd.Series(closes).rolling(20).std().values
    bu = m20 + 2*s; bl = m20 - 2*s
    bw = (bu - bl) / m20
    if volumes is not None:
        vma = pd.Series(volumes).rolling(VOL_MA).mean().values
    else:
        vma = np.ones(n) * 1e8

    cash = CAPITAL; hold = 0; first = False; bd = 0
    sigs = []; dc = [CAPITAL]; dh = [0]
    buys = sells = 0

    for i in range(n):
        p = closes[i]
        pr = hold * p / max(cash + hold * p, 1) if cash + hold * p > 0 else 0

        if i < 20:
            if i == 0:
                val = cash * 0.3; sh = int(val/p/100)*100
                if sh >= 100: cash -= sh * p * (1 + COMM + SLIP); hold += sh
            dc.append(cash); dh.append(hold)
            continue

        bf = m5[i] > m20[i] and m10[i] > m20[i]
        bh = m5[i] > m20[i] and m10[i] <= m20[i]
        t, base, bbm, mom = calc_target(p, m5[i], m10[i], m20[i], bu[i], bl[i], bf, bh, bw[i])

        bd = bd + 1 if bf else 0
        eff = t if first else min(t, INIT_CAP)
        if bf and bd < CONFIRM_DAYS:
            if not first: eff = min(eff, 0.40)
            elif first: eff = min(eff, 0.70)

        vOK = True
        if volumes is not None and vma[i] > 0:
            vOK = volumes[i] / vma[i] >= VOL_MIN

        if eff > pr + TRADE_THRESHOLD and cash > CAPITAL * 0.03 and vOK:
            tv = cash + hold * p; ts = int(eff * tv / p / 100) * 100
            bs2 = max(100, min(ts - hold, int(cash / p / 100) * 100))
            if bs2 >= 100:
                cash -= bs2 * p * (1 + COMM + SLIP)
                hold += bs2; first = True; buys += 1
                sigs.append((i, f"B{int(eff*100)}"))

        if base <= 0 and hold > 0:
            cash += hold * p * (1 - COMM - SLIP)
            sigs.append((i, "SA")); hold = 0; bd = 0; sells += 1

        dc.append(cash); dh.append(hold)

    nav = np.array([dc[j+1] + dh[j+1] * closes[j] for j in range(n)])
    tr = (nav[-1] - CAPITAL) / CAPITAL
    bh_ret = (closes[-1] - closes[0]) / closes[0]
    pk = np.maximum.accumulate(nav); mdd_n = np.max((pk - nav) / pk * 100) if len(nav) > 0 else 0
    pk2 = np.maximum.accumulate(closes); mdd_b = np.max((pk2 - closes) / pk2 * 100) if len(closes) > 0 else 0

    return {"nav": nav, "ret": tr, "mdd_n": mdd_n, "mdd_b": mdd_b, "bh": bh_ret,
            "excess": tr - bh_ret, "buys": buys, "sells": sells, "sigs": sigs}


# ============================ 走势图 ============================
def chart():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt; import matplotlib.dates as mdates
    try: plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei','Noto Sans CJK JP','SimHei'] + plt.rcParams['font.sans-serif']; plt.rcParams['axes.unicode_minus'] = False
    except: pass

    df = fetch(end="2026-05-30")
    if df.empty: print("❌ 无数据"); return
    c, d, v = df["close"].values, df["date"].values, df["volume"].values
    bt = backtest(c, v); nv = bt["nav"]; ss = bt["sigs"]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(d, nv/CAPITAL, "#e74c3c", lw=2, label=f"v4动量 {bt['ret']:.1%}")
    a1.plot(d, c/c[0], "#3498db", lw=1.5, alpha=0.7, label=f"ETF {bt['bh']:.1%}")
    a1.axhline(1, color="gray", ls="--", alpha=0.4)
    a1.set_ylabel("净值"); a1.legend(loc="upper left"); a1.grid(alpha=0.3)
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    a1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    info = f"v4: {bt["ret"]:.1%} ETF:{bt["bh"]:.1%} 回撤:{bt["mdd_n"]:.1f}% 交易:{bt["buys"]}买/{bt["sells"]}卖"
    a1.text(0.02, 0.95, info, transform=a1.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    for i, s in ss:
        if i < len(d):
            a1.scatter(d[i], nv[i]/CAPITAL, color="green" if "B" in s else "red",
                      marker="^" if "B" in s else "v", s=80, zorder=5)

    pk = np.maximum.accumulate(nv); dda = (pk - nv) / pk * 100
    pk2 = np.maximum.accumulate(c); ddb = (pk2 - c) / pk2 * 100
    a2.plot(dda, color="#e74c3c", alpha=0.5, lw=1, label=f"回撤 {bt["mdd_n"]:.1f}%")
    a2.plot(ddb, color="#3498db", alpha=0.3, lw=1, label=f"ETF回撤 {bt["mdd_b"]:.1f}%")
    a2.set_ylabel("回撤(%)"); a2.set_xlabel("交易日"); a2.legend(loc="upper left"); a2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(CHART_FILE, dpi=150, bbox_inches="tight"); plt.close()
    print(f"✅ 走势图: {CHART_FILE}")
    print(f"v4: {bt['ret']:.1%}  ETF:{bt['bh']:.1%}  超额:{bt['excess']:.1%}")
    print(f"回撤:{bt["mdd_n"]:.1f}%  交易:{bt["buys"]}买/{bt["sells"]}卖")


# ============================ 信号 ============================
def signals():
    today = datetime.now().strftime("%Y-%m-%d")
    df = fetch(end=today)
    if df.empty or len(df) < 20: print("❌ 数据不足"); return

    cp = df["close"].iloc[-1]; last = df["date"].iloc[-1].strftime("%Y-%m-%d")
    m5v = df["close"].rolling(5).mean().iloc[-1]
    m10v = df["close"].rolling(10).mean().iloc[-1]
    m20v = df["close"].rolling(20).mean().iloc[-1]
    s = df["close"].rolling(20).std().iloc[-1]
    bu = m20v + 2*s; bl = m20v - 2*s
    bw2 = (bu - bl) / m20v
    bf = m5v > m20v and m10v > m20v
    bh = m5v > m20v and m10v <= m20v

    t, base, bbm, mom = calc_target(cp, m5v, m10v, m20v, bu, bl, bf, bh, bw2)

    st = load()
    h = st["holdings"] if st else 0
    ca = st["cash"] if st else CAPITAL
    tv = ca + h * cp; pr = h * cp / max(tv, 1); bd = st.get("bull_days",0) if st else 0

    eff = t if(st and st.get("first_buy_done")) else min(t, INIT_CAP)
    if bf and bd < CONFIRM_DAYS and not(st and st.get("first_buy_done")):
        eff = min(eff, 0.40)

    dev = (m5v - m20v) / m20v if(bf or bh) else 0
    labels = {0:"空仓", 0.25:"轻仓", 0.40:"半仓", 0.70:"重仓", 0.90:"满仓"}
    tl = labels.get(t, f"{int(t*100)}%")

    print(f"\n{'='*60}")
    print(f"  科创50ETF 趋势看板 v4 动量版")
    print(f"  {today}  (数据: {last})")
    print(f"{'='*60}")
    print(f"\n📊 价:{cp:.3f}  MA5:{m5v:.3f} MA10:{m10v:.3f} MA20:{m20v:.3f}")
    print(f"  MA偏离:{dev:.1%}  布林:上{bu:.3f} 中{m20v:.3f} 下{bl:.3f}")
    print(f"  因子: 布林乘数={bbm:.2f} 动量加仓={mom:.0%}")
    print(f"  目标仓位: {tl}({int(t*100)}%)  基础:{int(base*100)}%×{bbm:.2f}+{mom:.0%}")
    print(f"\n💼 持仓{h}股(≈{h*cp:.0f}) 现金¥{ca:.0f} 总¥{tv:.0f} 实际{pr:.1%}")

    vr = df["volume"].iloc[-1] / max(df["volume"].rolling(VOL_MA).mean().iloc[-1], 1)
    ret = df["close"].pct_change()
    vratio = ret.rolling(VOLA_S).std().iloc[-1] / max(ret.rolling(VOLA_L).std().iloc[-1], 1e-8)
    vol_ok = vr >= VOL_MIN; vola_ok = vratio <= VOLA_MAX

    act = "⏸️ 无需操作"
    if eff > pr + TRADE_THRESHOLD and ca > CAPITAL * 0.03:
        if not vol_ok: act = f"⏸️ 缩量(vr={vr:.2f})暂缓"
        elif not vola_ok: act = f"⏸️ 波动异常({vratio:.2f})暂缓"
        else:
            tv2 = ca + h * cp; ts2 = int(eff * tv2 / cp / 100) * 100
            bs2 = max(100, min(ts2 - h, int(ca / cp / 100) * 100))
            if bs2 >= 100: act = f"🟢 买入→{int(eff*100)}%: ≤{cp:.3f}×{bs2}股 (¥{bs2*cp:.0f})"
    elif eff < pr - TRADE_THRESHOLD and h > 0:
        tv2 = ca + h * cp; ts2 = int(eff * tv2 / cp / 100) * 100
        ss2 = max(100, h - ts2)
        if ss2 >= 100: act = f"🔴 卖出→{int(eff*100)}%: ≥{cp:.3f}×{ss2}股 (¥{ss2*cp:.0f})"
    elif base <= 0 and pr > 0.02:
        act = f"🔴 空头清仓: ≥{cp:.3f}×{h}股"
    print(f"\n  {act}")

    new = {"price":round(cp,3),"holdings":h,"cash":round(ca,2),
           "trend_en":"FULL" if bf else("HALF" if bh else "ZERO"),
           "target_pos":t,"exec_pos":eff,"bb_mult":bbm,"mom_add":mom,
           "ma5":round(m5v,3),"ma10":round(m10v,3),"ma20":round(m20v,3),
           "vol_ratio":round(vr,2),"bw":round(bw2,3),
           "bull_days":bd+1 if bf else 0,
           "first_buy_done":st.get("first_buy_done",False) if st else False}
    save(new)


# ============================ 成交记录 ============================
def record(cmd, price=0):
    st = load() or {"holdings":0,"cash":CAPITAL}
    h, ca = st["holdings"], st["cash"]; cp = price or st.get("price", 1)
    t = st.get("exec_pos", st.get("target_pos", 0.4))

    if cmd == "buy":
        tv = ca + h * cp; ts = int(t * tv / cp / 100) * 100
        bs2 = max(100, min(ts - h, int(ca / cp / 100) * 100))
        if bs2 >= 100:
            cost = bs2 * cp * (1 + COMM + SLIP); ca -= cost; h += bs2
            log_ap("买入", cp, bs2, cost, ca, h, f"→{int(t*100)}%")
            print(f"✅ 买入→{int(t*100)}%: {cp:.3f}×{bs2} ¥{cost:.0f}")
    elif cmd == "sell":
        if h > 0:
            val = h * cp * (1 - COMM - SLIP); ca += val
            log_ap("清仓", cp, h, val, ca, 0, "清仓")
            print(f"✅ 清仓 {cp:.3f}×{h} ¥{val:.0f}"); h = 0
    elif cmd == "half":
        tv = ca + h * cp; ts = int(t * tv / cp / 100) * 100
        ss2 = max(100, h - ts)
        if ss2 >= 100:
            val = ss2 * cp * (1 - COMM - SLIP); ca += val; h -= ss2
            log_ap("减仓", cp, ss2, val, ca, h, f"→{int(t*100)}%")
            print(f"✅ 减仓→{int(t*100)}%: {ss2}股 ¥{val:.0f}")

    st["holdings"] = h; st["cash"] = round(ca, 2); st["first_buy_done"] = True
    save(st); print(f"  持仓{h}股 现金¥{ca:.0f}")


# ============================ 日志 ============================
def show_log():
    if not os.path.exists(LOG_FILE): print("ℹ️ 无日志"); return
    d = pd.read_csv(LOG_FILE)
    if d.empty: print("ℹ️ 无日志"); return
    print(f"\n{'='*65}\n  交易日志\n{'='*65}")
    for _, r in d.iterrows():
        ic = "🟢" if r["action"]=="买入" else ( "🔴" if r["action"]=="清仓" else "🟡")
        print(f"  {ic} {r['datetime']} {r['action']} {r['price']:.3f}×{int(r['shares']):,}  ¥{r['amount']:.0f}  {r.get('signal','')}")

def reset():
    for f in [STATE_FILE, LOG_FILE]:
        if os.path.exists(f): os.remove(f)
    print("✅ 已重置")


# ============================ Main ============================
if __name__ == "__main__":
    if len(sys.argv) == 1: signals()
    elif sys.argv[1] == "buy": record("buy", float(sys.argv[2]) if len(sys.argv)>2 else 0)
    elif sys.argv[1] == "sell": record("sell", float(sys.argv[2]) if len(sys.argv)>2 else 0)
    elif sys.argv[1] == "half": record("half", float(sys.argv[2]) if len(sys.argv)>2 else 0)
    elif sys.argv[1] == "log": show_log()
    elif sys.argv[1] == "chart": chart()
    elif sys.argv[1] == "status": print(json.dumps(load() or {}, ensure_ascii=False, indent=2))
    elif sys.argv[1] == "reset": reset()
    else: print("用法: python3 ma_trend.py [buy|sell|half|log|chart|status|reset]")
