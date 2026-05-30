#!/usr/bin/env python3
"""生成Pages看板"""
import sys, os, base64, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime
import numpy as np, pandas as pd, akshare as ak
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import matplotlib.dates as mdates
import ma_trend; from importlib import reload; reload(ma_trend)
CAP=100000; ODIR=os.path.dirname(os.path.abspath(__file__))

def main():
    df=ak.stock_zh_index_daily(symbol='sh000688')
    df['date']=pd.to_datetime(df['date']); df=df[df['date']>='2020-01-01']
    c=df['close'].values; v=df['volume'].values*1000; d=df['date'].values
    r=ma_trend.backtest(c,v)
    print(f"收益:{r['ret']:.1%} ETF:{r['bh']:.1%}")

    fig,(a1,a2)=plt.subplots(2,1,figsize=(12,6.5),gridspec_kw={"height_ratios":[3,1]})
    a1.plot(d,r["nav"]/CAP,"#e74c3c",lw=2,label=f"策略 {r['ret']:.1%}")
    a1.plot(d,c/c[0],"#3498db",lw=1.5,alpha=0.7,label=f"科创50指数 {r['bh']:.1%}")
    a1.axhline(1,color="gray",ls="--",alpha=0.4); a1.legend(loc="upper left"); a1.grid(alpha=0.3)
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    a1.set_title("科创50ETF v4 动量版",fontsize=12)
    info=f"策略: {r['ret']:.1%} 指数: {r['bh']:.1%} 回撤: {r['mdd_n']:.1f}%"
    a1.text(0.02,0.95,info,transform=a1.transAxes,fontsize=10,bbox=dict(boxstyle="round",facecolor="wheat",alpha=0.5))
    for i,s in r["sigs"]:
        if i<len(d): a1.scatter(d[i],r["nav"][i]/CAP,color="green" if"B"in s else"red",marker="^"if"B"in s else"v",s=60,zorder=5)
    pk=np.maximum.accumulate(r["nav"]); dda=(pk-r["nav"])/pk*100; pk2=np.maximum.accumulate(c); ddb=(pk2-c)/pk2*100
    a2.plot(dda,"#e74c3c",alpha=0.5); a2.plot(ddb,"#3498db",alpha=0.3)
    a2.set_xlabel("交易日"); a2.grid(alpha=0.3)
    plt.tight_layout()
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=130,bbox_inches="tight"); plt.close()
    img=base64.b64encode(buf.getvalue()).decode()

    cs=pd.Series(c); lp=c[-1]; m5v=cs.rolling(5).mean().iloc[-1]; m10v=cs.rolling(10).mean().iloc[-1]; m20v=cs.rolling(20).mean().iloc[-1]
    bu=m20v+2*cs.rolling(20).std().iloc[-1]; bl=m20v-2*cs.rolling(20).std().iloc[-1]; bw=(bu-bl)/m20v
    bf=m5v>m20v and m10v>m20v; bh=m5v>m20v and m10v<=m20v
    b=0.9 if(bf and m5v>m10v)else(0.7 if bf else(0.4 if bh else 0))
    bm=1.0
    if lp>=bu: bm=0.8
    elif lp<=bl: bm=1.15 if(bf or bh)else 0.85
    elif lp<=m20v: bm=0.9
    mo=0
    if bf or bh:
        dv=(m5v-m20v)/m20v
        if dv>0.08: mo=0.15
        elif dv>0.03: mo=0.10
        elif dv>0.01: mo=0.05
    t=b*bm+mo; t=max(0,min(1,t))
    if bw<0.12: t=min(t,0.5)
    tl={0:"空仓",0.25:"轻仓",0.4:"半仓",0.7:"重仓",0.9:"满仓"}.get(t,f"{int(t*100)}%")
    ms=f"+{mo:.0%}" if mo>0 else"0%"
    vs=pd.Series(df['volume'].values); vr=df['volume'].values[-1]/max(vs.rolling(20).mean().iloc[-1],1)
    sig="无需操作"
    if vr<0.8: sig=f"缩量({vr:.2f})暂缓买入"
    elif t>=0.9: sig="趋势满仓 - 继续持有"
    elif t>=0.7: sig="趋势重仓 - 继续持有"
    elif t>=0.4: sig="趋势半仓 - 视信号加仓"
    elif t<=0: sig="空头 - 观望"
    td=str(d[-1])[:10]

    html=f'''<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>科创50ETF 趋势看板 v4 动量版</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;background:#0f1923;color:#e0e6ed;padding:20px}}
.container{{max-width:1100px;margin:auto}}
h1{{font-size:22px;margin-bottom:4px}}
.sub{{color:#8899aa;font-size:13px;margin-bottom:20px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}}
.card{{background:#1a2533;border-radius:10px;padding:16px;border:1px solid #2a3a4a}}
.card .l{{font-size:12px;color:#8899aa}}
.card .v{{font-size:22px;font-weight:bold;margin-top:4px}}
.up{{color:#e74c3c}}.down{{color:#27ae60}}.mid{{color:#f0c040}}
.sb{{background:#1a2533;border-radius:10px;padding:16px;margin-bottom:20px;border:1px solid #2a3a4a}}
.cb{{background:#1a2533;border-radius:10px;padding:16px;margin-bottom:20px;border:1px solid #2a3a4a}}
.cb img{{width:100%;border-radius:6px}}
.mr{{display:flex;gap:16px;flex-wrap:wrap;font-size:14px;margin-bottom:10px}}
.mr span{{padding:4px 10px;border-radius:4px;background:#253545}}
.sig{{background:#253545;border-radius:8px;padding:12px;margin-top:10px;font-size:14px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:#253545;padding:8px 10px;text-align:left;font-weight:500}}
td{{padding:6px 10px;border-bottom:1px solid #1e2d3d}}
.bdg{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px}}
.b{{background:#e74c3c22;color:#e74c3c;border:1px solid #e74c3c55}}
.s{{background:#27ae6022;color:#27ae60;border:1px solid #27ae6055}}
.rf{{color:#5a7a9a;font-size:12px;margin-top:10px;text-align:center}}
.rd{{font-size:13px;line-height:1.8;color:#c0d0e0}}
.rd b{{color:#e74c3c}}.rd .g{{color:#27ae60}}
</style></head>
<body>
<div class="container">
<h1>科创50ETF 趋势看板 v4 动量版</h1>
<div class="sub">588000 | 科创50指数回测 2020-2026 | 递进仓位+布林带+MA动量</div>
<div class="cards">
<div class="card"><div class="l">策略累计</div><div class="v up">{r['ret']:.1%}</div></div>
<div class="card"><div class="l">指数累计</div><div class="v" style="color:#3498db">{r['bh']:.1%}</div></div>
<div class="card"><div class="l">超额收益</div><div class="v up">{r['excess']:.1%}</div></div>
<div class="card"><div class="l">最大回撤</div><div class="v mid">{r['mdd_n']:.1f}%</div></div>
<div class="card"><div class="l">指数回撤</div><div class="v mid">{r['mdd_b']:.1f}%</div></div>
<div class="card"><div class="l">交易</div><div class="v">{r['buys']}买/{r['sells']}卖</div></div>
</div>
<div class="sb"><div class="mr"><span>MA5: <b>{m5v:.0f}</b></span><span>MA10: <b>{m10v:.0f}</b></span><span>MA20: <b>{m20v:.0f}</b></span><span>动量: <b>{ms}</b></span><span>目标: <b>{tl}</b></span></div><div class="sig">{sig}</div></div>
<div class="cb"><h2>回测走势 (2020-2026)</h2><img src="data:image/png;base64,{img}" alt="走势图">
<div style="display:flex;gap:20px;margin-top:10px;font-size:13px;color:#8899aa">
<span>策略: <b style="color:#e74c3c">{r['ret']:.1%}</b></span>
<span>指数: <b style="color:#3498db">{r['bh']:.1%}</b></span>
<span>超额: <b>{r['excess']:.1%}</b></span>
<span>回撤: <b>{r['mdd_n']:.1f}%</b></span></div></div>
<div class="sb"><h2>交易日志</h2>
<table><tr><th>日期</th><th>方向</th><th>价格</th><th>数量</th></tr>
<tr><td>2026-02-02</td><td><span class="bdg b">买入</span></td><td>1.528</td><td>21,400</td></tr>
<tr><td>2026-02-03</td><td><span class="bdg s">清仓</span></td><td>1.548</td><td>41,600</td></tr>
<tr><td>2026-04-10</td><td><span class="bdg b">买入</span></td><td>1.437</td><td>28,300</td></tr>
<tr><td>2026-04-14</td><td><span class="bdg b">买入</span></td><td>1.479</td><td>41,300</td></tr></table></div>
<div class="rf">GitHub: wepoets1107/sci50-grid-backtest | {td}</div>
</div></body></html>'''
    with open(os.path.join(ODIR,"index.html"),"w",encoding="utf-8") as f: f.write(html)
    print("OK")

if __name__=="__main__":
    main()
