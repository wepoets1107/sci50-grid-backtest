#!/usr/bin/env python3
"""GitHub Actions: 生成完整看板"""
import numpy as np, pandas as pd, json, os, sys, base64, io
from datetime import datetime
import baostock as bs
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import matplotlib.dates as mdates

CAP=100000.0; COM=0.0001; SLI=0.0002; IC=0.60; CD=1; TT=0.10
VM=20; VN=0.8; VS=5; VL=20; VX=1.5

def fetch(s="2024-06-01",e=None):
    if e is None: e=datetime.now().strftime("%Y-%m-%d")
    bs.login(); rs=bs.query_history_k_data_plus("sh.588000","date,close,volume",start_date=s,end_date=e,frequency="d",adjustflag="2")
    d=[]
    while rs.next(): d.append(rs.get_row_data())
    bs.logout()
    df=pd.DataFrame(d,columns=["date","close","volume"]).astype({"close":float,"volume":float}); df["date"]=pd.to_datetime(df["date"]); return df

def ct(p,m5,m10,m20,bu,bl,bf,bh,bw):
    b=0.9 if(bf and m5>m10)else(0.7 if bf else(0.4 if bh else 0))
    bm=1.0
    if p>=bu: bm=0.8
    elif p<=bl: bm=1.15 if(bf or bh)else 0.85
    elif p<=m20: bm=0.9
    mo=0
    if bf or bh:
        dv=(m5-m20)/max(m20,1e-8)
        if dv>0.08: mo=0.15
        elif dv>0.03: mo=0.10
        elif dv>0.01: mo=0.05
    t=b*bm+mo; t=max(0,min(1,t))
    if bw<0.12: t=min(t,0.5)
    return t,b,bm,mo

def bt(c,v=None):
    n=len(c)
    m5=pd.Series(c).rolling(5).mean().values; m10=pd.Series(c).rolling(10).mean().values; m20=pd.Series(c).rolling(20).mean().values
    s2=pd.Series(c).rolling(20).std().values; bu=m20+2*s2; bl=m20-2*s2; bw=(bu-bl)/m20
    vma=pd.Series(v).rolling(VM).mean().values if v is not None else np.ones(n)*1e8
    ca=CAP; ho=0; fi=0; bd=0; sg=[]; bc2=0; sc=0; dc=[CAP]; dh=[0]
    for i in range(n):
        p=c[i]; pr=ho*p/max(ca+ho*p,1) if ca+ho*p>0 else 0
        if i<20:
            if i==0: val=ca*0.3; sh=int(val/p/100)*100
            if sh>=100: ca-=sh*p*(1+COM+SLI); ho+=sh
            dc.append(ca); dh.append(ho); continue
        bf=m5[i]>m20[i] and m10[i]>m20[i]; bh=m5[i]>m20[i] and m10[i]<=m20[i]
        t2,b2,bm,mo=ct(p,m5[i],m10[i],m20[i],bu[i],bl[i],bf,bh,bw[i])
        bd=bd+1 if bf else 0; e=t2 if fi else min(t2,IC)
        if bf and bd<CD: e=min(e,0.4 if not fi else 0.7)
        vOK=v[i]/vma[i]>=VN if(v is not None and vma[i]>0)else True
        if e>pr+TT and ca>CAP*0.03 and vOK:
            tv=ca+ho*p; ts=int(e*tv/p/100)*100; bs2=max(100,min(ts-ho,int(ca/p/100)*100))
            if bs2>=100: ca-=bs2*p*(1+COM+SLI); ho+=bs2; fi=1; bc2+=1; sg.append((i,f"B{int(e*100)}"))
        if b2<=0 and ho>0:
            ca+=ho*p*(1-COM-SLI); ho=0; bd=0; sc+=1; sg.append((i,"SA"))
        dc.append(ca); dh.append(ho)
    nv=np.array([dc[j+1]+dh[j+1]*c[j] for j in range(n)])
    tr=(nv[-1]-CAP)/CAP; bhr=(c[-1]-c[0])/c[0]
    pk=np.maximum.accumulate(nv); mn=np.max((pk-nv)/pk*100)
    pk2=np.maximum.accumulate(c); mb=np.max((pk2-c)/pk2*100)
    return {"nav":nv,"ret":tr,"mdd_n":mn,"mdd_b":mb,"bh":bhr,"excess":tr-bhr,"buys":bc2,"sells":sc,"sigs":sg,"fc":ca,"fh":ho}

def chart_b64(c,d,bt):
    nv=bt["nav"]; ss=bt["sigs"]
    f,(a1,a2)=plt.subplots(2,1,figsize=(12,6.5),gridspec_kw={"height_ratios":[3,1]})
    a1.plot(d,nv/CAP,"#e74c3c",lw=2,label=f"v4 {bt['ret']:.1%}")
    a1.plot(d,c/c[0],"#3498db",lw=1.5,alpha=0.7,label=f"ETF {bt['bh']:.1%}")
    a1.axhline(1,color="gray",ls="--",alpha=0.4); a1.set_ylabel("净值"); a1.legend(loc="upper left"); a1.grid(alpha=0.3)
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); a1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    a1.set_title("科创50ETF v4 动量版",fontsize=12)
    info=f"v4: {bt['ret']:.1%} ETF:{bt['bh']:.1%} 回撤:{bt['mdd_n']:.1f}% 交易:{bt['buys']}买/{bt['sells']}卖"
    a1.text(0.02,0.95,info,transform=a1.transAxes,fontsize=10,bbox=dict(boxstyle="round",facecolor="wheat",alpha=0.5))
    for i,s in ss:
        if i<len(d): a1.scatter(d[i],nv[i]/CAP,color="green" if"B"in s else"red",marker="^"if"B"in s else"v",s=80,zorder=5)
    pk=np.maximum.accumulate(nv); dda=(pk-nv)/pk*100; pk2=np.maximum.accumulate(c); ddb=(pk2-c)/pk2*100
    a2.plot(dda,"#e74c3c",alpha=0.5,lw=1,label=f"回撤 {bt['mdd_n']:.1f}%")
    a2.plot(ddb,"#3498db",alpha=0.3,lw=1,label=f"ETF回撤 {bt['mdd_b']:.1f}%")
    a2.set_xlabel("交易日"); a2.legend(loc="upper left"); a2.grid(alpha=0.3)
    plt.tight_layout(); buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=130,bbox_inches="tight"); plt.close()
    return base64.b64encode(buf.getvalue()).decode()

def gen_html(df,bt,chart_img):
    c=df["close"].values; d=df["date"].values
    td=str(d[-1])[:10]; lp=c[-1]; tv=bt["fc"]+bt["fh"]*lp
    pnl=tv-CAP; pp=bt["fh"]*lp/max(tv,1)*100
    m5v=pd.Series(c).rolling(5).mean().iloc[-1]; m10v=pd.Series(c).rolling(10).mean().iloc[-1]; m20v=pd.Series(c).rolling(20).mean().iloc[-1]
    bu=m20v+2*pd.Series(c).rolling(20).std().iloc[-1]; bl=m20v-2*pd.Series(c).rolling(20).std().iloc[-1]; bw2=(bu-bl)/m20v
    bf=m5v>m20v and m10v>m20v; bh=m5v>m20v and m10v<=m20v; ld=m5v>m10v
    t,b2,bm,mo=ct(lp,m5v,m10v,m20v,bu,bl,bf,bh,bw2)
    dev=(m5v-m20v)/m20v if(bf or bh)else 0; ms=f"+{mo:.0%}"if mo>0 else"0%"
    lb={"0":"空仓","0.25":"轻仓","0.4":"半仓","0.7":"重仓","0.9":"满仓"}
    tl=lb.get(str(t),f"{int(t*100)}%")
    vr=df["volume"].values[-1]/max(df["volume"].rolling(VM).mean().iloc[-1],1)
    vratio=pd.Series(c).pct_change().rolling(VS).std().iloc[-1]/max(pd.Series(c).pct_change().rolling(VL).std().iloc[-1],1e-8)
    sig="无需操作"; pr2=bt["fh"]*lp/max(tv,1)
    if t>pr2+TT and bt["fc"]>CAP*0.03:
        if vr<VN: sig=f"缩量({vr:.2f})暂缓"
        elif vratio>VX: sig=f"波动异常({vratio:.2f})暂缓"
        else:
            tv2=bt["fc"]+bt["fh"]*lp; ts=int(t*tv2/lp/100)*100; bs2=max(100,min(ts-bt["fh"],int(bt["fc"]/lp/100)*100))
            if bs2>=100: sig=f"买入→{int(t*100)}%: {lp:.3f}×{bs2}"
    elif t<pr2-TT and bt["fh"]>0:
        tv2=bt["fc"]+bt["fh"]*lp; ts=int(t*tv2/lp/100)*100; ss=max(100,bt["fh"]-ts)
        if ss>=100: sig=f"卖出→{int(t*100)}%: {lp:.3f}×{ss}"
    elif t<=0 and pr2>0.02: sig=f"空头清仓: {lp:.3f}"
    sc="#e74c3c" if"买入"in sig else("#27ae60" if"卖出"in sig or"清仓"in sig else"#8899aa")

    return f'''<!DOCTYPE html>
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
.card .v{{font-size:24px;font-weight:bold;margin-top:4px}}
.up{{color:#e74c3c}}.down{{color:#27ae60}}.mid{{color:#f0c040}}
.sb,.sb2{{background:#1a2533;border-radius:10px;padding:16px;margin-bottom:20px;border:1px solid #2a3a4a}}
.mr{{display:flex;gap:16px;flex-wrap:wrap;font-size:14px;margin-bottom:10px}}
.mr span{{padding:4px 10px;border-radius:4px;background:#253545}}
.sig{{background:#253545;border-radius:8px;padding:12px;margin-top:10px;font-size:14px;color:{sc}}}
.cb{{background:#1a2533;border-radius:10px;padding:16px;margin-bottom:20px;border:1px solid #2a3a4a}}
.cb img{{width:100%;border-radius:6px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:#253545;padding:8px 10px;text-align:left;font-weight:500}}
td{{padding:6px 10px;border-bottom:1px solid #1e2d3d;white-space:nowrap}}
td:nth-child(4),td:nth-child(5){{text-align:right}}
.bdg{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px}}
.b{{background:#e74c3c22;color:#e74c3c;border:1px solid #e74c3c55}}
.s{{background:#27ae6022;color:#27ae60;border:1px solid #27ae6055}}
.y{{background:#f0c04022;color:#f0c040;border:1px solid #f0c04055}}
.rf{{color:#5a7a9a;font-size:12px;margin-top:10px;text-align:center}}
.rd{{font-size:13px;line-height:1.8;color:#c0d0e0}}
.rd b{{color:#e74c3c}}.rd .g{{color:#27ae60}}.rd code{{background:#253545;padding:2px 6px;border-radius:3px;font-size:12px}}
.rd ul{{margin:8px 0 8px 20px}}.rd li{{margin:4px 0}}
</style></head>
<body>
<div class="container">
<h1>科创50ETF 趋势看板 v4 动量版</h1>
<div class="sub">588000 | 100,000 | 递进仓位+布林带+MA动量 | {td}</div>

<div class="cards">
<div class="card"><div class="l">当前价</div><div class="v up">{lp:.3f}</div></div>
<div class="card"><div class="l">策略净值</div><div class="v up">{tv/CAP:.2f}</div></div>
<div class="card"><div class="l">盈亏</div><div class="v up">+{int(pnl)}</div></div>
<div class="card"><div class="l">仓位</div><div class="v mid">{pp:.0f}%</div></div>
<div class="card"><div class="l">持仓</div><div class="v">{bt["fh"]:,}</div></div>
<div class="card"><div class="l">目标</div><div class="v" style="font-size:18px">{tl}</div></div>
</div>

<div class="sb">
<div class="mr">
<span>MA5: <b>{m5v:.3f}</b></span>
<span>MA10: <b>{m10v:.3f}</b></span>
<span>MA20: <b>{m20v:.3f}</b></span>
<span>动量: <b>{ms}</b></span>
<span>目标: <b>{int(t*100)}%</b></span>
</div>
<div class="sig">{sig}</div>
</div>

<div class="cb"><h2>回测走势</h2>
<img src="data:image/png;base64,{chart_img}" alt="走势图">
<div style="display:flex;gap:20px;margin-top:10px;font-size:13px;color:#8899aa;flex-wrap:wrap">
<span>v4: <b style="color:#e74c3c">{bt['ret']:.1%}</b></span>
<span>ETF: <b style="color:#3498db">{bt['bh']:.1%}</b></span>
<span>回撤: <b>{bt['mdd_n']:.1f}%</b></span>
<span>交易: {bt['buys']}买/{bt['sells']}卖</span>
</div></div>

<div class="sb"><h2>交易日志</h2>
<table><tr><th>日期</th><th>方向</th><th>价格</th><th>数量</th><th>金额</th><th>信号</th></tr>
<tr><td>2026-02-02</td><td><span class="bdg b">买入</span></td><td>1.528</td><td>21,400</td><td>19,564</td><td>买入63%</td></tr>
<tr><td>2026-02-03</td><td><span class="bdg s">清仓</span></td><td>1.548</td><td>41,600</td><td>51,069</td><td>空头出清</td></tr>
<tr><td>2026-04-10</td><td><span class="bdg b">买入</span></td><td>1.437</td><td>28,300</td><td>40,536</td><td>买入40%</td></tr>
<tr><td>2026-04-14</td><td><span class="bdg b">买入</span></td><td>1.479</td><td>41,300</td><td>50,745</td><td>买入100%</td></tr>
</table></div>

<div class="sb2"><h2>策略说明</h2>
<div class="rd">
<p><b>递进仓位 (MA趋势渐进出清)</b></p>
<ul>
<li><b class="g">满仓 90%</b> 双多头+MA5领先MA10</li>
<li><b class="g">重仓 70%</b> 双多头+MA5≤MA10</li>
<li style="color:#f0c040">半仓 40% 短多长平</li>
<li>轻仓 25% MA5>MA20+MA20斜率>0</li>
<li style="color:#e74c3c">空仓 0% 空头</li>
</ul>
<b>因子调节:</b> 布林带乘数 | MA动量加仓 | 初始上限60% | 量比<0.8暂缓<br>
<b>操作:</b> python3 ma_trend.py 查看信号
</div></div>
<div class="rf">GitHub: wepoets1107/sci50-grid-backtest | baostock</div>
</div></body></html>'''

def main():
    print(f"生成看板: {datetime.now()}")
    df=fetch()
    if df.empty: print("无数据"); sys.exit(1)
    c=df["close"].values; v=df["volume"].values; d=df["date"].values
    bt2=bt(c,v)
    print(f"收益:{bt2['ret']:.1%} ETF:{bt2['bh']:.1%} 回撤:{bt2['mdd_n']:.1f}%")
    img=chart_b64(c,d,bt2)
    html=gen_html(df,bt2,img)
    with open("index.html","w",encoding="utf-8") as f: f.write(html)
    # 保存走势图文件
    fig,(a1,a2)=plt.subplots(2,1,figsize=(12,6.5),gridspec_kw={"height_ratios":[3,1]})
    nv=bt2["nav"]; ss=bt2["sigs"]
    a1.plot(d,nv/CAP,"#e74c3c",lw=2,label=f"v4 {bt2['ret']:.1%}")
    a1.plot(d,c/c[0],"#3498db",lw=1.5,alpha=0.7,label=f"ETF {bt2['bh']:.1%}")
    a1.legend(loc="upper left"); a1.grid(alpha=0.3)
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    a1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    pk=np.maximum.accumulate(nv); dda=(pk-nv)/pk*100
    pk2=np.maximum.accumulate(c); ddb=(pk2-c)/pk2*100
    a2.plot(dda,"#e74c3c",alpha=0.5); a2.plot(ddb,"#3498db",alpha=0.3)
    a2.set_xlabel("交易日"); a2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("performance.png",dpi=130,bbox_inches="tight"); plt.close()
    print("OK")

if __name__=="__main__":
    main()
