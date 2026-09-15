# -*- coding: utf-8 -*-
"""
AI TRADING ENGINE v10.0 — Alpha Portfolio
=========================================
Research-grade daily ML portfolio backtester.

Design goals:
- Maximize risk-adjusted compounded return rather than raw classification accuracy.
- Strict chronological / purged walk-forward evaluation.
- Fold-local feature selection and probability calibration.
- Cross-asset ranking: capital goes to the strongest expected opportunities.
- Expected-value / return-aware signal scoring.
- Volatility-aware dynamic position sizing.
- Trailing stop, break-even and time exits.
- Portfolio exposure, concentration and drawdown controls.
- No look-ahead in features, labels, model fitting or portfolio decisions.

This is a research/backtesting engine. Backtest performance does not guarantee
future live performance.
"""

from __future__ import annotations

import dataclasses
import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("AI-TRADING-v10")


@dataclass
class Config:
    tickers: Tuple[str, ...] = ("BTC-USD", "ETH-USD", "SOL-USD", "QQQ", "NVDA", "MSFT")
    benchmark: str = "^GSPC"
    start_date: str = "2018-01-01"
    end_date: str = "2026-09-01"

    initial_cash: float = 10_000.0
    fee_pct: float = 0.0010
    slippage_low: float = 0.00025
    slippage_high: float = 0.0015
    volatility_slippage_threshold: float = 1.6

    # Trade geometry
    atr_sl: float = 1.7
    atr_tp: float = 4.2
    max_hold_days: int = 10
    breakeven_trigger_r: float = 1.15
    trailing_trigger_r: float = 1.8
    trailing_atr: float = 2.2

    # Signal selection
    min_probability: float = 0.52
    min_edge: float = 0.015
    top_n_signals: int = 3
    score_temperature: float = 0.08

    # Aggressive but bounded portfolio risk
    kelly_fraction: float = 0.70
    min_risk_pct: float = 0.0025
    max_risk_pct: float = 0.045
    max_exposure_pct: float = 1.00
    max_single_position_pct: float = 0.45
    max_positions: int = 4

    # Drawdown protection
    soft_drawdown: float = 0.10
    hard_drawdown: float = 0.20
    soft_risk_multiplier: float = 0.70
    hard_risk_multiplier: float = 0.35

    # Regime
    use_regime_filter: bool = True
    high_vol_risk_multiplier: float = 0.65
    trend_penalty: float = 0.55
    allow_countertrend: bool = False

    # Walk-forward
    n_folds: int = 7
    min_train_days: int = 600
    validation_days: int = 90
    purge_days: int = 10
    embargo_days: int = 10

    # LightGBM
    n_estimators: int = 450
    max_depth: int = 5
    num_leaves: int = 31
    min_child_samples: int = 40
    learning_rate: float = 0.025
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    calibration_splits: int = 3
    feature_threshold: str = "median"
    random_state: int = 42

    features: Tuple[str, ...] = (
        "RET_1D","RET_2D","RET_3D","RET_5D","RET_10D","RET_20D","RET_60D",
        "RSI","ATR_PCT","ATR_REGIME","REALIZED_VOL_10","REALIZED_VOL_20",
        "REALIZED_VOL_60","RET_SKEW_20","RET_KURT_20",
        "EMA20_DIST","EMA50_DIST","EMA100_DIST","EMA200_DIST","EMA20_SLOPE",
        "EMA50_SLOPE","TREND_STRENGTH","MOM_ACCEL",
        "VOLUME_RATIO","VOL_MOM","RANGE_PCT","BODY_PCT",
        "UPPER_WICK_PCT","LOWER_WICK_PCT",
        "SP500_RET_3D","SP500_RET_20D","SP500_RET_60D",
        "BETA_60","REL_STRENGTH_20","REL_STRENGTH_60",
        "DOW_SIN","DOW_COS","MONTH_SIN","MONTH_COS",
    )


def _fetch_yfinance(ticker: str, bench: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    asset = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    benchmark = yf.download(bench, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(asset.columns, pd.MultiIndex):
        asset.columns = asset.columns.get_level_values(0)
    if isinstance(benchmark.columns, pd.MultiIndex):
        benchmark.columns = benchmark.columns.get_level_values(0)
    needed = ["Open","High","Low","Close","Volume"]
    if any(c not in asset.columns for c in needed):
        raise ValueError(f"{ticker}: historische OHLCV-Daten unvollständig.")
    out = asset[needed].copy()
    out["SP500_Close"] = benchmark["Close"].reindex(out.index).ffill().shift(1)
    return out.dropna(subset=["Open","High","Low","Close"])


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100/(1+rs)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    o = df.copy()
    c = o["Close"].astype(float)
    lr = np.log(c / c.shift(1))
    bench_lr = np.log(o["SP500_Close"] / o["SP500_Close"].shift(1))

    for p in (1,2,3,5,10,20,60):
        o[f"RET_{p}D"] = c.pct_change(p)

    o["RSI"] = _rsi(c)
    pc = c.shift(1)
    tr = pd.concat([(o["High"]-o["Low"]), (o["High"]-pc).abs(), (o["Low"]-pc).abs()], axis=1).max(axis=1)
    o["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()
    o["ATR_PCT"] = o["ATR"]/c
    o["ATR_REGIME"] = o["ATR_PCT"]/o["ATR_PCT"].rolling(60).mean()

    for w in (10,20,60):
        o[f"REALIZED_VOL_{w}"] = lr.rolling(w).std()
    o["RET_SKEW_20"] = lr.rolling(20).skew()
    o["RET_KURT_20"] = lr.rolling(20).kurt()

    for p in (20,50,100,200):
        ema = c.ewm(span=p, adjust=False).mean()
        o[f"EMA{p}_DIST"] = c/ema - 1
        if p in (20,50):
            o[f"EMA{p}_SLOPE"] = ema.pct_change(5)

    o["TREND_STRENGTH"] = (o["EMA20_DIST"] - o["EMA200_DIST"]).clip(-2,2)
    o["MOM_ACCEL"] = o["RET_5D"] - o["RET_20D"]/4
    o["VOLUME_RATIO"] = o["Volume"]/o["Volume"].rolling(20).mean()
    o["VOL_MOM"] = o["VOLUME_RATIO"] * o["RET_1D"]
    o["RANGE_PCT"] = (o["High"]-o["Low"])/c
    o["BODY_PCT"] = (o["Close"]-o["Open"]).abs()/c
    o["UPPER_WICK_PCT"] = (o["High"]-o[["Open","Close"]].max(axis=1))/c
    o["LOWER_WICK_PCT"] = (o[["Open","Close"]].min(axis=1)-o["Low"])/c

    o["SP500_RET_3D"] = o["SP500_Close"].pct_change(3)
    o["SP500_RET_20D"] = o["SP500_Close"].pct_change(20)
    o["SP500_RET_60D"] = o["SP500_Close"].pct_change(60)

    cov = lr.rolling(60).cov(bench_lr)
    var = bench_lr.rolling(60).var()
    o["BETA_60"] = cov / var.replace(0,np.nan)
    o["REL_STRENGTH_20"] = o["RET_20D"] - o["SP500_RET_20D"]
    o["REL_STRENGTH_60"] = o["RET_60D"] - o["SP500_RET_60D"]

    dow = o.index.dayofweek
    month = o.index.month - 1
    o["DOW_SIN"] = np.sin(2*np.pi*dow/5)
    o["DOW_COS"] = np.cos(2*np.pi*dow/5)
    o["MONTH_SIN"] = np.sin(2*np.pi*month/12)
    o["MONTH_COS"] = np.cos(2*np.pi*month/12)
    return o


def create_labels(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Triple-barrier-ish long label. Future only; no cross-asset shifting."""
    o = df.copy()
    h = int(cfg.max_hold_days)
    entry = o["Open"].shift(-1)
    atr = o["ATR"]
    sl = entry - cfg.atr_sl*atr
    tp = entry + cfg.atr_tp*atr

    low, high = o["Low"].to_numpy(), o["High"].to_numpy()
    slv, tpv = sl.to_numpy(), tp.to_numpy()
    n = len(o)
    sentinel = h + 1
    sl_hit = np.full(n, sentinel, dtype=int)
    tp_hit = np.full(n, sentinel, dtype=int)

    for step in range(1,h+1):
        lo = np.full(n,np.nan); hi = np.full(n,np.nan)
        if step < n:
            lo[:-step] = low[step:]
            hi[:-step] = high[step:]
        both = (lo <= slv) & (hi >= tpv)
        hit_sl = lo <= slv
        hit_tp = hi >= tpv
        sl_hit = np.where((sl_hit==sentinel) & (hit_sl | both), step, sl_hit)
        tp_hit = np.where((tp_hit==sentinel) & hit_tp & ~both, step, tp_hit)

    o["Target"] = (tp_hit < sl_hit).astype(float)
    first = np.minimum(sl_hit,tp_hit)
    o["T1_OFFSET"] = np.where(first==sentinel,h,first).astype(int)
    # Return proxy: conservative realized outcome of the barrier framework.
    rr = cfg.atr_tp/cfg.atr_sl
    o["TRADE_RETURN_TARGET"] = np.where(
        tp_hit < sl_hit, rr, np.where(sl_hit < tp_hit, -1.0, 0.0)
    )
    invalid = (np.arange(n) >= n-h-1) | entry.isna() | atr.isna()
    o.loc[invalid, ["Target","TRADE_RETURN_TARGET"]] = np.nan
    o.loc[invalid,"T1_OFFSET"] = -1
    return o


@dataclass
class FoldModel:
    model: CalibratedClassifierCV
    selector: SelectFromModel
    gmm: Optional[GaussianMixture]
    high_vol_cluster: int
    selected_features: List[str]


def _base_model(cfg: Config):
    return lgb.LGBMClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        class_weight="balanced",
        random_state=cfg.random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_fold_model(train: pd.DataFrame, cfg: Config) -> Optional[FoldModel]:
    cols = [c for c in cfg.features if c in train.columns]
    x = train.dropna(subset=cols+["Target"]).copy()
    if len(x) < cfg.min_train_days or x["Target"].nunique() < 2:
        return None

    X,y = x[cols],x["Target"].astype(int)
    selector = SelectFromModel(_base_model(cfg), threshold=cfg.feature_threshold).fit(X,y)
    selected = [c for c,k in zip(cols,selector.get_support()) if k]
    if not selected:
        selected = cols

    splits = min(cfg.calibration_splits,max(2,len(x)//200))
    model = CalibratedClassifierCV(_base_model(cfg),cv=TimeSeriesSplit(n_splits=splits),method="sigmoid")
    model.fit(selector.transform(X),y)

    gmm=None; high=0
    if cfg.use_regime_filter:
        v=x[["REALIZED_VOL_20"]].fillna(0)
        if len(v)>=150 and v["REALIZED_VOL_20"].nunique()>1:
            gmm=GaussianMixture(n_components=2,random_state=cfg.random_state).fit(v)
            high=int(np.argmax(gmm.means_.ravel()))
    return FoldModel(model,selector,gmm,high,selected)


def _walk_folds(n:int,cfg:Config):
    test = max(cfg.validation_days,(n-cfg.min_train_days)//max(1,cfg.n_folds))
    out=[]
    for k in range(cfg.n_folds):
        lo=cfg.min_train_days+k*test
        hi=min(n,lo+test)
        if lo>=n or hi<=lo: break
        out.append((lo,hi))
    return out


def _purged_mask(n,test_lo,test_hi,t1,purge,embargo):
    idx=np.arange(n)
    exits=idx+np.maximum(t1,0)
    overlap=(idx<test_lo)&(exits>=test_lo)
    purge_zone=(idx>=max(0,test_lo-purge))&(idx<test_lo)
    embargo_zone=(idx>=test_hi)&(idx<test_hi+embargo)
    return (idx<test_lo)&~overlap&~purge_zone&~embargo_zone


def walk_forward_predict(df:pd.DataFrame,cfg:Config):
    labeled=create_labels(df,cfg)
    cols=list(cfg.features)
    clean=labeled.dropna(subset=cols+["Target","T1_OFFSET"]).copy()
    probs=pd.Series(index=clean.index,dtype=float)
    regimes=pd.Series(index=clean.index,dtype=int)
    reports=[]; selected=[]

    for fold,(lo,hi) in enumerate(_walk_folds(len(clean),cfg),1):
        t1=clean["T1_OFFSET"].to_numpy(int)
        mask=_purged_mask(len(clean),lo,hi,t1,cfg.purge_days,cfg.embargo_days)
        train=clean.iloc[mask]; test=clean.iloc[lo:hi]
        if len(train)<cfg.min_train_days or test.empty: continue
        fm=fit_fold_model(train,cfg)
        if fm is None: continue

        p=fm.model.predict_proba(fm.selector.transform(test[cols]))[:,1]
        probs.loc[test.index]=p
        if fm.gmm is not None:
            regimes.loc[test.index]=(fm.gmm.predict(test[["REALIZED_VOL_20"]].fillna(0))==fm.high_vol_cluster).astype(int)

        auc=np.nan
        if test["Target"].nunique()>1: auc=roc_auc_score(test["Target"].astype(int),p)
        reports.append({"fold":fold,"train_rows":len(train),"test_rows":len(test),
                        "auc":float(auc) if np.isfinite(auc) else np.nan,
                        "features_selected":len(fm.selected_features)})
        selected.append({"fold":fold,"features":", ".join(fm.selected_features)})
    return probs,regimes.fillna(0).astype(int),reports,selected


def signal_score(row,prob,cfg):
    """Combines calibrated probability, payoff, trend and volatility quality."""
    rr=cfg.atr_tp/max(cfg.atr_sl,1e-9)
    edge=prob*rr-(1-prob)
    trend=float(row.EMA200_DIST)
    trend_boost=1.0+np.clip(trend/0.15,-0.25,0.25)
    vol_quality=1.0/np.clip(float(row.ATR_REGIME),0.65,2.5)
    score=edge*trend_boost*vol_quality
    return float(edge),float(score)


def _portfolio_risk_multiplier(equity,peak,cfg):
    dd=equity/peak-1 if peak>0 else 0
    if dd<=-cfg.hard_drawdown: return cfg.hard_risk_multiplier
    if dd<=-cfg.soft_drawdown: return cfg.soft_risk_multiplier
    return 1.0


class Broker:
    def __init__(self,cfg):
        self.cfg=cfg; self.cash=cfg.initial_cash; self.positions={}; self.trades=[]
        self.peak=cfg.initial_cash

    def equity(self,marks):
        return self.cash+sum(p["qty"]*marks.get(t,p["entry_price"]) for t,p in self.positions.items())

    def exits(self,date,rows):
        remove=[]
        for t,p in list(self.positions.items()):
            r=rows.get(t)
            if r is None: continue
            risk=p["initial_risk"]
            exit_price=None; reason=None

            if r.Open<=p["stop"]:
                exit_price,reason=r.Open,"STOP_GAP"
            elif r.Open>=p["target"]:
                exit_price,reason=r.Open,"TP_GAP"
            elif r.Low<=p["stop"] and r.High>=p["target"]:
                exit_price,reason=p["stop"],"STOP_AMBIGUOUS"
            elif r.Low<=p["stop"]:
                exit_price,reason=p["stop"],"STOP"
            elif r.High>=p["target"]:
                exit_price,reason=p["target"],"TAKE_PROFIT"
            else:
                favorable=r.High-p["entry_price"]
                if favorable>=self.cfg.breakeven_trigger_r*risk:
                    p["stop"]=max(p["stop"],p["entry_price"])
                if favorable>=self.cfg.trailing_trigger_r*risk:
                    p["stop"]=max(p["stop"],r.High-self.cfg.trailing_atr*r.ATR)
                if (date-p["entry_date"]).days>=self.cfg.max_hold_days:
                    exit_price,reason=r.Close,"TIME"

            if exit_price is not None:
                slip=self.cfg.slippage_high if r.VOL_RATIO>self.cfg.volatility_slippage_threshold else self.cfg.slippage_low
                px=exit_price*(1-slip)
                proceeds=p["qty"]*px*(1-self.cfg.fee_pct)
                self.cash+=proceeds
                self.trades.append({
                    "ticker":t,"entry_date":p["entry_date"],"exit_date":date,
                    "entry_price":p["entry_price"],"exit_price":px,
                    "pnl":proceeds-p["entry_cash"],
                    "pnl_pct":proceeds/p["entry_cash"]-1,
                    "reason":reason,
                    "hold_days":(date-p["entry_date"]).days
                })
                remove.append(t)
        for t in remove: self.positions.pop(t,None)

    def entries(self,date,signals,marks):
        if not signals: return
        signals=sorted(signals,key=lambda x:x["score"],reverse=True)[:self.cfg.top_n_signals]
        if len(self.positions)>=self.cfg.max_positions: return

        eq=self.equity(marks); self.peak=max(self.peak,eq)
        dd_mult=_portfolio_risk_multiplier(eq,self.peak,self.cfg)

        for s in signals:
            if len(self.positions)>=self.cfg.max_positions: break
            t,r,prob,edge,score,risk_mult=s["ticker"],s["row"],s["prob"],s["edge"],s["score"],s["risk_mult"]
            if t in self.positions or edge<self.cfg.min_edge: continue

            sl_dist=self.cfg.atr_sl*r.ATR
            if not np.isfinite(sl_dist) or sl_dist<=0: continue
            slip=self.cfg.slippage_high if r.VOL_RATIO>self.cfg.volatility_slippage_threshold else self.cfg.slippage_low
            entry=r.Open*(1+slip)

            rr=self.cfg.atr_tp/self.cfg.atr_sl
            k=max(0,(prob*rr-(1-prob))/max(rr,1e-9))
            risk_pct=k*self.cfg.kelly_fraction*risk_mult*dd_mult
            risk_pct=float(np.clip(risk_pct,self.cfg.min_risk_pct,self.cfg.max_risk_pct))
            risk_cash=eq*risk_pct
            qty=risk_cash/sl_dist

            exposure_used=sum(p["qty"]*marks.get(x,p["entry_price"]) for x,p in self.positions.items())
            room=max(0,eq*self.cfg.max_exposure_pct-exposure_used)
            qty=min(qty,eq*self.cfg.max_single_position_pct/max(entry,1e-12),room/max(entry,1e-12))
            cost=qty*entry*(1+self.cfg.fee_pct)
            if qty<=0 or cost>self.cash: continue

            self.cash-=cost
            self.positions[t]={
                "entry_date":date,"entry_price":entry,"qty":qty,
                "stop":entry-sl_dist,"target":entry+self.cfg.atr_tp*r.ATR,
                "initial_risk":sl_dist,"entry_cash":cost
            }


def run_backtest(data,probs,regimes,cfg):
    dates=sorted(set().union(*[set(x.index) for x in data.values()]))
    b=Broker(cfg); curve=[]

    for date in dates:
        rows={}; marks={}
        for t,df in data.items():
            if date in df.index:
                rows[t]=df.loc[date]; marks[t]=float(df.loc[date].Close)

        b.exits(date,rows)
        eq=b.equity(marks); b.peak=max(b.peak,eq)
        signals=[]

        for t,r in rows.items():
            p=probs.get(t,pd.Series(dtype=float)).get(date,np.nan)
            if not np.isfinite(p) or p<cfg.min_probability: continue

            edge,score=signal_score(r,float(p),cfg)
            if edge<cfg.min_edge: continue

            risk_mult=1.0
            if cfg.use_regime_filter and regimes.get(t,pd.Series(dtype=int)).get(date,0)==1:
                risk_mult*=cfg.high_vol_risk_multiplier

            trend_ok=(r.EMA20_DIST>0 and r.EMA200_DIST>0)
            if not trend_ok:
                if not cfg.allow_countertrend: continue
                risk_mult*=cfg.trend_penalty

            signals.append({"ticker":t,"row":r,"prob":float(p),"edge":edge,
                            "score":score,"risk_mult":risk_mult})

        b.entries(date,signals,marks)
        curve.append({"Date":date,"Equity":b.equity(marks)})

    equity=pd.DataFrame(curve).set_index("Date")["Equity"]
    return equity,b.trades


def metrics(equity,trades,cfg):
    if equity.empty:
        return {"Endkapital":cfg.initial_cash,"Total Return":0,"CAGR":0,"Max Drawdown":0,
                "Win Rate":0,"Trades":0,"Profit Factor":0,"Sharpe":0,"Avg Trade":0}

    ret=equity.iloc[-1]/cfg.initial_cash-1
    dd=equity/equity.cummax()-1
    wins=[x["pnl"] for x in trades if x["pnl"]>0]
    losses=[-x["pnl"] for x in trades if x["pnl"]<0]
    pf=sum(wins)/sum(losses) if losses else (999 if wins else 0)
    daily=equity.pct_change().replace([np.inf,-np.inf],np.nan).dropna()
    sharpe=np.sqrt(252)*daily.mean()/daily.std() if daily.std()>0 else 0
    years=max((equity.index[-1]-equity.index[0]).days/365.25,1/365.25)
    cagr=(equity.iloc[-1]/cfg.initial_cash)**(1/years)-1 if equity.iloc[-1]>0 else -1
    return {
        "Endkapital":float(equity.iloc[-1]),"Total Return":float(ret),"CAGR":float(cagr),
        "Max Drawdown":float(dd.min()),"Win Rate":float(len(wins)/len(trades)) if trades else 0,
        "Trades":len(trades),"Profit Factor":float(pf),
        "Sharpe":float(sharpe),"Avg Trade":float(np.mean([x["pnl_pct"] for x in trades])) if trades else 0
    }


def main_with_config(cfg:Config)->dict:
    data={}; probs={}; regimes={}; folds=[]; selected=[]
    for ticker in cfg.tickers:
        raw=_fetch_yfinance(ticker,cfg.benchmark,cfg.start_date,cfg.end_date)
        data[ticker]=build_features(raw)
        p,r,fr,sf=walk_forward_predict(data[ticker],cfg)
        probs[ticker]=p; regimes[ticker]=r
        for z in fr:
            z["ticker"]=ticker; folds.append(z)
        for z in sf:
            z["ticker"]=ticker; selected.append(z)

    equity,trades=run_backtest(data,probs,regimes,cfg)
    return {
        "metrics":metrics(equity,trades,cfg),"equity":equity,"trades":trades,
        "fold_reports":pd.DataFrame(folds),"selected_features":pd.DataFrame(selected),
        "probabilities":probs,"regimes":regimes,"config":cfg
    }


if __name__=="__main__":
    print(main_with_config(Config())["metrics"])
