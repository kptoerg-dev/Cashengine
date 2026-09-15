import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ai_trading_engine_v10 import Config, main_with_config

st.set_page_config(page_title="AI Alpha Lab v10", page_icon="📈", layout="wide")

st.title("🤖 AI Alpha Lab v10")
st.caption("Return-focused ML portfolio research • Purged Walk-Forward • Dynamic Risk • Cross-Asset Ranking")

with st.sidebar:
    st.header("📈 Universum")
    assets=st.multiselect(
        "Assets",
        ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","QQQ","SPY","NVDA","MSFT","AAPL","AMZN","META","TSLA"],
        default=["BTC-USD","ETH-USD","SOL-USD","QQQ","NVDA","MSFT"]
    )
    initial=st.number_input("Startkapital ($)",1000.0,1_000_000.0,10000.0,1000.0)

    st.header("🔥 Alpha / Aggressivität")
    profile=st.select_slider("Risikoprofil",
        ["Defensiv","Balanced","Aggressiv","Alpha Max"],value="Aggressiv")
    profiles={
        "Defensiv":(.40,.002,.025),
        "Balanced":(.55,.0025,.035),
        "Aggressiv":(.70,.0025,.045),
        "Alpha Max":(.85,.004,.060)
    }
    kelly,minrisk,maxrisk=profiles[profile]
    minprob=st.slider("Mindestwahrscheinlichkeit",0.48,0.68,0.52,0.01)
    minedge=st.slider("Minimaler Expected Edge",0.0,0.10,0.015,0.005)
    topn=st.slider("Beste Signale pro Tag",1,6,3)

    st.header("🎯 Trade Engine")
    sl=st.slider("Stop Loss — ATR",1.0,3.0,1.7,0.1)
    tp=st.slider("Take Profit — ATR",2.0,7.0,4.2,0.1)
    hold=st.slider("Max. Haltedauer (Tage)",3,30,10)
    be=st.slider("Break-even ab R",0.5,2.5,1.15,0.05)
    trail=st.slider("Trailing ab R",1.0,4.0,1.8,0.1)
    trailatr=st.slider("Trailing Abstand ATR",1.0,4.0,2.2,0.1)

    st.header("💰 Portfolio")
    maxpos=st.slider("Max. Positionen",1,10,4)
    exposure=st.slider("Max. Exposure",0.25,1.0,1.0,0.05)
    single=st.slider("Max. Einzelposition",0.10,0.80,0.45,0.05)

    st.header("🛡️ Risiko-Regeln")
    softdd=st.slider("Soft Drawdown",0.05,0.25,0.10,0.01)
    harddd=st.slider("Hard Drawdown",0.10,0.40,0.20,0.01)
    regime=st.checkbox("Volatilitäts-Regime",True)
    highvol=st.slider("Risiko bei hoher Volatilität",0.25,1.0,0.65,0.05)
    counter=st.checkbox("Countertrend erlauben",False)
    trendpen=st.slider("Countertrend Risiko",0.20,1.0,0.55,0.05)

    st.header("🧠 Walk-Forward")
    folds=st.slider("Folds",3,10,7)
    train=st.slider("Min. Trainingstage",300,1500,600,50)
    validation=st.slider("OOS-Testfenster",30,180,90,10)
    purge=st.slider("Purge",0,30,10)
    embargo=st.slider("Embargo",0,30,10)
    trees=st.slider("LightGBM Trees",150,900,450,25)
    lr=st.slider("Learning Rate",0.005,0.08,0.025,0.005)

    st.header("📅 Zeitraum")
    start=st.date_input("Start",pd.Timestamp("2018-01-01").date())
    end=st.date_input("Ende",pd.Timestamp.today().date())
    run=st.button("🚀 ALPHA BACKTEST STARTEN",type="primary",use_container_width=True)

if not assets:
    st.warning("Bitte mindestens ein Asset wählen.")
    st.stop()

if run:
    if start>=end:
        st.error("Startdatum muss vor dem Enddatum liegen.")
        st.stop()

    cfg=Config(
        tickers=tuple(assets),initial_cash=initial,min_probability=minprob,min_edge=minedge,
        top_n_signals=topn,kelly_fraction=kelly,min_risk_pct=minrisk,max_risk_pct=maxrisk,
        atr_sl=sl,atr_tp=tp,max_hold_days=hold,breakeven_trigger_r=be,
        trailing_trigger_r=trail,trailing_atr=trailatr,max_positions=maxpos,
        max_exposure_pct=exposure,max_single_position_pct=single,
        soft_drawdown=softdd,hard_drawdown=harddd,use_regime_filter=regime,
        high_vol_risk_multiplier=highvol,allow_countertrend=counter,trend_penalty=trendpen,
        n_folds=folds,min_train_days=train,validation_days=validation,
        purge_days=purge,embargo_days=embargo,n_estimators=trees,learning_rate=lr,
        start_date=str(start),end_date=str(end)
    )

    with st.spinner("Alpha Engine arbeitet: Daten → Features → Purged Walk-Forward → Portfolio-Simulation …"):
        try:
            result=main_with_config(cfg)
        except Exception as e:
            st.exception(e)
            st.stop()

    m=result["metrics"]
    st.subheader("🏆 Performance")
    c=st.columns(8)
    c[0].metric("Endkapital",f"${m['Endkapital']:,.0f}")
    c[1].metric("Rendite",f"{m['Total Return']*100:.1f}%")
    c[2].metric("CAGR",f"{m['CAGR']*100:.1f}%")
    c[3].metric("Max DD",f"{m['Max Drawdown']*100:.1f}%")
    c[4].metric("Sharpe",f"{m['Sharpe']:.2f}")
    c[5].metric("Profit Factor",f"{m['Profit Factor']:.2f}")
    c[6].metric("Win Rate",f"{m['Win Rate']*100:.1f}%")
    c[7].metric("Trades",m["Trades"])

    eq=result["equity"]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=eq.index,y=eq.values,mode="lines",name="Equity"))
    fig.update_layout(template="plotly_dark",height=470,hovermode="x unified",
                      title="Compound Equity Curve",xaxis_title="Datum",yaxis_title="Portfolio ($)")
    st.plotly_chart(fig,use_container_width=True)

    tab1,tab2,tab3=st.tabs(["💹 Trades","🔬 Walk-Forward","🧬 Feature Selection"])
    with tab1:
        trades=pd.DataFrame(result["trades"])
        if trades.empty:
            st.warning("Keine Trades. Senke Mindestwahrscheinlichkeit / Edge oder erweitere das Universum.")
        else:
            display=trades.copy()
            display["pnl_pct"]=display["pnl_pct"]*100
            st.dataframe(display,use_container_width=True,hide_index=True)
            st.download_button("⬇️ Trades CSV",trades.to_csv(index=False).encode(), "trades_v10.csv","text/csv")
    with tab2:
        f=result["fold_reports"]
        if f.empty: st.warning("Keine gültigen OOS-Folds.")
        else:
            st.dataframe(f,use_container_width=True,hide_index=True)
            st.caption("AUC ist nur ein Diagnosewert. Entscheidend ist die OOS-Portfolio-Performance.")
    with tab3:
        s=result["selected_features"]
        if s.empty: st.info("Keine Feature-Daten.")
        else: st.dataframe(s,use_container_width=True,hide_index=True)

    st.divider()
    st.info("⚠️ Ein Backtest kann keine zukünftige Rendite garantieren. Besonders bei Alpha Max sind Drawdowns und Modellinstabilität bewusst höher.")

else:
    st.info("👈 Universum und Risiko einstellen und den Alpha-Backtest starten.")
