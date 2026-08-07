"""Lazy local HTML report for one prepared dataset."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .low_activity import ATOM_SECONDS, LowActivityResult, json_ready


OVERVIEW_MINUTES = 15
DETAIL_PADDING_MINUTES = 30


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>低活跃与交易机制过滤审计</title>
<style>
:root{color-scheme:light;--bg:#edf2f7;--panel:#fff;--panel2:#f7f9fc;--plot:#fff;--line:#d4dee9;--text:#10233d;--body:#2d425f;--muted:#63758c;--accent:#1769c2;--soft:#e7f2ff;--volume:#0f9f83;--volumeFill:rgba(15,159,131,.13);--lock:#a23bb9;--lockFill:rgba(162,59,185,.14);--circuit:#e37a12;--circuitFill:rgba(227,122,18,.15);--auction:#c44725;--auctionFill:rgba(196,71,37,.13);--uncertain:#7b8798;--uncertainFill:rgba(123,135,152,.13);--confirm:#1769c2;--up:#aeb7c2;--upFill:#f4f6f8;--down:#505b68;--downFill:#7e8996}
:root[data-theme=dark]{color-scheme:dark;--bg:#080a0e;--panel:#0e1116;--panel2:#090c11;--plot:#090c11;--line:#252b34;--text:#f4f7fb;--body:#d4dbe5;--muted:#98a4b4;--accent:#72b9ff;--soft:#142235;--volume:#4de1bd;--volumeFill:rgba(77,225,189,.12);--lock:#d887ee;--lockFill:rgba(216,135,238,.13);--circuit:#ff9c3d;--circuitFill:rgba(255,156,61,.14);--auction:#ff6f61;--auctionFill:rgba(255,111,97,.12);--uncertain:#8995a5;--uncertainFill:rgba(137,149,165,.12);--confirm:#72b9ff;--up:#9ca7b4;--upFill:#d9dee5;--down:#ff788e;--downFill:#d24a63}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:1rem/1.55 "Segoe UI","Microsoft YaHei",system-ui,sans-serif}.shell{max-width:1880px;margin:auto;padding:18px 22px 48px}.top{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;margin-bottom:14px}h1{font-size:1.72rem;line-height:1.2;margin:0 0 6px}.lede{max-width:88ch;color:var(--body);font-size:.94rem}.theme{display:flex;gap:3px;padding:3px;border:1px solid var(--line);border-radius:10px;background:var(--panel)}button{font:inherit}.theme button,.small{border:1px solid transparent;border-radius:7px;background:transparent;color:var(--body);font-weight:700;cursor:pointer;padding:7px 11px}.theme button[aria-pressed=true]{border-color:var(--accent);background:var(--soft);color:var(--text)}.notice{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px;color:var(--body)}.metrics,.legend{display:flex;gap:8px;flex-wrap:wrap}.metric{border:1px solid var(--line);border-radius:999px;background:var(--panel2);padding:5px 9px;font-size:.86rem}.legend{margin-top:10px;color:var(--muted);font-size:.82rem}.legend span{display:inline-flex;align-items:center;gap:6px}.swatch{width:13px;height:13px;border:1px solid currentColor;border-radius:3px}.swatch.volume{color:var(--volume);background:var(--volumeFill)}.swatch.lock{color:var(--lock);background:var(--lockFill)}.swatch.circuit{color:var(--circuit);background:var(--circuitFill)}.sections{display:flex;flex-direction:column;gap:14px;margin-top:15px}.section{border:1px solid var(--line);border-radius:14px;background:var(--panel);overflow:hidden}.section>summary{list-style:none;display:flex;align-items:center;justify-content:space-between;gap:15px;min-height:76px;padding:14px 16px;cursor:pointer}.section>summary::-webkit-details-marker{display:none}.section>summary:hover{background:var(--panel2)}.title{display:flex;gap:11px;align-items:center}.mark{width:10px;height:36px;border-radius:5px;background:var(--volume)}[data-family=mechanism] .mark{background:linear-gradient(var(--lock) 0 48%,var(--circuit) 52%)}h2{font-size:1.1rem;margin:0}.copy{color:var(--muted);font-size:.83rem;margin-top:3px}.expand{color:var(--accent);font-weight:750}.expand:after{content:" ＋"}.section[open] .expand:after{content:" −"}.content{border-top:1px solid var(--line)}.loading{display:grid;place-items:center;min-height:220px;color:var(--muted)}.toolbar{display:flex;justify-content:space-between;gap:10px;align-items:center;padding:9px 14px;background:var(--panel2);border-bottom:1px solid var(--line)}.actions{display:flex;gap:6px;flex-wrap:wrap}.small{border-color:var(--line);background:var(--panel);padding:5px 9px}.chart{height:690px;background:var(--plot)}.events{border-top:1px solid var(--line);padding:12px 14px}.event{display:grid;grid-template-columns:minmax(190px,.4fr) minmax(300px,1fr);gap:12px;border-top:1px solid var(--line);padding:9px 0;color:var(--body);font-size:.84rem}.event:first-child{border-top:0}.event button{text-align:left;border:0;background:transparent;color:var(--accent);font:inherit;font-weight:750;cursor:pointer}.event code{color:var(--muted)}.foot{padding:10px 14px;border-top:1px solid var(--line);color:var(--muted);font-size:.8rem}@media(max-width:850px){.shell{padding:12px}.top,.toolbar,.section>summary{align-items:flex-start;flex-direction:column}.chart{height:610px}.event{grid-template-columns:1fr}}
</style></head><body><main class="shell"><div class="top"><div><h1>V4.4 低活跃与交易机制过滤审计</h1><div class="lede">每个标题在展开时才加载对应行情。通用低成交量使用「临时缓冲—恢复回填—确认排除」；K200 交易机制保持独立原因。</div></div><div class="theme"><button id="light" aria-pressed="true">浅色 Light</button><button id="dark" aria-pressed="false">深色 Dark</button></div></div>
<section class="notice"><div id="metrics" class="metrics"></div><div class="legend"><span><i class="swatch volume"></i>通用低成交量</span><span><i class="swatch lock"></i>涨跌停锁价候选</span><span><i class="swatch circuit"></i>熔断／集合竞价候选</span></div></section>
<div class="sections"><details class="section" data-section="universal" data-family="universal"><summary><div class="title"><i class="mark"></i><div><h2>通用低成交量</h2><div class="copy">84小时早期可信正成交量中位数 × 20%；准备阶段完成缓冲分类，短暂恢复保留，连续30分钟标记排除。</div></div></div><span class="expand">展开并加载</span></summary><div class="content"><div class="loading">等待加载…</div></div></details>
<details class="section" data-section="mechanism" data-family="mechanism"><summary><div class="title"><i class="mark"></i><div><h2>K200 涨跌停与熔断候选</h2><div class="copy">仅使用 OHLCV 与成交量推断；各类型保持独立颜色、证据和基准排除状态。</div></div></div><span class="expand">展开并加载</span></summary><div class="content"><div class="loading">等待加载…</div></div></details></div>
<p class="foot">审计标记规则：过滤程序一次完成疑似、恢复与确认分类；短暂恢复原子保留，达到30分钟的低成交段保留审计标记。V4.4 默认「全部」策略使用同一连续段内全部有限 TR15 原子；可选「排除标记」策略忽略 baseline_excluded 原子并向前补足 BH。两种策略独立记录和排名。灰色长停候选只展示，但仍参加通用低成交检测。</p></main>
<script>const SUMMARY=__SUMMARY__;let plotlyPromise=null;const loaded={};const themes={light:{template:'plotly_white',paper:'#fff',plot:'#fff',ink:'#10233d',grid:'rgba(100,116,139,.17)',up:'#aeb7c2',upFill:'#f4f6f8',down:'#505b68',downFill:'#7e8996',volume:'#0f9f83',volumeFill:'rgba(15,159,131,.13)',lock:'#a23bb9',lockFill:'rgba(162,59,185,.14)',circuit:'#e37a12',circuitFill:'rgba(227,122,18,.15)',auction:'#c44725',auctionFill:'rgba(196,71,37,.13)',uncertain:'#7b8798',uncertainFill:'rgba(123,135,152,.13)',confirm:'#1769c2'},dark:{template:'plotly_dark',paper:'#0e1116',plot:'#090c11',ink:'#f4f7fb',grid:'rgba(148,163,184,.13)',up:'#9ca7b4',upFill:'#d9dee5',down:'#ff788e',downFill:'#d24a63',volume:'#4de1bd',volumeFill:'rgba(77,225,189,.12)',lock:'#d887ee',lockFill:'rgba(216,135,238,.13)',circuit:'#ff9c3d',circuitFill:'rgba(255,156,61,.14)',auction:'#ff6f61',auctionFill:'rgba(255,111,97,.12)',uncertain:'#8995a5',uncertainFill:'rgba(137,149,165,.12)',confirm:'#72b9ff'}};let theme='light';const pct=v=>(100*Number(v||0)).toFixed(2)+'%',num=v=>Number(v||0).toLocaleString('zh-CN',{maximumFractionDigits:2}),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
document.getElementById('metrics').innerHTML=`<span class=metric>数据 <b>${esc(SUMMARY.instrument)}</b></span><span class=metric>确认标记 <b>${num(SUMMARY.baseline_excluded_minutes)} 分钟</b></span><span class=metric>恢复回填原子 <b>${num(SUMMARY.buffer_reinserted_atom_count)}</b></span><span class=metric>通用区间 <b>${num(SUMMARY.universal_event_count)}</b></span><span class=metric>锁价候选 <b>${num(SUMMARY.price_lock_event_count)}</b></span><span class=metric>熔断候选 <b>${num(SUMMARY.circuit_event_count)}</b></span>`;
function loadScript(src){return new Promise((ok,bad)=>{const s=document.createElement('script');s.src=src;s.onload=ok;s.onerror=bad;document.head.appendChild(s)})}function ensurePlotly(){return plotlyPromise||(plotlyPromise=loadScript('assets/plotly.min.js'))}function color(e,t){if(e.event_type==='universal_low_volume')return {line:t.volume,fill:t.volumeFill};if(String(e.event_type).includes('lock'))return {line:t.lock,fill:t.lockFill};if(e.event_type==='circuit_breaker_candidate')return {line:t.circuit,fill:t.circuitFill};return {line:t.uncertain,fill:t.uncertainFill}}
function shapes(events,t){const out=[];for(const e of events){const c=color(e,t);if(e.event_type==='circuit_breaker_candidate'&&e.halt_end){out.push({type:'rect',xref:'x',yref:'paper',x0:e.x0,x1:e.halt_x,y0:0,y1:1,fillcolor:t.circuitFill,line:{color:t.circuit,width:.6},layer:'below'},{type:'rect',xref:'x',yref:'paper',x0:e.auction_x,x1:e.x1,y0:0,y1:1,fillcolor:t.auctionFill,line:{color:t.auction,width:.6},layer:'below'})}else out.push({type:'rect',xref:'x',yref:'paper',x0:e.x0,x1:e.x1,y0:0,y1:1,fillcolor:c.fill,line:{color:c.line,width:.6},layer:'below'});if(e.confirmation_x!=null)out.push({type:'line',xref:'x',yref:'paper',x0:e.confirmation_x,x1:e.confirmation_x,y0:0,y1:1,line:{color:t.confirm,width:1.1,dash:'dot'}})}return out}
function hover(bars){return bars.map((r,i)=>{const p=bars[i-1],change=p&&Number(p.close)>0?(Number(r.close)/Number(p.close)-1)*100:null;return `序号 ${num(r.source_index+1)}<br>时间 ${esc(r.datetime)}${r.end_time?` → ${esc(r.end_time)}`:''}<br>open ${Number(r.open).toFixed(4)}<br>high ${Number(r.high).toFixed(4)}<br>low ${Number(r.low).toFixed(4)}<br>close ${Number(r.close).toFixed(4)}<br>涨跌幅（相对前收） ${Number.isFinite(change)?(change>=0?'+':'')+change.toFixed(2)+'%':'—'}<br><b>成交量 ${num(r.volume)}</b><br>成交笔数 ${num(r.trade_count)}${r.low_activity_state?`<br>low_activity_state ${esc(r.low_activity_state)}`:''}${r.pending_buffer_start?`<br>pending_buffer_start ${esc(r.pending_buffer_start)}`:''}${r.pending_buffer_count?`<br>pending_buffer_count ${num(r.pending_buffer_count)}`:''}${r.buffer_reinserted?'<br>buffer_reinserted true':''}${r.buffer_confirmed_excluded?'<br>buffer_confirmed_excluded true':''}${r.recovery_confirmation_time?`<br>recovery_confirmation_time ${esc(r.recovery_confirmation_time)}`:''}${r.reason_codes?`<br>过滤原因 ${esc(r.reason_codes)}`:''}`})}
function traces(bars,t){const text=hover(bars);return [{type:'candlestick',x:bars.map(r=>r.x),open:bars.map(r=>r.open),high:bars.map(r=>r.high),low:bars.map(r=>r.low),close:bars.map(r=>r.close),text,hoverinfo:'text',increasing:{line:{color:t.up,width:.8},fillcolor:t.upFill},decreasing:{line:{color:t.down,width:.8},fillcolor:t.downFill},yaxis:'y'},{type:'bar',x:bars.map(r=>r.x),y:bars.map(r=>r.volume),text,hoverinfo:'text',marker:{color:t.volume,line:{width:0}},yaxis:'y2'}]}
function layout(data,t,events){return {template:t.template,paper_bgcolor:t.paper,plot_bgcolor:t.plot,font:{color:t.ink},margin:{l:70,r:30,t:28,b:48},showlegend:false,xaxis:{type:'linear',rangeslider:{visible:false},gridcolor:t.grid},yaxis:{domain:[.24,1],gridcolor:t.grid,title:'price'},yaxis2:{domain:[0,.18],gridcolor:t.grid,title:'volume'},shapes:shapes(events,t),hovermode:'x unified',dragmode:'zoom'}}
function eventRows(data){return data.events.length?data.events.map(e=>`<div class=event><button data-event="${esc(e.event_id)}">${esc(e.label)} · ${esc(e.start)} → ${esc(e.end)}</button><div>${esc(e.reason)}<br><code>确认 ${esc(e.confirmation_time)} · ${e.apply_to_baseline?'可被「排除标记」策略使用':'仅展示'}</code></div></div>`).join(''):'<div class=event>当前数据没有命中该类型。</div>'}function detailEvent(e,bars){const nearest=time=>bars.reduce((best,row)=>Math.abs(new Date(row.datetime)-new Date(time))<Math.abs(new Date(best.datetime)-new Date(time))?row:best,bars[0]);const a=nearest(e.start),b=nearest(e.end),c=nearest(e.confirmation_time),out={...e,x0:a.x-.48,x1:b.x+.48,confirmation_x:c.x};if(e.event_type==='circuit_breaker_candidate'){out.halt_x=nearest(e.halt_end).x;out.auction_x=nearest(e.call_auction_start).x}return out}
function renderSection(id){const details=document.querySelector(`[data-section=${id}]`),data=window['FILTER_'+id.toUpperCase()],content=details.querySelector('.content');content.innerHTML=`<div class=toolbar><span>15分钟总览；点击下方事件查看15秒原子</span><div class=actions><button class=small data-all>全区间</button></div></div><div class=chart></div><div class=events>${eventRows(data)}</div>`;const gd=content.querySelector('.chart');const draw=(bars,events)=>Plotly.react(gd,traces(bars,themes[theme]),layout(data,themes[theme],events),{responsive:true,displaylogo:false,doubleClick:'reset+autosize'});const overview=()=>draw(data.bars,data.events);content.querySelector('[data-all]').onclick=overview;content.querySelectorAll('[data-event]').forEach(b=>b.onclick=()=>{const e=data.events.find(x=>x.event_id===b.dataset.event),bars=data.details[e.event_id]||[];if(bars.length)draw(bars,[detailEvent(e,bars)])});overview();gd.on('plotly_doubleclick',()=>{setTimeout(overview,0);return false});loaded[id]={draw:()=>overview()}}
document.querySelectorAll('.section').forEach(d=>d.addEventListener('toggle',async()=>{if(!d.open||loaded[d.dataset.section])return;try{await ensurePlotly();await loadScript(`sections/${d.dataset.section}.js`);renderSection(d.dataset.section)}catch(e){d.querySelector('.content').innerHTML='<div class=loading>加载失败：'+esc(e.message)+'</div>'}}));function setTheme(next){theme=next;document.documentElement.dataset.theme=next==='dark'?'dark':'';document.getElementById('light').setAttribute('aria-pressed',next==='light');document.getElementById('dark').setAttribute('aria-pressed',next==='dark');Object.values(loaded).forEach(x=>x.draw())}document.getElementById('light').onclick=()=>setTheme('light');document.getElementById('dark').onclick=()=>setTheme('dark');
</script></body></html>'''


def _overview(atoms: pd.DataFrame) -> list[dict[str, Any]]:
    frame = atoms.copy()
    frame["segment"] = frame["datetime"].diff().dt.total_seconds().ne(ATOM_SECONDS).cumsum()
    atoms_per_bar = OVERVIEW_MINUTES * 60 // ATOM_SECONDS
    frame["group"] = frame.groupby("segment", sort=False).cumcount().floordiv(atoms_per_bar)
    bars = (
        frame.groupby(["segment", "group"], sort=False)
        .agg(
            source_index=("source_index", "first"), datetime=("datetime", "first"),
            end_time=("datetime", "last"), open=("open", "first"),
            high=("high", "max"), low=("low", "min"), close=("close", "last"),
            volume=("volume", "sum"), trade_count=("trade_count", "sum"),
            reason_codes=("filter_reason_codes", lambda values: "|".join(sorted({part for value in values for part in str(value).split("|") if part}))),
        )
        .reset_index(drop=True)
    )
    bars["x"] = np.arange(len(bars), dtype=int)
    return json_ready(bars.to_dict("records"))


def _event_coordinates(events: list[dict[str, Any]], bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for event in events:
        row = dict(event)
        overlap = [bar for bar in bars if pd.Timestamp(bar["end_time"]) >= pd.Timestamp(event["start"]) and pd.Timestamp(bar["datetime"]) <= pd.Timestamp(event["end"])]
        if overlap:
            row["x0"], row["x1"] = float(overlap[0]["x"]) - .48, float(overlap[-1]["x"]) + .48
            confirmation = min(bars, key=lambda bar: abs(pd.Timestamp(bar["datetime"]) - pd.Timestamp(event["confirmation_time"])))
            row["confirmation_x"] = float(confirmation["x"])
            if event["event_type"] == "circuit_breaker_candidate":
                halt = min(bars, key=lambda bar: abs(pd.Timestamp(bar["datetime"]) - pd.Timestamp(event["halt_end"])))
                auction = min(bars, key=lambda bar: abs(pd.Timestamp(bar["datetime"]) - pd.Timestamp(event["call_auction_start"])))
                row["halt_x"], row["auction_x"] = float(halt["x"]), float(auction["x"])
        output.append(json_ready(row))
    return output


def _details(atoms: pd.DataFrame, events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        start = pd.Timestamp(event["start"]) - pd.Timedelta(minutes=DETAIL_PADDING_MINUTES)
        end = pd.Timestamp(event["end"]) + pd.Timedelta(minutes=DETAIL_PADDING_MINUTES)
        rows = atoms.loc[atoms["datetime"].between(start, end), [
            "source_index", "datetime", "open", "high", "low", "close",
            "volume", "trade_count", "filter_reason_codes", "low_activity_state",
            "pending_buffer_start", "pending_buffer_count", "buffer_reinserted",
            "buffer_confirmed_excluded", "recovery_confirmation_time",
            "low_activity_confirmation_time", "baseline_excluded_from",
            "confirmed_low_activity_active",
        ]].copy()
        rows["x"] = np.arange(len(rows), dtype=int)
        output[str(event["event_id"])] = json_ready(rows.to_dict("records"))
    return output


def _write_script(path: Path, name: str, payload: dict[str, Any]) -> None:
    path.write_text(
        f"window.FILTER_{name.upper()}="
        + json.dumps(json_ready(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + ";\n",
        encoding="utf-8",
    )


def build_audit_report(result: LowActivityResult, output: Path, plotly_source: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    (output / "sections").mkdir(parents=True, exist_ok=True)
    (output / "assets").mkdir(parents=True, exist_ok=True)
    atoms = result.atoms.copy()
    atoms["source_index"] = np.arange(len(atoms), dtype=int)
    bars = _overview(atoms)
    universal = [event for event in result.events if event["family"] == "universal_low_volume"]
    mechanism = [event for event in result.events if event["family"] == "k200_market_mechanism"]
    payloads = {
        "universal": {
            "bars": bars,
            "events": _event_coordinates(universal, bars),
            "details": _details(atoms, universal),
        },
        "mechanism": {
            "bars": bars,
            "events": _event_coordinates(mechanism, bars),
            "details": _details(atoms, mechanism),
        },
    }
    (output / "index.html").write_text(
        HTML.replace("__SUMMARY__", json.dumps(json_ready(result.summary), ensure_ascii=False, separators=(",", ":"))),
        encoding="utf-8",
    )
    for name, payload in payloads.items():
        _write_script(output / "sections" / f"{name}.js", name, payload)
    if not plotly_source.is_file():
        raise FileNotFoundError(f"local Plotly asset is required: {plotly_source}")
    shutil.copy2(plotly_source, output / "assets" / "plotly.min.js")
    return {
        "index": str(output / "index.html"),
        "index_bytes": (output / "index.html").stat().st_size,
        "section_bytes": {
            name: (output / "sections" / f"{name}.js").stat().st_size
            for name in payloads
        },
        "startup_embeds_section_payload": False,
    }
