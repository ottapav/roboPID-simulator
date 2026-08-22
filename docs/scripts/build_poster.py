"""
SPIN poster in the layout of the reference "PID TUNING" infographic, sized to
print on a single A4 page.

Row structure of the main table mirrors the reference:
    badge | step response | phase portraits | checklist

All curves are real, from the reference implementation:
  * step responses  (steps.json)   - the two runs behind the portraits
  * phase portraits (portraits.json), truncated at the epsilon-disc per
    Definition 1, with a ring gauge for limit vs. measured count.
"""
from __future__ import annotations
import json, math
from pathlib import Path

P = json.load(open("/home/claude/portraits.json"))
S = json.load(open("/home/claude/steps.json"))

RED, NAVY = "#e11b22", "#1b3a6b"
GREEN, PURPL, BLUE, GREY = "#2e9e4f", "#6b2e9e", "#1f5fa9", "#546474"
AMBER = "#c77a06"          # K_p band identity (green is reserved for "safe/raise")

BAND = [
    dict(k="Gamma0", sym="&Gamma;<sub>0</sub>", gain="K<sub>i</sub>", col=PURPL,
         axes="&int;e vs e", lim="0.50", q="2",
         sees="Slow cycling &mdash; the loop hunts around setpoint"),
    dict(k="Gamma1", sym="&Gamma;<sub>1</sub>", gain="K<sub>p</sub>", col=AMBER,
         axes="e vs &Delta;e", lim="0.75", q="3",
         sees="Ringing &mdash; decaying oscillation after the step"),
    dict(k="Gamma2", sym="&Gamma;<sub>2</sub>", gain="K<sub>d</sub>", col=BLUE,
         axes="&Delta;e vs &Delta;&sup2;e", lim="1.00", q="4",
         sees="Buzzing &mdash; fast chatter on the transient"),
]


def portrait(rec, col, size=80):
    """Real portrait curve (red = over limit, green = within) plus a spiral
    gauge, in the band colour, showing the angle the curve ACTUALLY sweeps:
    it begins at the curve's first angle and ends at its last, so it is not
    aligned to the axes and its length is the turn index itself."""
    s_, c = size / 3.0, size / 2
    over = rec["N"] > rec["Nbar"]
    curve_col = RED if over else GREEN

    def Pt(x, y):
        return f"{c + x*s_:.1f},{c - y*s_:.1f}"

    poly = " ".join(Pt(x, y) for x, y in rec["pts"])
    tail = " ".join(Pt(x, y) for x, y in (rec.get("tail") or []))

    # --- angle actually swept by the counted arc ---
    pts = rec["pts"]
    a = [math.atan2(y, x) for x, y in pts]
    unw, off = [a[0]], 0.0
    for k in range(1, len(a)):
        d = a[k] - a[k-1]
        if d > math.pi:
            off -= 2 * math.pi
        elif d < -math.pi:
            off += 2 * math.pi
        unw.append(a[k] + off)
    a0, sweep = unw[0], unw[-1] - unw[0]

    r0, r1 = 1.16 * s_, 1.36 * s_
    n = max(28, int(abs(sweep) / 0.045))
    gp = []
    for i in range(n + 1):
        t = i / n
        ang = a0 + sweep * t
        r = r0 + (r1 - r0) * t
        gp.append(f"{c + r*math.cos(ang):.1f},{c - r*math.sin(ang):.1f}")
    gauge = (f'<polyline points="{" ".join(gp)}" fill="none" stroke="{col}" '
             f'stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round"/>'
             f'<circle cx="{c + r0*math.cos(a0):.1f}" cy="{c - r0*math.sin(a0):.1f}" '
             f'r="2.6" fill="{col}"/>')

    return f'''<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}">
  <line x1="{c-1.0*s_:.0f}" y1="{c}" x2="{c+1.0*s_:.0f}" y2="{c}" stroke="#dde3e9"/>
  <line x1="{c}" y1="{c-1.0*s_:.0f}" x2="{c}" y2="{c+1.0*s_:.0f}" stroke="#dde3e9"/>
  <circle cx="{c}" cy="{c}" r="{0.1*s_:.1f}" fill="none" stroke="#96a4b2" stroke-dasharray="2 2"/>
  {f'<polyline points="{tail}" fill="none" stroke="#c4ccd5" stroke-width="1.4" stroke-dasharray="3 2"/>' if tail else ''}
  <polyline points="{poly}" fill="none" stroke="{curve_col}" stroke-width="2.6"
            stroke-linejoin="round" stroke-linecap="round"/>
  <circle cx="{c}" cy="{c}" r="2.4" fill="{NAVY}"/>
  {gauge}
</svg>'''


def step_plot(w=196, h=88):
    """Both step responses in one plot: over limit vs within limit."""
    tmax = max(S["ringing"]["t"][-1], S["quiet"]["t"][-1])
    ymin, ymax = -0.08, 1.70
    ml, mr, mt, mb = 22, 6, 8, 16
    pw, ph = w - ml - mr, h - mt - mb

    def X(t):
        return ml + pw * t / tmax

    def Y(v):
        return mt + ph * (ymax - v) / (ymax - ymin)

    def line(rec, colour, width):
        pts = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in zip(rec["t"], rec["y"]))
        return (f'<polyline points="{pts}" fill="none" stroke="{colour}" '
                f'stroke-width="{width}" stroke-linejoin="round"/>')

    return f'''<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">
  <line x1="{ml}" y1="{Y(1.0):.1f}" x2="{w-mr}" y2="{Y(1.0):.1f}"
        stroke="#b9c4cf" stroke-dasharray="4 3"/>
  <line x1="{ml}" y1="{mt}" x2="{ml}" y2="{h-mb}" stroke="#8895a3"/>
  <line x1="{ml}" y1="{Y(0):.1f}" x2="{w-mr}" y2="{Y(0):.1f}" stroke="#8895a3"/>
  {line(S["quiet"], GREEN, 2.4)}
  {line(S["ringing"], RED, 2.4)}
  <text x="{ml-4}" y="{Y(1.0)+3:.1f}" font-size="8" fill="{GREY}" text-anchor="end"
        font-family="Arial">sp</text>
  <text x="{w-mr}" y="{h-4}" font-size="8" fill="{GREY}" text-anchor="end"
        font-family="Arial">time</text>
  <line x1="{w-66}" y1="{mt+5}" x2="{w-54}" y2="{mt+5}" stroke="{RED}" stroke-width="2.4"/>
  <text x="{w-51}" y="{mt+8}" font-size="8.5" fill="{RED}" font-family="Arial"
        font-weight="bold">over limit</text>
  <line x1="{w-66}" y1="{mt+16}" x2="{w-54}" y2="{mt+16}" stroke="{GREEN}" stroke-width="2.4"/>
  <text x="{w-51}" y="{mt+19}" font-size="8.5" fill="{GREEN}" font-family="Arial"
        font-weight="bold">within limit</text>
</svg>'''


def rows():
    out = []
    for i, b in enumerate(BAND):
        r = next(d for d in P["ringing"] if d["name"] == b["k"])
        q = next(d for d in P["quiet"] if d["name"] == b["k"])
        first = i == 0
        stepcell = (f'<td class="stepcell" rowspan="3">{step_plot()}'
                    f'<div class="gains"><span class="gr">'
                    f'K = ({S["ringing"]["gains"][0]}, {S["ringing"]["gains"][1]}, '
                    f'{S["ringing"]["gains"][2]})</span><br><span class="gq">'
                    f'K = ({S["quiet"]["gains"][0]}, {S["quiet"]["gains"][1]}, '
                    f'{S["quiet"]["gains"][2]})</span></div>'
                    f'<div class="stepnote">Both portraits come from these two runs.</div></td>') if first else ""
        out.append(f'''
      <tr>
        <td class="term" style="--c:{b['col']}">
          <span class="badge">{b['sym']}</span>
          <span class="tname">{b['gain']} band</span>
          <span class="taxes">{b['axes']}</span>
        </td>
        {stepcell}
        <td class="plotcell"><div class="pair">
          <figure>{portrait(r, b['col'])}<figcaption class="pn bad">
            N<sub>{i}</sub> = {r['N']}</figcaption></figure>
          <figure>{portrait(q, b['col'])}<figcaption class="pn ok">
            N<sub>{i}</sub> = {q['N']}</figcaption></figure>
        </div></td>
        <td class="effect"><ul>
          <li><i>&#10004;</i><span>{b['sees']}</span></li>
          <li><i>&#10004;</i><span>Limit <b>N&#772;<sub>{i}</sub> = {b['lim']}</b>
              <span class="q">= {b['q']} quarter turns</span></span></li>
          <li><i>&#10004;</i><span>Over limit: cut <b>{b['gain']}</b>
              {inline_arrow('dn')}, raise bands below {inline_arrow('up')}</span></li>
        </ul></td>
      </tr>''')
    return "\n".join(out)


RULE = [("record grows", "DDD", "runaway &mdash; halve all", 1),
        ("&Gamma;<sub>0</sub> loops", "dnn", "slow cycling &mdash; less reset", 0),
        ("&Gamma;<sub>1</sub> loops", "Udn", "ringing &mdash; less gain", 0),
        ("&Gamma;<sub>2</sub> loops", "UUd", "buzzing &mdash; less rate", 0),
        ("none loops", "UUU", "all quiet &mdash; tighten", 0)]


def inline_arrow(direction, size=15):
    """Small solid arrow for inline use in prose (Reading column), heavier
    than a text glyph so it reads clearly at print size."""
    col = GREEN if direction == "up" else RED
    if direction == "up":
        d = "M7 0 L14 8 L10 8 L10 15 L4 15 L4 8 L0 8 Z"
    else:
        d = "M7 15 L0 7 L4 7 L4 0 L10 0 L10 7 L14 7 Z"
    return (f'<svg viewBox="0 0 14 15" width="{size}" height="{size*15/14:.1f}" '
            f'class="inarrow"><path d="{d}" fill="{col}"/></svg>')


def glyph(t, w=22, h=25):
    """Heavy arrow glyphs for the rule table: up / down / halve / hold."""
    if t == "U":
        return (f'<svg viewBox="0 0 26 30" width="{w}" height="{h}" class="gl">'
                f'<path d="M13 2 L25 15 L18.5 15 L18.5 28 L7.5 28 L7.5 15 L1 15 Z" '
                f'fill="{GREEN}"/></svg>')
    if t == "d":
        return (f'<svg viewBox="0 0 26 30" width="{w}" height="{h}" class="gl">'
                f'<path d="M13 28 L1 15 L7.5 15 L7.5 2 L18.5 2 L18.5 15 L25 15 Z" '
                f'fill="{RED}"/></svg>')
    if t == "D":
        return (f'<svg viewBox="0 0 26 30" width="{w}" height="{h}" class="gl">'
                f'<path d="M2.5 4 L13 14 L23.5 4" fill="none" stroke="{RED}" '
                f'stroke-width="5.4" stroke-linecap="round" stroke-linejoin="round"/>'
                f'<path d="M2.5 16 L13 26 L23.5 16" fill="none" stroke="{RED}" '
                f'stroke-width="5.4" stroke-linecap="round" stroke-linejoin="round"/></svg>')
    return (f'<svg viewBox="0 0 26 30" width="{w}" height="{h}" class="gl">'
            f'<circle cx="13" cy="15" r="4.2" fill="#b7c2cd"/></svg>')


def rule_rows():
    out = []
    for cond, toks, why, alarm in RULE:
        cells = "".join(f'<td class="a">{glyph(t)}</td>' for t in toks)
        out.append(f'<tr{" class=alarm" if alarm else ""}><td class="cn">{cond}</td>'
                   f'{cells}<td class="why">{why}</td></tr>')
    return "\n".join(out)


HTML = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SPIN TUNING</title>
<style>
@page{{size:A4 portrait;margin:7mm}}
:root{{--red:{RED};--navy:{NAVY};--green:{GREEN};--purple:{PURPL};--blue:{BLUE};--grey:{GREY};--amber:{AMBER}}}
*{{box-sizing:border-box}}
body{{margin:0;background:#e9edf1;font-family:Arial,Helvetica,"Liberation Sans",sans-serif;
  color:#16202b;font-size:13px;line-height:1.4}}
.sheet{{max-width:742px;margin:0 auto;background:#fff;padding:9px 13px 10px}}
h1{{font-size:clamp(36px,7.4vw,42px);color:var(--red);text-align:center;margin:0;
  font-weight:800;line-height:1}}
.sub{{text-align:center;font-size:12.5px;color:var(--navy);font-weight:700;
  letter-spacing:.1em;margin:3px 0 1px}}
.sub em{{font-style:normal;color:var(--red)}}
.tag{{text-align:center;font-size:12.5px;color:var(--grey);margin:0 0 3px}}
h2{{color:var(--navy);font-size:15px;font-weight:800;margin:6px 0 3px}}
.diagram{{width:100%;height:auto;display:block}}
.key{{font-size:11px;color:var(--grey);margin:0 0 4px}}
.key .sw{{display:inline-block;width:9px;height:9px;border-radius:2px;margin:0 3px 0 7px;vertical-align:middle}}
.key .cr{{color:var(--red)}} .key .cg{{color:var(--green)}}
.eqwrap{{display:flex;gap:10px;align-items:center;margin:6px 0 0}}
.eq{{flex:1 1 auto;border:2px solid var(--navy);border-radius:5px;padding:6px 10px;
  text-align:center;font-size:15px}}
.eq b{{font-weight:700}}
.eqnote{{font-size:10.5px;color:var(--grey);margin-top:3px}}
.legend{{flex:0 0 172px;font-size:11.5px;line-height:1.6}}
.ki{{color:var(--purple)}} .kp{{color:var(--amber)}} .kd{{color:var(--blue)}}
table.grid{{width:100%;border-collapse:collapse;border:1px solid #c3cdd8}}
table.grid th{{background:var(--navy);color:#fff;font-size:11.5px;padding:5px 5px;
  text-transform:uppercase;letter-spacing:.04em}}
table.grid td{{border:1px solid #c3cdd8;padding:4px 5px;vertical-align:middle}}
.term{{width:82px;text-align:center}}
.badge{{display:inline-block;width:36px;height:36px;line-height:36px;border-radius:50%;
  background:var(--c);color:#fff;font-size:16px;font-weight:700}}
.tname{{display:block;font-size:11.5px;font-weight:700;color:var(--c);margin-top:3px}}
.taxes{{display:block;font-size:10.5px;color:var(--grey);font-family:ui-monospace,Consolas,monospace}}
.stepcell{{width:206px;text-align:center;background:#fafbfc}}
.gains{{font-family:ui-monospace,Consolas,monospace;font-size:10.5px;margin-top:2px}}
.gr{{color:var(--red)}} .gq{{color:var(--green)}}
.stepnote{{font-size:10.5px;color:var(--grey);margin-top:6px;line-height:1.35}}
.plotcell{{width:228px}}
.pair{{display:flex;gap:6px;justify-content:center}}
.pair figure{{margin:0;text-align:center}}
.pn{{font-family:ui-monospace,Consolas,monospace;font-size:11px;font-weight:700;margin-top:1px}}
.pn.bad{{color:var(--red)}} .pn.ok{{color:var(--green)}}
.effect ul{{margin:0;padding:0;list-style:none}}
.effect li{{font-size:12px;line-height:1.35;margin:3px 0;display:flex;gap:5px;align-items:baseline}}
.effect i{{color:var(--navy);font-size:10px;font-style:normal;flex:none}}
.q{{color:var(--grey);font-size:11px}}
b.dn{{color:var(--red)}} b.up{{color:var(--green)}}
.inarrow{{vertical-align:middle;margin:0 1px;position:relative;top:-1px}}
.guide{{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin-top:5px}}
.gstep{{border:2px solid #9db6d4;border-radius:5px;padding:4px 6px;font-size:11px;
  line-height:1.3;text-align:center}}
.gnum{{display:block;width:20px;height:20px;line-height:20px;border-radius:50%;
  background:var(--navy);color:#fff;font-weight:700;font-size:11.5px;margin:0 auto 3px}}
.gstep b:first-of-type{{display:block;color:var(--navy);font-size:12.5px;margin-bottom:2px}}
.bottom{{display:grid;grid-template-columns:1fr 218px;gap:12px;align-items:start;margin-top:6px}}
table.rule{{width:100%;border-collapse:collapse;border:1px solid #c3cdd8}}
table.rule th{{color:#fff;font-size:11px;padding:5px 4px;text-transform:uppercase}}
th.hc{{background:var(--navy);text-align:left;padding-left:7px}}
th.hi{{background:var(--purple)}} th.hp{{background:var(--amber)}} th.hd{{background:var(--blue)}}
th.hw{{background:var(--navy)}}
table.rule td{{border:1px solid #c3cdd8;padding:3px 4px;font-size:11.5px}}
.cn{{font-family:ui-monospace,Consolas,monospace;font-size:11.5px;padding-left:7px !important}}
.a{{text-align:center;width:34px;padding:1px 2px !important}}
.gl{{display:block;margin:0 auto}}
.why{{color:var(--grey);font-size:11.5px}}
tr.alarm td{{background:#fdf0f0}}
.tips{{border:2px solid var(--red);border-radius:5px;padding:6px 9px}}
.tips h4{{margin:0 0 5px;color:var(--red);font-size:12.5px}}
.tips ul{{margin:0;padding:0}}
.tips li{{list-style:none;font-size:11px;line-height:1.25;margin:0 0 2px;
  padding-left:13px;position:relative}}
.tips li::before{{content:"\\2714";position:absolute;left:0;color:var(--navy);font-size:9px}}
.foot{{margin-top:6px;border-top:2px solid var(--navy);padding-top:4px;font-size:10px;
  color:var(--grey);display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}}
.foot a{{color:var(--blue)}}
.cite{{margin-top:3px;font-size:10.5px;color:#333f4a}}
.cite i{{font-style:italic}}
.cite a{{color:var(--blue);font-style:normal}}
@media(max-width:700px){{
  table.grid,table.grid tbody,table.grid tr,table.grid td{{display:block;width:auto}}
  table.grid thead{{display:none}}
  table.grid tr{{border-bottom:2px solid #c3cdd8}}
  .guide,.bottom{{grid-template-columns:1fr}}
  .eqwrap{{flex-wrap:wrap}}
}}
@media print{{body{{background:#fff}}.sheet{{max-width:none;padding:0}}}}
</style></head><body><div class="sheet">

  <h1>SPIN TUNING</h1>
  <div class="sub"><em>S</em>TEP-RESPONSE <em>P</em>HASE-PORTRAIT <em>IN</em>SPECTION</div>
  <p class="tag">Tune a PID loop from one step response. No model, no relay test, no optimizer.</p>

  <svg class="diagram" viewBox="0 -8 640 158">
    <defs>
      <marker id="k" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
        <path d="M0 1 L7 4 L0 7 z" fill="{NAVY}"/></marker>
    </defs>
    <g font-family="Arial" font-size="10.5" fill="{NAVY}">
      <text x="2" y="26" font-weight="700">SETPOINT</text>
      <path d="M4 54 h13 v-14 h18" fill="none" stroke="{BLUE}" stroke-width="2"/>
      <text x="4" y="68">r(t)</text>
      <line x1="38" y1="40" x2="77" y2="40" stroke="{NAVY}" stroke-width="1.6" marker-end="url(#k)"/>
      <circle cx="90" cy="40" r="11" fill="none" stroke="{NAVY}" stroke-width="1.6"/>
      <path d="M83 33 L97 47 M83 47 L97 33" stroke="{NAVY}" stroke-width="1.2"/>
      <text x="77" y="24" font-size="11">+</text><text x="73" y="66" font-size="11">&minus;</text>
      <line x1="102" y1="40" x2="160" y2="40" stroke="{NAVY}" stroke-width="1.6" marker-end="url(#k)"/>
      <text x="108" y="26" font-weight="700" fill="{NAVY}">ERROR</text>
      <text x="114" y="58">e(t)</text>
      <rect x="162" y="20" width="118" height="40" rx="4" fill="#fff" stroke="{NAVY}" stroke-width="2"/>
      <text x="197" y="37" font-size="13" font-weight="700" fill="{NAVY}">PID</text>
      <text x="178" y="53" font-size="11" font-weight="700" fill="{PURPL}">I</text>
      <text x="192" y="53" font-size="11">+</text>
      <text x="206" y="53" font-size="11" font-weight="700" fill="{AMBER}">P</text>
      <text x="220" y="53" font-size="11">+</text>
      <text x="234" y="53" font-size="11" font-weight="700" fill="{BLUE}">D</text>
      <line x1="280" y1="40" x2="332" y2="40" stroke="{NAVY}" stroke-width="1.6" marker-end="url(#k)"/>
      <text x="288" y="32">u(t)</text>
      <rect x="334" y="20" width="112" height="40" rx="4" fill="#fff" stroke="{NAVY}" stroke-width="2"/>
      <text x="355" y="45" font-size="12" font-weight="700">PROCESS</text>
      <line x1="446" y1="40" x2="512" y2="40" stroke="{NAVY}" stroke-width="1.6" marker-end="url(#k)"/>
      <text x="518" y="26" font-weight="700">OUTPUT</text>
      <path d="M518 56 q10 -20 20 -13 t22 -2" fill="none" stroke="{BLUE}" stroke-width="2"/>
      <text x="518" y="70">y(t)</text>
      <path d="M478 40 L478 84 L90 84 L90 53" fill="none" stroke="{NAVY}" stroke-width="1.6" marker-end="url(#k)"/>
      <path d="M132 40 L132 98" fill="none" stroke="{NAVY}" stroke-width="1.6"
            stroke-dasharray="4 3" marker-end="url(#k)"/>
      <rect x="60" y="100" width="140" height="34" rx="4" fill="#eef3f9" stroke="{NAVY}" stroke-width="1.6"/>
      <text x="74" y="115" font-size="10.5" fill="{NAVY}" font-weight="700">RECORD ERROR</text>
      <text x="74" y="128" font-size="10.5" fill="{NAVY}">cut at settling</text>
      <line x1="202" y1="117" x2="234" y2="117" stroke="{NAVY}" stroke-width="1.6"
            stroke-dasharray="4 3" marker-end="url(#k)"/>
      <rect x="236" y="100" width="148" height="34" rx="4" fill="#eef3f9" stroke="{NAVY}" stroke-width="1.6"/>
      <text x="250" y="115" font-size="10.5" fill="{NAVY}" font-weight="700">THREE PORTRAITS</text>
      <text x="250" y="128" font-size="10.5" fill="{NAVY}">count the turns</text>
      <line x1="386" y1="117" x2="418" y2="117" stroke="{NAVY}" stroke-width="1.6"
            stroke-dasharray="4 3" marker-end="url(#k)"/>
      <rect x="420" y="100" width="134" height="34" rx="4" fill="#eef3f9" stroke="{NAVY}" stroke-width="1.6"/>
      <text x="434" y="115" font-size="10.5" fill="{NAVY}" font-weight="700">APPLY THE RULE</text>
      <text x="434" y="128" font-size="10.5" fill="{NAVY}">move one gain</text>
      <path d="M556 117 L602 117 L602 4 L221 4 L221 18" fill="none" stroke="{NAVY}"
            stroke-width="1.6" stroke-dasharray="4 3" marker-end="url(#k)"/>
      <text x="420" y="-1" font-size="10" fill="{NAVY}">new gains</text>
    </g>
  </svg>

  <div class="eqwrap">
    <div class="eq">
      u(t) = <b class="ki">K<sub>i</sub></b>&int;e(&tau;)d&tau; +
      <b class="kp">K<sub>p</sub></b>e(t) +
      <b class="kd">K<sub>d</sub></b>de(t)/dt
      <div class="eqnote">each gain multiplies one signal &mdash; and each portrait
        plots that same signal</div>
    </div>
    <div class="legend">
      <b class="ki">K<sub>i</sub></b> Integral &rarr; <b class="ki">&Gamma;<sub>0</sub></b><br>
      <b class="kp">K<sub>p</sub></b> Proportional &rarr; <b class="kp">&Gamma;<sub>1</sub></b><br>
      <b class="kd">K<sub>d</sub></b> Derivative &rarr; <b class="kd">&Gamma;<sub>2</sub></b>
    </div>
  </div>

  <h2>WHAT EACH PORTRAIT SHOWS</h2>
  <p class="key">
    <span class="sw" style="background:var(--purple)"></span>K<sub>i</sub>
    <span class="sw" style="background:var(--amber)"></span>K<sub>p</sub>
    <span class="sw" style="background:var(--blue)"></span>K<sub>d</sub>
    &mdash; the band, used for the outer angle arc. &nbsp;
    <span class="sw" style="background:var(--red)"></span><b class="cr">red</b> curve = over
    the limit, cut that gain. &nbsp;
    <span class="sw" style="background:var(--green)"></span><b class="cg">green</b> curve =
    within limit, safe to raise.</p>
  <table class="grid">
    <thead><tr><th>Portrait</th><th>Step response</th>
      <th>Pachner plots &mdash; over / within</th><th>Reading</th></tr></thead>
    <tbody>
{rows()}
    </tbody>
  </table>

  <h2>TUNING GUIDE (SPIN ITERATION)</h2>
  <div class="guide">
    <div class="gstep"><span class="gnum">1</span><b>Step it</b>
      Close the loop with any gains, step the setpoint, record <b>e</b></div>
    <div class="gstep"><span class="gnum">2</span><b>Trim it</b>
      Screen for runaway, cut where |e| last leaves <b>2%</b> of peak</div>
    <div class="gstep"><span class="gnum">3</span><b>Count it</b>
      Draw the three portraits, count turns
      <b>N<sub>0</sub> N<sub>1</sub> N<sub>2</sub></b></div>
    <div class="gstep"><span class="gnum">4</span><b>Move it</b>
      Use the table below to move
      <b>K<sub>i</sub> K<sub>p</sub> K<sub>d</sub></b>, then repeat</div>
  </div>

  <div class="bottom">
    <table class="rule">
      <thead><tr><th class="hc">Rule &mdash; first matching row wins</th>
        <th class="hi">K<sub>i</sub></th><th class="hp">K<sub>p</sub></th>
        <th class="hd">K<sub>d</sub></th><th class="hw">Reading</th></tr></thead>
      <tbody>
{rule_rows()}
      </tbody>
    </table>
    <div class="tips">
      <h4>TIPS</h4>
      <ul>
        <li>Cut only the <b>lowest</b> objecting band &mdash; higher ones may be leakage</li>
        <li>Bands below the fault were just measured safe, so raise them</li>
        <li>Counts carry no units: the same limits fit every loop</li>
        <li>Any start works: the screen halves gains until the record settles</li>
        <li>No stopping test: stop at any iteration</li>
      </ul>
    </div>
  </div>

  <div class="foot">
    <span>&uarr; &times;&gamma; &nbsp;&darr; &divide;&gamma; &nbsp;&dArr; halve
      &nbsp;&middot; hold &nbsp;|&nbsp; &gamma; = 1/(1&minus;&beta;), &beta; = 0.1;
      N&#772; = (0.5, 0.75, 1.0), &epsilon; = 0.1, &delta; = 0.02 &nbsp;|&nbsp;
      solid = counted arc, grey dash = discarded tail, outer arc = the angle actually swept</span>
  </div>
  <div class="cite">
    Pachner, Otta, Dost&aacute;l, Havlena &mdash; &ldquo;Model-Free PID Tuning by
    Step-Response Inspection,&rdquo; submitted to <i>Journal of Process Control</i>, 2026.
    &nbsp;&middot;&nbsp; CTU Prague
    &nbsp;&middot;&nbsp; <a href="https://robopid.uceeb.cvut.cz">robopid.uceeb.cvut.cz</a>
  </div>
</div></body></html>
'''

Path("/mnt/user-data/outputs/spin_poster.html").write_text(HTML, encoding="utf-8")
print("written", len(HTML))
