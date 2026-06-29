"""
Scenario 01 — 3.3V Power Rail Brownout
Generates oscilloscope-style waveform PNGs for the HTML scenario.

Run from this directory:
    python simulate_power_rail_01.py

Output: waveforms/scope_tp1_clean.png
        waveforms/scope_tp3_noisy.png
        waveforms/scope_tp3_sag_zoom.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

np.random.seed(42)

OUT = os.path.join(os.path.dirname(__file__), "waveforms")
os.makedirs(OUT, exist_ok=True)

SCOPE_BG  = "#0d0d0d"
GRID_CLR  = "#1c2a1c"
TRACE_CLR = "#00FF41"
WARN_CLR  = "#FF6B35"
NOM_CLR   = "#444444"


def scope_style(ax, title):
    ax.set_facecolor(SCOPE_BG)
    ax.figure.patch.set_facecolor(SCOPE_BG)
    for sp in ax.spines.values():
        sp.set_color("#2a2a2a")
    ax.tick_params(colors="#666666", labelsize=7)
    ax.grid(True, color=GRID_CLR, linewidth=0.6)
    ax.set_title(title, color="#999999", fontsize=9, pad=6, fontfamily="monospace")
    ax.set_xlabel("Time (ms)", color="#666666", fontsize=8, fontfamily="monospace")
    ax.set_ylabel("Voltage (V)", color="#666666", fontsize=8, fontfamily="monospace")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))


# ── Plot 1: TP1 — clean input rail ───────────────────────────────────────────
t1 = np.linspace(0, 2.0, 4000)
v1 = 3.30 + 0.006 * np.random.randn(len(t1))

fig, ax = plt.subplots(figsize=(6.5, 2.8))
ax.plot(t1, v1, color=TRACE_CLR, linewidth=0.7)
ax.axhline(3.30, color=NOM_CLR,  ls="--", lw=0.8, label="3.30V nominal",   alpha=0.8)
ax.axhline(2.97, color=WARN_CLR, ls="--", lw=0.9, label="2.97V brownout",  alpha=0.9)
ax.set_ylim(2.80, 3.60)
scope_style(ax, "CH1  TP1 — Power Input  |  50mV/div  |  0.2ms/div")
ax.legend(fontsize=7, facecolor="#111", edgecolor="#333",
          labelcolor="#999", loc="upper right", framealpha=0.85)
plt.tight_layout(pad=0.5)
plt.savefig(os.path.join(OUT, "scope_tp1_clean.png"), dpi=130, bbox_inches="tight")
plt.close()


# ── Plot 2: TP3 — noisy MCU VCC with brownout sags ──────────────────────────
t2    = np.linspace(0, 2.0, 8000)
ripple = 0.025 * np.sin(2 * np.pi * 3.0 * t2)
noise  = 0.010 * np.random.randn(len(t2))
sag    = np.zeros_like(t2)
for center in [0.38, 1.08, 1.72]:
    sag += 0.42 * np.exp(-((t2 - center) ** 2) / (0.0025 ** 2))
v2 = 3.30 - ripple - sag + noise

fig, ax = plt.subplots(figsize=(6.5, 2.8))
ax.plot(t2, v2, color=TRACE_CLR, linewidth=0.7)
ax.axhline(3.30, color=NOM_CLR,  ls="--", lw=0.8, label="3.30V nominal",  alpha=0.8)
ax.axhline(2.97, color=WARN_CLR, ls="--", lw=0.9, label="2.97V brownout", alpha=0.9)
ax.fill_between(t2, v2, 2.97, where=(v2 < 2.97),
                color="#FF4444", alpha=0.30, label="Brownout zone")
ax.set_ylim(2.70, 3.60)
scope_style(ax, "CH1  TP3 — MCU VCC Pin  |  100mV/div  |  0.2ms/div")
ax.legend(fontsize=7, facecolor="#111", edgecolor="#333",
          labelcolor="#999", loc="upper right", framealpha=0.85)
plt.tight_layout(pad=0.5)
plt.savefig(os.path.join(OUT, "scope_tp3_noisy.png"), dpi=130, bbox_inches="tight")
plt.close()


# ── Plot 3: TP3 zoomed — single sag event ────────────────────────────────────
t3    = np.linspace(0.28, 0.52, 3000)
sag3  = 0.42 * np.exp(-((t3 - 0.38) ** 2) / (0.0025 ** 2))
v3    = 3.30 - 0.025 * np.sin(2 * np.pi * 3.0 * t3) - sag3 + 0.008 * np.random.randn(len(t3))

fig, ax = plt.subplots(figsize=(6.5, 2.8))
ax.plot(t3, v3, color=TRACE_CLR, linewidth=1.0)
ax.axhline(3.30, color=NOM_CLR,  ls="--", lw=0.8, alpha=0.8)
ax.axhline(2.97, color=WARN_CLR, ls="--", lw=0.9, label="2.97V brownout", alpha=0.9)
ax.fill_between(t3, v3, 2.97, where=(v3 < 2.97),
                color="#FF4444", alpha=0.40, label="Brownout zone")

vmin = v3.min()
tmin = t3[v3.argmin()]
ax.annotate(
    f"V_min ≈ {vmin:.2f}V",
    xy=(tmin, vmin), xytext=(tmin + 0.025, vmin - 0.04),
    color="#FFCC00", fontsize=8, fontfamily="monospace",
    arrowprops=dict(arrowstyle="->", color="#FFCC00", lw=0.9),
)
# sag width annotation
mask = sag3 > 0.21
if mask.any():
    tl, tr = t3[mask][0], t3[mask][-1]
    ax.annotate("", xy=(tr, 3.10), xytext=(tl, 3.10),
                arrowprops=dict(arrowstyle="<->", color="#FFCC00", lw=0.8))
    ax.text((tl + tr) / 2, 3.13, f"≈{(tr - tl)*1000:.0f}µs",
            ha="center", color="#FFCC00", fontsize=7, fontfamily="monospace")

ax.set_ylim(2.70, 3.60)
scope_style(ax, "CH1  TP3 — Sag Event Zoomed  |  50mV/div  |  20µs/div")
ax.legend(fontsize=7, facecolor="#111", edgecolor="#333",
          labelcolor="#999", loc="upper right", framealpha=0.85)
plt.tight_layout(pad=0.5)
plt.savefig(os.path.join(OUT, "scope_tp3_sag_zoom.png"), dpi=130, bbox_inches="tight")
plt.close()


# ── Console analysis ──────────────────────────────────────────────────────────
I  = 0.150   # peak MCU current (A)
Rt = 0.800   # total trace resistance (Ω)
E  = 2.000   # C2 degraded ESR (Ω)
Vi = 3.300   # input voltage (V)
Vb = 2.970   # brownout threshold (V)

print("=" * 54)
print("  SCENARIO 01 — POWER RAIL ANALYSIS")
print("  Board: PCBA-REV2   Net: +3V3")
print("=" * 54)

print("\n[MULTIMETER READINGS]")
print(f"  TP1 (Input)   : {Vi:.3f} V  ← nominal, no fault here")
print(f"  TP2 (Mid-rail): {Vi - I*0.5:.3f} V  ← minor drop across first trace segment")
print(f"  TP3 (MCU VCC) : {Vi - I*Rt:.3f} V  ← DC avg (transients hidden from MM!)")
print(f"  TP4 (GND)     : 0.000 V  ← reference confirmed")

print("\n[CIRCUIT ANALYSIS — Ohm's Law: V = I × R]")
print(f"  Peak MCU current (I)      : {I*1000:.0f} mA")
print(f"  Trace resistance (R_trace): {Rt:.2f} Ω")
print(f"  C2 ESR (degraded)         : {E:.2f} Ω  (spec < 0.1 Ω for MLCC)")
print(f"  Voltage drop — trace      : I×R  = {I:.3f} × {Rt} = {I*Rt:.3f} V")
print(f"  Voltage drop — C2 ESR     : I×E  = {I:.3f} × {E} = {I*E:.3f} V")
print(f"  Total sag                 : {I*(Rt+E):.3f} V")
print(f"  V_MCU during sag          : {Vi - I*(Rt+E):.3f} V  ←  BELOW {Vb} V brownout")
print(f"  Headroom margin           : {(Vi - I*(Rt+E)) - Vb:.3f} V  (negative = BROWNOUT)")

print("\n[OSCILLOSCOPE FINDINGS]")
print("  TP1: Flat 3.30V, < 15mV noise — source is clean")
print("  TP3: Periodic sags to ~2.88V at ~3ms intervals")
print("       Sag duration ≈ 30µs — correlates with MCU active cycles")
print("       Crosses 2.97V brownout threshold → MCU resets")

print("\n[DIAGNOSIS]")
print("  ROOT CAUSE : C2 (100nF bypass) has degraded ESR (2.0Ω)")
print("               Cannot supply transient charge during MCU load steps")
print("  FIX        : Replace C2 with low-ESR MLCC (X5R/X7R, 100nF)")
print("               Add 10µF MLCC bulk cap near U1 VCC pin")
print("               Verify trace width — 0.8Ω suggests under-spec or damage")

print("\n[CAD NOTE — Altium Designer / OrCAD / Eagle]")
print("  Check C2 BOM: electrolytic may have been placed vs. MLCC footprint")
print("  Run DRC: verify net +3V3 trace width on copper layer")
print("  IPC-2221: ≥ 0.25mm trace width for 150mA (1oz Cu, 10°C rise)")
print("  In Altium: Tools → PCB Inspector → check net impedance on +3V3 net")

print(f"\nWaveforms → waveforms/")
print("  scope_tp1_clean.png")
print("  scope_tp3_noisy.png")
print("  scope_tp3_sag_zoom.png")
