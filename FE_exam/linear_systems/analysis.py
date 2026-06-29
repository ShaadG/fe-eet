"""
FE Exam — Topic 7: Linear Systems
Generates 4 educational plots for practice.html

Run: python3 analysis.py
Output: plots/ folder with 4 PNG files
"""

import numpy as np
import matplotlib.pyplot as plt
import os

OUT = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.edgecolor': '#aaaaaa',
    'grid.color': '#dddddd',
    'text.color': '#222222',
})


# ── Q1: RC Transient Response ────────────────────────────────────────────────
R = 2000       # 2 kΩ
C = 0.5e-6     # 0.5 µF
tau = R * C    # τ = RC = 1 ms
V = 10.0

t = np.linspace(0, 5e-3, 2000)
vc = V * (1 - np.exp(-t / tau))
v_at_2ms = V * (1 - np.exp(-2))   # t = 2τ → 8.647 V

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t * 1000, vc, color='#1565C0', linewidth=2.2,
        label=r'$v_C(t) = 10\left(1 - e^{-t/\tau}\right)$')
ax.axvline(tau * 1000, color='#C62828', linestyle='--', linewidth=1.2, alpha=0.8,
           label=f'τ = {tau*1000:.1f} ms  (63.2% = 6.32 V)')
ax.axhline(v_at_2ms, color='#2E7D32', linestyle='--', linewidth=1.2, alpha=0.8,
           label=f't = 2 ms:  v_C = {v_at_2ms:.2f} V  ← answer')
ax.plot(2, v_at_2ms, 'o', color='#2E7D32', markersize=9, zorder=5)
ax.annotate(f'{v_at_2ms:.2f} V', xy=(2, v_at_2ms), xytext=(2.3, v_at_2ms - 0.9),
            fontsize=9, color='#2E7D32',
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.0))
ax.set_xlabel('Time  t  (ms)', fontsize=11)
ax.set_ylabel('Capacitor Voltage  $v_C$  (V)', fontsize=11)
ax.set_title('Q1 — RC Transient Response\n'
             'R = 2 kΩ,  C = 0.5 µF,  τ = RC = 1 ms,  V = 10 V step', fontsize=11)
ax.legend(fontsize=9, loc='lower right')
ax.set_ylim(0, 11)
ax.grid(True, alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'q1_rc_transient.png'), dpi=140, bbox_inches='tight')
plt.close()


# ── Q2: Series RLC Resonance — Impedance vs Frequency ────────────────────────
R2 = 5.0       # Ω
L2 = 50e-3     # H
C2 = 20e-6     # F
w0 = 1 / np.sqrt(L2 * C2)   # 1000 rad/s
Q_val = w0 * L2 / R2          # 10
BW = w0 / Q_val               # 100 rad/s

omega = np.linspace(100, 10000, 6000)
Z_mag = np.abs(R2 + 1j * omega * L2 + 1 / (1j * omega * C2))

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(omega, Z_mag, color='#1565C0', linewidth=2.0, label='|Z(jω)|')
ax.axvline(w0, color='#C62828', linestyle='--', linewidth=1.3, alpha=0.9,
           label=f'ω₀ = 1/√(LC) = {w0:.0f} rad/s  →  Z_min = R = {R2} Ω')
ax.axvline(w0 - BW / 2, color='#E65100', linestyle=':', linewidth=1.1, alpha=0.8)
ax.axvline(w0 + BW / 2, color='#E65100', linestyle=':', linewidth=1.1, alpha=0.8,
           label=f'Half-power pts  (BW = ω₀/Q = {BW:.0f} rad/s)')
ax.annotate(f'Q = ω₀L/R = {Q_val:.0f}', xy=(w0 * 1.05, R2 * 1.3),
            fontsize=10, color='#C62828', fontweight='bold')
ax.set_xlabel('Angular Frequency  ω  (rad/s)', fontsize=11)
ax.set_ylabel('Impedance  |Z|  (Ω)', fontsize=11)
ax.set_title('Q2 — Series RLC Resonance\n'
             'R = 5 Ω,  L = 50 mH,  C = 20 µF', fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(100, 10000)
ax.grid(True, alpha=0.5, which='both')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'q2_rlc_resonance.png'), dpi=140, bbox_inches='tight')
plt.close()


# ── Q3: Laplace Transform Pair — f(t)=3sin(2t) ↔ F(s)=6/(s²+4) ──────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: time domain
t3 = np.linspace(0, 6, 1000)
ft = 3 * np.sin(2 * t3)
axes[0].plot(t3, ft, color='#1565C0', linewidth=2.0,
             label='f(t) = 3 sin(2t) u(t)')
axes[0].axhline(0, color='#555555', linewidth=0.7)
axes[0].fill_between(t3, ft, 0, alpha=0.08, color='#1565C0')
axes[0].set_xlabel('Time  t  (s)', fontsize=11)
axes[0].set_ylabel('Amplitude', fontsize=11)
axes[0].set_title('Time Domain\nf(t) = 3 sin(2t) u(t)', fontsize=11)
axes[0].legend(fontsize=9)
axes[0].set_ylim(-3.8, 3.8)
axes[0].grid(True, alpha=0.5)

# Right: pole-zero map of F(s) = 6/(s²+4) = 6/((s+j2)(s-j2))
ax2 = axes[1]
ax2.axhline(0, color='#555555', linewidth=0.8)
ax2.axvline(0, color='#555555', linewidth=0.8)
# poles at s = ±j2
ax2.plot(0, 2, 'rx', markersize=13, markeredgewidth=2.5, label='Poles  s = ±j2')
ax2.plot(0, -2, 'rx', markersize=13, markeredgewidth=2.5)
ax2.annotate('  s = +j2', xy=(0, 2), fontsize=9, color='#C62828')
ax2.annotate('  s = −j2', xy=(0, -2), fontsize=9, color='#C62828')
ax2.set_xlim(-4, 4)
ax2.set_ylim(-4, 4)
ax2.set_xlabel('Real  σ', fontsize=11)
ax2.set_ylabel('Imaginary  jω', fontsize=11)
ax2.set_title('s-Domain Pole Map\nF(s) = 6 / (s² + 4)', fontsize=11)
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.5)
ax2.set_aspect('equal')

plt.suptitle('Q3 — Laplace Transform:  f(t) = 3 sin(2t)  ↔  F(s) = 6/(s²+4)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'q3_laplace.png'), dpi=140, bbox_inches='tight')
plt.close()


# ── Q4: Transfer Function Pole-Zero Map — H(s)=50/(s²+7s+50) ────────────────
# Denominator: s² + 7s + 50 = 0  →  s = (-7 ± sqrt(49-200))/2 = -3.5 ± j*sqrt(37.75)
sigma_p = -7 / 2
omega_p = np.sqrt(50 - (7 / 2) ** 2)   # sqrt(37.75) ≈ 6.14
wn = np.sqrt(50)                         # natural frequency ≈ 7.07
zeta = 7 / (2 * wn)                      # damping ratio ≈ 0.495

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.axhline(0, color='#555555', linewidth=0.8)
ax.axvline(0, color='#555555', linewidth=0.8, label='jω axis (stability boundary)')

# ωn circle
theta = np.linspace(0, 2 * np.pi, 400)
ax.plot(wn * np.cos(theta), wn * np.sin(theta), color='#1565C0',
        linestyle='--', linewidth=1.0, alpha=0.4,
        label=f'|s| = ωₙ = √50 ≈ {wn:.2f} rad/s')

# Poles
ax.plot(sigma_p, omega_p, 'rx', markersize=15, markeredgewidth=2.8, zorder=5,
        label=f'Poles: {sigma_p:.1f} ± j{omega_p:.2f}')
ax.plot(sigma_p, -omega_p, 'rx', markersize=15, markeredgewidth=2.8, zorder=5)
ax.annotate(f'  −3.5 + j{omega_p:.2f}', xy=(sigma_p, omega_p),
            xytext=(sigma_p + 0.3, omega_p + 0.5), fontsize=9, color='#C62828')
ax.annotate(f'  −3.5 − j{omega_p:.2f}', xy=(sigma_p, -omega_p),
            xytext=(sigma_p + 0.3, -omega_p - 0.8), fontsize=9, color='#C62828')

# Line from origin to pole (shows angle and ωn)
ax.plot([0, sigma_p], [0, omega_p], color='gray', linewidth=0.8, linestyle=':')

ax.text(-6.5, 0.4, f'ζ = {zeta:.3f}  (underdamped, 0 < ζ < 1)', fontsize=8.5,
        color='#555555', style='italic')

ax.set_xlim(-9, 3)
ax.set_ylim(-9, 9)
ax.set_xlabel('Real Axis  σ', fontsize=11)
ax.set_ylabel('Imaginary Axis  jω', fontsize=11)
ax.set_title('Q4 — Transfer Function Pole-Zero Map\n'
             'H(s) = 50 / (s² + 7s + 50)', fontsize=11)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'q4_transfer_fn.png'), dpi=140, bbox_inches='tight')
plt.close()


print("=" * 50)
print("  Linear Systems — Plot Generation Complete")
print("=" * 50)
print()
print("Saved to plots/")
print("  q1_rc_transient.png")
print("  q2_rlc_resonance.png")
print("  q3_laplace.png")
print("  q4_transfer_fn.png")
print()

# Verification calcs
print("[Q1] τ = RC =", R, "×", C, "=", tau*1000, "ms")
print(f"     v_C(2ms) = 10(1 - e^-2) = {v_at_2ms:.4f} V")
print()
print(f"[Q2] ω₀ = 1/√(LC) = {w0:.2f} rad/s")
print(f"     Q  = ω₀L/R = {Q_val:.2f}")
print(f"     BW = ω₀/Q  = {BW:.2f} rad/s")
print()
print("[Q3] L{3sin(2t)} = 3 × ω/(s²+ω²) = 3 × 2/(s²+4) = 6/(s²+4)")
print()
print(f"[Q4] poles: s = -3.5 ± j{omega_p:.4f}")
print(f"     ζ = {zeta:.4f}  (underdamped)")
