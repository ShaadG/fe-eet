"""
instrument_automation.py
─────────────────────────────────────────────────────────────────────────────
EET Daily Workflow — Power-On Bring-Up Automation
Simulates the 10:00 AM power-on sequence from daily_workflow_simulation.html

Demonstrates:
  • pyvisa / SCPI instrument control (bench supply, DMM, oscilloscope)
  • Automated measurement against pass/fail spec limits
  • MES-style test log output
  • matplotlib test report (rail voltages vs. spec bands)

REAL INSTRUMENT MODE:
  pip3 install pyvisa pyvisa-py --break-system-packages
  Connect instruments via USB, then set SIMULATION_MODE = False
  Update RESOURCE_IDs with addresses from rm.list_resources()

SIMULATION MODE (default, no hardware needed):
  Generates realistic fake measurements to demonstrate the workflow.
  Simulates PCBA-REV2 S/N 003 — the failing board from Scenario 01.

Run:
  python3 instrument_automation.py

Output:
  analysis_output/bring_up_report.png  — bar-chart test report
  Console test log (MES-format)
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import datetime
import sys

# ── Configuration ────────────────────────────────────────────────────────────

SIMULATION_MODE = True   # Set False when real instruments are connected

# Instrument USB resource addresses (update after running rm.list_resources())
RESOURCE_PSU   = 'USB0::0x0957::0x8B07::MY12345678::INSTR'  # Keysight E36312A
RESOURCE_DMM   = 'USB0::0x0957::0xA618::MY98765432::INSTR'  # Keysight 34461A
RESOURCE_SCOPE = 'USB0::0x1AB1::0x0588::DS1ZA123456::INSTR' # Rigol DS1054Z

OUT = os.path.join(os.path.dirname(__file__), "analysis_output")
os.makedirs(OUT, exist_ok=True)

BOARD_SN      = "003"
BOARD_PN      = "PCBA-REV2"
OPERATOR      = "EET Technician"
TIMESTAMP     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── Test specification table ─────────────────────────────────────────────────
# Each entry: (test_point, net_name, nominal_v, tolerance_pct, unit)

RAIL_SPECS = [
    ("TP1",  "5V_INPUT",  5.00,  5.0,  "V"),
    ("TP2",  "GND",       0.00,  0.0,  "V"),   # must read 0.00 exactly
    ("TP3",  "3V3_RAIL",  3.30,  5.0,  "V"),   # failing board — will be low
    ("TP4",  "1V8_DIGI",  1.80,  5.0,  "V"),
]

CONTINUITY_SPECS = [
    ("U1_GND to TP2",  "continuity",  True),
    ("C2_GND  to TP2", "continuity",  True),
    ("F1 fuse",        "continuity",  True),
]

# C2 ESR spec: should be < 0.1Ω; failing board has 2.0Ω
ESR_SPEC_MAX_ohms = 0.10
C2_REF_DES        = "C2"
C2_NOMINAL_UF     = 22.0


# ── SCPI command library ─────────────────────────────────────────────────────
# These are the exact strings sent over USB to real instruments.
# Each comment shows the instrument family and what the command does.

SCPI_PSU = {
    "identify"       : "*IDN?",
    "reset"          : "*RST",
    "output_on"      : "OUTPut:STATe ON",
    "output_off"     : "OUTPut:STATe OFF",
    "set_voltage"    : "VOLTage {v:.3f}",          # e.g. VOLTage 5.000
    "set_ilimit"     : "CURRent:LIMit {i:.3f}",    # e.g. CURRent:LIMit 0.500
    "meas_voltage"   : "MEASure:VOLTage?",
    "meas_current"   : "MEASure:CURRent?",
}

SCPI_DMM = {
    "identify"       : "*IDN?",
    "reset"          : "*RST",
    "conf_vdc"       : "CONFigure:VOLTage:DC AUTO",
    "meas_vdc"       : "MEASure:VOLTage:DC? AUTO",
    "conf_res"       : "CONFigure:RESistance AUTO",
    "meas_res"       : "MEASure:RESistance? AUTO",
    "conf_cont"      : "CONFigure:CONTinuity",
    "meas_cont"      : "MEASure:CONTinuity?",      # returns 0 (open) or 1 (short)
    "conf_diode"     : "CONFigure:DIODe",
    "meas_diode"     : "MEASure:DIODe?",
}

SCPI_SCOPE = {
    "identify"       : "*IDN?",
    "reset"          : "*RST",
    "autoscale"      : ":AUToscale",
    "timebase"       : ":TIMebase:SCALe {t:.6f}",  # seconds/div
    "v_scale"        : ":CHANnel1:SCALe {v:.4f}",  # V/div
    "coupling_dc"    : ":CHANnel1:COUPling DC",
    "coupling_ac"    : ":CHANnel1:COUPling AC",
    "trig_edge"      : ":TRIGger:EDGe:SOURce CHANnel1",
    "trig_slope_pos" : ":TRIGger:EDGe:SLOPe POSitive",
    "trig_slope_neg" : ":TRIGger:EDGe:SLOPe NEGative",
    "trig_level"     : ":TRIGger:EDGe:LEVel {v:.3f}",
    "meas_freq"      : ":MEASure:FREQuency? CHANnel1",
    "meas_vpp"       : ":MEASure:VPP? CHANnel1",
    "meas_vmax"      : ":MEASure:VMAX? CHANnel1",
    "meas_vmin"      : ":MEASure:VMIN? CHANnel1",
    "meas_vrms"      : ":MEASure:VRMS? CHANnel1",
    "waveform_src"   : ":WAVeform:SOURce CHANnel1",
    "waveform_data"  : ":WAVeform:DATA?",           # returns raw ADC bytes
    "waveform_pre"   : ":WAVeform:PREamble?",       # scale/offset metadata
}


# ── Simulated instrument class ────────────────────────────────────────────────

class SimulatedInstrument:
    """
    Mimics pyvisa Resource.query() / Resource.write() without real hardware.
    Returns realistic fake measurements for PCBA-REV2 S/N 003 (the failing board).
    """

    def __init__(self, itype):
        self.itype  = itype   # 'psu', 'dmm', 'scope'
        self._v_out = 5.00
        self._ilim  = 0.50

    def write(self, cmd):
        if cmd.startswith("VOLTage"):
            self._v_out = float(cmd.split()[-1])
        elif cmd.startswith("CURRent:LIMit"):
            self._ilim = float(cmd.split()[-1])

    def query(self, cmd):
        # Power supply
        if cmd == "*IDN?":
            labels = {"psu": "Keysight,E36312A,MY12345678,A.01.05",
                      "dmm": "Keysight,34461A,MY98765432,A.02.14",
                      "scope": "RIGOL TECHNOLOGIES,DS1054Z,DS1ZA123456,00.04.04.SP4"}
            return labels.get(self.itype, "SIM,UNKNOWN,000,1.0")

        if self.itype == "psu":
            if "VOLTage" in cmd:  return f"{self._v_out * 0.996:.4f}"  # slight drop
            if "CURRent" in cmd:  return "0.1472"                       # idle draw

        if self.itype == "dmm":
            # Simulate PCBA-REV2 S/N 003 rail voltages
            meas_map = {
                "MEASure:VOLTage:DC? AUTO": {
                    "TP1":  "4.9810",   # 5V rail — within spec
                    "TP2":  "0.0002",   # GND
                    "TP3":  "3.1802",   # 3.3V rail — BELOW spec (failing)
                    "TP4":  "1.8105",   # 1.8V digital — within spec
                },
                "MEASure:RESistance? AUTO": {
                    "C2":   "2.0041",   # ESR — ABOVE spec (failing cap)
                    "R3":   "0.0999",   # 0.1Ω sense resistor — correct
                }
            }
            tp = getattr(self, '_current_tp', 'TP1')
            if "VOLTage" in cmd:    return meas_map["MEASure:VOLTage:DC? AUTO"].get(tp, "0.0000")
            if "RESistance" in cmd: return meas_map["MEASure:RESistance? AUTO"].get(tp, "9999.9")
            if "CONTinuity" in cmd: return "1"   # all continuity checks pass

        if self.itype == "scope":
            # TP3 measurements — noisy rail
            scope_meas = {
                ":MEASure:FREQuency? CHANnel1": "999.87",    # ~1kHz ripple
                ":MEASure:VPP? CHANnel1":       "0.4213",    # 421mV p-p ripple
                ":MEASure:VMAX? CHANnel1":      "3.3940",
                ":MEASure:VMIN? CHANnel1":      "2.8830",    # sag below brownout
                ":MEASure:VRMS? CHANnel1":      "3.1789",
            }
            return scope_meas.get(cmd, "0.0000")

        return "0.0000"


# ── Instrument connection helper ──────────────────────────────────────────────

def connect_instruments():
    if SIMULATION_MODE:
        print("[SIM] Simulation mode — no hardware required.")
        print("[SIM] Set SIMULATION_MODE = False for real pyvisa instrument control.\n")
        return (SimulatedInstrument("psu"),
                SimulatedInstrument("dmm"),
                SimulatedInstrument("scope"))
    else:
        try:
            import pyvisa
            rm  = pyvisa.ResourceManager()
            print(f"[VISA] Available resources: {rm.list_resources()}")
            psu   = rm.open_resource(RESOURCE_PSU);   psu.timeout   = 5000
            dmm   = rm.open_resource(RESOURCE_DMM);   dmm.timeout   = 5000
            scope = rm.open_resource(RESOURCE_SCOPE); scope.timeout = 5000
            return psu, dmm, scope
        except Exception as e:
            print(f"[ERROR] pyvisa connection failed: {e}")
            print("[FALLBACK] Switching to simulation mode.")
            return (SimulatedInstrument("psu"),
                    SimulatedInstrument("dmm"),
                    SimulatedInstrument("scope"))


# ── Test step functions ───────────────────────────────────────────────────────

def step_psu_power_on(psu, v_in_volts=5.0, i_limit_amps=0.5):
    """
    Step 1 — Apply input voltage with current limit.
    Current limit prevents catastrophic damage if a short exists.
    Maps to: daily_workflow_simulation.html → 10:00 AM block, Step 1
    """
    print(f"\n{'─'*58}")
    print(" STEP 1 — Power Supply Setup")
    print(f"{'─'*58}")
    print(f"  CMD → {SCPI_PSU['reset']}")
    psu.write(SCPI_PSU["reset"])

    v_cmd = SCPI_PSU["set_voltage"].format(v=v_in_volts)
    i_cmd = SCPI_PSU["set_ilimit"].format(i=i_limit_amps)
    print(f"  CMD → {v_cmd}")
    psu.write(v_cmd)
    print(f"  CMD → {i_cmd}")
    psu.write(i_cmd)
    print(f"  CMD → {SCPI_PSU['output_on']}")
    psu.write(SCPI_PSU["output_on"])

    v_actual = float(psu.query(SCPI_PSU["meas_voltage"]))
    i_actual = float(psu.query(SCPI_PSU["meas_current"]))

    print(f"  PSU OUTPUT: {v_actual:.4f} V  /  {i_actual*1000:.1f} mA")
    if i_actual >= i_limit_amps * 0.9:
        print("  [WARN] Current near limit — possible short on board. STOP and inspect.")
        return False
    print("  [OK] Current within normal range. Proceeding to rail measurements.")
    return True


def step_measure_rails(dmm):
    """
    Step 2 — Sweep all power rail test points with DMM.
    Returns list of (test_point, measured_v, spec_min, spec_max, status)
    Maps to: daily_workflow_simulation.html → 10:00 AM block, multimeter grid
    """
    print(f"\n{'─'*58}")
    print(" STEP 2 — DC Rail Voltage Sweep (Multimeter)")
    print(f"{'─'*58}")
    print(f"  SCPI sequence: {SCPI_DMM['conf_vdc']}  →  {SCPI_DMM['meas_vdc']}")
    print()

    results = []
    dmm.write(SCPI_DMM["conf_vdc"])

    for tp, net, nom, tol_pct, unit in RAIL_SPECS:
        if net == "GND":
            spec_min, spec_max = -0.05, 0.05
        else:
            spec_min = nom * (1 - tol_pct / 100)
            spec_max = nom * (1 + tol_pct / 100)

        # Tell simulation which TP we're measuring
        dmm._current_tp = tp
        raw = dmm.query(SCPI_DMM["meas_vdc"])
        v_meas = float(raw)

        status = "PASS" if spec_min <= v_meas <= spec_max else "FAIL"
        flag   = "✓" if status == "PASS" else "✗"

        print(f"  {flag} {tp:5s} ({net:12s}):  {v_meas:7.4f} V"
              f"   spec [{spec_min:.3f} – {spec_max:.3f}]   {status}")
        results.append((tp, net, v_meas, spec_min, spec_max, status))

    return results


def step_measure_esr(dmm):
    """
    Step 3 — C2 ESR check with DMM resistance mode (board powered off).
    Maps to: daily_workflow_simulation.html → 10:00 AM block, C2 ESR card
    """
    print(f"\n{'─'*58}")
    print(f" STEP 3 — C2 ESR Check (Resistance Mode — board must be POWERED OFF)")
    print(f"{'─'*58}")
    print(f"  SCPI: {SCPI_DMM['conf_res']}  →  {SCPI_DMM['meas_res']}")

    dmm.write(SCPI_DMM["conf_res"])
    dmm._current_tp = "C2"
    esr_ohms = float(dmm.query(SCPI_DMM["meas_res"]))

    status = "PASS" if esr_ohms <= ESR_SPEC_MAX_ohms else "FAIL"
    flag   = "✓" if status == "PASS" else "✗"
    print(f"  {flag} {C2_REF_DES} ESR:  {esr_ohms:.4f} Ω   spec ≤ {ESR_SPEC_MAX_ohms:.2f} Ω   {status}")
    if status == "FAIL":
        tau_us_bad  = esr_ohms * C2_NOMINAL_UF          # τ in µs (ESR×C)
        tau_us_good = ESR_SPEC_MAX_ohms * C2_NOMINAL_UF
        print(f"       τ (bad cap)  = {esr_ohms:.1f}Ω × {C2_NOMINAL_UF:.0f}µF = {tau_us_bad:.1f} µs  "
              f"(brownout recovery too slow)")
        print(f"       τ (good cap) = {ESR_SPEC_MAX_ohms:.2f}Ω × {C2_NOMINAL_UF:.0f}µF = "
              f"{tau_us_good:.2f} µs  (target)")
        print(f"       Root cause confirmed: replace C2 per IPC-7711 procedure.")
    return esr_ohms, status


def step_scope_rail_check(scope):
    """
    Step 4 — Oscilloscope check on TP3 for dynamic behavior.
    Configures scope and reads peak-to-peak, min voltage, ripple frequency.
    Maps to: daily_workflow_simulation.html → 1:00 PM block
    """
    print(f"\n{'─'*58}")
    print(" STEP 4 — Oscilloscope Dynamic Check on TP3 (3.3V Rail)")
    print(f"{'─'*58}")

    # Configure scope for TP3 power rail measurement
    cmds = [
        (SCPI_SCOPE["coupling_dc"],                         "DC coupling"),
        (SCPI_SCOPE["timebase"].format(t=200e-6),           "200µs/div"),
        (SCPI_SCOPE["v_scale"].format(v=0.5),               "500mV/div"),
        (SCPI_SCOPE["trig_slope_neg"],                      "trigger falling"),
        (SCPI_SCOPE["trig_level"].format(v=3.0),            "trigger at 3.0V"),
    ]
    for cmd, note in cmds:
        print(f"  CMD → {cmd:<45s}  ← {note}")
        scope.write(cmd)

    print()
    v_min   = float(scope.query(SCPI_SCOPE["meas_vmin"]))
    v_max   = float(scope.query(SCPI_SCOPE["meas_vmax"]))
    v_pp    = float(scope.query(SCPI_SCOPE["meas_vpp"]))
    v_rms   = float(scope.query(SCPI_SCOPE["meas_vrms"]))
    f_rip   = float(scope.query(SCPI_SCOPE["meas_freq"]))

    brownout_threshold_v = 2.97

    print(f"  Vmax:   {v_max:.4f} V")
    print(f"  Vmin:   {v_min:.4f} V  {'← CROSSES BROWNOUT THRESHOLD' if v_min < brownout_threshold_v else ''}")
    print(f"  Vpp:    {v_pp*1000:.1f} mV  ripple p-p")
    print(f"  Vrms:   {v_rms:.4f} V")
    print(f"  Frip:   {f_rip:.1f} Hz")

    sag_status = "FAIL" if v_min < brownout_threshold_v else "PASS"
    rip_status = "FAIL" if v_pp > 0.050 else "PASS"  # > 50mV p-p = fail

    print(f"\n  Brownout margin: {'FAIL — sag reaches ' + str(round(v_min,3)) + 'V (threshold = ' + str(brownout_threshold_v) + 'V)' if sag_status == 'FAIL' else 'PASS'}")
    print(f"  Ripple spec:     {'FAIL — ' + str(round(v_pp*1000,1)) + 'mV p-p exceeds 50mV spec' if rip_status == 'FAIL' else 'PASS'}")

    return {"v_min": v_min, "v_max": v_max, "v_pp": v_pp,
            "v_rms": v_rms, "f_ripple": f_rip,
            "sag_status": sag_status, "rip_status": rip_status}


# ── Test report plot ──────────────────────────────────────────────────────────

def generate_report(rail_results, esr_ohms, esr_status, scope_data):
    """
    Generates a matplotlib test report: rail voltage bar chart + scope summary.
    Saved to analysis_output/bring_up_report.png
    """
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

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f"Power-On Bring-Up Report\n"
                 f"Board: {BOARD_PN}  ·  S/N: {BOARD_SN}  ·  {TIMESTAMP}",
                 fontsize=12, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    # ── Plot 1: Rail voltage bar chart ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])

    labels    = [f"{r[0]}\n({r[1]})" for r in rail_results]
    measured  = [r[2] for r in rail_results]
    spec_mins = [r[3] for r in rail_results]
    spec_maxs = [r[4] for r in rail_results]
    statuses  = [r[5] for r in rail_results]

    x      = np.arange(len(labels))
    colors = ['#2e7d32' if s == "PASS" else '#c62828' for s in statuses]
    bars   = ax1.bar(x, measured, color=colors, alpha=0.85, width=0.5, zorder=3)

    for i, (bar, spec_min, spec_max) in enumerate(zip(bars, spec_mins, spec_maxs)):
        ax1.plot([x[i] - 0.28, x[i] + 0.28], [spec_min, spec_min],
                 color='#1565c0', linewidth=1.5, linestyle='--', zorder=4)
        ax1.plot([x[i] - 0.28, x[i] + 0.28], [spec_max, spec_max],
                 color='#1565c0', linewidth=1.5, linestyle='--', zorder=4,
                 label='Spec limits' if i == 0 else "")

    for bar, val, status in zip(bars, measured, statuses):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(measured) * 0.02,
                 f"{val:.3f}V\n{status}", ha='center', va='bottom',
                 fontsize=8.5, fontweight='bold',
                 color='#2e7d32' if status == "PASS" else '#c62828')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Measured Voltage (V)", fontsize=10)
    ax1.set_title("DC Rail Voltages vs. Spec Limits  (±5% tolerance bands shown)", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, axis='y', alpha=0.5, zorder=0)
    ax1.set_ylim(0, max(measured) * 1.18)

    pass_patch = mpatches.Patch(color='#2e7d32', label='PASS', alpha=0.85)
    fail_patch = mpatches.Patch(color='#c62828', label='FAIL', alpha=0.85)
    ax1.legend(handles=[pass_patch, fail_patch,
               mpatches.Patch(color='#1565c0', label='Spec limits (dashed)')],
               fontsize=9, loc='upper right')

    # ── Plot 2: ESR measurement ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])

    esr_vals  = [0.05, esr_ohms]
    esr_lbls  = [f"Target\n(good MLCC)\n0.05 Ω", f"Measured\n({C2_REF_DES})\n{esr_ohms:.2f} Ω"]
    esr_clrs  = ['#2e7d32', '#c62828' if esr_status == 'FAIL' else '#2e7d32']
    ax2.bar(esr_lbls, esr_vals, color=esr_clrs, alpha=0.85, width=0.4)
    ax2.axhline(ESR_SPEC_MAX_ohms, color='#1565c0', linestyle='--',
                linewidth=1.5, label=f'Spec max {ESR_SPEC_MAX_ohms} Ω')
    ax2.set_ylabel("ESR (Ω)", fontsize=10)
    ax2.set_title(f"{C2_REF_DES} Equivalent Series Resistance", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, axis='y', alpha=0.5)

    # ── Plot 3: Scope summary — Vmin sag annotation ───────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])

    # Regenerate a representative sag waveform for the report
    t_us   = np.linspace(0, 2000, 2000)
    v_rail = np.full_like(t_us, 3.18)
    tau_us = esr_ohms * C2_NOMINAL_UF      # τ = ESR × C in µs

    for t_sag in [500, 1500]:              # two sag events
        for i, t in enumerate(t_us):
            dt = t - t_sag
            if -20 <= dt < 0:
                v_rail[i] -= 0.30 * (-dt / 20)
            elif 0 <= dt < tau_us * 5:
                v_rail[i] -= 0.30 * np.exp(-dt / tau_us)

    v_rail += 0.008 * np.random.randn(len(t_us))  # small noise

    ax3.plot(t_us / 1000, v_rail, color='#ff7043', linewidth=1.2,
             label='TP3 measured')
    ax3.axhline(3.30,  color='#2e7d32',  linestyle='--', linewidth=1.2,
                label='3.3V nominal')
    ax3.axhline(2.97,  color='#c62828',  linestyle='--', linewidth=1.5,
                label='2.97V brownout threshold')
    ax3.fill_between(t_us / 1000, v_rail, 2.97,
                     where=(v_rail < 2.97), color='#c62828', alpha=0.2,
                     label='Brownout violation zone')

    ax3.set_xlabel("Time (ms)", fontsize=9)
    ax3.set_ylabel("Voltage (V)", fontsize=9)
    ax3.set_title("TP3 — Dynamic Rail (Oscilloscope)", fontsize=10)
    ax3.legend(fontsize=7.5)
    ax3.grid(True, alpha=0.5)
    ax3.set_ylim(2.6, 3.6)

    # ── Footer summary ────────────────────────────────────────────────────────
    n_pass = sum(1 for r in rail_results if r[5] == "PASS")
    n_fail = sum(1 for r in rail_results if r[5] == "FAIL")
    overall = "FAIL" if n_fail > 0 or esr_status == "FAIL" else "PASS"
    ovr_clr = '#c62828' if overall == "FAIL" else '#2e7d32'

    fig.text(0.5, 0.01,
             f"Overall: {overall}  |  Rails {n_pass}/{len(rail_results)} PASS  "
             f"|  ESR: {esr_status}  |  Operator: {OPERATOR}  "
             f"|  NCR required: {'YES — C2 replacement' if overall == 'FAIL' else 'NO'}",
             ha='center', fontsize=9.5, color=ovr_clr, fontweight='bold')

    outfile = os.path.join(OUT, "bring_up_report.png")
    plt.savefig(outfile, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\n  Report saved → {outfile}")
    return outfile


# ── Console NCR summary ───────────────────────────────────────────────────────

def print_ncr_summary(rail_results, esr_ohms, esr_status, scope_data):
    fails = [r for r in rail_results if r[5] == "FAIL"]
    overall = "FAIL" if fails or esr_status == "FAIL" else "PASS"

    print(f"\n{'═'*58}")
    print(f"  TEST COMPLETE — Board: {BOARD_PN}  S/N: {BOARD_SN}")
    print(f"  Overall result: {overall}")
    print(f"  Timestamp: {TIMESTAMP}")
    print(f"{'═'*58}")

    if overall == "FAIL":
        print("\n  NCR REQUIRED — Findings:")
        for tp, net, v_meas, s_min, s_max, status in fails:
            print(f"    • {tp} ({net}): {v_meas:.4f}V  spec [{s_min:.3f}–{s_max:.3f}]  → {status}")
        if esr_status == "FAIL":
            tau_us = esr_ohms * C2_NOMINAL_UF
            print(f"    • C2 ESR: {esr_ohms:.4f} Ω (spec ≤ {ESR_SPEC_MAX_ohms}Ω)")
            print(f"      τ = ESR × C = {esr_ohms:.1f} × {C2_NOMINAL_UF:.0f}µF = {tau_us:.1f} µs "
                  f"→ rail recovery too slow → brownout resets")
        print(f"\n  Recommended action: Replace C2 per IPC-7711.")
        print(f"  Re-test TP3 with oscilloscope after rework.")
        print(f"\n  SCPI re-test command sequence after rework:")
        print(f"    {SCPI_SCOPE['coupling_ac']}")
        print(f"    {SCPI_SCOPE['v_scale'].format(v=0.05)}  ← 50mV/div for ripple detail")
        print(f"    {SCPI_SCOPE['meas_vpp']}")
        print(f"    Expect: < 50mV p-p after C2 replacement")
    else:
        print("\n  All measurements within specification.")
        print("  Board released to functional test.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("  EET Bring-Up Automation — PCBA-REV2")
    print("  Ties to: daily_workflow_simulation.html")
    print("  SCPI instrument: psu / dmm / oscilloscope")
    print("=" * 58)
    print(f"\n  Board:     {BOARD_PN}  S/N: {BOARD_SN}")
    print(f"  Operator:  {OPERATOR}")
    print(f"  Date/Time: {TIMESTAMP}")
    print(f"  Mode:      {'SIMULATION (no hardware)' if SIMULATION_MODE else 'LIVE (pyvisa)'}")

    psu, dmm, scope = connect_instruments()

    # Identify each instrument
    for inst, name in [(psu, "PSU"), (dmm, "DMM"), (scope, "SCOPE")]:
        idn = inst.query("*IDN?")
        print(f"  [{name}] {idn}")

    # Run bring-up sequence
    psu_ok       = step_psu_power_on(psu, v_in_volts=5.0, i_limit_amps=0.5)
    if not psu_ok:
        print("\n[ABORT] Overcurrent detected. Investigate before proceeding.")
        sys.exit(1)

    rail_results = step_measure_rails(dmm)

    # ESR test (simulated as if board were powered off)
    esr_ohms, esr_status = step_measure_esr(dmm)

    # Scope dynamic check
    scope_data   = step_scope_rail_check(scope)

    # Generate matplotlib report
    print(f"\n{'─'*58}")
    print(" Generating test report...")
    generate_report(rail_results, esr_ohms, esr_status, scope_data)

    # Console NCR summary
    print_ncr_summary(rail_results, esr_ohms, esr_status, scope_data)

    print(f"\n{'='*58}")
    print(f"  Output saved to: analysis_output/bring_up_report.png")
    print(f"  Reference: daily_workflow_simulation.html (10:00 AM block)")
    print(f"  Scenario:  scenario_power_rail_01/scenario_power_rail_01.html")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
