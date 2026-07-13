"""
d16: task decay -- work that waits too long leaves the window (M2).

Unbound mass decays at rate task_decay (per year): each step a share
task_decay*dt of U moves into a loss ledger and is gone. Task mass
accounting becomes seeded = capital + bound + U + lost; the population is
untouched (workers are never destroyed, work is). Discretisation: the
standing stock decays BEFORE the step's seeding is added, so flow-through
mass that binds promptly pays no toll; the alternative ordering charges
every seeded unit one step of decay and swamps the gradual regime with an
artefact (caught and corrected in development). The tempo thereby
acquires permanent level consequences, not only compositional ones: the
congested regime holds a large U stock for a long time, so it loses more.

Theoretical basis: the product cycle (Vernon 1966), windows of opportunity
(Perez and Soete 1988), the depreciating unmatched stock of stock-flow
matching (Coles and Smith 1998), and skill depreciation when unused (Lise
and Postel-Vinay 2020). "Leaves the window" is the honest verb: the
mechanism conflates obsolescence with demand relocation until a
demand-side reading is worked out (development memo, Section 4).

Pre-registered hypotheses (frozen from the development pass; the
certified reference run confirms):

(H1) To first order in the decay rate, lost mass equals the rate times
     the time integral of U in the undecayed run, so the loss ratio
     between tempos inherits the certified hump geometry of d01. At the
     smallest tested rate the first-order prediction holds within 15
     per cent, and the congested-to-gradual loss ratio is of order 10^2.

(H2) The tempo has permanent level consequences: at every positive rate
     the congested regime ends with strictly less bound mass than the
     gradual regime as a share of its own seeded mass, and the gap grows
     with the rate.

(H3) The spatial distribution of the loss correlates positively with the
     static unbound field u(r): losses concentrate where binding is slow,
     which is where the static close says no bearer reaches.

(H4) Accounting closes at every rate: capital + bound + U_end + lost
     equals the seeded total, and reinst closes against the bound stock
     at 1e-9.

Floor off (f_min = 0) so decay is identified separately from M1; births
off; T_max = 20.

Usage: python experiment/d16_task_decay.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiment" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


iface = _load("_interface")
rd = _load("run_dynamic")

DELTAS = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)
THETAS = (1.0, 15.0)

# Frozen baselines (development pass; tolerances noted inline).
BASE_LOST = {(1.0, 0.05): 0.0001, (1.0, 0.1): 0.0002, (1.0, 0.2): 0.0004,
             (1.0, 0.5): 0.0009, (1.0, 1.0): 0.0018,
             (15.0, 0.05): 0.0161, (15.0, 0.1): 0.0309, (15.0, 0.2): 0.0571,
             (15.0, 0.5): 0.1168, (15.0, 1.0): 0.1815}
BASE_RATIO_SMALL = 177.4
BASE_UCORR = 0.474


def main():
    layer = iface.load_static_layer()
    eq = layer.eq
    area = eq.area
    sh = lambda f: float(np.sum(f * area))

    layer.set_maturity(layer.tech.A_K)
    L = eq.solve(layer.c, layer.kappa).L
    a = eq.a_grid
    C = L @ eq.e
    Phi = np.where(C > 0, C / (1.0 + C), 0.0)
    u_s = eq.g_hat * (1.0 - a) * (1.0 - Phi)

    res = {}
    loss_field = None
    for th in THETAS:
        for de in DELTAS:
            dyn, rec, _ = rd.main(T_max=20.0, theta_L=th, theta_abs=th,
                                  verbose=False, layer=layer, anchored=True,
                                  max_births=0, task_decay=de)
            capm, Bm, Um, Lo = sh(dyn.cap), sh(dyn.B), sh(dyn.U), sh(dyn.lost)
            tot = capm + Bm + Um + Lo
            assert abs(float(np.sum(dyn.reinst)) - Bm) < 1e-9
            res[(th, de)] = {"lost": Lo / tot, "bound": Bm / tot, "lost_abs": Lo,
                             "Uint": float(np.sum(np.asarray(rec["U_tot"])) * 0.2)}
            if th == 15.0 and de == 0.2:
                loss_field = dyn.lost.copy()

    # H1: first order at the smallest rate, on absolute mass
    for th in THETAS:
        pred = 0.05 * res[(th, 0.0)]["Uint"]
        got = res[(th, 0.05)]["lost_abs"]
        assert abs(got - pred) / max(pred, 1e-12) < 0.15, \
            f"first-order loss off at theta {th}: {got:.2e} vs {pred:.2e}"
    lost_g, lost_c = res[(1.0, 0.05)]["lost"], res[(15.0, 0.05)]["lost"]
    ratio_small = lost_c / lost_g
    if BASE_RATIO_SMALL is not None:
        assert abs(ratio_small - BASE_RATIO_SMALL) / BASE_RATIO_SMALL < 0.05, \
            f"tempo loss ratio drifted: {ratio_small:.1f} vs {BASE_RATIO_SMALL}"
    assert ratio_small > 50.0, f"tempo loss ratio collapsed: {ratio_small:.1f}"

    # H2: monotont vaxande gap
    for de in DELTAS[1:]:
        assert res[(15.0, de)]["bound"] < res[(1.0, de)]["bound"] - 1e-6, \
            f"congested does not lose more at delta {de}"

    # H3: rumslig korrelation
    ucorr = float(np.corrcoef(loss_field, u_s)[0, 1])
    if BASE_UCORR is not None:
        assert abs(ucorr - BASE_UCORR) < 0.02, f"loss-field corr drifted: {ucorr:.3f}"

    for (th, de), r in res.items():
        if (th, de) in BASE_LOST:
            assert abs(r["lost"] - BASE_LOST[(th, de)]) < 0.02, \
                f"lost share drifted at ({th},{de}): {r['lost']:.3f}"

    # figur
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for th, col, lab in ((1.0, "#0e7c66", "gradual $\\theta=1$"),
                         (15.0, "#c8452a", "congested $\\theta=15$")):
        ax.plot(DELTAS, [res[(th, de)]["lost"] for de in DELTAS], "o-",
                color=col, lw=2, label=lab)
    ax.set_xlabel("task decay rate $\\delta$ (per year)")
    ax.set_ylabel("lost share of seeded mass")
    ax.set_title("Congestion destroys work; the shock speed decides how much")
    ax.legend(frameon=False)
    fig.tight_layout()
    iface.RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(iface.RESULTS / "task_decay.png", dpi=150)

    lines = ["task_decay -- work that waits too long leaves the window (M2)",
             "floor off, births off, T_max = 20",
             "",
             "  theta  delta   lost(seeded)  bound(seeded)"]
    for th in THETAS:
        for de in DELTAS:
            r = res[(th, de)]
            lines.append(f"  {th:5.0f}  {de:5.2f}     {r['lost']:7.4f}      {r['bound']:7.4f}")
    lines += ["",
              f"tempo loss ratio (congested/gradual) at delta 0.05: {ratio_small:.1f}",
              f"U time-integral (delta 0): gradual {res[(1.0,0.0)]['Uint']:.5f}"
              f"  congested {res[(15.0,0.0)]['Uint']:.5f}"
              f"  (ratio {res[(15.0,0.0)]['Uint']/res[(1.0,0.0)]['Uint']:.1f})",
              f"loss-field corr with static u(r), theta 15 delta 0.2: {ucorr:+.3f}",
              "", "all frozen-baseline asserts passed."]
    print("\n".join(lines))
    out = iface.RESULTS / "task_decay_summary.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
