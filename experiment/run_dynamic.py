"""
run_dynamic.py
--------------
THE FULL DYNAMIC MODEL: all continuous layers coupled in one time loop over a
technology trajectory A_K(t), with the binding flow allocated to occupations as
grain growth, bound mass accumulating as a persistent stock that feeds back into
place value, AND occupational birth with mid-run structural rebuild.

One unifying principle. A single FIT field drives binding, allocation, and
birth:
        FIT_o(r) = exp(-(under_o + lam_over*over_o)/ell) * exp(-d(r, mu_o)/rho)
the SYMMETRIC capability match (under- and over-qualification both penalised)
times locality to the occupation's core. This is the attachment primitive of
the static paper, eq. `attachment` (= model.regime._fit = Equilibrium.e); the
static and dynamic layers share it verbatim, so an over-qualified generalist
far from its core neither attaches in the static capacity nor binds, wins, or
blocks a birth here.

  binding capacity  C(r) = sum_o L_o FIT_o(r)        (poor fit binds weakly)
  binding flow      iota = lambda_b Phi(C) U,  Phi=C/(1+C)
  allocation        share_o = FIT_o / sum_active FIT  (best fit wins the mass)
  birth gap         max_o FIT_o(r) < e*               (no capable+near home)

Birth needs BOTH a gap AND carrying capacity ("baerkraft"): a new occupation
appears only where a fit-gap coincides with strong potential -- value-weighted
seeded mass above a threshold -- so occupations are not spawned everywhere a gap
exists, only where the area can sustain one. e* anchored in the readiness scale
(well-qualified := deficit < ell, e* = e^-1).

Stocks: A_K(t) scalar; L_o(t) population (conserved sum); U(r,t) unbound seeded
mass; B(r,t) accumulated bound mass (persistent, feeds density); grain_o(r) each
occupation's share of B; reinst_o reinstated employment.

FLAGGED SIMPLIFICATIONS: (S1) newborn readiness e_new=exp(-dist/ell), a local
bump not a full q-vector; (S2) W is one bundle-value integral (presence x place
value); (S3) birth = fit-gap + carrying-capacity threshold, periodic check, capped.
The capacity C(r) = sum_o L_o FIT_o(r) is the static paper's attachment capacity
(same primitive, same parameters). eta=1, survival on. No committed code touched.
"""
import importlib.util, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from model.equilibrium import Equilibrium
from model.technology import Technology
_spec = importlib.util.spec_from_file_location("_setup", REPO / "scripts" / "_setup.py")
_setup = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_setup)
R, TAU, BETA, GAMMA = 18.0, 0.08, 0.5, 0.5
NMIN = 0.01  # irreducible baseline density: caps marginal value beta*Pi*n^(b-1)
             # at empty cells (n->0) so empty regions are not infinitely valuable
OUT = REPO / "scripts"


def set_AK(eq, A_K, g0_grid, g0_task, h_grid=None, h_task=None):
    # h is a human-productivity field on the gate: labour's effective cost per
    # task unit is Pi/h, so capital operates where s_K phi_K > h R / Pi.
    # h = None (uniform 1) is the baseline absolute-productivity gate.
    Rg = R if h_grid is None else R*h_grid
    Rt = R if h_task is None else R*h_task
    a_grid = 1.0/(1.0+np.exp(-(A_K*g0_grid - Rg/eq.pi_cell)/TAU))
    a_task = 1.0/(1.0+np.exp(-(A_K*g0_task - Rt/eq.pi_task)/TAU))
    eq.a_grid = a_grid; eq.a_task = a_task
    eq.D_o = np.bincount(eq.row_of, weights=eq.b_w*a_task, minlength=eq.n_occ)
    return a_grid


class Dyn:
    def __init__(self, eq, inp, L0, ell, rho, lam_over=1.0):
        self.eq = eq; self.inp = inp; self.ell = ell; self.rho = rho
        self.ncell = eq.area.size; self.area = eq.area
        self.gx = inp.grid.chi*np.cos(inp.grid.xi)
        self.gy = inp.grid.chi*np.sin(inp.grid.xi)
        self.Pi = eq.pi_cell
        self.n0 = eq.n_occ
        self.mu = eq.mu.copy()
        # Attachment primitive, recomputed here for the (possibly growing)
        # occupation set. At construction this EQUALS the static eq.e
        # (model.regime._fit, the static paper's eq. `attachment`): delta =
        # under-qualification PLUS lam_over * over-qualification, times locality.
        # lam_over=0 and rho->inf recover the one-sided, non-local readiness.
        cap, grid = inp.cap, inp.grid; keys = list(cap.v_gate.keys())
        under = np.zeros((eq.n_occ, grid.xi.size)); over = np.zeros_like(under)
        for k in keys:
            oc = inp.occ[k].to_numpy()[:, None]; qk = cap.q(k, grid.xi, grid.chi)[None, :]
            under += cap.v[k]*np.maximum(qk-oc, 0.0); over += cap.v[k]*np.maximum(oc-qk, 0.0)
        self.lam_over = lam_over
        self.E = np.exp(-(under + lam_over*over)/ell)   # match readiness (binds best where capability MATCHES)
        # locality matrix LOC_o(r) = exp(-d(r,mu_o)/rho); FIT = E_match*LOC
        d = np.sqrt((self.gx[None,:]-self.mu[:,0:1])**2 + (self.gy[None,:]-self.mu[:,1:2])**2)
        self.LOC = np.exp(-d/rho)
        self.FIT = self.E*self.LOC
        self.L = L0.copy()
        self.reinst = np.zeros(eq.n_occ)
        self.sw = np.bincount(eq.row_of, weights=eq.b_w, minlength=eq.n_occ)
        self.original = L0 * self.sw          # pre-shock task mass per occupation (size)
        self.B = np.zeros(self.ncell)
        self.U = np.zeros(self.ncell)
        self.cell_of = eq.cell_of; self.row_of = eq.row_of; self.b_w = eq.b_w
        self.grain = {}
        self.birth_log = []

    @property
    def n_occ(self): return self.L.size

    def bundle_density(self):
        Lorig = self.L[:self.n0]
        return (np.bincount(self.cell_of, weights=Lorig[self.row_of]*self.b_w,
                            minlength=self.ncell) / self.area)

    def density(self):
        return self.bundle_density() + self.B

    def capacity(self):
        return self.L @ self.FIT                         # fit-weighted (poor fit binds weakly)

    def place_value(self, n):
        neff = np.maximum(n, NMIN)                        # density floor (A10 fix):
        nb1 = neff**(BETA-1.0)                            # no divergence as n->0
        return BETA*self.Pi*nb1                           # value density per cell

    def values(self, n):
        # W_o on the SAME scale as the static density_and_value (which calibrates
        # kappa). Strip term = occupation's own non-automated bundle value, exactly
        # as in equilibrium.py; reinstated term = value of accumulated grain. Earlier
        # this multiplied a density by cell area and mis-scaled W ~871x, leaving the
        # re-sort distance/noise-driven; this restores value to the mobility logit.
        with np.errstate(divide="ignore", invalid="ignore"):
            nb1 = np.where(n > 0, n**(BETA-1.0), 0.0)
        a_task = self.eq.a_task; pi_task = self.eq.pi_task
        strip_wD = self.b_w*(1.0-a_task)*pi_task
        W = np.zeros(self.n_occ)
        W[:self.n0] = BETA*np.bincount(self.row_of, weights=strip_wD*nb1[self.cell_of],
                                       minlength=self.n0)
        for o, g in self.grain.items():                   # reinstated grain value
            W[o] += BETA*float(np.sum(g*self.Pi*nb1*self.area))
        return W

    def distances(self):
        return np.sqrt(((self.mu[:,None,:]-self.mu[None,:,:])**2).sum(-1))

    def add_occupation(self, mu_new, t):
        d = np.hypot(self.gx-mu_new[0], self.gy-mu_new[1])
        e_new = np.exp(-d/self.ell)                       # (S1) local capability bump
        loc_new = np.exp(-d/self.rho)
        self.mu = np.vstack([self.mu, mu_new])
        self.E = np.vstack([self.E, e_new[None,:]])
        self.LOC = np.vstack([self.LOC, loc_new[None,:]])
        self.FIT = np.vstack([self.FIT, (e_new*loc_new)[None,:]])
        self.L = np.append(self.L, 0.0)
        self.reinst = np.append(self.reinst, 0.0)
        self.original = np.append(self.original, 1e-4)   # small founding mass so a newborn can begin absorbing
        self.birth_log.append((t, mu_new[0], mu_new[1]))
        return self.n_occ-1


def softmax_target(dyn, W, c, kappa):
    d = dyn.distances()
    Um = (W[None,:]-c*d)/kappa; Um -= Um.max(1, keepdims=True)
    P = np.exp(Um); P /= P.sum(1, keepdims=True)
    return dyn.L @ P


def main(T_max=20.0, dt=0.2, theta_L=3.0, lam_b=1.0, rho=0.5, theta_abs=3.0, lam_over=1.0,
         match_beta=3.0, T_shock=5.0, birth_every=10, carry_thresh=0.002, max_births=40,
         verbose=True, ESTAR=np.exp(-1.0), R_TASK=0.5, L_min=2e-4, layer=None,
         survival_gate=True, ca_lambda=0.0, binding_law="match_allocated"):
    # survival_gate: gate seeding by (1 - a) (baseline True). False seeds the
    #   full gradient ring, including the capital-dominated core.
    # ca_lambda: comparative-advantage variant. h(r) = exp(ca_lambda*(1 - phihat)),
    #   phihat the unit-amplitude technology shape: human productivity high where
    #   the machine is weak, entering the gate as s_K phi_K > h R / Pi.
    #   ca_lambda = 0 recovers the uniform-R absolute gate.
    # binding_law: "match_allocated" (eq. claim + size-rate cap, baseline) or
    #   "size_multiplies" (iota_o ~ M_o * FIT_o, the conflated alternative of
    #   manuscript sec. 3.3, kept for the d05 comparison).
    # The static layer enters through experiment/_interface.py (frozen inputs,
    # calibrated technology, shared attachment primitive, mobility reference);
    # numbered d-scripts pass their cached layer, standalone use loads it here.
    if layer is None:
        _ispec = importlib.util.spec_from_file_location(
            "_interface", REPO / "experiment" / "_interface.py")
        _iface = importlib.util.module_from_spec(_ispec)
        _ispec.loader.exec_module(_iface)
        layer = _iface.load_static_layer()
    inp, L0, occ = layer.inp, layer.L0, layer.occ
    tech, ell = layer.tech, layer.ell; A_final = tech.A_K
    eq = layer.eq
    g0_grid, g0_task = layer.g0_grid, layer.g0_task
    if ca_lambda > 0.0:
        h_grid = np.exp(ca_lambda*(1.0 - g0_grid))
        h_task = np.exp(ca_lambda*(1.0 - g0_task))
    else:
        h_grid = h_task = None
    set_AK(eq, 0.0, g0_grid, g0_task, h_grid, h_task)
    kappa, c = layer.kappa, layer.c
    dyn = Dyn(eq, inp, L0, ell, rho, lam_over=lam_over)
    # Technology maturation in CALENDAR time: a logistic (S-curve) diffusion that
    # rises 5%->95% over T_shock years, centred at T_shock/2. t is now in years, so
    # theta_L and theta_abs are calendar timescales and the outcome is governed by the
    # ratio of the shock tempo (T_shock) to the redistribution tempo (theta_abs/theta_L).
    k_shock = 5.88/T_shock                                        # 5%->95% spans T_shock years
    A_of = lambda t: A_final/(1.0+np.exp(-k_shock*(t-0.5*T_shock)))
    ts = np.arange(0.0, T_max+dt, dt); GammaD_prev = None
    rec = {k: [] for k in ("t","A_K","U_tot","B_tot","n_occ","emp_newborn","Lsum",
                           "cap_util")}
    if verbose:
        print(f"  full dynamic run (fit-weighted): theta_L={theta_L}, lam_b={lam_b}, rho={rho}")
        print(f"  {'t':>5} {'A_K':>6} {'U_tot':>8} {'B_tot':>8} {'n_occ':>6} {'emp_new':>8} {'Lsum':>7}")
    for it, t in enumerate(ts):
        A_K = A_of(t); a_grid = set_AK(eq, A_K, g0_grid, g0_task, h_grid, h_task)
        C = dyn.capacity(); n = dyn.density()
        pv = dyn.place_value(n); W = dyn.values(n)       # value field + source values
        GammaD = float(np.sum(dyn.L[:dyn.n0]*eq.D_o))
        dGamma = 0.0 if GammaD_prev is None else max((GammaD-GammaD_prev)/dt, 0.0)
        GammaD_prev = GammaD
        surv = (1.0-a_grid) if survival_gate else np.ones_like(a_grid)
        sdot = GAMMA*dGamma*eq.g_hat*surv
        dyn.U = dyn.U + dt*sdot                                   # seeding fills the unbound stock
        # MATCH-ALLOCATED, SIZE-RATE-LIMITED binding. The unbound mass is ATTRACTED to
        # the best match: each occupation's claim on a cell's available mass is its
        # FIT^match_beta share (match_beta sharpens so the best match dominates the
        # claim). But each occupation can only BIND up to a size-limited capacity
        # c_o = (dt/theta_abs)*M_o per step; what a saturated occupation cannot staff
        # this step returns to the unbound stock and is offered to the next-best match
        # later. Match thus sets WHERE the mass goes, size only HOW FAST it binds; small
        # best-match occupations saturate and expose lower matches, large occupations do
        # not dominate unless they actually match. (Replaces the M*FIT/C_M inertia, in
        # which size multiplied match and the absolute capture tracked size at corr 0.99.)
        M = dyn.original + dyn.reinst                             # current task mass (grows endogenously)
        avail = dyn.U*dyn.area                                    # unbound mass available per cell
        if binding_law == "size_multiplies":
            # the conflated alternative: size MULTIPLIES match in the claim and
            # no per-occupation cap -- absorption ~ M_o FIT_o share at a global
            # rate dt/theta_abs. Kept only for the sec. 3.3 comparison (d05).
            Wm = M[:, None]*dyn.FIT; Wms = Wm.sum(0)
            with np.errstate(divide="ignore", invalid="ignore"):
                share = np.where(Wms > 0, Wm/Wms[None, :], 0.0)
            t_o = share*avail[None, :]
            f_glob = min(1.0, dt/theta_abs)
            dyn.reinst[:] = dyn.reinst + t_o.sum(1)*f_glob
            absorbed = t_o.sum(0)*f_glob
            absorbed_tot = float(absorbed.sum())
            dyn.U = dyn.U - absorbed/dyn.area
            dyn.B = dyn.B + absorbed/dyn.area
            f = np.full(dyn.n_occ, f_glob)                        # for the newborn grain below
        else:
            Wb = dyn.FIT**match_beta; Wsum = Wb.sum(0)           # match-share weights (sharpened)
            with np.errstate(divide="ignore", invalid="ignore"):
                claim = np.where(Wsum > 0, avail/Wsum, 0.0)       # available mass per unit match-weight
            t_o = Wb*claim[None, :]                               # match-share claim, per occ per cell (mass)
            des = t_o.sum(1)                                      # desired intake per occupation
            capo = (dt/theta_abs)*M                               # size-limited capacity this step
            with np.errstate(divide="ignore", invalid="ignore"):
                f = np.where(des > 1e-15, np.minimum(1.0, capo/des), 0.0)  # absorbed fraction (rate cap)
            dyn.reinst[:] = dyn.reinst + des*f                    # absorb min(match-claim, size-cap)
            absorbed = (t_o*f[:, None]).sum(0)                    # total mass bound per cell
            absorbed_tot = float((des*f).sum())
            dyn.U = dyn.U - absorbed/dyn.area                     # residual stays unbound -> cascades
            dyn.B = dyn.B + absorbed/dyn.area                     # bound density (feeds n)
        for o in range(dyn.n0, dyn.n_occ):                       # grain for newborns (their own claim)
            dyn.grain[o] = dyn.grain.get(o, np.zeros(dyn.ncell)) + t_o[o]*f[o]/dyn.area
        # birth: fit-gap AND carrying capacity (value-weighted seeded mass)
        if it % birth_every == 0 and it > 0 and len(dyn.birth_log) < max_births:
            fit_best = dyn.FIT.max(0)
            gap = (fit_best < ESTAR).astype(float)
            # carrying capacity = seeded WORK weighted by the BOUNDED price Pi
            # (NOT the divergent marginal value beta*Pi*n^(b-1), which blows up at
            # empty cells and would birth spurious occupations on the empty rim).
            potential = (dyn.B + dyn.U)*dyn.Pi*gap
            j = int(np.argmax(potential))
            local = np.exp(-np.hypot(dyn.gx-dyn.gx[j], dyn.gy-dyn.gy[j])/0.25)
            carry = float(np.sum(potential*local*dyn.area))   # carrying capacity
            if carry > carry_thresh:
                # STAFFING VIABILITY (soft, mobility-consistent): a newborn starts
                # small, so it need not dominate established occupations -- it must
                # only attract at least a minimum viable employment L_min through the
                # ordinary re-sort. We compute the newborn's prospective softmax
                # inflow given its prospective value, and birth only if it clears
                # L_min. Far un-staffable niches (empty rim) attract ~0 and stay
                # stillborn; near niches with real carrying capacity clear it.
                W_new_prosp = float(np.sum((dyn.B+dyn.U)*gap*pv*local*dyn.area))
                dj = np.hypot(dyn.mu[:dyn.n0,0]-dyn.gx[j], dyn.mu[:dyn.n0,1]-dyn.gy[j])
                d_all = dyn.distances()[:dyn.n0,:]
                Um = (W[None,:]-c*d_all)/kappa; m = Um.max(1, keepdims=True)
                Z = np.exp(Um-m).sum(1)
                w_new = np.exp((W_new_prosp-c*dj)/kappa - m[:,0])
                T_new = float(np.sum(dyn.L[:dyn.n0]*w_new/(Z+w_new)))
                if T_new > L_min:                            # would attract viable employment
                    dyn.add_occupation(np.array([dyn.gx[j], dyn.gy[j]]), t)
        # re-sort population (moving origin), incl. newborns (W after any birth)
        W = dyn.values(dyn.density())
        Tgt = softmax_target(dyn, W, c, kappa)
        dyn.L = dyn.L + (dt/theta_L)*(Tgt - dyn.L)
        emp_new = float(np.sum(dyn.L[dyn.n0:])) if dyn.n_occ > dyn.n0 else 0.0
        rec["t"].append(t); rec["A_K"].append(A_K)
        # capacity utilisation: mass absorbed this step over the aggregate
        # capacity bound (dt/theta_abs) * M_tot -- the share of task mass at
        # binding caps; 1 in the deep slow limit (Prop. 2 regime).
        rec["cap_util"].append(absorbed_tot/((dt/theta_abs)*float(M.sum())))
        rec["U_tot"].append(float(np.sum(dyn.U*dyn.area)))
        rec["B_tot"].append(float(np.sum(dyn.B*dyn.area)))
        rec["n_occ"].append(dyn.n_occ); rec["emp_newborn"].append(emp_new)
        rec["Lsum"].append(float(np.sum(dyn.L)))
        if verbose and it % 25 == 0:
            print(f"  {t:5.1f} {A_K:6.3f} {rec['U_tot'][-1]:8.4f} {rec['B_tot'][-1]:8.4f} "
                  f"{dyn.n_occ:6d} {emp_new:8.5f} {rec['Lsum'][-1]:7.4f}")
    return dyn, {k: np.array(v) for k, v in rec.items()}, occ


if __name__ == "__main__":
    dyn, rec, occ = main()
    print(f"\n  births: {len(dyn.birth_log)}; final newborn employment {rec['emp_newborn'][-1]:.5f}; "
          f"population conserved {rec['Lsum'][0]:.4f} -> {rec['Lsum'][-1]:.4f}")
    if dyn.birth_log:
        for (tb,bx,by) in dyn.birth_log[:8]:
            print(f"    birth t={tb:.1f} at ({bx:+.2f},{by:+.2f}), chi={np.hypot(bx,by):.2f}")
