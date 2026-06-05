"""
test_elsm_identical.py — prova de equivalência numérica do e-LSM
================================================================

Este teste contém uma CÓPIA VERBATIM do motor da célula 74 do notebook
``Cópia_de_Level_Set_Algorithms.ipynb`` (funções/classes usando GLOBAIS de
módulo, exatamente como no notebook) e a executa lado a lado com o motor
refatorado de ``elsm_model.py`` (que lê os mesmos hiperparâmetros de um objeto
``ELSMConfig``).

Sobre o MESMO stream sintético, MESMO seed e MESMOS hiperparâmetros, exige:
  - centros/sigmas de inicialização idênticos (k-means + clip),
  - u₀ (DLSM) idêntico,
  - previsões de treino e teste idênticas (max |diff| ≤ 1e-12),
  - nº final de regras, regras criadas e fundidas idênticos.

Além disso, confere que os defaults de ``ELSMConfig`` são EXATAMENTE os literais
da célula 74.

Uso:
    python test_elsm_identical.py
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

import elsm_model as mod


# ===========================================================================
# REFERÊNCIA — CÓPIA VERBATIM DA CÉLULA 74 (hiperparâmetros como GLOBAIS)
# ===========================================================================

# ---- initial structure ----
N_INIT_RULES   = 10
SIGMA_SCALE    = 1.10

# ---- rule creation gates ----
TAU_CREATE     = 0.94
TAU_ERR_GATE   = 0.90
E_CREATE       = 0.0022

# ---- rule shape limits ----
SIGMA_MIN      = 0.008
SIGMA_MAX      = 0.12

# ---- RLS core parameters ----
ALPHA          = 1e4
LAMBDA         = 1.0

# ---- reliability dynamics ----
BETA           = 0.985
ETA            = 0.30

# ---- merging control ----
MERGE_DIST     = 0.11
CLIP_YHAT      = True

# ---- soft brakes (no hard max rules) ----
CREATE_COOLDOWN = 45
TAU_CREATE_HARD = 0.33
ERR_GATE_HARD   = 0.55
GROWTH_GAMMA    = 0.035

# ---- cheap merging ----
MERGE_EVERY     = 12

EPS            = 1e-12
SEED           = 0


def safe_sum(x, eps=1e-12):
    s = float(np.sum(x))
    return s if abs(s) > eps else eps


def build_d_from_tau(tau, eps=1e-12):
    tau = np.asarray(tau, float).ravel()
    s = safe_sum(tau, eps)
    d = np.empty(2 * tau.size, float)
    d[0::2] = (tau**2) / s
    d[1::2] = tau / s
    return d


def gaussian_tau_vec(x, centers, sigmas):
    x = np.asarray(x, float).ravel()
    C = np.asarray(centers, float)
    S = np.maximum(np.asarray(sigmas, float), EPS)
    z = (x[None, :] - C) / S
    dist = np.mean(z*z, axis=1)
    return np.exp(-0.5 * dist)


def kmeans_init_rules(Xtr, n_rules, sigma_scale=1.0, seed=0):
    km = KMeans(n_clusters=n_rules, n_init=10, random_state=seed)
    km.fit(Xtr)
    centers = km.cluster_centers_
    labels = km.labels_

    global_std = Xtr.std(axis=0) + 1e-3
    sigmas = np.zeros_like(centers)
    for i in range(n_rules):
        Xi = Xtr[labels == i]
        sigmas[i] = (Xi.std(axis=0) + 1e-3) if Xi.shape[0] >= 5 else global_std
    sigmas = np.clip(sigmas * float(sigma_scale), SIGMA_MIN, SIGMA_MAX)
    return centers, sigmas


def dlsm_fit(X, y, centers, sigmas):
    D = []
    for x in X:
        tau = gaussian_tau_vec(x, centers, sigmas)
        D.append(build_d_from_tau(tau))
    D = np.vstack(D)
    u, *_ = np.linalg.lstsq(D, y, rcond=1e-12)
    return u


class EvolvingRules:
    def __init__(self, centers=None, sigmas=None):
        self.c, self.s, self.n, self.r = [], [], [], []
        if centers is not None:
            for i in range(centers.shape[0]):
                self.c.append(centers[i].copy())
                self.s.append(sigmas[i].copy())
                self.n.append(1)
                self.r.append(1.0)

    def num_rules(self):
        return len(self.c)

    def raw_tau(self, x):
        if self.num_rules() == 0:
            return np.array([])
        C = np.vstack(self.c)
        S = np.vstack(self.s)
        return gaussian_tau_vec(x, C, S)

    def effective_tau(self, x):
        if self.num_rules() == 0:
            return np.array([])
        return self.raw_tau(x) * np.asarray(self.r, float)

    def winner_raw(self, x):
        if self.num_rules() == 0:
            return None, 0.0
        tau = self.raw_tau(x)
        i = int(np.argmax(tau))
        return i, float(tau[i])

    def add_rule(self, x):
        p = x.size
        self.c.append(x.copy())
        self.s.append(np.full(p, SIGMA_MAX))  # start wide
        self.n.append(1)
        self.r.append(1.0)

    def delete_rule(self, j):
        for arr in (self.c, self.s, self.n, self.r):
            del arr[j]

    def update_rule(self, i, x):
        ni = self.n[i]
        ci = self.c[i]
        ci_new = ci + (x - ci) / (ni + 1)

        si = self.s[i]
        diff1 = x - ci
        diff2 = x - ci_new
        var = (si**2 * ni + diff1 * diff2) / (ni + 1)
        si_new = np.sqrt(np.maximum(var, SIGMA_MIN**2))
        si_new = np.minimum(si_new, SIGMA_MAX)

        self.c[i], self.s[i], self.n[i] = ci_new, si_new, ni + 1

    def update_reliability(self, i, e_raw):
        self.r[i] = BETA * self.r[i] + (1 - BETA) * np.exp(-(e_raw*e_raw) / (2*ETA*ETA))

    def try_merge_winner_neighbor(self, i_win):
        if MERGE_DIST is None or MERGE_DIST <= 0 or self.num_rules() < 2:
            return None

        C = np.vstack(self.c)
        ci = C[i_win]
        d = np.linalg.norm(C - ci[None, :], axis=1)
        d[i_win] = np.inf
        j = int(np.argmin(d))
        if float(d[j]) >= MERGE_DIST:
            return None

        ni, nj = self.n[i_win], self.n[j]
        w = ni + nj

        self.c[i_win] = (ni*self.c[i_win] + nj*self.c[j]) / w
        self.s[i_win] = np.minimum(np.maximum((ni*self.s[i_win] + nj*self.s[j]) / w, SIGMA_MIN), SIGMA_MAX)
        self.n[i_win] = w
        self.r[i_win] = max(self.r[i_win], self.r[j])

        return (i_win, j, ni, nj)


class eRLSM:
    def __init__(self, rules: EvolvingRules, u_init):
        self.rules = rules
        self.u = u_init.astype(float).copy()
        self.P = ALPHA * np.eye(self.u.size, dtype=float)
        self._cooldown = 0
        self._step_id = 0

    def _ensure_dims(self):
        N = self.rules.num_rules()
        if self.u.size != 2*N:
            raise RuntimeError(f"Dim mismatch: u={self.u.size}, expected {2*N}")

    def expand_params_for_new_rule(self, init_w):
        old = self.u.size
        new = old + 2
        u_new = np.zeros((new,), float)
        u_new[:old] = self.u
        u_new[new-2] = 0.0
        u_new[new-1] = float(init_w)
        self.u = u_new

        P_new = np.zeros((new, new), float)
        P_new[:old, :old] = self.P
        P_new[old:, old:] = ALPHA * np.eye(2)
        self.P = P_new

    def apply_merge_event(self, i_keep, j_remove, n_keep_before, n_remove_before):
        ik = 2*i_keep
        jk = 2*j_remove
        w = n_keep_before + n_remove_before

        vk, wk = self.u[ik], self.u[ik+1]
        vj, wj = self.u[jk], self.u[jk+1]
        self.u[ik]   = (n_keep_before*vk + n_remove_before*vj) / w
        self.u[ik+1] = (n_keep_before*wk + n_remove_before*wj) / w

        del_idx = [jk, jk+1]
        self.u = np.delete(self.u, del_idx)
        self.P = np.delete(np.delete(self.P, del_idx, axis=0), del_idx, axis=1)

        self.rules.delete_rule(j_remove)

    def _yhat_unclipped(self, x):
        tau_eff = self.rules.effective_tau(x)
        d = build_d_from_tau(tau_eff)
        return float(d @ self.u)

    def step(self, x, y):
        self._step_id += 1
        x = np.asarray(x, float).ravel()
        y = float(y)

        i_win, tau_max = self.rules.winner_raw(x)
        if i_win is None:
            tau_max = 0.0

        yhat_unclipped = self._yhat_unclipped(x)
        e_create_raw = y - yhat_unclipped

        N = max(self.rules.num_rules(), 1)
        tau_create_eff = max(TAU_CREATE_HARD, TAU_CREATE - GROWTH_GAMMA * np.log1p(N))
        err_gate_eff   = max(ERR_GATE_HARD,  TAU_ERR_GATE - GROWTH_GAMMA * np.log1p(N))

        create = False
        if self._cooldown <= 0:
            create = (tau_max < tau_create_eff) or ((abs(e_create_raw) > E_CREATE) and (tau_max < err_gate_eff))

        if create:
            self.rules.add_rule(x)
            self.expand_params_for_new_rule(init_w=y)
            i_win = self.rules.num_rules() - 1
            yhat_unclipped = self._yhat_unclipped(x)
            self._cooldown = CREATE_COOLDOWN
        else:
            self._cooldown = max(self._cooldown - 1, 0)

        self._ensure_dims()

        yhat_out = float(np.clip(yhat_unclipped, 0.0, 1.0)) if CLIP_YHAT else float(yhat_unclipped)

        e_raw = y - yhat_unclipped

        tau_eff = self.rules.effective_tau(x)
        d = build_d_from_tau(tau_eff)
        Pd = self.P @ d
        denom = LAMBDA + float(d @ Pd)
        if abs(denom) < EPS:
            denom = EPS if denom >= 0 else -EPS
        self.P = (1.0 / LAMBDA) * (self.P - (1.0 / denom) * np.outer(Pd, Pd))
        self.u = self.u + (self.P @ d) * e_raw

        if i_win is None:
            i_win = int(np.argmax(self.rules.raw_tau(x)))
        self.rules.update_rule(i_win, x)
        self.rules.update_reliability(i_win, e_raw)

        merged_event = None
        if (MERGE_DIST is not None) and (MERGE_DIST > 0) and (self._step_id % MERGE_EVERY == 0):
            merged_event = self.rules.try_merge_winner_neighbor(i_win)
            if merged_event is not None:
                ik, jr, nk, nr = merged_event
                self.apply_merge_event(ik, jr, nk, nr)

        self._ensure_dims()
        return yhat_out, bool(create), bool(merged_event)


# ===========================================================================
# DRIVERS
# ===========================================================================

def _run_reference(X_train, y_train, X_test, y_test):
    centers0, sigmas0 = kmeans_init_rules(X_train, N_INIT_RULES, SIGMA_SCALE, SEED)
    u0 = dlsm_fit(X_train, y_train, centers0, sigmas0)
    model = eRLSM(EvolvingRules(centers0, sigmas0), u0)

    yhat_tr = np.empty(len(X_train))
    for k, (x, yy) in enumerate(zip(X_train, y_train)):
        yhat_tr[k], _, _ = model.step(x, yy)

    created = merged = 0
    yhat_te = np.empty(len(X_test))
    for k, (x, yy) in enumerate(zip(X_test, y_test)):
        yhat_te[k], dc, dm = model.step(x, yy)
        created += int(dc)
        merged += int(dm)

    return dict(centers0=centers0, sigmas0=sigmas0, u0=u0,
                yhat_tr=yhat_tr, yhat_te=yhat_te,
                n_rules=model.rules.num_rules(), created=created, merged=merged)


def _run_module(X_train, y_train, X_test, y_test, cfg):
    centers0, sigmas0 = mod.kmeans_init_rules(X_train, cfg.n_init_rules, cfg.sigma_scale, cfg.seed, cfg)
    u0 = mod.dlsm_fit(X_train, y_train, centers0, sigmas0)
    model = mod.eRLSM(cfg, mod.EvolvingRules(cfg, centers0, sigmas0), u0)

    yhat_tr = np.empty(len(X_train))
    for k, (x, yy) in enumerate(zip(X_train, y_train)):
        yhat_tr[k], _, _ = model.step(x, yy)

    created = merged = 0
    yhat_te = np.empty(len(X_test))
    for k, (x, yy) in enumerate(zip(X_test, y_test)):
        yhat_te[k], dc, dm = model.step(x, yy)
        created += int(dc)
        merged += int(dm)

    return dict(centers0=centers0, sigmas0=sigmas0, u0=u0,
                yhat_tr=yhat_tr, yhat_te=yhat_te,
                n_rules=model.rules.num_rules(), created=created, merged=merged)


def main() -> int:
    print("=" * 70)
    print("Teste de equivalência: célula 74 (verbatim) vs elsm_model")
    print("=" * 70)

    # 1) Defaults do ELSMConfig devem bater com os literais da célula 74
    cfg = mod.ELSMConfig()
    expected = dict(
        n_init_rules=N_INIT_RULES, sigma_scale=SIGMA_SCALE,
        tau_create=TAU_CREATE, tau_err_gate=TAU_ERR_GATE, e_create=E_CREATE,
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, alpha=ALPHA, lam=LAMBDA,
        beta=BETA, eta=ETA, merge_dist=MERGE_DIST, merge_every=MERGE_EVERY,
        create_cooldown=CREATE_COOLDOWN, tau_create_hard=TAU_CREATE_HARD,
        err_gate_hard=ERR_GATE_HARD, growth_gamma=GROWTH_GAMMA,
        clip_yhat=CLIP_YHAT, seed=SEED,
    )
    mismatches = {k: (getattr(cfg, k), v) for k, v in expected.items() if getattr(cfg, k) != v}
    assert not mismatches, f"ELSMConfig defaults divergem da célula 74: {mismatches}"
    print("  [OK] ELSMConfig defaults == literais da célula 74")

    # 2) Dados sintéticos normalizados (testa o motor cru, sem normalização)
    rng = np.random.RandomState(123)
    X_train = rng.rand(300, 3)
    y_train = (0.4 * X_train[:, 0] + 0.3 * X_train[:, 1] + 0.3 * X_train[:, 2]
               + 0.02 * rng.randn(300))
    X_test = rng.rand(150, 3)
    y_test = (0.4 * X_test[:, 0] + 0.3 * X_test[:, 1] + 0.3 * X_test[:, 2]
              + 0.02 * rng.randn(150))

    ref = _run_reference(X_train, y_train, X_test, y_test)
    out = _run_module(X_train, y_train, X_test, y_test, cfg)

    def _maxdiff(a, b):
        return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

    d_centers = _maxdiff(ref["centers0"], out["centers0"])
    d_sigmas = _maxdiff(ref["sigmas0"], out["sigmas0"])
    d_u0 = _maxdiff(ref["u0"], out["u0"])
    d_tr = _maxdiff(ref["yhat_tr"], out["yhat_tr"])
    d_te = _maxdiff(ref["yhat_te"], out["yhat_te"])

    print(f"  max|Δ centers0| = {d_centers:.2e}")
    print(f"  max|Δ sigmas0|  = {d_sigmas:.2e}")
    print(f"  max|Δ u0|       = {d_u0:.2e}")
    print(f"  max|Δ yhat_tr|  = {d_tr:.2e}")
    print(f"  max|Δ yhat_te|  = {d_te:.2e}")
    print(f"  regras finais ref={ref['n_rules']} mod={out['n_rules']}")
    print(f"  criadas  ref={ref['created']} mod={out['created']}")
    print(f"  fundidas ref={ref['merged']} mod={out['merged']}")

    tol = 1e-12
    ok = (d_centers <= tol and d_sigmas <= tol and d_u0 <= tol
          and d_tr <= tol and d_te <= tol
          and ref["n_rules"] == out["n_rules"]
          and ref["created"] == out["created"]
          and ref["merged"] == out["merged"])

    print("=" * 70)
    if ok:
        print("RESULTADO: IDÊNTICO ✓ (todas as diferenças ≤ 1e-12)")
        return 0
    print("RESULTADO: DIVERGÊNCIA ✗ — o port NÃO é idêntico. Investigar.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
