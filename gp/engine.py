"""Models and decisions. Demand from sales that stop at the shelf (censored Poisson regression: a stockout day says demand was
at least what sold), shelf life from lots that spoiled, sold out or hit their date (interval- and right-censored Weibull by
category, with transit exposure), a markdown policy by dynamic programming, and order quantities by a perishable
newsvendor chosen in simulation. Every decision is scored against the simulation truth, which no model sees.
"""
import functools
import math

import numpy as np
from scipy import optimize, sparse
from scipy.special import gammainc, gammaln

from . import world as W

VERSION = "gp-1.0"
ACTIONS = np.array([0.0, 0.2, 0.3, 0.4, 0.5])
HOLDOUT = 28


# ---------------------------------------------------------------- demand

def design(pairs_s, pairs_k, days, temp, promo, discount):
    """Rows: one per (pair, day). Columns: pair, category x weekday, category x temperature, category x log price ratio,
    category x flyer. Arrays are [B, D]."""
    B, D = discount.shape
    cat = np.repeat(W.SKU_CAT[pairs_k], D)
    pair = np.repeat(np.arange(B), D)
    dw = np.tile(W.dow(np.asarray(days)), B)
    t = np.tile((temp - 22) / 10, B)
    pr = promo.ravel()
    lp = np.log((1 - discount.ravel()) * np.where(pr, 0.75, 1.0))
    n, C = B * D, len(W.CAT_NAMES)
    cols, vals, rows = [], [], []
    def add(c, v):
        rows.append(np.arange(n)); cols.append(c); vals.append(v)
    add(pair, np.ones(n))
    off = B
    m = dw > 0
    rows.append(np.arange(n)[m]); cols.append(off + cat[m] * 6 + dw[m] - 1); vals.append(np.ones(m.sum()))
    off += C * 6
    add(off + cat, t)
    off += C
    add(off + cat, lp)
    off += C
    add(off + cat, pr.astype(float))
    off += C
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, off))


def fit_poisson(X, y, censored):
    """Maximum likelihood with a log link. A censored row contributes P(demand >= y) instead of P(demand = y)."""
    y = y.astype(float)
    cz = censored & (y > 0)

    def f(beta):
        eta = np.clip(X @ beta, -20, 12)
        lam = np.exp(eta)
        nll = np.where(cz, 0.0, lam - y * eta + gammaln(y + 1))
        g = np.where(cz, 0.0, lam - y)
        if cz.any():
            yc, lc = y[cz], lam[cz]
            q = np.maximum(gammainc(yc, lc), 1e-300)
            nll[cz] = -np.log(q)
            g[cz] = -np.exp((yc - 1) * np.log(lc) - lc - gammaln(yc) - np.log(q)) * lc
        return nll.sum(), X.T @ g
    res = optimize.minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 3000})
    return res.x


def coef_blocks(beta, B):
    C = len(W.CAT_NAMES)
    off = B
    dow = np.c_[np.zeros(C), beta[off:off + C * 6].reshape(C, 6)]
    off += C * 6
    temp = beta[off:off + C]
    off += C
    elast = -beta[off:off + C]
    off += C
    promo = beta[off:off + C]
    return {"pair": beta[:B], "dow": dow, "temp": temp, "elast": elast, "promo": promo}


def dispersion(mu, y):
    v = ((y - mu) ** 2 - mu).sum()
    return float(np.clip((mu ** 2).sum() / max(v, 1e-9), 0.5, 200))


def predict(cb, b, k, days, temp, promo, discount=0.0):
    """Mean demand from fitted coefficients for pair index b (item k) on days."""
    c = W.SKU_CAT[k]
    d = np.asarray(days)
    pr = np.asarray(promo, dtype=float)
    lp = np.log((1 - np.asarray(discount)) * np.where(pr > 0, 0.75, 1.0))
    return np.exp(cb["pair"][b] + cb["dow"][c, W.dow(d)] + cb["temp"][c] * (np.asarray(temp) - 22) / 10 - cb["elast"][c] * lp + cb["promo"][c] * pr)


@functools.lru_cache(maxsize=8)
def demand_models(seed, upto=W.HIST):
    """Fit on days [0, upto): the censored model and the naive one that reads sales as demand."""
    w = W.world(seed)
    h = w["hist"]
    days = np.arange(upto)
    X = design(h["s"], h["k"], days, w["temp"][:upto], w["promo"][h["k"]][:, :upto], h["discount"][:, :upto])
    y = h["sales"][:, :upto].ravel()
    cens = h["stockout"][:, :upto].ravel().astype(bool)
    bc = fit_poisson(X, y, cens)
    bn = fit_poisson(X, y, np.zeros_like(cens))
    mu = np.exp(X @ bc)
    k = dispersion(mu[~cens], y[~cens])
    B = len(h["s"])
    return {"censored": coef_blocks(bc, B), "naive": coef_blocks(bn, B), "k": k}


def baseline(sales, d):
    """Same weekday, last four weeks, from sales."""
    past = [d - 7 * i for i in range(1, 5) if d - 7 * i >= 0]
    return sales[:, past].mean(1)


@functools.lru_cache(maxsize=8)
def demand_report(seed):
    """Fit on the first 22 weeks, score on the last four against true demand (which a stockout hides from the data)."""
    w = W.world(seed)
    h = w["hist"]
    cut = W.HIST - HOLDOUT
    m = demand_models(seed, cut)
    days = np.arange(cut, W.HIST)
    B = len(h["s"])
    truth = W.demand_mean(w, h["s"][:, None], h["k"][:, None], days[None, :], discount=h["discount"][:, cut:])
    out = {}
    for name in ("censored", "naive"):
        p = predict(m[name], np.arange(B)[:, None], h["k"][:, None], days[None, :], w["temp"][days][None, :], w["promo"][h["k"]][:, cut:W.HIST], h["discount"][:, cut:])
        out[name] = p
    out["baseline"] = np.stack([baseline(h["sales"], d) for d in days], 1) * (1 - 0)       # the rule's view: sales, not demand
    wape = {n: float(np.abs(v - truth).sum() / truth.sum()) for n, v in out.items()}
    bias = {n: float(v.sum() / truth.sum() - 1) for n, v in out.items()}
    cats = {}
    for c, name in enumerate(W.CAT_NAMES):
        sel = W.SKU_CAT[h["k"]] == c
        cats[name] = {"elasticity_censored": round(float(m["censored"]["elast"][c]), 2), "elasticity_naive": round(float(m["naive"]["elast"][c]), 2),
                      "elasticity_true": round(float(w["elast"][W.SKU_CAT == c].mean()), 2), "stockout_days": round(float(h["stockout"][sel].mean()), 3)}
    return {"wape": wape, "bias": bias, "k": m["k"], "categories": cats, "holdout_days": HOLDOUT}


# ---------------------------------------------------------------- shelf life

def lot_table(seed, upto=W.HIST):
    """Every lot received before `upto`: category, label, exposure, and what happened to it: spoiled in (a, a+1], or still good
    after c days (sold out, pulled at its date, or still on the shelf)."""
    w = W.world(seed)
    h = w["hist"]
    end = {}
    for b, r, age, kind, units in h["lots"]:
        end[(b, r)] = (age, kind)
    rec = []
    B = len(h["s"])
    for b in range(B):
        k = h["k"][b]
        for r in np.nonzero(h["received"][b, :upto] > 0)[0]:
            age, kind = end.get((b, r), (W.HIST - 1 - r, "on_hand"))
            if r + age >= upto:              # not finished by the cut: still good at the cut
                age, kind = upto - 1 - r, "on_hand"
            lo = age if kind == "spoiled" else age + 1 if kind in ("date_expired", "on_hand") else age
            rec.append((b, int(k), int(r), float(h["dh"][b, r]), lo, kind == "spoiled", float(h["life"][b, r])))
    a = np.array(rec, dtype=float)
    return {"b": a[:, 0].astype(int), "k": a[:, 1].astype(int), "r": a[:, 2].astype(int), "dh": a[:, 3], "t": a[:, 4], "spoiled": a[:, 5].astype(bool), "true_life": a[:, 6]}


def fit_life(lt):
    """Per category: log scale = log(label) + mu - b * exposure/100, Weibull shape kappa."""
    out = {}
    for c, name in enumerate(W.CAT_NAMES):
        sel = W.SKU_CAT[lt["k"]] == c
        lab, dh, t, sp = W.LABEL[lt["k"][sel]], lt["dh"][sel], lt["t"][sel], lt["spoiled"][sel]
        if sp.sum() < 30:            # too few spoiled lots to learn from: trust the date code
            out[name] = {"mu": 1.0, "b": 0.0, "shape": 30.0, "lots": int(sel.sum()), "spoiled": int(sp.sum()), "fallback": "label"}
            continue

        def nll(p):
            mu, b, lk = p
            kap = math.exp(lk)
            sc = lab * np.exp(mu - b * dh / 100)
            s0 = np.exp(-(np.maximum(t, 0) / sc) ** kap)
            s1 = np.exp(-((t + 1) / sc) ** kap)
            return -(np.log(np.maximum(s0[sp] - s1[sp], 1e-300)).sum() + np.log(np.maximum(s0[~sp], 1e-300)).sum())
        best = min((optimize.minimize(nll, [m0, 0.2, math.log(3)], method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-6}) for m0 in (0.0, 0.3)), key=lambda o: o.fun)
        mu, b, lk = best.x
        out[name] = {"mu": float(mu), "b": float(b), "shape": float(math.exp(lk)), "lots": int(sel.sum()), "spoiled": int(sp.sum())}
    return out


def survival(params, k, dh, t):
    """P(a lot of item k with this exposure is still good after t days)."""
    p = params[W.CAT_NAMES[W.SKU_CAT[k]]]
    sc = W.LABEL[k] * np.exp(p["mu"] - p["b"] * np.asarray(dh) / 100)
    return np.exp(-(np.maximum(np.asarray(t, dtype=float), 0) / sc) ** p["shape"])


def true_survival(w, k, dh, t):
    shape = w["shape"][k]
    sc = W.life_scale(W.LABEL[k], W.SKU_CAT[k], np.asarray(dh)) / math.gamma(1 + 1 / shape)
    return np.exp(-(np.maximum(np.asarray(t, dtype=float), 0) / sc) ** shape)


@functools.lru_cache(maxsize=8)
def life_models(seed, upto=W.HIST):
    return fit_life(lot_table(seed, upto))


@functools.lru_cache(maxsize=8)
def life_report(seed):
    """Fit on lots received in the first 22 weeks; score P(spoils before its date) on the last four weeks' lots against their real lives."""
    w = W.world(seed)
    cut = W.HIST - HOLDOUT
    params = life_models(seed, cut)
    lt = lot_table(seed)
    sel = lt["r"] >= cut
    k, dh, life = lt["k"][sel], lt["dh"][sel], lt["true_life"][sel]
    actual = life < W.LABEL[k]
    p_model = 1 - survival_vec(params, k, dh, W.LABEL[k])
    brier_model = float(((p_model - actual) ** 2).mean())
    brier_label = float((actual.astype(float) ** 2).mean())           # the date code says every lot lasts to its date: p = 0
    hot = dh > 120
    return {"lots": int(sel.sum()), "spoil_before_date": float(actual.mean()), "predicted": float(p_model.mean()), "brier_model": brier_model, "brier_label": brier_label,
            "high_exposure_lots": int(hot.sum()), "high_exposure_spoil": float(actual[hot].mean()) if hot.any() else None, "high_exposure_predicted": float(p_model[hot].mean()) if hot.any() else None,
            "params": params}


def survival_vec(params, k, dh, t):
    out = np.empty(len(k))
    for i in range(len(k)):
        out[i] = survival(params, int(k[i]), dh[i], t[i])
    return out


# ---------------------------------------------------------------- markdown

def nb_pmf(mu, k, n):
    """pmf over 0..n-1 and the tail mass at n (P(D >= n))."""
    x = np.arange(n)
    p = k / (k + mu)
    logp = gammaln(x + k) - gammaln(k) - gammaln(x + 1) + k * np.log(p) + x * np.log1p(-p)
    pm = np.exp(logp)
    return pm, max(0.0, 1 - pm.sum())


def markdown_dp(mu_full, hazard, price, elast, k, n0, actions=ACTIONS, disposal=W.DISPOSAL):
    """Value V_t(n): the best expected revenue minus disposal from day t with n units. Each day: pick a discount, demand is
    negative binomial around the discounted mean, sales are min(n, demand); then the lot spoils with probability hazard[t]
    (all that is left is thrown out); after the last day what is left is pulled at its date."""
    H = len(mu_full)
    N = int(n0) + 1
    V = -disposal * np.arange(N, dtype=float)
    policy = np.zeros((H, N))
    for t in range(H - 1, -1, -1):
        best = np.full(N, -np.inf)
        arg = np.zeros(N)
        for a in actions:
            mu = mu_full[t] * (1 - a) ** (-elast)
            pm, tail = nb_pmf(mu, k, N)
            val = np.zeros(N)
            for n in range(N):
                d = np.arange(n)
                sold = np.r_[d, n]
                probs = np.r_[pm[:n], 1 - pm[:n].sum()]
                left = n - sold
                cont = hazard[t] * (-disposal * left) + (1 - hazard[t]) * V[left]
                val[n] = (probs * (price * (1 - a) * sold + cont)).sum()
            better = val > best + 1e-9
            best = np.where(better, val, best)
            arg = np.where(better, a, arg)
        V, policy[t] = best, arg
    return policy, float(V[int(n0)])


# ---------------------------------------------------------------- scoring helpers

def score(out, k):
    """Totals over the simulated days, averaged over Monte Carlo rows sharing an item."""
    rev = out["revenue"].sum(1)
    waste = out["spoiled"].sum(1) + out["dated"].sum(1)
    cogs = out["received"].sum(1) * W.COST[k]
    return {"revenue": rev, "waste_units": waste, "waste_cost": waste * W.COST[k] + waste * W.DISPOSAL, "gross_margin": rev - cogs - waste * W.DISPOSAL,
            "lost_units": (out["demand"] - out["sales"]).sum(1), "sold": out["sales"].sum(1), "demand": out["demand"].sum(1)}


def exposure_model(seed):
    """Degree-hours a lot picks up, against the day's heat above 26 C: least squares on the logged history."""
    w = W.world(seed)
    h = w["hist"]
    x = np.maximum(w["temp"][:W.HIST] - 26, 0)
    y = h["dh"].mean(0)
    slope, icpt = np.polyfit(x, y, 1)
    return {"intercept": float(icpt), "per_degree_over_26": float(slope)}


def hazards(params, k, dh, H):
    """P(a lot spoils at the end of day t | still good), t = 0..H-1 (age t)."""
    s = np.array([survival(params, k, dh, t) for t in range(H + 1)])
    return 1 - s[1:] / np.maximum(s[:-1], 1e-12)


# ---------------------------------------------------------------- replenishment

CR_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def store_rows(seed, s):
    h = W.world(seed)["hist"]
    return np.nonzero(h["s"] == s)[0]           # pair indices of store s, in item order


def forecast_store(seed, s, days, temps, items=None):
    m = demand_models(seed)
    w = W.world(seed)
    rows = store_rows(seed, s)
    d = np.asarray(days)
    items = range(W.NK) if items is None else items
    return np.stack([predict(m["censored"], rows[k], k, d, temps, w["promo"][k, d]) for k in items])     # [items, H]


def baseline_targets(seed, s, days, items=None):
    """The chain's rule: up to 1.6x the same weekday's sales over the last four weeks of history."""
    h = W.world(seed)["hist"]
    rows = store_rows(seed, s)[list(range(W.NK)) if items is None else list(items)]
    out = np.zeros((len(rows), len(days)))
    for j, d in enumerate(days):
        past = [x for x in (d - 7 * i for i in range(1, 9)) if x < W.HIST][:4]
        out[:, j] = 1.6 * h["sales"][rows][:, past].mean(1)
    return out


def nb_quantile(mu, k, q):
    from scipy.stats import nbinom
    return nbinom.ppf(q, k, k / (k + mu))


def simulate_store(seed, s, days, state, arrivals, world, reps, rng_seed, items=None, targets=None, counted=None, discount=None, temps=None):
    """Store s over `days` from `state` (units, real lives and exposure by item and age, [K, A]) plus today's `arrivals`
    (units, life, exposure per item). world='truth' draws real demand and real lives; a dict {mu [items, H], life, expo} is the
    model's view, in which the lives of what is already on the shelf are drawn given their age. `temps` is the forecast the
    model plans with; the truth always runs on the real weather. Orders up to `targets`
    [items, H] less `counted(...)`; `discount(j, units_on_hand)` sets the price. -> per-item means over reps."""
    w = W.world(seed)
    items = np.arange(W.NK) if items is None else np.asarray(items)
    n_it = len(items)
    kk = np.tile(items, reps)
    it = np.tile(np.arange(n_it), reps)
    r = np.random.default_rng(rng_seed)
    inv0 = {n: np.tile(state[n][items], (reps, 1)) for n in ("inv", "life", "dh")}
    aq, al, adh = (np.tile(arrivals[n][items], reps) for n in ("qty", "life", "dh"))
    temps = w["temp"] if temps is None else temps
    if world != "truth":
        lp = world["life"]
        for c, name in enumerate(W.CAT_NAMES):
            p = lp[name]
            for a in range(W.A_MAX):
                sel = (W.SKU_CAT[kk] == c) & (inv0["inv"][:, a] > 0)
                if not sel.any():
                    continue
                sc = W.LABEL[kk[sel]] * np.exp(p["mu"] - p["b"] * inv0["dh"][sel, a] / 100)
                u = r.random(sel.sum()) * np.exp(-(a / sc) ** p["shape"])
                inv0["life"][sel, a] = sc * (-np.log(np.maximum(u, 1e-300))) ** (1 / p["shape"])
        sc = W.LABEL[kk] * np.exp(np.array([lp[W.CAT_NAMES[c]]["mu"] for c in W.SKU_CAT[kk]]) - np.array([lp[W.CAT_NAMES[c]]["b"] for c in W.SKU_CAT[kk]]) * adh / 100)
        al = sc * r.weibull(np.array([lp[W.CAT_NAMES[c]]["shape"] for c in W.SKU_CAT[kk]]))

    def order(j, d, inv, life, dh):
        q = np.zeros(len(kk)) if targets is None else np.maximum(targets[it, j] - counted(inv, life, dh, kk), 0)
        return q + (aq if j == 0 else 0)

    def dh_fn(j, d, n):
        if j == 0:
            return np.where(aq > 0, adh, W.lot_dh(r, w["temp"][d], n) if world == "truth" else world["expo"]["intercept"])
        return W.lot_dh(r, w["temp"][d], n) if world == "truth" else np.full(n, world["expo"]["intercept"] + world["expo"]["per_degree_over_26"] * max(temps[d] - 26, 0))

    def life_fn(j, d, dh):
        if world == "truth":
            v = W.draw_life(r, kk, dh, w["shape"][kk])
        else:
            lp = world["life"]
            v = np.empty(len(kk))
            for c, name in enumerate(W.CAT_NAMES):
                sel = W.SKU_CAT[kk] == c
                p = lp[name]
                v[sel] = W.LABEL[kk[sel]] * np.exp(p["mu"] - p["b"] * dh[sel] / 100) * r.weibull(p["shape"], sel.sum())
        return np.where((aq > 0) & (j == 0), al, v)

    def lam(j, d, a):
        if world == "truth":
            return W.demand_mean(w, s, kk, d, discount=a)          # the real weather, whatever the forecast said
        m = demand_models(seed)
        el = m["censored"]["elast"][W.SKU_CAT[kk]]
        return world["mu"][it, j] * (1 - a) ** (-el)
    disc = (lambda j, d, inv, life, dh: np.zeros(len(kk))) if discount is None else (lambda j, d, inv, life, dh: discount(j, inv.sum(1)))
    o = W.simulate(lam, kk, r, order, disc, life_fn, dh_fn, days, inv0=inv0)
    sc = score(o, kk)
    agg = lambda v: v.reshape(reps, n_it).mean(0)     # noqa: E731
    out = {k_: agg(v) for k_, v in sc.items()}
    out["daily_discount"] = o["discount"].reshape(reps, n_it, -1).mean(0)
    out["daily_sold"] = o["sales"].reshape(reps, n_it, -1).mean(0)
    out["daily_waste"] = (o["spoiled"] + o["dated"]).reshape(reps, n_it, -1).mean(0)
    return out


def counted_label(inv, life, dh, kk):
    ages = np.arange(W.A_MAX)[None, :]
    return (inv * (ages + 1 < W.LABEL[kk][:, None])).sum(1)


def counted_model(params):
    def f(inv, life, dh, kk):
        tot = np.zeros(len(kk))
        for a in range(W.A_MAX):
            have = inv[:, a] > 0
            if not have.any():
                continue
            idx = np.nonzero(have)[0]
            ok = a + 1 < W.LABEL[kk[idx]]
            p_ok = np.array([survival(params, int(kk[i]), dh[i, a], a + 1) / max(survival(params, int(kk[i]), dh[i, a], a), 1e-12) for i in idx])
            tot[idx] += inv[idx, a] * p_ok * ok
        return tot
    return f


def model_targets(mu, k, cr_by_item):
    return np.stack([nb_quantile(mu[i], k, cr_by_item[i]) for i in range(len(mu))])


# ---------------------------------------------------------------- decisions, shared by the API and the evaluation

def world_shelf(seed, s, arrival=None):
    """The shelf of store s at the start of today from the generator (real lives included), plus an arrival for item 0."""
    st = W.world(seed)["hist"]["state"]
    rows = store_rows(seed, s)
    shelf = {n: st[n][rows].copy() for n in ("inv", "life", "dh")}
    arr = {"qty": np.zeros(W.NK), "life": np.full(W.NK, np.inf), "dh": np.zeros(W.NK)}
    if arrival:
        k, qty, dh = arrival
        arr["qty"][k], arr["dh"][k] = qty, dh
        arr["life"][k] = float(W.draw_life(np.random.default_rng([seed, s, k, W.TODAY, int(qty), int(round(dh))]), np.array([k]), np.array([dh]), W.world(seed)["shape"][[k]])[0])
    return shelf, arr


def markdown_plan(seed, s, k, st, arr, temps, lot_dh, lot_age, days_left, reps=1500):
    """The dynamic program on the model, then every policy run against the truth."""
    w = W.world(seed)
    H = days_left
    n0 = int(st["inv"][k].sum() + arr["qty"][k])
    d = list(range(W.TODAY, W.TODAY + H))
    m, lp = demand_models(seed), life_models(seed)
    mu = forecast_store(seed, s, d, temps[d], items=[k])[0]
    el = float(m["censored"]["elast"][W.SKU_CAT[k]])
    hz = hazards(lp, k, lot_dh, lot_age + H)[lot_age:]
    pol, value = markdown_dp(mu, hz, W.PRICE[k], el, m["k"], n0)
    tmu = W.demand_mean(w, s, k, np.array(d))
    ts = np.array([true_survival(w, k, lot_dh, x) for x in range(lot_age, lot_age + H + 1)])
    tpol, _ = markdown_dp(tmu, 1 - ts[1:] / np.maximum(ts[:-1], 1e-12), W.PRICE[k], w["elast"][k], W.K_TRUE, n0)
    pols = {"no_markdown": lambda j, n: np.zeros(len(n)), "rule_30_on_last_day": lambda j, n: np.full(len(n), 0.3 if j == H - 1 else 0.0),
            "rule_30_from_day_two": lambda j, n: np.full(len(n), 0.3 if j >= 1 else 0.0), "optimised": lambda j, n: pol[j, np.minimum(n, n0).astype(int)],
            "optimised_with_true_parameters": lambda j, n: tpol[j, np.minimum(n, n0).astype(int)]}
    cogs = float(n0 * W.COST[k])
    scores = {}
    for name, p in pols.items():
        o = simulate_store(seed, s, d, st, arr, "truth", reps, 11, items=[k], discount=p, temps=temps)
        scores[name] = {"revenue": round(float(o["revenue"][0]), 2), "sold": round(float(o["sold"][0]), 1), "waste_units": round(float(o["waste_units"][0]), 1),
                        "gross_margin": round(float(o["revenue"][0] - cogs - o["waste_units"][0] * W.DISPOSAL), 2),
                        "daily_discount": [round(float(x), 3) for x in o["daily_discount"][0]], "daily_sold": [round(float(x), 1) for x in o["daily_sold"][0]]}
    path, n = [], n0
    for j in range(H):
        a = float(pol[j, int(min(n, n0))])
        sold = min(n, mu[j] * (1 - a) ** (-el))
        path.append({"day_index": d[j], "discount": a, "units_at_open": round(n), "expected_sales": round(float(sold), 1), "p_lot_spoils_tonight": round(float(hz[j]), 3), "forecast_full_price": round(float(mu[j]), 1)})
        n = max(n - sold, 0)
    return {"n0": n0, "elasticity": el, "value": value, "path": path, "scores": scores, "cogs": cogs}


def replenish_plan(seed, s, st, arr, temps, normal_temps, days, reps=300):
    """Service levels chosen in the model's own world, then the plan, the chain's rule and the plan without the weather
    update scored against the truth; and the margin-against-waste sweep across service levels."""
    m, lp, expo = demand_models(seed), life_models(seed), exposure_model(seed)
    d = list(range(W.TODAY, W.TODAY + days))
    mu = forecast_store(seed, s, d, temps[d])
    cnt = counted_model(lp)
    mw = {"mu": mu, "life": lp, "expo": expo}

    def targets(mean, crs):
        t = model_targets(mean, m["k"], crs)
        t[:, 0] = 0                               # today's delivery is already on the shelf
        return t
    trial = {cr: simulate_store(seed, s, d, st, arr, mw, 80, 1, targets=targets(mu, [cr] * W.NK), counted=cnt, temps=temps)["gross_margin"] for cr in CR_GRID}
    cr_item = [max(CR_GRID, key=lambda x: trial[x][k]) for k in range(W.NK)]
    tg = targets(mu, cr_item)
    base = baseline_targets(seed, s, d)
    base[:, 0] = 0
    tg_n = targets(forecast_store(seed, s, d, normal_temps[d]), cr_item)

    def on_truth(t, counted):
        o = simulate_store(seed, s, d, st, arr, "truth", reps, 7, targets=t, counted=counted, temps=temps)
        return o, {"gross_margin": round(float(o["gross_margin"].sum()), 2), "waste_units": round(float(o["waste_units"].sum())), "waste_cost": round(float(o["waste_cost"].sum()), 2),
                   "lost_sales_units": round(float(o["lost_units"].sum())), "fill_rate": round(float(o["sold"].sum() / o["demand"].sum()), 3)}
    po, plan = on_truth(tg, cnt)
    bo, rule = on_truth(base, counted_label)
    _, normal = on_truth(tg_n, cnt)
    sweep = [{"service_level": cr, **on_truth(targets(mu, [cr] * W.NK), cnt)[1]} for cr in CR_GRID]
    return {"mu": mu, "cr_item": cr_item, "targets": tg, "rule_targets": base, "plan_items": po, "rule_items": bo,
            "summary": {"plan": plan, "chain_rule": rule, "plan_with_normal_weather": normal}, "tradeoff": sweep}
