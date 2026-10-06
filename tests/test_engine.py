"""The shelf simulator, the fits and the decisions, without a database."""
import numpy as np
from scipy import sparse

from gp import engine as E, world as W


def test_shelf_conserves_units_and_sells_oldest_first():
    k = np.array([0, 3, 12])
    r = np.random.default_rng(0)
    inv0 = {"inv": np.zeros((3, W.A_MAX)), "life": np.full((3, W.A_MAX), np.inf), "dh": np.zeros((3, W.A_MAX))}
    inv0["inv"][:, 2], inv0["life"][:, 2] = 10, 50.0
    o = W.simulate(lambda j, d, a: np.full(3, 3.0), k, r, lambda j, d, inv, life, dh: np.full(3, 6.0), lambda j, d, inv, life, dh: np.zeros(3),
                   lambda j, d, dh: np.full(3, 2.5), lambda j, d, n: np.zeros(n), list(range(6)), inv0=inv0)
    assert np.allclose(10 + o["received"].sum(1), o["sales"].sum(1) + o["spoiled"].sum(1) + o["dated"].sum(1) + o["state"]["inv"].sum(1))
    assert (o["sales"] <= o["demand"]).all() and o["spoiled"].sum() > 0             # 2.5-day lots spoil when demand is short
    one = W.simulate(lambda j, d, a: np.full(1, 4.0), k[:1], np.random.default_rng(1), lambda j, d, inv, life, dh: np.full(1, 5.0), lambda j, d, inv, life, dh: np.zeros(1),
                     lambda j, d, dh: np.full(1, 9.0), lambda j, d, n: np.zeros(n), [0], inv0={n: v[:1].copy() for n, v in inv0.items()})
    left = one["state"]["inv"][0]
    assert left[1] == 5 or one["sales"][0, 0] > 10                                  # today's 5 untouched unless the old 10 sold out


def test_censored_poisson_recovers_demand_that_stockouts_hide():
    r = np.random.default_rng(2)
    n = 4000
    x = r.integers(0, 2, n)
    lam = np.exp(2.5 + 0.4 * x)
    demand = r.poisson(lam)
    cap = r.integers(8, 20, n)
    sales = np.minimum(demand, cap)
    X = sparse.csr_matrix(np.c_[np.ones(n), x])
    cens = demand >= cap
    good = E.fit_poisson(X, sales, cens)
    naive = E.fit_poisson(X, sales, np.zeros(n, bool))
    assert abs(good[0] - 2.5) < 0.05 and abs(good[1] - 0.4) < 0.06 and naive[0] + naive[1] < good[0] + good[1] - 0.1


def test_shelf_life_fit_recovers_the_curve_from_censored_lots():
    seed = 31
    lp = E.life_models(seed)
    w = W.world(seed)
    for k, dh in ((0, 40.0), (0, 250.0), (15, 60.0)):
        model = E.survival(lp, k, dh, W.LABEL[k])
        truth = float(E.true_survival(w, k, dh, W.LABEL[k]))
        assert abs(model - truth) < 0.12, (k, dh, model, truth)
    assert lp["dairy"].get("fallback") == "label"                                  # dairy never spoils in history: the date code stands


def test_markdown_program_discounts_only_when_it_pays():
    mu = np.full(4, 20.0)
    calm, _ = E.markdown_dp(mu, np.zeros(4), 5.0, 2.0, 10.0, 40)
    rush, _ = E.markdown_dp(mu, np.array([0.0, 0.9, 0.9, 0.9]), 5.0, 2.0, 10.0, 200)
    assert calm[0, 40] == 0 and rush[0, 200] >= 0.3
    _, v_small = E.markdown_dp(mu, np.full(4, 0.2), 5.0, 2.0, 10.0, 30)
    _, v_big = E.markdown_dp(mu, np.full(4, 0.2), 5.0, 2.0, 10.0, 60)
    assert v_big >= v_small


def test_order_targets_rise_with_service_level_and_count_only_surviving_stock():
    mu = np.array([[10.0, 30.0]])
    assert (E.model_targets(mu, 9.0, [0.9]) > E.model_targets(mu, 9.0, [0.5])).all()
    lp = E.life_models(12)
    inv = np.zeros((1, W.A_MAX))
    inv[0, 3] = 20
    dh = np.zeros((1, W.A_MAX))
    dh[0, 3] = 300.0
    kk = np.array([0])
    assert E.counted_model(lp)(inv, None, dh, kk)[0] < E.counted_label(inv, None, dh, kk)[0] == 20
