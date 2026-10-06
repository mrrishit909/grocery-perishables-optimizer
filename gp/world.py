"""The simulated grocery chain. Twelve stores, twenty perishable items, twenty-six weeks of history made by the chain's usual
ordering rule, and the truth the models never see: real demand (sales stop at the shelf, so a stockout hides it), each lot's
real shelf life (shortened by heat in transit and on hot days), the price response, and the weather to come.

Days count from Friday 2026-03-27 (day 0). Day 182 is "today", Friday 2026-09-25.
"""
import datetime
import functools
import math

import numpy as np

DAY0 = datetime.date(2026, 3, 27)
HIST = 182
TODAY = HIST
FUTURE = 28
DOW = np.array([1.15, 1.35, 1.25, 0.85, 0.80, 0.85, 0.95])        # Fri, Sat, Sun, Mon, Tue, Wed, Thu
DOW_NAMES = ["Fri", "Sat", "Sun", "Mon", "Tue", "Wed", "Thu"]
K_TRUE = 9.0                                                        # demand dispersion: var = mu + mu^2 / k
DISPOSAL = 0.20                                                     # $ per unit thrown away
HEAT_DH = 8.0                                                       # extra degree-hours per degree above 26 C on the day a lot ships
# category: price elasticity, demand change per 10 C above 22 C (log), shelf-life loss per 100 degree-hours (log), Weibull shape
CATS = {"berries": (2.2, 0.25, 0.35, 3.0), "greens": (1.5, 0.10, 0.30, 3.5), "dairy": (0.8, 0.00, 0.10, 5.0), "bakery": (1.6, -0.10, 0.05, 6.0),
        "meat": (1.2, 0.08, 0.25, 4.0), "prepared": (1.8, 0.15, 0.30, 4.0), "produce": (0.9, 0.05, 0.15, 3.5)}
SKUS = [("Strawberries 1 lb", "berries", 4.99, 2.45, 5, 30), ("Blueberries pint", "berries", 4.49, 2.20, 8, 22), ("Raspberries 6 oz", "berries", 3.99, 1.95, 4, 14),
        ("Baby spinach 5 oz", "greens", 3.49, 1.55, 6, 18), ("Romaine hearts 3 ct", "greens", 3.99, 1.70, 8, 16), ("Spring mix 5 oz", "greens", 4.29, 1.85, 5, 15),
        ("Whole milk gallon", "dairy", 3.79, 2.55, 12, 45), ("Greek yogurt 32 oz", "dairy", 5.49, 3.05, 21, 14), ("Heavy cream pint", "dairy", 3.99, 2.35, 14, 9),
        ("Sourdough loaf", "bakery", 4.99, 1.85, 3, 20), ("Croissants 4 ct", "bakery", 5.99, 2.15, 2, 12), ("Bagels 6 ct", "bakery", 4.49, 1.60, 4, 13),
        ("Ground beef 1 lb", "meat", 5.99, 3.95, 3, 26), ("Chicken breast 2 lb", "meat", 8.99, 5.40, 3, 22), ("Salmon fillet 1 lb", "meat", 11.99, 7.60, 2, 9),
        ("Caesar salad bowl", "prepared", 6.99, 2.95, 2, 10), ("Sushi roll tray", "prepared", 8.99, 4.20, 1, 8), ("Cut watermelon cup", "prepared", 4.99, 2.10, 3, 12),
        ("Bananas 3 lb", "produce", 1.99, 1.05, 5, 40), ("Avocados 4 ct", "produce", 5.49, 3.10, 4, 14)]
STORES = ["Elm Street", "Harbor Point", "Maple Grove", "Riverside", "Oak Park", "Cedar Hills", "Lakeview", "Summit", "Westgate", "Brookfield", "Pinecrest", "Northshore"]
NS, NK = len(STORES), len(SKUS)
CAT_NAMES = list(CATS)
SKU_CAT = np.array([CAT_NAMES.index(s[1]) for s in SKUS])
PRICE = np.array([s[2] for s in SKUS])
COST = np.array([s[3] for s in SKUS])
LABEL = np.array([s[4] for s in SKUS])
A_MAX = int(LABEL.max()) + 1
HEATWAVE = [24.0, 25.0, 36.5, 38.0, 38.5, 37.5, 36.0, 33.0, 29.0, 27.0, 26.0, 25.0, 24.0, 24.0]   # forecast from day 182, updated Friday


def date(d):
    return DAY0 + datetime.timedelta(days=int(d))


def dow(d):
    return np.asarray(d) % 7


def nb(rng, mu, k):
    """Negative binomial draws as gamma-Poisson: mean mu, variance mu + mu^2 / k."""
    return rng.poisson(rng.gamma(k, np.maximum(mu, 1e-9) / k))


def life_scale(label, cat, dh, mu=math.log(1.2), b=None):
    """Weibull scale of a lot's shelf life in days. Truth uses mu = log 1.2 and the category's b."""
    b = np.array([CATS[c][2] for c in CAT_NAMES])[cat] if b is None else b
    return label * np.exp(mu - b * dh / 100.0)


@functools.lru_cache(maxsize=8)
def world(seed=12):
    r = np.random.default_rng(seed)
    size = r.uniform(0.6, 1.6, NS)
    size[0] = 1.2
    base = np.array([s[5] for s in SKUS])[None, :] * size[:, None] * r.lognormal(0, 0.15, (NS, NK))      # units/day at full price, 22 C, average weekday
    days = HIST + FUTURE
    t = np.arange(days)
    temp = 14 + 16 * np.sin(np.pi * t / 220) + np.convolve(r.normal(0, 1.6, days + 3), np.ones(4) / 2, "same")[:days]
    forecast_normal = temp[HIST:].copy()
    forecast_normal[:FUTURE] = 24 + r.normal(0, 1, FUTURE)
    temp[HIST:] = np.r_[HEATWAVE, np.full(FUTURE - len(HEATWAVE), 24.0)] + r.normal(0, 1.2, FUTURE)            # what the coming weeks really bring
    promo = (r.random((NK, days // 7 + 1)) < 0.06)[:, t // 7]                                                    # chain-wide weekly flyer
    elast = np.array([CATS[c][0] for c in CAT_NAMES])[SKU_CAT] * r.uniform(0.9, 1.1, NK)
    heat_d = np.array([CATS[c][1] for c in CAT_NAMES])[SKU_CAT]
    shape = np.array([CATS[c][3] for c in CAT_NAMES])[SKU_CAT]
    w = {"seed": seed, "size": size, "base": base, "temp": temp, "forecast_normal": forecast_normal, "promo": promo, "elast": elast, "heat_d": heat_d, "shape": shape}
    w["hist"] = history(w, r)
    return w


def demand_mean(w, s, k, d, temp=None, discount=0.0):
    """True mean demand of store s, item k on day d (arrays broadcast)."""
    temp = w["temp"][d] if temp is None else temp
    promo = w["promo"][k, d]
    price_ratio = (1 - discount) * np.where(promo, 0.75, 1.0)
    return w["base"][s, k] * DOW[dow(d)] * np.exp(w["heat_d"][k] * (temp - 22) / 10) * np.where(promo, 1.3, 1.0) * price_ratio ** (-w["elast"][k])


def lot_dh(rng, temp_day, n):
    """Degree-hours above 4 C a lot picked up in transit and in the back room: usual, sometimes a fault, more on hot days."""
    dh = rng.gamma(2.0, 20.0, n) + HEAT_DH * np.maximum(temp_day - 26, 0)
    return dh + np.where(rng.random(n) < 0.06, rng.uniform(120, 380, n), 0)


def draw_life(rng, k, dh, shape):
    """Real shelf life in days: Weibull around the scale, with mean equal to the scale."""
    return life_scale(LABEL[k], SKU_CAT[k], dh) * rng.weibull(shape) / np.array([math.gamma(1 + 1 / x) for x in np.atleast_1d(shape)])


def simulate(lam_fn, k, rng, order_fn, disc_fn, life_fn, dh_fn, days, inv0=None, record=False):
    """One shelf per row of k (store-item pairs, or Monte Carlo copies), day by day: today's lot arrives, the price is set,
    demand is drawn and sold oldest first, then lots past their real shelf life are thrown out as spoiled and lots at their
    label date are pulled. -> per-day arrays [B, days]."""
    B = len(k)
    inv = np.zeros((B, A_MAX)) if inv0 is None else inv0["inv"].copy()       # units by age in days
    life = np.full((B, A_MAX), np.inf) if inv0 is None else inv0["life"].copy()
    dh = np.zeros((B, A_MAX)) if inv0 is None else inv0["dh"].copy()
    label = LABEL[k]
    out = {n: np.zeros((B, len(days))) for n in ("demand", "sales", "revenue", "spoiled", "dated", "received", "discount", "stockout", "on_hand")}
    lots = [] if record else None
    for j, d in enumerate(days):
        q = np.maximum(np.round(order_fn(j, d, inv, life, dh)), 0)
        inv[:, 0], dh[:, 0] = q, dh_fn(j, d, B)
        life[:, 0] = life_fn(j, d, dh[:, 0])
        a = disc_fn(j, d, inv, life, dh)
        D = nb(rng, lam_fn(j, d, a), K_TRUE)
        avail = inv.sum(1)
        left = D.astype(float)
        before = inv.copy()
        for age in range(A_MAX - 1, -1, -1):
            take = np.minimum(inv[:, age], left)
            inv[:, age] -= take
            left -= take
        sold = avail - inv.sum(1)
        price = PRICE[k] * (1 - a)
        out["demand"][:, j], out["sales"][:, j], out["revenue"][:, j] = D, sold, sold * price
        out["received"][:, j], out["discount"][:, j], out["stockout"][:, j] = q, a, D > avail
        ages = np.arange(A_MAX)[None, :]
        spoil = (inv > 0) & (ages + 1 >= life)
        dated = (inv > 0) & ~spoil & (ages + 1 >= label[:, None])
        out["spoiled"][:, j] = (inv * spoil).sum(1)
        out["dated"][:, j] = (inv * dated).sum(1)
        if record:
            soldout = (before > 0) & (inv == 0)
            for b_, age in zip(*np.nonzero(soldout | spoil | dated)):
                kind = "spoiled" if spoil[b_, age] else "date_expired" if dated[b_, age] else "sold_out"
                lots.append((b_, d - age, age, kind, inv[b_, age]))
        inv[spoil | dated] = 0
        out["on_hand"][:, j] = inv.sum(1)
        inv, life, dh = np.roll(inv, 1, 1), np.roll(life, 1, 1), np.roll(dh, 1, 1)
        inv[:, 0], life[:, 0], dh[:, 0] = 0, np.inf, 0
    out["state"] = {"inv": inv, "life": life, "dh": dh}
    out["lots"] = lots
    return out


def history(w, r):
    """Twenty-six weeks under the chain's rule: order up to 1.1x the same weekday's average sales of the last four weeks;
    on a lot's last labelled day, half the time a clerk marks it down 20-50%. Every lot's real life and exposure are kept
    (the life is truth; the exposure is what the temperature logger recorded)."""
    S, K = np.meshgrid(np.arange(NS), np.arange(NK), indexing="ij")
    s, k = S.ravel(), K.ravel()
    B = len(s)
    sales_log, lives, dhs = np.zeros((B, HIST)), np.zeros((B, HIST)), np.zeros((B, HIST))
    rows = np.arange(B)

    def order(j, d, inv, life, dh):
        past = [d - 7 * i for i in range(1, 5) if d - 7 * i >= 0]
        target = 1.6 * (sales_log[:, past].mean(1) if past else w["base"][s, k] * DOW[dow(d)])
        return np.maximum(target - inv[:, :-1].sum(1), 0)

    def disc(j, d, inv, life, dh):
        last = inv[rows, LABEL[k] - 1] > 0
        return np.where(last & (r.random(B) < 0.5), r.choice([0.2, 0.3, 0.4, 0.5], B), 0.0)

    def dh_fn(j, d, n):
        dhs[:, d] = lot_dh(r, w["temp"][d], n)
        return dhs[:, d]

    def life_fn(j, d, dh):
        lives[:, d] = draw_life(r, k, dh, w["shape"][k])
        return lives[:, d]

    res = {n: np.zeros((B, HIST)) for n in ("demand", "sales", "revenue", "spoiled", "dated", "received", "discount", "stockout", "on_hand")}
    lots, state = [], None
    for d in range(HIST):           # one day at a time: the rule reads yesterday's sales
        o = simulate(lambda j, dd, a: demand_mean(w, s, k, dd, discount=a), k, r, order, disc, life_fn, dh_fn, [d], inv0=state, record=True)
        for n in res:
            res[n][:, d] = o[n][:, 0]
        sales_log[:, d] = o["sales"][:, 0]
        lots += o["lots"]
        state = o["state"]
    res.update(state=state, lots=lots, s=s, k=k, life=lives, dh=dhs)
    return res
