"""Public API (modular monolith). Blueprint services map to: POS ingest (the history load), lot inventory (lots with their
temperature logs), demand forecast, shelf-life model, replenishment (perishable newsvendor), markdown optimizer (dynamic
program), vision shelf (counts from a detector, reconciled against the system) and waste accounting.

    uvicorn gp.api:app          python -m core.jobs gp.api      # the worker
"""
import datetime
import uuid

import numpy as np
from fastapi import Depends, Header
from fastapi.encoders import jsonable_encoder
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from core import audit, db, jobs
from core.app import Ctx, Problem, create_app, run

from . import engine as E, world as W

READ = {"chain:read", "jobs:read"}
WORK = READ | {"lots:write", "vision:write", "forecast:run", "markdown:plan", "replenishment:plan"}
PERMISSIONS = {"viewer": READ, "planner": WORK, "store_manager": WORK | {"chain:load", "markdown:approve", "weather:write", "audit:read"}}
app = create_app("grocery-perishables-optimizer", PERMISSIONS)
auth = app.state.auth
IdemKey = Header(None, alias="Idempotency-Key")
DEEP = 0.4                 # markdowns this deep wait for a store manager
REPS = 300


def worker_ctx(job):
    return Ctx(job["tenant_id"], uuid.UUID(job["payload"]["actor_id"]), "worker", "system")


def sid(t, i):
    return uuid.uuid5(t, f"store-{i}")


def kid(t, k):
    return uuid.uuid5(t, f"sku-{k}")


def dnum(d):
    return (d - W.DAY0).days


def money(v):
    return round(float(v), 2)


def chain(c):
    g = c.execute("SELECT * FROM chain").fetchone()
    if not g:
        raise Problem(409, "no_chain", "load the chain first: POST /v1/chain:load")
    return g


def store_no(c, name):
    r = c.execute("SELECT store_no FROM store WHERE name = %s", [name]).fetchone()
    if not r:
        raise Problem(404, "store_not_found", f"no store '{name}'")
    return r["store_no"]


def sku_no(c, name):
    r = c.execute("SELECT sku_no FROM sku WHERE name = %s", [name]).fetchone()
    if not r:
        raise Problem(404, "sku_not_found", f"no item '{name}'")
    return r["sku_no"]


def true_life(w, seed, s, k, r, qty, dh, source):
    """The simulation truth for a lot: history lots carry the life the generator drew; a delivery gets one drawn from its
    exposure with a seed fixed by the lot, so every run sees the same lot."""
    if source == "history":
        return float(w["hist"]["life"][s * W.NK + k, r])
    rng = np.random.default_rng([seed, s, k, r, int(qty), int(round(dh))])
    return float(W.draw_life(rng, np.array([k]), np.array([dh]), w["shape"][[k]])[0])


def shelf(c, g, s):
    """The store's shelf from the database: units, real lives and exposure by item and age, and today's arrivals."""
    w = W.world(g["seed"])
    today = dnum(g["today"])
    st = {n: np.zeros((W.NK, W.A_MAX)) for n in ("inv", "life", "dh")}
    st["life"][:] = np.inf
    arr = {"qty": np.zeros(W.NK), "life": np.full(W.NK, np.inf), "dh": np.zeros(W.NK)}
    lots = c.execute("""SELECT l.lot_id, k.sku_no, l.received_on, l.qty_received, l.qty_on_hand, l.exposure_dh, l.source, l.label_expiry
                          FROM inventory_lot l JOIN sku k USING (sku_id) JOIN store s USING (store_id)
                         WHERE s.store_no = %s AND l.status = 'on_hand' AND l.qty_on_hand > 0 ORDER BY l.received_on""", [s]).fetchall()
    for lot in lots:
        k, r = lot["sku_no"], dnum(lot["received_on"])
        age = today - r
        life = true_life(w, g["seed"], s, k, r, lot["qty_received"], lot["exposure_dh"], lot["source"])
        if age == 0:
            arr["qty"][k] += lot["qty_on_hand"]
            arr["life"][k], arr["dh"][k] = life, lot["exposure_dh"]
        elif 0 < age < W.A_MAX:
            st["inv"][k, age] += lot["qty_on_hand"]
            st["life"][k, age], st["dh"][k, age] = life, lot["exposure_dh"]
    return st, arr, lots


def weather(g, kind=None):
    """Forecast temperatures from today for 14 days: the one in force, or 'normal' / 'heatwave'."""
    kind = kind or g["weather"]["kind"]
    w = W.world(g["seed"])
    temps = w["temp"].copy()
    fc = np.array(g["weather"]["days"]) if kind == g["weather"]["kind"] else (w["forecast_normal"] if kind == "normal" else np.array(W.HEATWAVE))
    temps[W.TODAY:W.TODAY + len(fc)] = fc
    return temps, kind


# ---------------------------------------------------------------- load (POS ingest)

class LoadIn(BaseModel):
    seed: int = Field(12, ge=0, le=10 ** 6)


@app.post("/v1/chain:load", status_code=202, tags=["chain"], summary="(+) Load the invented chain: 12 stores, 20 perishable items, 26 weeks of sales, lots and waste; fit the demand and shelf-life models")
def load_chain(body: LoadIn, ctx: Ctx = Depends(auth("chain:load")), idem: str | None = IdemKey):
    def work(c):
        if c.execute("SELECT 1 FROM chain").fetchone():
            raise Problem(409, "chain_exists")
        return 202, jobs.enqueue(c, ctx, "chain.load", body.model_dump())
    return run(ctx, idem, body, work)


@jobs.handler("chain.load")
def load_job(c, job):
    t, seed = job["tenant_id"], job["payload"]["seed"]
    w = W.world(seed)
    h = w["hist"]
    c.execute("INSERT INTO chain (id, tenant_id, name, seed, today, weather) VALUES (%s,%s,'FreshWay Markets',%s,%s,%s)",
              [uuid.uuid5(t, "chain"), t, seed, W.date(W.TODAY), Jsonb({"kind": "normal", "days": [round(float(x), 1) for x in w["forecast_normal"]], "issued": "2026-09-24"})])
    db.load(c, "store", ["store_id", "tenant_id", "name", "store_no"], [(sid(t, i), t, n, i) for i, n in enumerate(W.STORES)])
    db.load(c, "sku", ["sku_id", "tenant_id", "name", "category", "price", "unit_cost", "label_days", "sku_no"], [(kid(t, k), t, x[0], x[1], x[2], x[3], x[4], k) for k, x in enumerate(W.SKUS)])
    s_, k_ = h["s"], h["k"]
    sales = [(sid(t, s_[b]), kid(t, k_[b]), W.date(d), int(h["sales"][b, d]), money(W.PRICE[k_[b]] * (1 - h["discount"][b, d]) * (0.75 if w["promo"][k_[b], d] else 1)), float(h["discount"][b, d]), bool(h["stockout"][b, d]), t)
             for b in range(len(s_)) for d in range(W.HIST)]
    db.load(c, "sale", ["store_id", "sku_id", "day", "units", "price", "discount", "stockout", "tenant_id"], sales)
    end = {(b, r): (age, kind, units) for b, r, age, kind, units in h["lots"]}
    state = h["state"]["inv"]
    lots, exps, waste = [], [], []
    for b in range(len(s_)):
        k = int(k_[b])
        for r in np.nonzero(h["received"][b] > 0)[0]:
            lid = uuid.uuid5(t, f"lot-{b}-{r}")
            age_now = W.HIST - r
            on = int(state[b, age_now]) if age_now < W.A_MAX else 0
            age, kind, units = end.get((b, r), (0, "sold_out", 0))
            status, qoh = ("on_hand", on) if on > 0 else (kind, 0)
            lots.append((lid, t, sid(t, s_[b]), kid(t, k), W.date(r), int(h["received"][b, r]), qoh, W.date(r + W.LABEL[k] - 1), round(float(h["dh"][b, r]), 1), status, "history"))
            if status in ("spoiled", "date_expired") and units > 0:
                day = W.date(r + age)
                exps.append((lid, day, status, int(units), t))
                waste.append((uuid.uuid5(t, f"waste-{b}-{r}"), t, sid(t, s_[b]), kid(t, k), lid, day, int(units), status, money(units * W.COST[k]), ))
    db.load(c, "inventory_lot", ["lot_id", "tenant_id", "store_id", "sku_id", "received_on", "qty_received", "qty_on_hand", "label_expiry", "exposure_dh", "status", "source"], lots)
    db.load(c, "expiration_event", ["lot_id", "day", "kind", "units", "tenant_id"], exps)
    db.load(c, "waste_event", ["id", "tenant_id", "store_id", "sku_id", "lot_id", "day", "units", "reason", "cost"], waste)
    dr, lr = E.demand_report(seed), E.life_report(seed)
    audit.record(c, worker_ctx(job), "chain.loaded", "chain", uuid.uuid5(t, "chain"), {"stores": W.NS, "items": W.NK, "sales_rows": len(sales), "lots": len(lots)})
    return {"stores": W.NS, "items": W.NK, "days": W.HIST, "sales_rows": len(sales), "lots": len(lots), "waste_events": len(waste),
            "history": {"stockout_days": round(float(h["stockout"].mean()), 3), "demand_lost_to_stockouts": round(float(1 - h["sales"].sum() / h["demand"].sum()), 3),
                        "waste_share_of_units": round(float((h["spoiled"].sum() + h["dated"].sum()) / h["received"].sum()), 4), "spoiled_units": int(h["spoiled"].sum()), "date_expired_units": int(h["dated"].sum())},
            "models": {"version": E.VERSION, "demand": {"holdout_days": dr["holdout_days"], "wape": {k: round(v, 3) for k, v in dr["wape"].items()}, "bias": {k: round(v, 3) for k, v in dr["bias"].items()}, "dispersion_k": round(dr["k"], 1), "categories": dr["categories"]},
                       "shelf_life": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in lr.items() if k != "params"} | {"categories": {n: {a: (round(b, 3) if isinstance(b, float) else b) for a, b in p.items()} for n, p in lr["params"].items()}}},
            "simulation_truth": {"what": "what the data hide, which no model sees", "elasticity": {n: round(float(w["elast"][W.SKU_CAT == c].mean()), 2) for c, n in enumerate(W.CAT_NAMES)}, "dispersion_k": W.K_TRUE,
                                 "demand_lost_to_stockouts": round(float(1 - h["sales"].sum() / h["demand"].sum()), 3)}}


@app.get("/v1/chain", tags=["chain"], summary="(+) The chain today: stores, items, stock on hand, waste so far, the weather in force")
def get_chain(ctx: Ctx = Depends(auth("chain:read"))):
    with db.tx(ctx.tenant_id) as c:
        g = chain(c)
        oh = c.execute("SELECT count(*) AS lots, coalesce(sum(qty_on_hand), 0) AS units FROM inventory_lot WHERE status = 'on_hand'").fetchone()
        wst = c.execute("SELECT reason, sum(units) AS units, sum(cost) AS cost FROM waste_event GROUP BY reason ORDER BY reason").fetchall()
    return jsonable_encoder({"name": g["name"], "today": g["today"], "weekday": W.DOW_NAMES[dnum(g["today"]) % 7], "stores": W.NS, "items": W.NK, "on_hand": oh, "waste": wst, "weather": g["weather"]})


# ---------------------------------------------------------------- lot inventory

class Reading(BaseModel):
    ts: datetime.datetime
    celsius: float


class LotIn(BaseModel):
    store: str = Field(max_length=40)
    sku: str = Field(max_length=60)
    qty: int
    received_on: datetime.date
    label_expiry: datetime.date | None = None
    temperatures: list[Reading] = Field(default_factory=list, max_length=2000, description="the lot's logger in transit, one reading per interval")


class LotsIn(BaseModel):
    lots: list[LotIn] = Field(min_length=1, max_length=500)


def exposure(readings):
    """Degree-hours above 4 C: each reading counts for the time until the next one."""
    if len(readings) < 2:
        return None
    rs = sorted(readings, key=lambda x: x.ts)
    return sum(max(a.celsius - 4, 0) * (b.ts - a.ts).total_seconds() / 3600 for a, b in zip(rs, rs[1:]))


@app.post("/v1/inventory/lots", status_code=201, tags=["inventory"], summary="Receive lots with their temperature logs; each is checked on its own and its exposure computed from the log")
def receive_lots(body: LotsIn, ctx: Ctx = Depends(auth("lots:write")), idem: str | None = IdemKey):
    def work(c):
        g = chain(c)
        stores = {r["name"]: r for r in c.execute("SELECT store_id, name, store_no FROM store")}
        skus = {r["name"]: r for r in c.execute("SELECT sku_id, name, sku_no, label_days FROM sku")}
        accepted, rejected = [], []
        for i, lot in enumerate(body.lots):
            st, sk = stores.get(lot.store), skus.get(lot.sku)
            dh = exposure(lot.temperatures)
            expiry = lot.label_expiry or (lot.received_on + datetime.timedelta(days=sk["label_days"] - 1) if sk else None)
            why = ("unknown store" if not st else "unknown item" if not sk else "quantity must be positive" if lot.qty <= 0
                   else "received in the future" if lot.received_on > g["today"] else "label expiry before receipt" if expiry < lot.received_on
                   else "temperature log needs at least two readings" if dh is None else "a reading outside -30..60 C" if any(not -30 <= x.celsius <= 60 for x in lot.temperatures) else None)
            if why:
                rejected.append({"index": i, "store": lot.store, "sku": lot.sku, "why": why})
                continue
            lid = uuid.uuid7()
            c.execute("""INSERT INTO inventory_lot (lot_id, tenant_id, store_id, sku_id, received_on, qty_received, qty_on_hand, label_expiry, exposure_dh, status, source)
                         VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'on_hand','delivery')""", [lid, ctx.tenant_id, st["store_id"], sk["sku_id"], lot.received_on, lot.qty, lot.qty, expiry, round(dh, 1)])
            db.load(c, "temperature_sample", ["lot_id", "ts", "celsius", "tenant_id"], [(lid, x.ts, x.celsius, ctx.tenant_id) for x in lot.temperatures])
            lp = E.life_models(g["seed"])
            age_label = (expiry - lot.received_on).days + 1
            p_spoil = 1 - E.survival(lp, sk["sku_no"], dh, age_label)
            accepted.append({"lot_id": lid, "store": lot.store, "sku": lot.sku, "qty": lot.qty, "exposure_dh": round(dh, 1), "label_expiry": expiry, "p_spoils_before_label": round(float(p_spoil), 3),
                             "median_life_days": round(float(W.LABEL[sk["sku_no"]] * np.exp(lp[W.CAT_NAMES[W.SKU_CAT[sk["sku_no"]]]]["mu"] - lp[W.CAT_NAMES[W.SKU_CAT[sk["sku_no"]]]]["b"] * dh / 100) * np.log(2) ** (1 / lp[W.CAT_NAMES[W.SKU_CAT[sk["sku_no"]]]]["shape"])), 1)})
        audit.record(c, ctx, "lots.received", "inventory_lot", "batch", {"accepted": len(accepted), "rejected": len(rejected)})
        return 201, {"accepted": accepted, "rejected": rejected}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- vision shelf

class Facing(BaseModel):
    facing: str = Field(max_length=20)
    count: int = Field(ge=0, le=10000)
    confidence: float = Field(ge=0, le=1)


class VisionIn(BaseModel):
    store: str
    sku: str
    facings: list[Facing] = Field(min_length=1, max_length=100)
    min_confidence: float = Field(0.6, ge=0, le=1)


@app.post("/v1/vision/count", status_code=201, tags=["vision"], summary="Reconcile a shelf camera's counts with the system: confident facings are counted, doubtful ones sent for a manual check, a real gap books shrink oldest lot first")
def vision_count(body: VisionIn, ctx: Ctx = Depends(auth("vision:write")), idem: str | None = IdemKey):
    def work(c):
        g = chain(c)
        s, k = store_no(c, body.store), sku_no(c, body.sku)
        lots = c.execute("""SELECT l.lot_id, l.qty_on_hand FROM inventory_lot l JOIN store s USING (store_id) JOIN sku k USING (sku_id)
                             WHERE s.store_no = %s AND k.sku_no = %s AND l.status = 'on_hand' ORDER BY l.received_on FOR UPDATE""", [s, k]).fetchall()
        system = sum(x["qty_on_hand"] for x in lots)
        sure = [f for f in body.facings if f.confidence >= body.min_confidence]
        doubt = [f for f in body.facings if f.confidence < body.min_confidence]
        counted = sum(f.count for f in sure)
        gap = system - counted
        tol = max(5, round(0.05 * system))
        action = "within tolerance"
        if doubt:
            action = "manual check: low-confidence facings, no adjustment"
        elif gap > tol:
            left = gap
            for x in lots:
                take = min(left, x["qty_on_hand"])
                if take:
                    c.execute("UPDATE inventory_lot SET qty_on_hand = qty_on_hand - %s, status = CASE WHEN qty_on_hand - %s = 0 THEN 'sold_out' ELSE status END WHERE lot_id = %s", [take, take, x["lot_id"]])
                    c.execute("INSERT INTO waste_event (id, tenant_id, store_id, sku_id, lot_id, day, units, reason, cost) VALUES (%s,%s,%s,%s,%s,%s,%s,'shrink',%s)",
                              [uuid.uuid7(), ctx.tenant_id, sid(ctx.tenant_id, s), kid(ctx.tenant_id, k), x["lot_id"], g["today"], take, money(take * W.COST[k])])
                    left -= take
            action = f"booked {gap} units of shrink"
        audit.record(c, ctx, "vision.reconciled", "sku", f"{body.store}/{body.sku}", {"system": system, "counted": counted, "action": action})
        return 201, {"store": body.store, "sku": body.sku, "system_on_hand": system, "camera_count": counted, "gap": gap, "tolerance": tol, "doubtful_facings": [f.facing for f in doubt], "action": action,
                     "note": "detections come from a simulated shelf camera; no image model runs here"}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- forecast and expiration risk

class WeatherIn(BaseModel):
    kind: str = Field(pattern="^(normal|heatwave)$")
    days: list[float] = Field(min_length=7, max_length=14)
    issued: str = Field(max_length=40)


@app.post("/v1/weather/forecast", status_code=201, tags=["forecast"], summary="(+) A new weather forecast for the next one to two weeks; forecasts, risk and plans use it from now on")
def set_weather(body: WeatherIn, ctx: Ctx = Depends(auth("weather:write")), idem: str | None = IdemKey):
    def work(c):
        g = chain(c)
        if any(not -20 <= x <= 50 for x in body.days):
            raise Problem(422, "implausible_temperature")
        c.execute("UPDATE chain SET weather = %s WHERE id = %s", [Jsonb(body.model_dump()), g["id"]])
        audit.record(c, ctx, "weather.updated", "chain", g["id"], {"kind": body.kind, "max": max(body.days)})
        return 201, {"kind": body.kind, "days": body.days, "hot_days": sum(1 for x in body.days if x >= 33)}
    return run(ctx, idem, body, work)


@app.get("/v1/forecast", tags=["forecast"], summary="Demand by day for an item at a store: mean and 80% band from the censored model and the weather in force, the chain rule's number, and the simulation truth")
def forecast(store: str, sku: str, days: int = 7, ctx: Ctx = Depends(auth("chain:read"))):
    with db.tx(ctx.tenant_id) as c:
        g = chain(c)
        s, k = store_no(c, store), sku_no(c, sku)
    w = W.world(g["seed"])
    m = E.demand_models(g["seed"])
    temps, kind = weather(g)
    d = list(range(W.TODAY, W.TODAY + min(days, 14)))
    mu = E.forecast_store(g["seed"], s, d, temps[d], items=[k])[0]
    base = E.baseline_targets(g["seed"], s, d, items=[k])[0] / 1.6
    norm = E.forecast_store(g["seed"], s, d, weather(g, "normal")[0][d], items=[k])[0]
    truth = W.demand_mean(w, s, k, np.array(d))
    rows = [{"day": W.date(x), "weekday": W.DOW_NAMES[x % 7], "temp": round(float(temps[x]), 1), "mean": round(float(mu[j]), 1), "p10": float(E.nb_quantile(mu[j], m["k"], 0.1)), "p90": float(E.nb_quantile(mu[j], m["k"], 0.9)),
             "if_normal_weather": round(float(norm[j]), 1), "same_weekday_sales": round(float(base[j]), 1), "truth_mean": round(float(truth[j]), 1)} for j, x in enumerate(d)]
    return jsonable_encoder({"store": store, "sku": sku, "weather": kind, "model": E.VERSION, "elasticity": round(float(m["censored"]["elast"][W.SKU_CAT[k]]), 2), "dispersion_k": round(m["k"], 1), "days": rows,
                             "week": {"model": round(float(mu.sum())), "if_normal_weather": round(float(norm.sum())), "same_weekday_sales": round(float(base.sum())), "truth": round(float(truth.sum()))}})


@app.get("/v1/expiration/risk", tags=["inventory"], summary="Each item's stock on hand at a store: the units likely to spoil or pass their date before they sell at full price, by the shelf-life model, by the date code alone, and in the simulation truth")
def expiration_risk(store: str, ctx: Ctx = Depends(auth("chain:read"))):
    with db.tx(ctx.tenant_id) as c:
        g = chain(c)
        s = store_no(c, store)
        st, arr, lots = shelf(c, g, s)
    seed = g["seed"]
    temps, kind = weather(g)
    H = int(W.LABEL.max())
    d = list(range(W.TODAY, W.TODAY + H))
    lp, expo = E.life_models(seed), E.exposure_model(seed)
    mu = E.forecast_store(seed, s, d, temps[d])
    model = E.simulate_store(seed, s, d, st, arr, {"mu": mu, "life": lp, "expo": expo}, 200, 3, temps=temps)
    label_view = E.simulate_store(seed, s, d, {**st, "life": np.full_like(st["life"], np.inf)}, {**arr, "life": np.full(W.NK, np.inf)}, {"mu": mu, "life": {n: {"mu": 9.0, "b": 0.0, "shape": 50.0} for n in W.CAT_NAMES}, "expo": expo}, 200, 3, temps=temps)
    truth = E.simulate_store(seed, s, d, st, arr, "truth", 200, 3, temps=temps)
    on = st["inv"].sum(1) + arr["qty"]
    items = [{"sku": W.SKUS[k][0], "category": W.SKUS[k][1], "on_hand": int(on[k]), "expected_waste_model": round(float(model["waste_units"][k]), 1), "expected_waste_date_code": round(float(label_view["waste_units"][k]), 1),
              "waste_cost_model": money(model["waste_units"][k] * W.COST[k]), "truth": round(float(truth["waste_units"][k]), 1)} for k in range(W.NK) if on[k] > 0]
    items.sort(key=lambda x: -x["waste_cost_model"])
    lot_rows = []
    for lot in lots:
        k = lot["sku_no"]
        age = dnum(g["today"]) - dnum(lot["received_on"])
        to_label = dnum(lot["label_expiry"]) - dnum(lot["received_on"]) + 1
        p = 1 - E.survival(lp, k, lot["exposure_dh"], to_label) / max(E.survival(lp, k, lot["exposure_dh"], age), 1e-12)
        lot_rows.append({"lot_id": lot["lot_id"], "sku": W.SKUS[k][0], "age_days": age, "qty_on_hand": lot["qty_on_hand"], "exposure_dh": lot["exposure_dh"], "label_expiry": lot["label_expiry"], "p_spoils_before_label": round(float(p), 3)})
    lot_rows.sort(key=lambda x: -x["p_spoils_before_label"] * x["qty_on_hand"])
    return jsonable_encoder({"store": store, "weather": kind, "horizon_days": H, "items": items, "lots": lot_rows[:12],
                             "totals": {"model": round(float(model["waste_units"].sum())), "date_code": round(float(label_view["waste_units"].sum())), "truth": round(float(truth["waste_units"].sum()))},
                             "how": "no reorders and no markdowns: what is on the shelf now, sold oldest first against the forecast; the model draws each lot's life from the fitted shelf-life curve given its exposure and age, the date-code view assumes every lot lasts to its date"})


# ---------------------------------------------------------------- markdown

class MarkdownIn(BaseModel):
    store: str
    sku: str


@app.post("/v1/markdown/solve", status_code=201, tags=["markdown"], summary="Discounts by day for an item's stock: a dynamic program over units on hand with the forecast, the fitted price response and the spoilage hazard of the largest lot; scored with the alternatives against the truth")
def markdown(body: MarkdownIn, ctx: Ctx = Depends(auth("markdown:plan")), idem: str | None = IdemKey):
    def work(c):
        g = chain(c)
        s, k = store_no(c, body.store), sku_no(c, body.sku)
        st, arr, lots = shelf(c, g, s)
        seed = g["seed"]
        w = W.world(seed)
        mine = [x for x in lots if x["sku_no"] == k]
        if not mine:
            raise Problem(409, "nothing_on_hand")
        big = max(mine, key=lambda x: x["qty_on_hand"])
        age = dnum(g["today"]) - dnum(big["received_on"])
        H = dnum(big["label_expiry"]) - dnum(g["today"]) + 1
        temps, kind = weather(g)
        r = E.markdown_plan(seed, s, k, st, arr, temps, big["exposure_dh"], age, H)
        n0, value, cogs, scores, el = r["n0"], r["value"], r["cogs"], r["scores"], r["elasticity"]
        path = [{"day": W.date(p["day_index"]), "weekday": W.DOW_NAMES[p["day_index"] % 7], **{x: y for x, y in p.items() if x != "day_index"}} for p in r["path"]]
        acts = []
        for p in path:
            if p["discount"] <= 0:
                continue
            status = "pending_approval" if p["discount"] >= DEEP else "planned"
            aid = uuid.uuid7()
            c.execute("INSERT INTO markdown_action (action_id, tenant_id, store_id, sku_id, day, discount, status, plan, created_by) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                      [aid, ctx.tenant_id, sid(ctx.tenant_id, s), kid(ctx.tenant_id, k), p["day"], p["discount"], status, Jsonb(jsonable_encoder({"expected_value": value, "path": path})), ctx.actor_id])
            acts.append({"action_id": aid, "day": p["day"], "discount": p["discount"], "status": status})
        audit.record(c, ctx, "markdown.planned", "sku", f"{body.store}/{body.sku}", {"units": n0, "days": H, "actions": len(acts), "pending": sum(a["status"] == "pending_approval" for a in acts)})
        return 201, {"store": body.store, "sku": body.sku, "units_on_hand": n0, "days_to_label": H, "largest_lot": {"qty": big["qty_on_hand"], "exposure_dh": big["exposure_dh"], "age_days": age},
                     "weather": kind, "elasticity_used": round(el, 2), "expected_value_model": money(value), "path": path, "actions": acts, "cost_of_stock": money(cogs),
                     "on_the_truth": scores, "how": "dynamic program over units on hand: each day a discount of 0-50%, negative binomial demand around the forecast scaled by the fitted price response, sales up to stock, then the largest lot spoils with the fitted hazard for its exposure and age; what is left at the date is pulled. Every policy is then run 1,500 times against real demand and the lot's real life"}
    return run(ctx, idem, body, work)


class ApproveIn(BaseModel):
    note: str = Field(min_length=3, max_length=300)


@app.post("/v1/markdown/{action_id}/approve", tags=["markdown"], summary="(+) A store manager signs off a deep markdown; never the person who planned it")
def approve_markdown(action_id: uuid.UUID, body: ApproveIn, ctx: Ctx = Depends(auth("markdown:plan")), idem: str | None = IdemKey):
    def work(c):
        a = c.execute("SELECT * FROM markdown_action WHERE action_id = %s FOR UPDATE", [action_id]).fetchone()
        if not a:
            raise Problem(404, "action_not_found")
        if a["created_by"] == ctx.actor_id:
            raise Problem(409, "four_eyes", "the planner cannot approve their own markdown")
        if "markdown:approve" not in PERMISSIONS[ctx.role]:
            raise Problem(403, "forbidden", "markdowns of 40% or more need a store manager")
        if a["status"] != "pending_approval":
            raise Problem(409, "not_pending", f"action is {a['status']}")
        c.execute("UPDATE markdown_action SET status = 'approved', approved_by = %s, approved_at = now() WHERE action_id = %s", [ctx.actor_id, action_id])
        audit.record(c, ctx, "markdown.approved", "markdown_action", action_id, {"discount": a["discount"], "day": a["day"], "note": body.note})
        return 200, {"action_id": action_id, "status": "approved", "discount": a["discount"], "day": a["day"], "approved_by": ctx.actor_name}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- replenishment

class ReplenishIn(BaseModel):
    store: str
    days: int = Field(7, ge=3, le=10)


@app.post("/v1/replenishment/solve", status_code=201, tags=["replenishment"], summary="Order-up-to levels for every item for the coming days: a perishable newsvendor on the censored demand forecast and the weather in force, counting only stock the shelf-life model expects to last, its service level chosen per item by simulating the model's own world; scored against the chain's rule on the truth")
def replenish(body: ReplenishIn, ctx: Ctx = Depends(auth("replenishment:plan")), idem: str | None = IdemKey):
    def work(c):
        g = chain(c)
        s = store_no(c, body.store)
        st, arr, _ = shelf(c, g, s)
        seed = g["seed"]
        temps, kind = weather(g)
        d = list(range(W.TODAY, W.TODAY + body.days))
        r = E.replenish_plan(seed, s, st, arr, temps, weather(g, "normal")[0], body.days, REPS)
        mu, cr_item, tg, base, po, bo, summary, sweep = r["mu"], r["cr_item"], r["targets"], r["rule_targets"], r["plan_items"], r["rule_items"], r["summary"], r["tradeoff"]
        plan_score, base_score = summary["plan"], summary["chain_rule"]
        lines = [{"sku": W.SKUS[k][0], "category": W.SKUS[k][1], "service_level": cr_item[k], "order_up_to": [int(v) for v in tg[k, 1:]], "chain_rule_up_to": [int(round(v)) for v in base[k, 1:]],
                  "forecast_week": round(float(mu[k].sum())), "margin_plan": money(po["gross_margin"][k]), "margin_rule": money(bo["gross_margin"][k]),
                  "waste_plan": round(float(po["waste_units"][k]), 1), "waste_rule": round(float(bo["waste_units"][k]), 1)} for k in range(W.NK)]
        pid = uuid.uuid7()
        c.execute("INSERT INTO replenishment_plan (plan_id, tenant_id, store_id, horizon_start, days, params, lines, summary, created_by) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                  [pid, ctx.tenant_id, sid(ctx.tenant_id, s), W.date(d[1]), body.days - 1, Jsonb({"weather": kind, "service_levels": cr_item, "model": E.VERSION}), Jsonb(jsonable_encoder(lines)), Jsonb(summary), ctx.actor_id])
        audit.record(c, ctx, "replenishment.planned", "replenishment_plan", pid, {"store": body.store, "weather": kind, "margin_plan": plan_score["gross_margin"], "margin_rule": base_score["gross_margin"]})
        return 201, {"plan_id": pid, "store": body.store, "weather": kind, "from": W.date(d[1]), "days": body.days - 1, "lines": lines, "on_the_truth": summary, "tradeoff": sweep,
                     "how": f"order up to the service-level quantile of next day's negative binomial demand (censored forecast, {kind} weather), less stock the fitted shelf-life model expects to survive the day; each item's service level is the one that earned most when the plan was simulated in the model's own world; then the plan, the chain's rule (1.6x same-weekday sales, label dates) and the plan without the weather update are run {REPS} times against real demand, real lives and the real weather"}
    return run(ctx, idem, body, work)
