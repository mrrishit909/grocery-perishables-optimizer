"""Integration, end-to-end and security tests against a real Postgres."""
import psycopg
import pytest

from core import db, jobs, scenario

from .conftest import bearer

MANAGER, PLANNER, VIEWER, OTHER = bearer("fresh", "store_manager"), bearer("fresh", "planner"), bearer("fresh", "viewer"), bearer("other", "store_manager")


def test_nothing_works_before_a_chain_is_loaded(client):
    assert client.get("/v1/chain", headers=VIEWER).json()["code"] == "no_chain"
    assert client.post("/v1/chain:load", headers=PLANNER, json={}).status_code == 403


@pytest.fixture(scope="module")
def demo(client):
    results, _ = scenario.run(client, drain=jobs.drain)
    return results


def one(sql, params=(), tenant=None):
    with db.tx(tenant) as c:
        return c.execute(sql, params).fetchone()


def test_demo_journey(demo):
    load = demo["load"]["result"]
    assert load["stores"] == 12 and one("SELECT count(*) AS n FROM sale")["n"] == 12 * 20 * 182 and one("SELECT count(*) AS n FROM inventory_lot")["n"] == load["lots"] + 2
    d = load["models"]["demand"]
    assert d["wape"]["censored"] < d["wape"]["naive"] < d["wape"]["baseline"]
    assert load["models"]["shelf_life"]["brier_model"] < load["models"]["shelf_life"]["brier_label"]
    assert len(demo["lots"]["accepted"]) == 2 and len(demo["lots"]["rejected"]) == 4 and demo["lots"]["accepted"][0]["exposure_dh"] > 100
    assert demo["vision"]["gap"] == 22 and demo["vision"]["action"].startswith("booked") and one("SELECT sum(units) AS n FROM waste_event WHERE reason = 'shrink'")["n"] == 22
    risk = demo["risk"]
    straw = risk["items"][0]
    assert straw["sku"] == "Strawberries 1 lb" and straw["expected_waste_model"] > straw["expected_waste_date_code"]
    md = demo["md"]
    t = md["on_the_truth"]
    assert md["units_on_hand"] == 244 and t["optimised"]["gross_margin"] > t["no_markdown"]["gross_margin"] and t["optimised"]["waste_units"] < t["no_markdown"]["waste_units"]
    assert md["path"][0]["discount"] >= 0.4 and md["actions"][0]["status"] == "pending_approval"
    assert demo["self"]["code"] == "four_eyes" and demo["approve"]["status"] == "approved"
    assert demo["fc"]["weather"] == "heatwave" and demo["fc"]["week"]["model"] > demo["fc"]["week"]["if_normal_weather"]
    p = demo["plan"]["on_the_truth"]
    assert p["plan"]["gross_margin"] > p["chain_rule"]["gross_margin"] and p["plan"]["waste_units"] < p["chain_rule"]["waste_units"]
    assert len(demo["plan"]["lines"]) == 20 and one("SELECT count(*) AS n FROM replenishment_plan")["n"] == 1
    assert demo["audit"]["chain_valid"] and {"chain.loaded", "lots.received", "vision.reconciled", "markdown.planned", "markdown.approved", "weather.updated", "replenishment.planned"} <= {e["action"] for e in demo["audit"]["events"]}


def test_validation_replay_and_approval_rules(client, demo):
    assert client.get("/v1/forecast?store=Atlantis&sku=Strawberries%201%20lb", headers=VIEWER).json()["code"] == "store_not_found"
    assert client.post("/v1/weather/forecast", headers=MANAGER, json={"kind": "heatwave", "days": [80] * 7, "issued": "x"}).json()["code"] == "implausible_temperature"
    assert client.post("/v1/weather/forecast", headers=PLANNER, json={"kind": "normal", "days": [20] * 7, "issued": "x"}).status_code == 403
    aid = demo["md"]["actions"][0]["action_id"]
    assert client.post(f"/v1/markdown/{aid}/approve", headers=MANAGER, json={"note": "again"}).json()["code"] == "not_pending"
    h = {**PLANNER, "Idempotency-Key": "vision-replay"}
    body = {"store": "Oak Park", "sku": "Whole milk gallon", "facings": [{"facing": "F1", "count": 3, "confidence": 0.4}]}
    a, b = client.post("/v1/vision/count", headers=h, json=body), client.post("/v1/vision/count", headers=h, json=body)
    assert a.json() == b.json() and b.headers.get("Idempotent-Replay") == "true" and a.json()["action"].startswith("manual check")
    assert client.post("/v1/vision/count", headers=h, json={**body, "store": "Summit"}).json()["code"] == "idempotency_key_reused"


def test_tenant_isolation_and_append_only_audit(client, demo):
    assert client.get("/v1/chain", headers=OTHER).json()["code"] == "no_chain"
    assert client.get("/v1/expiration/risk?store=Elm%20Street", headers=OTHER).json()["code"] == "no_chain"
    with db.tx() as c:
        other = c.execute("SELECT DISTINCT tenant_id FROM api_tokens WHERE actor_name LIKE '%%Corner Basket%%'").fetchone()["tenant_id"]
    assert one("SELECT count(*) AS n FROM sale", tenant=other)["n"] == 0 and one("SELECT count(*) AS n FROM markdown_action", tenant=other)["n"] == 0
    with pytest.raises(psycopg.errors.RaiseException), db.tx() as c:
        c.execute("UPDATE audit_events SET action = 'edited'")
