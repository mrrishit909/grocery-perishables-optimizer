"""Held-out evaluation: chains the demo never uses (seeds 21, 31, 41). Each is built and modelled the way the demo is, and
every decision is scored against the simulation truth: real demand (which stockouts hide), each lot's real shelf life, the
real price response and the real weather.

    python -m gp.evaluate > docs/evaluation.md
"""
import time

from . import engine as E, world as W

SEEDS = (21, 31, 41)
CASES = [(0, 0, 260, 150.0), (1, 5, 120, 140.0), (2, 12, 90, 60.0)]      # store, item, overstock units, exposure (degree-hours)


def main():
    t0 = time.time()
    out = ["# Evaluation", "", f"Three chains the demo never uses (seeds {', '.join(map(str, SEEDS))}). Models are fitted on the first 22 weeks and",
           "scored on the last four, or fitted on all 26 weeks and their decisions scored on the coming days, against the simulation truth.",
           "Reproduce with `python -m gp.evaluate`.", ""]
    out += ["## Demand: four held-out weeks against true demand", "",
            "| Chain | Censored model WAPE | bias | Naive (sales as demand) WAPE | bias | Same weekday, 4 weeks WAPE | bias | Berries price response: censored / naive / true |",
            "|---|---|---|---|---|---|---|---|"]
    life_rows, md_rows, rp_rows = [], [], []
    for seed in SEEDS:
        dr = E.demand_report(seed)
        b = dr["categories"]["berries"]
        out.append(f"| {seed} | {dr['wape']['censored']:.1%} | {dr['bias']['censored']:+.1%} | {dr['wape']['naive']:.1%} | {dr['bias']['naive']:+.1%} | {dr['wape']['baseline']:.1%} | {dr['bias']['baseline']:+.1%} | {b['elasticity_censored']} / {b['elasticity_naive']} / {b['elasticity_true']} |")
        lr = E.life_report(seed)
        life_rows.append(f"| {seed} | {lr['lots']} | {lr['spoil_before_date']:.1%} | {lr['predicted']:.1%} | {lr['brier_model']:.3f} | {lr['brier_label']:.3f} | {lr['high_exposure_spoil']:.0%} / {lr['high_exposure_predicted']:.0%} |")
        w = W.world(seed)
        temps_normal = w["temp"].copy()
        temps_normal[W.TODAY:] = w["forecast_normal"]
        for s, k, qty, dh in CASES:
            st, arr = E.world_shelf(seed, s, (k, qty, dh))
            sc = E.markdown_plan(seed, s, k, st, arr, temps_normal, dh, 0, int(W.LABEL[k]), reps=1000)["scores"]
            md_rows.append(f"| {seed} | {W.STORES[s]}: {W.SKUS[k][0]}, {qty} extra | " + " | ".join(f"${sc[p]['gross_margin']:,.0f} ({sc[p]['waste_units']:.0f})" for p in ("no_markdown", "rule_30_on_last_day", "rule_30_from_day_two", "optimised", "optimised_with_true_parameters")) + " |")
        temps_heat = w["temp"].copy()
        temps_heat[W.TODAY:W.TODAY + len(W.HEATWAVE)] = W.HEATWAVE
        for s in (0, 1, 2):
            st, arr = E.world_shelf(seed, s)
            sm = E.replenish_plan(seed, s, st, arr, temps_heat, temps_normal, 7, reps=200)["summary"]
            rp_rows.append(f"| {seed} | {W.STORES[s]} | " + " | ".join(f"${sm[p]['gross_margin']:,.0f} · {sm[p]['waste_units']} wasted · {sm[p]['fill_rate']:.1%} filled" for p in ("chain_rule", "plan_with_normal_weather", "plan")) + " |")
    out += ["", "## Shelf life: lots received in the four held-out weeks", "",
            "| Chain | Lots | Real lives shorter than the date | Model's average P | Brier, model | Brier, date code | Lots over 120 degree-hours: real / predicted |", "|---|---|---|---|---|---|---|"] + life_rows
    out += ["", "## Markdown: an overstocked lot from today to its date, gross margin (units wasted), 1,000 runs on the truth", "",
            "| Chain | Case | No markdown | 30% on the last day | 30% from day two | Optimised | Optimised with true parameters |", "|---|---|---|---|---|---|---|"] + md_rows
    out += ["", "## Replenishment through the heatwave: six days of orders, 200 runs on the truth", "",
            "| Chain | Store | Chain's rule | Plan without the weather update | Plan |", "|---|---|---|---|---|"] + rp_rows
    out += ["", f"Run time {time.time() - t0:.0f} s on one core."]
    print("\n".join(out))


if __name__ == "__main__":
    main()
