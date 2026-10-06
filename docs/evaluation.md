# Evaluation

Three chains the demo never uses (seeds 21, 31, 41). Models are fitted on the first 22 weeks and
scored on the last four, or fitted on all 26 weeks and their decisions scored on the coming days, against the simulation truth.
Reproduce with `python -m gp.evaluate`.

## Demand: four held-out weeks against true demand

| Chain | Censored model WAPE | bias | Naive (sales as demand) WAPE | bias | Same weekday, 4 weeks WAPE | bias | Berries price response: censored / naive / true |
|---|---|---|---|---|---|---|---|
| 21 | 6.8% | -5.8% | 10.6% | -9.7% | 25.8% | -7.3% | 1.66 / 0.98 / 2.29 |
| 31 | 6.7% | -5.7% | 10.7% | -9.8% | 24.8% | -6.9% | 2.4 / 1.04 / 2.16 |
| 41 | 7.2% | -6.2% | 11.1% | -10.2% | 25.4% | -8.3% | 2.02 / 0.94 / 2.23 |

## Shelf life: lots received in the four held-out weeks

| Chain | Lots | Real lives shorter than the date | Model's average P | Brier, model | Brier, date code | Lots over 120 degree-hours: real / predicted |
|---|---|---|---|---|---|---|
| 21 | 6301 | 42.2% | 36.2% | 0.232 | 0.422 | 76% / 67% |
| 31 | 6279 | 39.4% | 37.2% | 0.231 | 0.394 | 80% / 71% |
| 41 | 6282 | 39.6% | 36.0% | 0.222 | 0.396 | 81% / 67% |

## Markdown: an overstocked lot from today to its date, gross margin (units wasted), 1,000 runs on the truth

| Chain | Case | No markdown | 30% on the last day | 30% from day two | Optimised | Optimised with true parameters |
|---|---|---|---|---|---|---|
| 21 | Elm Street: Strawberries 1 lb, 260 extra | $65 (128) | $65 (128) | $291 (17) | $266 (3) | $310 (19) |
| 21 | Harbor Point: Spring mix 5 oz, 120 extra | $35 (61) | $35 (61) | $64 (33) | $35 (61) | $57 (41) |
| 21 | Maple Grove: Ground beef 1 lb, 90 extra | $139 (8) | $139 (8) | $95 (2) | $139 (8) | $139 (8) |
| 31 | Elm Street: Strawberries 1 lb, 260 extra | $188 (111) | $188 (111) | $347 (9) | $334 (3) | $380 (19) |
| 31 | Harbor Point: Spring mix 5 oz, 120 extra | $-11 (68) | $-11 (68) | $40 (36) | $-11 (68) | $66 (14) |
| 31 | Maple Grove: Ground beef 1 lb, 90 extra | $222 (1) | $222 (1) | $170 (0) | $222 (1) | $222 (1) |
| 41 | Elm Street: Strawberries 1 lb, 260 extra | $622 (7) | $611 (1) | $346 (0) | $480 (0) | $591 (1) |
| 41 | Harbor Point: Spring mix 5 oz, 120 extra | $247 (23) | $255 (14) | $195 (1) | $243 (2) | $256 (4) |
| 41 | Maple Grove: Ground beef 1 lb, 90 extra | $111 (15) | $96 (7) | $58 (3) | $111 (15) | $112 (13) |

## Replenishment through the heatwave: six days of orders, 200 runs on the truth

| Chain | Store | Chain's rule | Plan without the weather update | Plan |
|---|---|---|---|---|
| 21 | Elm Street | $6,638 · 267 wasted · 81.6% filled | $7,647 · 41 wasted · 80.4% filled | $7,841 · 55 wasted · 84.2% filled |
| 21 | Harbor Point | $6,735 · 249 wasted · 80.8% filled | $7,817 · 28 wasted · 79.4% filled | $8,171 · 37 wasted · 84.2% filled |
| 21 | Maple Grove | $7,329 · 283 wasted · 82.8% filled | $8,546 · 35 wasted · 81.6% filled | $8,799 · 53 wasted · 85.9% filled |
| 31 | Elm Street | $6,709 · 241 wasted · 88.1% filled | $7,439 · 33 wasted · 83.8% filled | $7,586 · 46 wasted · 87.3% filled |
| 31 | Harbor Point | $3,398 · 133 wasted · 81.7% filled | $3,882 · 20 wasted · 80.1% filled | $3,938 · 28 wasted · 83.1% filled |
| 31 | Maple Grove | $6,946 · 253 wasted · 83.9% filled | $7,788 · 34 wasted · 80.7% filled | $7,932 · 48 wasted · 84.0% filled |
| 41 | Elm Street | $6,569 · 233 wasted · 88.3% filled | $7,345 · 30 wasted · 83.5% filled | $7,476 · 42 wasted · 86.7% filled |
| 41 | Harbor Point | $7,129 · 339 wasted · 88.4% filled | $8,372 · 32 wasted · 83.2% filled | $8,538 · 45 wasted · 86.7% filled |
| 41 | Maple Grove | $3,728 · 179 wasted · 87.7% filled | $4,319 · 27 wasted · 82.9% filled | $4,404 · 35 wasted · 86.5% filled |

Run time 23 s on one core.
