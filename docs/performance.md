# Performance

Docker stack (one uvicorn process, one worker, Postgres 16) after the demo: 12 stores, 20 items, 43,680 sales rows, 40,832 lots,
33,136 units of waste in history. Reproduce with `make demo`, then `make load-test`.

## Reads: the strawberry forecast at Elm Street and the chain summary, closed loop, 10 s per level


| Concurrency | Requests | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|
| 1 | 1733 | 173.3 req/s | 5.5 ms | 8.5 ms | 11.6 ms | 0.00% |
| 8 | 1831 | 182.4 req/s | 37.6 ms | 70.7 ms | 120.1 ms | 0.00% |
| 32 | 1546 | 152.6 req/s | 166.9 ms | 249.4 ms | 300.5 ms | 0.00% |

Target from the blueprint: p95 under 500 ms for reads: met. A forecast call fits nothing; the demand model is fitted once per chain
and kept in memory, so each call is a few dot products plus the database checks of store and item.

## Compute (in-process, one core)

| Work | Time |
|---|---|
| Generate the chain: weather, demand, 26 weeks of the ordering rule, every lot's life | 0.1 s |
| Censored demand fit (43,680 rows, 303 coefficients, L-BFGS with an analytic gradient) | 0.8 s |
| Shelf-life fit, seven categories, interval- and right-censored Weibull | 0.6 s |
| Markdown dynamic program, 266 units, 5 days, 5 discounts | 0.09 s |
| Markdown plan with five policies scored 1,500 times each on the truth | 0.2 s |
| Replenishment for 20 items: six service levels in the model's world, then three plans and the sweep on the truth | 2.6 s |
| Chain load job (generate, fit, write 43,680 sales and 40,830 lots) | a few seconds |
| Evaluation: three chains, nine markdown cases, nine store-weeks | about 22 s |

Not measured: hundreds of stores and thousands of items. The demand fit grows with rows and would want minibatches or a
per-category split; replenishment simulates every item of a store together and scales linearly with items and runs.
