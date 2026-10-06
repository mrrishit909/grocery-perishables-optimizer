-- Domain model. Names follow the blueprint's schema; additions are marked (+). Days are calendar dates; the chain's clock is "today".

CREATE TABLE chain (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants UNIQUE, name text NOT NULL, seed int NOT NULL, today date NOT NULL, weather jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now());   -- (+) seed, clock and the weather forecast in force
SELECT enable_tenant_rls('chain');

CREATE TABLE store (store_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, name text NOT NULL, store_no int NOT NULL, UNIQUE (tenant_id, store_no));
SELECT enable_tenant_rls('store');

CREATE TABLE sku (sku_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, name text NOT NULL, category text NOT NULL, price numeric NOT NULL CHECK (price > 0), unit_cost numeric NOT NULL CHECK (unit_cost > 0), label_days int NOT NULL CHECK (label_days > 0), sku_no int NOT NULL, UNIQUE (tenant_id, sku_no));
SELECT enable_tenant_rls('sku');

CREATE TABLE inventory_lot (lot_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, store_id uuid NOT NULL REFERENCES store, sku_id uuid NOT NULL REFERENCES sku, received_on date NOT NULL, qty_received int NOT NULL CHECK (qty_received > 0), qty_on_hand int NOT NULL CHECK (qty_on_hand >= 0), label_expiry date NOT NULL, exposure_dh double precision NOT NULL, status text NOT NULL CHECK (status IN ('on_hand', 'sold_out', 'spoiled', 'date_expired')), source text NOT NULL, CHECK (label_expiry >= received_on));   -- (+) exposure: degree-hours above 4 C from the lot's logger
SELECT enable_tenant_rls('inventory_lot');
CREATE INDEX lot_on_hand ON inventory_lot (tenant_id, store_id, sku_id) WHERE status = 'on_hand';

CREATE TABLE temperature_sample (lot_id uuid NOT NULL REFERENCES inventory_lot, ts timestamptz NOT NULL, celsius real NOT NULL, tenant_id uuid NOT NULL REFERENCES tenants, PRIMARY KEY (lot_id, ts));
SELECT enable_tenant_rls('temperature_sample');

CREATE TABLE sale (store_id uuid NOT NULL REFERENCES store, sku_id uuid NOT NULL REFERENCES sku, day date NOT NULL, units int NOT NULL, price numeric NOT NULL, discount real NOT NULL, stockout boolean NOT NULL, tenant_id uuid NOT NULL REFERENCES tenants, PRIMARY KEY (store_id, sku_id, day));   -- (+) stockout: the shelf ran empty, so units is a floor on demand
SELECT enable_tenant_rls('sale');

CREATE TABLE expiration_event (lot_id uuid NOT NULL REFERENCES inventory_lot, day date NOT NULL, kind text NOT NULL CHECK (kind IN ('spoiled', 'date_expired')), units int NOT NULL, tenant_id uuid NOT NULL REFERENCES tenants, PRIMARY KEY (lot_id));
SELECT enable_tenant_rls('expiration_event');

CREATE TABLE waste_event (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, store_id uuid NOT NULL REFERENCES store, sku_id uuid NOT NULL REFERENCES sku, lot_id uuid REFERENCES inventory_lot, day date NOT NULL, units int NOT NULL, reason text NOT NULL, cost numeric NOT NULL);
SELECT enable_tenant_rls('waste_event');
CREATE INDEX waste_by_day ON waste_event (tenant_id, day);

CREATE TABLE forecast (store_id uuid NOT NULL REFERENCES store, sku_id uuid NOT NULL REFERENCES sku, target_day date NOT NULL, issued_at timestamptz NOT NULL, mean real NOT NULL, p10 real NOT NULL, p90 real NOT NULL, model_version text NOT NULL, weather text NOT NULL, tenant_id uuid NOT NULL REFERENCES tenants, PRIMARY KEY (store_id, sku_id, target_day, issued_at));
SELECT enable_tenant_rls('forecast');

CREATE TABLE replenishment_plan (plan_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, store_id uuid NOT NULL REFERENCES store, horizon_start date NOT NULL, days int NOT NULL, params jsonb NOT NULL, lines jsonb NOT NULL, summary jsonb NOT NULL, created_by uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now());
SELECT enable_tenant_rls('replenishment_plan');

CREATE TABLE markdown_action (action_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, store_id uuid NOT NULL REFERENCES store, sku_id uuid NOT NULL REFERENCES sku, day date NOT NULL, discount real NOT NULL CHECK (discount >= 0 AND discount < 1), status text NOT NULL CHECK (status IN ('planned', 'pending_approval', 'approved')), plan jsonb NOT NULL, created_by uuid NOT NULL, approved_by uuid, approved_at timestamptz);   -- (+) deep markdowns wait for a store manager
SELECT enable_tenant_rls('markdown_action');
