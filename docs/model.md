# How the simulator works

Everything here was written against the PostgreSQL 16 source, with the
version differences for 13 to 18 noted where the simulator cares about them.
File and function names are the ones to grep for if you want to check a claim.

## Time and workload

Every rate (inserts, updates, deletes per table, and xids for the cluster) is
piecewise constant per hour. The simulator never asks for a rate directly. It
keeps the cumulative count at each hour boundary and interpolates, so "how
many dead rows by time t" and "when does this table cross its threshold" are
both exact lookups, one with `bisect`. That's what makes it fast enough to run
a quarter in a few seconds: a table's state is only computed when an event
looks at it.

A fitted rate is `base * (1 + growth * days) * hod[hour] * dow[weekday]` in
the profile's time zone. The fitter tries flat, daily and weekly shapes and
keeps the lowest BIC. Growth is fitted on daily totals and only kept when its
t-stat is above 3 with at least 14 days of data. Fitting it on hourly points
found trends in pure noise, because hourly residuals are strongly
autocorrelated.

## Launcher

`autovacuum.c`: `launcher_determine_sleep`, `rebuild_database_list`,
`do_start_worker`, `launch_worker`.

- The database list gets `next_worker` times spread across one naptime. The
  launcher sleeps until the soonest one (minimum 100ms, maximum 300s).
- At wake-up with a free slot it picks a database: the one with the oldest
  datfrozenxid if any is past `autovacuum_freeze_max_age`, otherwise the
  least recently autovacuumed one that isn't due within the next naptime.
  That database's `next_worker` becomes now + naptime.
- With no free slot it just waits. A worker exiting wakes it, but it only
  launches if a `next_worker` time has come due, so a busy cluster loses
  launch opportunities instead of queueing them.
- With one database and nothing to do, the simulator skips ahead to the
  first naptime tick after the next threshold crossing. A real worker would
  start, find nothing and exit, which changes nothing we track.

## Workers

`do_autovacuum`, `table_recheck_autovac`, `relation_needs_vacanalyze`.

A worker builds its table list once, when it starts: every table over a
threshold at that moment, in pg_class order (the profile's `oid`). It then
walks the list, rechecks each table right before touching it, and skips any
table another worker has claimed. A table that crosses its threshold while
the list is being worked waits for the next worker.

Thresholds:

```
vacuum   n_dead_tup         > vacuum_threshold + vacuum_scale_factor * reltuples
                              (capped at autovacuum_vacuum_max_threshold in 18)
insert   n_ins_since_vacuum > insert_threshold + insert_scale_factor * reltuples
                              (times the unfrozen share of the table in 18)
analyze  n_mod_since_analyze > analyze_threshold + analyze_scale_factor * reltuples
forced   age(relfrozenxid)  > autovacuum_freeze_max_age
```

A table with `autovacuum_enabled = off`, or a cluster with `autovacuum = off`,
still gets the forced anti-wraparound vacuum.

## The shared cost budget

`AutoVacuumUpdateCostLimit`, `autovac_recalculate_workers_for_balance`,
`vacuum_delay_point`.

Each running worker whose table has no per-table cost settings gets
`cost_limit / n` (integer division), where `n` counts those same workers.
A table with its own `autovacuum_vacuum_cost_limit` or
`autovacuum_vacuum_cost_delay` takes its worker out of the pool, and that
worker gets the full value. So three workers on a 200 limit each run at
66, six workers at 33, and total vacuum I/O stays the same: adding workers
without raising the limit doesn't make anything finish sooner. It also
means a per-table cost setting adds I/O on top of the server-wide budget,
which surprises people.

A worker in failsafe mode stops sleeping but keeps its balance flag, so it
still takes a share from everyone else.

Throughput per job:

```
seconds per credit = overhead * delay * min(1/limit, 4/balance) + 1/raw_rate
```

`balance` is the limit plus half a page's average cost (the overshoot before
the check fires). The `4/balance` term is the 4x cap on a single nap, which
only matters for silly small limits. `raw_rate` is how fast vacuum gets
through credits when nothing throttles it, and `overhead` is how much longer
`pg_usleep` really sleeps. Both get fitted from logs. Whenever a worker starts
or stops, every running job's rate changes and its finish time is
rescheduled.

## What one vacuum costs

`lazy_scan_heap`, `lazy_vacuum`, `lazy_check_wraparound_failsafe`.

Pages scanned:

- Non-aggressive: every page not marked all-visible, plus all-visible pages
  vacuum reads anyway because they're in a run shorter than
  `SKIP_PAGES_THRESHOLD` (32). The simulator keeps not-all-visible pages in
  two piles: a contiguous tail from appends, and pages scattered over the
  table's "active" region by updates and deletes. In the active region, if
  a page is skippable with probability q, the share of skippable pages in
  runs of 32 or more is `q^31 * (32 - 31q)`, and the rest get read.
- Aggressive (age past `vacuum_freeze_table_age`): every page not
  all-frozen, same rule for short runs. An anti-wraparound vacuum is usually
  aggressive as well, but only because `freeze_max_age` is normally above
  `freeze_table_age`. Give a table a low `autovacuum_freeze_max_age` and its
  forced runs are logged as "automatic vacuum to prevent wraparound" without
  "aggressive". The source calls that a corner case; the first lab run hit
  it.
- 18 adds eager scanning: a normal vacuum also reads all-visible but not
  frozen pages until it has frozen 20% of them, or failed on 3% of each
  4096-page region (`vacuum_max_eager_freeze_failure_rate`).

Index passes happen when there are dead items, unless the bypass applies
(fewer than 2% of pages have dead items and fewer than about 5.6M of them)
or the failsafe triggered. Each pass reads every page of every index. Up to
16 the number of passes is dead items * 6 bytes over `autovacuum_work_mem`
(capped at 1GB). From 17 the TID store is a radix tree; the simulator charges
48 bytes per heap page with dead items. That figure is an estimate.

Buffer accounting, checked against real 16 logs:

- reads = heap pages scanned + index pages + second heap pass, split into
  hits and misses by the table's `cache_hit`
- one extra hit per scanned page for the free space map lookup. On a cached
  table the logged hits come out at almost exactly 2x pages scanned, which
  is how this turned up
- dirtied = the not-all-visible pages that are clean by the time vacuum
  gets to them, plus all-visible pages it freezes, plus index pages that lost
  entries (non-HOT dead rows only, HOT ones have no index entries)

Which not-all-visible pages are clean comes from checkpoint timing. A page
changed more than `checkpoint_timeout` ago has been written out by a
checkpoint, and setting its all-visible bit dirties it again. One changed
more recently is usually still dirty in shared buffers, so it costs nothing
extra. With changes spread over the D seconds since the last vacuum, the
clean share is `1 - checkpoint_timeout / D`. On the append-only lab table,
whose vacuum interval stretched from 122s to 650s as it grew, the logs had
0.07%, 29% and 54% of scanned pages dirtied; the formula gives 0, 30 and 54.
A per-table constant fitted on the early runs was off by 3x on the later
ones.

Any vacuum, aggressive or not, freezes the pages it reads whose rows are all
older than the freeze cutoff. With the default 50M `vacuum_freeze_min_age`
that's rarely many pages. On top of that, 16 freezes some younger pages
opportunistically (`freeze_fraction`, fitted from runs where nothing old
could have been scanned).

Credits are `hits * page_hit + misses * page_miss + dirtied * page_dirty`,
the same formula the server meters against the limit.

ANALYZE reads up to 300 * `default_statistics_target` pages under the same
throttle, then spends CPU on the sample (`analyze_row_seconds` per row) with
no throttle at all.

## What a vacuum leaves behind

- Dead rows that existed when it started are removed, minus any pinned by a
  long transaction. Rows that died during the run stay.
- `n_dead_tup` is overwritten with what vacuum counted as recently dead
  (`pgstat_report_vacuum`). Rows that died behind the scan never show up, so
  after a long run the stat is low and the next vacuum comes late. The
  simulator assumes vacuum saw half of them. ANALYZE overwrites it again with
  a sample estimate, which corrects it.
- `reltuples` is updated, which moves every threshold.
- relfrozenxid moves if the run was aggressive or happened to read every
  page that isn't all-frozen. Since 15 the new value is the oldest xid still
  unfrozen after the run; before that it was the freeze cutoff. The simulator tracks
  the oldest unfrozen xid per table and starts it at relfrozenxid, since it
  can't see inside the heap.

## Space

A table's file grows when live plus dead rows no longer fit. Space freed by
vacuum gets reused first. The file never shrinks. Bloat is
`1 - (live rows / rows per page) / pages`, so a table that bloated during a
bad week keeps that bloat for the rest of the forecast, the same as it would
in production without `VACUUM FULL` or `pg_repack`.

Rows per page come from the average row width in `pg_stats`, plus the tuple
header and line pointer, and fillfactor. Without a width, the profiler
assumes the table is packed right now, which hides existing bloat.

## Long transactions

A `pin` in the profile (`start_hours`, `duration_hours`) holds the xmin
horizon. While it's held, vacuum can't remove anything that died after it
started, can't advance relfrozenxid past it, and leaves pages with those
rows not-all-visible. Because the pinned rows still count in `n_dead_tup`,
autovacuum keeps picking the same table every naptime and getting nowhere,
which is exactly what happens on a real server with an idle-in-transaction
session or a stale replication slot.

## Not modelled

- TOAST tables. The main table's numbers include none of it.
- Index bloat. Index sizes scale with the heap.
- Truncation of empty pages at the end of the heap.
- Autovacuum getting cancelled by a conflicting lock (DDL, `LOCK TABLE`).
  On a busy schema this matters and the simulator will be optimistic.
- Multixact wraparound. Only xid age is tracked.
- System catalogs, and their share of datfrozenxid.
- Shared buffers and OS cache as dynamic things. `cache_hit` is fixed per
  table.
- WAL volume, checkpoints, and replication lag caused by vacuum.
- Space reclaimed by on-access pruning of non-HOT dead rows. Dead rows take
  their full width until vacuum, so heap growth is a little pessimistic for
  tables whose pages get pruned often.
