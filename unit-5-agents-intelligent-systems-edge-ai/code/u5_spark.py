# Task A4 / B3.3 - RDD pipeline: per-host alert statistics from raw connection records (NSL-KDD lines),
# batch-scored with the compressed edge model. Same shape as the A4 air-quality pipeline:
# (key, timestamp, value) -> per-key mean and maximum -> ten "noisiest" keys.
import time, numpy as np

N_HOSTS = 5000


CHUNK = 50000


def make_generator(lines_bc, ranges_bc):
    """Narrow source: partition p synthesises records [start, end) by resampling raw NSL-KDD test lines and
    tagging each with a host id and a timestamp. Randomness is seeded per fixed-size chunk of the global
    record index, so the data set is identical whatever the number of partitions."""
    def gen(pidx, _it):
        lines = lines_bc.value
        start, end = ranges_bc.value[pidx]
        for c in range(start // CHUNK, (end - 1) // CHUNK + 1 if end > start else 0):
            rng = np.random.default_rng(1000 + c)
            idx = rng.integers(0, len(lines), CHUNK); hosts = rng.integers(0, N_HOSTS, CHUNK)
            lo, hi = max(start, c * CHUNK), min(end, (c + 1) * CHUNK)
            for g in range(lo, hi):
                k = g - c * CHUNK
                yield f'{hosts[k]},{g},{lines[idx[k]]}'
    return gen


def parse(line):
    """'host,ts,<41 KDD fields>,label,difficulty' -> (host, ts, fields)."""
    p = line.split(',')
    return int(p[0]), int(p[1]), p[2:43]


def make_featuriser(vocab_bc, numeric_idx, log_idx):
    def featurise(rec):
        host, ts, f = rec
        vocab = vocab_bc.value
        x = [float(f[i]) for i in numeric_idx]
        for j in log_idx: x[j] = float(np.log1p(x[j]))
        for c, col in ((1, 'protocol_type'), (2, 'service'), (3, 'flag')):
            oh = [0.0] * len(vocab[col])
            k = vocab[col].get(f[c])
            if k is not None: oh[k] = 1.0
            x += oh
        return host, ts, x
    return featurise


def make_scorer(model_bc):
    def score_partition(it):
        model = model_bc.value
        batch = []
        for rec in it:
            batch.append(rec)
            if len(batch) == 20000:
                yield from _score(model, batch); batch = []
        if batch: yield from _score(model, batch)
    return score_partition


def _score(model, batch):
    X = np.array([r[2] for r in batch], dtype=np.float32)
    s = model.score(X)
    for (host, ts, _), v in zip(batch, s):
        yield host, (1, int(v >= 0.5), float(v), float(v))          # (count, alerts, sum score, max score)


def merge(a, b):
    return a[0] + b[0], a[1] + b[1], a[2] + b[2], max(a[3], b[3])


def build_pipeline(sc, lines, enc, model, n_records, partitions, cache=False):
    lines_bc = sc.broadcast(lines)
    bounds = [n_records * i // partitions for i in range(partitions + 1)]
    per_bc = sc.broadcast([(bounds[i], bounds[i + 1]) for i in range(partitions)])
    vocab_bc = sc.broadcast({c: {v: i for i, v in enumerate(enc.vocab[c])} for c in enc.vocab})
    numeric_idx = [i for i in range(41) if i not in (1, 2, 3)]
    log_idx = [enc.numeric.index(c) for c in enc.LOG]
    model_bc = sc.broadcast(model)
    raw = sc.parallelize(range(partitions), partitions).mapPartitionsWithIndex(make_generator(lines_bc, per_bc))
    records = raw.map(parse)                                         # narrow
    feats = records.map(make_featuriser(vocab_bc, numeric_idx, log_idx))   # narrow
    if cache == 'feats': feats.cache()                               # persist BEFORE children are defined,
    scored = feats.mapPartitions(make_scorer(model_bc))              # narrow  (PySpark otherwise fuses the
    if cache == 'scored': scored.cache()                             #          narrow chain into one task)
    per_host = scored.reduceByKey(merge, numPartitions=partitions)   # WIDE: shuffle by host
    stats = per_host.mapValues(lambda v: (v[0], v[1], v[1] / v[0], v[2] / v[0], v[3]))  # narrow
    return feats, scored, per_host, stats


def cached_mb(sc):
    infos = sc._jsc.sc().getRDDStorageInfo()
    return sum(i.memSize() for i in infos) / 2 ** 20


def run_once(sc, lines, enc, model, n, P, cache=False):
    """cache: False, 'scored' (small (host, 4-number) records re-used by both actions) or 'feats'
    (122-float feature vectors: large, and only the scoring step reads them)."""
    feats, scored, per_host, stats = build_pipeline(sc, lines, enc, model, n, P, cache)
    t0 = time.perf_counter()
    total_alerts = scored.map(lambda kv: kv[1][1]).sum()             # action 1
    t1 = time.perf_counter()
    top10 = stats.takeOrdered(10, key=lambda kv: -kv[1][1])          # action 2 (re-uses scored)
    t2 = time.perf_counter()
    shuffles = stats.toDebugString().decode().count('ShuffledRDD')
    mem = cached_mb(sc)
    for r in list(sc._jsc.getPersistentRDDs().values()): r.unpersist()
    return dict(n=n, partitions=P, cache=cache or 'none', action1_s=t1 - t0, action2_s=t2 - t1, total_s=t2 - t0,
                alerts=int(total_alerts), shuffles=shuffles, cached_MB=mem, top10=top10)
