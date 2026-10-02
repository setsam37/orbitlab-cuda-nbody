import math,statistics
def summarize(times,n,mode):
    if not times or any(not math.isfinite(t) or t<=0 for t in times) or n<1 or mode not in ('force','step','end-to-end'):raise ValueError('invalid timing workload')
    median=statistics.median(times)
    return dict(median_ms=median,min_ms=min(times),max_ms=max(times),interactions_per_second=n*(n-1)/(median/1000) if mode=='force' else None)
def speedup(cpu,gpu):
    if any(cpu[key]!=gpu[key] for key in ('n','precision','mode')):raise ValueError('incompatible workloads')
    if any(not math.isfinite(r['median_ms']) or r['median_ms']<=0 for r in (cpu,gpu)):raise ValueError('invalid timing')
    return cpu['median_ms']/gpu['median_ms']
