"""Analysis functions kept separate from plotting and experiment execution."""
import csv,math
def convergence_ratios(errors):
    if len(errors)<2 or any(not math.isfinite(e) or e<=0 for e in errors):raise ValueError('errors must be finite and positive')
    return [a/b for a,b in zip(errors,errors[1:])]
def invariant_summary(rows,energy_scale,momentum_scale,angular_scale):
    if not rows or any(not math.isfinite(s) or s<=0 for s in (energy_scale,momentum_scale,angular_scale)):raise ValueError('positive characteristic scales required')
    if any(not math.isfinite(v) for r in rows for v in r.values()):raise ValueError('nonfinite diagnostics')
    initial=rows[0]
    energy=[abs(r['energy']-initial['energy'])/energy_scale for r in rows]
    vector=lambda r,keys:math.sqrt(sum((r[k]-initial[k])**2 for k in keys))
    return dict(max_energy=max(energy),final_energy=energy[-1],max_momentum=max(vector(r,('px','py','pz'))/momentum_scale for r in rows),max_angular_momentum=max(vector(r,('lx','ly','lz'))/angular_scale for r in rows))
def load_numeric_csv(path):
    with open(path,newline='') as file:
        rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(file)]
    if not rows or any(not math.isfinite(v) for row in rows for v in row.values()):raise ValueError('empty or nonfinite data')
    return rows
