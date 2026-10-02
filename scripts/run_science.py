"""Run the fixed scientific experiments; fail on invalid numerical results."""
import argparse,json,math,subprocess
from pathlib import Path
from science import load_numeric_csv,invariant_summary,convergence_ratios

def run(exe,output,cpu_only=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    commands=[];epsilon=.01;speed=math.sqrt(.5/(1+epsilon**2)**1.5);period=math.pi/speed
    def simulate(name,**kwargs):
        folder=output/name
        args=[str(exe),'simulate','--output',str(folder)]
        for key,value in kwargs.items():args.extend(['--'+key.replace('_','-'),str(value)])
        p=subprocess.run(args,capture_output=True,text=True);commands.append(dict(command=args,stdout=p.stdout,stderr=p.stderr,exit_code=p.returncode))
        if p.returncode:raise RuntimeError(p.stdout+p.stderr)
        return folder
    convergence=[]
    for steps in (128,256,512):
        folder=simulate(f'convergence-{steps}',precision='double',steps=steps,dt=period/steps,sample_every=steps)
        rows=load_numeric_csv(folder/'trajectory.csv');initial=rows[:2];final=rows[-2:]
        error=math.sqrt(sum((a[k]-b[k])**2 for a,b in zip(initial,final) for k in ('x','y','z','vx','vy','vz')))
        convergence.append(dict(steps=steps,dt=period/steps,error=error))
    ratios=convergence_ratios([r['error'] for r in convergence])
    if not all(3.5<=r<=4.5 for r in ratios):raise RuntimeError(f'second-order convergence failed: {ratios}')
    backends=('cpu',) if cpu_only else ('cpu','cuda-basic','cuda-tiled');invariants={}
    for backend in backends:
        for precision in ('float','double'):
            name=f'{backend}-{precision}'
            folder=simulate(name,backend=backend,precision=precision,steps=5120,dt=period/512,sample_every=1)
            rows=load_numeric_csv(folder/'diagnostics.csv')
            summary=invariant_summary(rows,abs(rows[0]['energy']),2*speed,abs(rows[0]['lz']))
            if summary['max_energy']>1e-3 or summary['max_momentum']>1e-5 or summary['max_angular_momentum']>1e-3:raise RuntimeError(f'invariant check failed: {name}: {summary}')
            invariants[name]=summary
    simulate('euler-double',precision='double',integrator='euler',steps=5120,dt=period/512,sample_every=1)
    # A short agreement run uses exactly the float-serialized starting state.
    init=simulate('short-initial',fixture='cloud',n=17,steps=0,precision='float')/'initial-state.csv'
    short={}
    for backend in backends:
        folder=simulate(f'short-{backend}',backend=backend,precision='double',initial_state=init,steps=10,dt=.001,sample_every=1)
        short[backend]=load_numeric_csv(folder/'trajectory.csv')
    ref=short['cpu'];agreement={}
    for backend,rows in short.items():
        max_ratio=0
        for a,b in zip(rows,ref):
            for keys in (('x','y','z'),('vx','vy','vz')):
                err=math.sqrt(sum((a[k]-b[k])**2 for k in keys));scale=math.sqrt(sum(b[k]**2 for k in keys))
                max_ratio=max(max_ratio,err/(1e-10+1e-9*scale))
        if max_ratio>1:raise RuntimeError(f'short-run agreement failed: {backend}')
        agreement[backend]=max_ratio
    references=[]
    for steps in (4096,8192):
        folder=simulate(f'reference-{steps}',precision='double',initial_state=init,steps=steps,dt=.1/steps,sample_every=steps)
        references.append(load_numeric_csv(folder/'trajectory.csv')[-17:])
    ref_difference=math.sqrt(sum((a[k]-b[k])**2 for a,b in zip(*references) for k in ('x','y','z','vx','vy','vz')))
    coarse=simulate('reference-coarse',precision='double',initial_state=init,steps=128,dt=.1/128,sample_every=128)
    coarse_rows=load_numeric_csv(coarse/'trajectory.csv')[-17:]
    coarse_error=math.sqrt(sum((a[k]-b[k])**2 for a,b in zip(coarse_rows,references[-1]) for k in ('x','y','z','vx','vy','vz')))
    if ref_difference>coarse_error*.01:raise RuntimeError('refined reference is not sufficiently stable for coarse-error comparison')
    results=dict(period=period,convergence=convergence,convergence_ratios=ratios,invariants=invariants,short_agreement_max_normalized=agreement,reference_refinement_difference=ref_difference,reference_coarse_error=coarse_error)
    (output/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    (output/'commands.json').write_text(json.dumps(commands,indent=2))
    print(json.dumps(results,indent=2))
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('results/science'));p.add_argument('--cpu-only',action='store_true');a=p.parse_args()
    run(a.exe.resolve(),a.output.resolve(),a.cpu_only)
