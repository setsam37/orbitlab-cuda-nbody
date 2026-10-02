import argparse,csv,datetime,hashlib,json,platform,subprocess
from pathlib import Path
from benchmarks import summarize,speedup
from provenance import observe_build
def run(exe,output,cpu_only=False):
    exe=Path(exe).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    summaries=[];commands=[]
    configs=[]
    for precision,ns,blocks in [('float',[128,512,1003,1024,2048,4096],[64,128,256]),('double',[512,2048],[128])]:
        for n in ns:
            for mode in ('force','step','end-to-end'):
                configs.append((precision,n,mode,'cpu',128))
                if not cpu_only:
                    for backend in ('cuda-basic','cuda-tiled'):
                        for block in ([128] if mode=='end-to-end' else blocks):configs.append((precision,n,mode,backend,block))
    print(f'Benchmark configurations: {len(configs)}; 5 batches each',flush=True)
    for precision,n,mode,backend,block in configs:
        folder=output/f'{backend}-{precision}-{n}-{mode}-{block}'
        args=[str(exe),'benchmark','--precision',precision,'--n',str(n),'--mode',mode,'--backend',backend,'--block-size',str(block),'--output',str(folder)]
        p=subprocess.run(args,capture_output=True,text=True);commands.append(dict(command=args,stdout=p.stdout,stderr=p.stderr,exit_code=p.returncode))
        if p.returncode:raise RuntimeError(p.stdout+p.stderr)
        with (folder/'batches.csv').open(newline='') as f:raw=list(csv.DictReader(f))
        if len(raw)!=5:raise ValueError('expected five raw batches')
        result=dict(n=n,precision=precision,mode=mode,backend=backend,block_size=block,**summarize([float(r['per_call_ms']) for r in raw],n,mode))
        result['max_acceleration_error']=max(float(r['max_acceleration_error']) for r in raw);summaries.append(result)
    cpu={(r['n'],r['precision'],r['mode']):r for r in summaries if r['backend']=='cpu'}
    for r in summaries:r['cpu_over_backend']=speedup(cpu[(r['n'],r['precision'],r['mode'])],r)
    basic={(r['n'],r['precision'],r['mode'],r['block_size']):r for r in summaries if r['backend']=='cuda-basic'}
    for r in summaries:
        key=(r['n'],r['precision'],r['mode'],r['block_size'])
        r['basic_over_tiled']=basic[key]['median_ms']/r['median_ms'] if r['backend']=='cuda-tiled' else None
    fields=list(summaries[0])
    with (output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(summaries)
    (output/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False))
    (output/'commands.json').write_text(json.dumps(commands,indent=2))
    def command(args):
        try:
            p=subprocess.run(args,capture_output=True,text=True);return dict(command=args,exit_code=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip())
        except FileNotFoundError:return dict(command=args,status='unavailable')
    root=Path(__file__).resolve().parents[1]
    sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ('include','src','cuda','scripts','tests') for p in (root/directory).rglob('*') if p.suffix in ('.hpp','.cpp','.cuh','.cu','.py')}
    build=observe_build(exe)
    environment=dict(run_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),platform=platform.platform(),cpu=command(['lscpu']),gpu=command(['nvidia-smi','--query-gpu=name,compute_cap,driver_version','--format=csv,noheader']),nvcc=command([build['cuda_compiler'],'--version']) if build['cuda_compiler'] else dict(status='unknown'),compiler=command([build['cxx_compiler'],'--version']) if build['cxx_compiler'] else dict(status='unknown'),cmake=command(['cmake','--version']),build=build,seed=37,epsilon=.01,dt=.001,batches=5,current_source_sha256=sources)
    (output/'environment.json').write_text(json.dumps(environment,indent=2))
    print('BENCHMARK_SWEEP_OK',len(summaries),'configurations',flush=True)
    for r in summaries:
        if r['precision']=='float' and r['n']==4096 and r['block_size']==128:print(json.dumps(r),flush=True)
    return summaries
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('results/benchmarks'));p.add_argument('--cpu-only',action='store_true');a=p.parse_args();run(a.exe,a.output,a.cpu_only)
