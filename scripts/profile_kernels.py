"""Optional Nsight Compute runs, separate from ordinary benchmark timing."""
import argparse,json,subprocess,sys
from pathlib import Path
def profile(exe,output,tool_cmd=None,backend='cuda-basic'):
    command=(tool_cmd or ['ncu'])+['--csv','--section','MemoryWorkloadAnalysis','--section','Occupancy','--launch-count','1','--kernel-name','regex:.*force.*',str(exe),'benchmark','--backend',backend,'--n','4096','--precision','float','--block-size','128','--mode','force','--output',str(output)]
    try:
        p=subprocess.run(command,capture_output=True,text=True,timeout=120)
    except FileNotFoundError as e:return dict(status='unavailable',command=command,stderr=str(e),stdout='',exit_code=None)
    except subprocess.TimeoutExpired as e:return dict(status='failed',command=command,stderr=str(e),stdout='',exit_code=None)
    status='ok' if p.returncode==0 else ('unavailable' if 'ERR_NVGPUCTRPERM' in p.stdout+p.stderr else 'failed')
    return dict(status=status,command=command,stdout=p.stdout,stderr=p.stderr,exit_code=p.returncode)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('results/profiling'));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    failed=False
    for backend in ('cuda-basic','cuda-tiled'):
        result=profile(a.exe.resolve(),a.output/backend,backend=backend)
        (a.output/(backend+'.json')).write_text(json.dumps(result,indent=2))
        print(backend,result['status'],result['stderr'][-1500:]+result['stdout'][-1500:])
        failed=failed or result['status']=='failed'
    sys.exit(1 if failed else 0)
