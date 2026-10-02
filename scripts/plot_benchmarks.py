import argparse,json
from pathlib import Path
def plot(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(folder);rows=json.loads((folder/'summary.json').read_text())
    for mode in ('force','step','end-to-end'):
        fig,ax=plt.subplots(figsize=(7,4))
        for backend in ('cpu','cuda-basic','cuda-tiled'):
            data=sorted((r for r in rows if r['mode']==mode and r['backend']==backend and r['precision']=='float' and r['block_size']==128),key=lambda r:r['n'])
            if data:ax.loglog([r['n'] for r in data],[r['median_ms'] for r in data],'o-',label=backend)
        ax.set(xlabel='Particles N',ylabel='Median milliseconds per call/step',title=f'Float32 {mode}: block size 128');ax.legend();fig.tight_layout();fig.savefig(folder/(mode+'-scaling.png'),dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for backend in ('cuda-basic','cuda-tiled'):
        data=sorted((r for r in rows if r['mode']=='force' and r['backend']==backend and r['precision']=='float' and r['n']==4096),key=lambda r:r['block_size'])
        if data:ax.plot([r['block_size'] for r in data],[r['median_ms'] for r in data],'o-',label=backend)
    ax.set(xlabel='Threads per block',ylabel='Median force time (ms)',title='Block size and tiling: N=4096, float32');ax.legend();fig.tight_layout();fig.savefig(folder/'block-size.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for precision in ('float','double'):
        data=sorted((r for r in rows if r['mode']=='force' and r['backend']=='cuda-tiled' and r['precision']==precision and r['n'] in (512,2048) and r['block_size']==128),key=lambda r:r['n'])
        if data:ax.plot([r['n'] for r in data],[r['median_ms'] for r in data],'o-',label=precision)
    ax.set(xlabel='Particles N',ylabel='Median force time (ms)',title='Precision comparison: tiled, block size128');ax.legend();fig.tight_layout();fig.savefig(folder/'precision.png',dpi=180);plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',default='results/benchmarks');a=p.parse_args();plot(a.input)
