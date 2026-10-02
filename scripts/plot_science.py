import argparse,json
from pathlib import Path
from science import load_numeric_csv
def plot(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(folder);summary=json.loads((folder/'summary.json').read_text())
    data=summary['convergence'];fig,ax=plt.subplots(figsize=(6,4))
    ax.loglog([r['dt'] for r in data],[r['error'] for r in data],'o-',label='Measured Verlet error')
    first=data[0];ax.loglog([r['dt'] for r in data],[first['error']*(r['dt']/first['dt'])**2 for r in data],'--',label='Second-order slope')
    ax.set(xlabel='Time step (dimensionless)',ylabel='Combined position/velocity error',title='Softened circular orbit: one period');ax.legend();fig.tight_layout();fig.savefig(folder/'convergence.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for name,label in (('cpu-double','Velocity Verlet (float64)'),('euler-double','Forward Euler (float64)')):
        rows=load_numeric_csv(folder/name/'diagnostics.csv');e0=rows[0]['energy']
        ax.plot([r['time']/summary['period'] for r in rows],[(r['energy']-e0)/abs(e0) for r in rows],label=label)
    ax.set(xlabel='Orbital periods',ylabel='Signed relative energy error',title='Same softened model and time step');ax.legend();fig.tight_layout();fig.savefig(folder/'energy-comparison.png',dpi=180);plt.close(fig)
    rows=load_numeric_csv(folder/'cpu-double'/'trajectory.csv');fig,ax=plt.subplots(figsize=(5,5))
    for body in (0,1):
        body_rows=[r for r in rows if r['id']==body];ax.plot([r['x'] for r in body_rows],[r['y'] for r in body_rows],label=f'Body {body}')
    ax.set(xlabel='x (dimensionless)',ylabel='y (dimensionless)',title='Equal masses: ten softened orbital periods',aspect='equal');ax.legend();fig.tight_layout();fig.savefig(folder/'orbit.png',dpi=180);plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',default='results/science');a=p.parse_args();plot(a.input)
