import argparse,math
from pathlib import Path
from science import load_numeric_csv
def group_frames(rows):
    if not rows:raise ValueError('empty trajectory')
    required=('step','time','id','x','y','z','vx','vy','vz');groups={}
    for row in rows:
        if any(k not in row or not math.isfinite(row[k]) for k in required):raise ValueError('missing or nonfinite trajectory values')
        if row['step']<0 or row['step']!=int(row['step']) or row['id']<0 or row['id']!=int(row['id']) or row['time']<0:raise ValueError('invalid step, ID, or time')
        groups.setdefault(int(row['step']),[]).append(row)
    frames=[];ids=None;last_time=-1
    for step,bodies in sorted(groups.items()):
        bodies=sorted(bodies,key=lambda r:r['id']);current=[int(r['id']) for r in bodies]
        if current!=list(range(len(current))):raise ValueError('missing or duplicate body ID')
        if ids is None:ids=current
        if current!=ids:raise ValueError('body set changes across frames')
        time=bodies[0]['time']
        if any(r['time']!=time for r in bodies) or time<=last_time:raise ValueError('inconsistent or non-increasing frame time')
        frames.append(dict(step=step,time=time,bodies=bodies));last_time=time
    return frames
def animate(trajectory,output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    all_frames=group_frames(load_numeric_csv(trajectory))
    indexes=np.linspace(0,len(all_frames)-1,min(240,len(all_frames)),dtype=int)
    frames=[all_frames[i] for i in indexes]
    positions=np.array([[[r['x'],r['y']] for r in f['bodies']] for f in frames])
    limit=max(.6,float(np.max(np.abs(positions)))*1.15)
    fig,ax=plt.subplots(figsize=(5,5));ax.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect='equal',xlabel='x (dimensionless)',ylabel='y (dimensionless)',title='OrbitLab: softened two-body gravity')
    colors=['#2463a6','#db8b21'];scatter=ax.scatter(positions[0,:,0],positions[0,:,1],s=70,c=[colors[i%2] for i in range(positions.shape[1])])
    trails=[ax.plot([],[],color=colors[i%2],alpha=.5,lw=1)[0] for i in range(positions.shape[1])];label=ax.text(.03,.97,'',transform=ax.transAxes,va='top')
    def update(k):
        scatter.set_offsets(positions[k])
        for body,trail in enumerate(trails):trail.set_data(positions[max(0,k-24):k+1,body,0],positions[max(0,k-24):k+1,body,1])
        label.set_text(f"t = {frames[k]['time']:.2f} | step {frames[k]['step']}")
        return [scatter,label,*trails]
    animation=FuncAnimation(fig,update,frames=len(frames),interval=1000/24,blit=True)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();animation.save(output,writer=PillowWriter(fps=24));plt.close(fig)
    print('ANIMATION_SAVED',output,len(frames),'frames')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trajectory',required=True);p.add_argument('--output',default='results/orbit.gif');a=p.parse_args();animate(a.trajectory,a.output)
