import os
import csv,io,json,pathlib
import cv2,h5py,numpy as np
from PIL import Image,ImageDraw
base=pathlib.Path(os.environ['PIPER_DATA_WORKSPACE']);root=base/'outputs/piper_65_trimmed';out=base/'outputs/quality_review';out.mkdir(exist_ok=True)
es=json.loads((root/'manifest.json').read_text())['episodes'];rows=[];sheets=[];panels=[]
def pct(x,p):return float(np.percentile(x,p))
for e in es:
 with h5py.File(root/e['file']) as f:
  q=f['observations/qpos'][:];a=f['action'][:];t=f['timestamp'][:];n=len(t)
  dt=np.diff(t);v=np.abs(np.diff(q[:,:6],axis=0))/dt[:,None]
  err=np.abs(a[:,:6]-q[:,:6])*180/np.pi
  camdiff=np.abs(f['timing/cam0/capture_monotonic'][:]-f['timing/cam1/capture_monotonic'][:])*1000
  skew=(f['timing/anchor_monotonic'][:]-f['timing/action_sent_monotonic'][:])*1000
  nums=[]
  def inspect(name,obj):
   if isinstance(obj,h5py.Dataset) and obj.dtype.kind in 'fiu':
    if not np.all(np.isfinite(obj[:])):nums.append(name)
  f.visititems(inspect)
  frameissues={}
  for cam in ['cam0','cam1']:
   delta=np.diff(f[f'timing/{cam}/frame_number'][:]);frameissues[cam]=dict(reused=int(np.sum(delta==0)),backwards=int(np.sum(delta<0)),skips=int(np.sum(np.maximum(delta-1,0))))
  sm=np.median(np.lib.stride_tricks.sliding_window_view(np.pad(q,((2,2),(0,0)),mode='edge'),5,axis=0),axis=-1)
  speed=np.abs(sm[3:]-sm[:-3])/(t[3:]-t[:-3])[:,None];moving=(speed[:,:6].max(axis=1)>.03)|(speed[:,6]>.005)
  marks=np.diff(np.r_[True,moving,True].astype(int));starts=np.where(marks==-1)[0];ends=np.where(marks==1)[0]
  pauses=[dict(start_seconds=float(s/30),duration_seconds=float((z-s)/30)) for s,z in zip(starts,ends) if z-s>=21]
  # Hysteresis transitions, requiring at least three samples in the new state.
  grip=sm[:,6];state='open' if grip[0]>.06 else 'closed';events=[]
  for i in range(2,n):
   nextstate='open' if np.all(grip[i-2:i+1]>.065) else 'closed' if np.all(grip[i-2:i+1]<.035) else state
   if nextstate!=state:events.append(dict(time_seconds=i/30,state=nextstate));state=nextstate
  row=dict(id=e['id'],frames=n,seconds=n/30,created=e['created'],nonfinite=nums,nonmonotonic_timestamp=int(np.sum(dt<=0)),camera_skew_p95_ms=pct(camdiff,95),camera_skew_max_ms=float(camdiff.max()),action_age_p95_ms=pct(skew,95),action_age_max_ms=float(skew.max()),joint_error_p95_deg=pct(err,95),joint_error_max_deg=float(err.max()),joint_speed_p99_deg_s=pct(v*180/np.pi,99),joint_speed_max_deg_s=float(v.max()*180/np.pi),joint_path_rad=float(np.abs(np.diff(sm[:,:6],axis=0)).sum()),frameissues=frameissues,pauses=pauses,gripper_events=events,close_count=sum(x['state']=='closed' for x in events),open_count=sum(x['state']=='open' for x in events))
  rows.append(row)
  panel=Image.new('RGB',(1600,196),'white');draw=ImageDraw.Draw(panel)
  draw.text((5,3),f"Ep {e['id']:03d} | {n/30:.1f}s | close/open {row['close_count']}/{row['open_count']} | joint error p95/max {row['joint_error_p95_deg']:.1f}/{row['joint_error_max_deg']:.1f} deg",fill='black')
  ids=np.linspace(0,n-1,8).astype(int)
  for j,k in enumerate(ids):
   im=Image.open(io.BytesIO(bytes(f['observations/images/cam0'][k]))).resize((200,150));panel.paste(im,(j*200,43));draw.text((j*200+4,25),f'{k/30:.1f}s',fill='black')
  panels.append(panel)
 print('screened',e['id'],flush=True)
for i in range(0,len(panels),6):
 group=panels[i:i+6];sheet=Image.new('RGB',(1600,196*len(group)),'white')
 for j,panel in enumerate(group):sheet.paste(panel,(0,j*196))
 name=f'overview_{i//6+1:02d}.jpg';sheet.save(out/name,quality=94);sheets.append(name)
(out/'metrics.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
fields=['id','frames','seconds','created','camera_skew_p95_ms','camera_skew_max_ms','action_age_p95_ms','action_age_max_ms','joint_error_p95_deg','joint_error_max_deg','joint_speed_p99_deg_s','joint_speed_max_deg_s','joint_path_rad','close_count','open_count']
with (out/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
print(json.dumps(dict(sheets=sheets,episodes=len(es))),flush=True)
