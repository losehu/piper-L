import os
import copy, csv, hashlib, io, json, pathlib, subprocess
import h5py
import numpy as np
from PIL import Image, ImageDraw

base=pathlib.Path(os.environ['PIPER_DATA_WORKSPACE'])
src=pathlib.Path(os.environ['PIPER_SOURCE_DIR'])
out=base/'outputs/piper_65_trimmed'
(out/'raw').mkdir(parents=True,exist_ok=True)
(out/'provenance').mkdir(exist_ok=True)
manifest=json.loads((src/'manifest.json').read_text())
episodes=[e for e in manifest['episodes'] if e['status']=='accepted']
assert len(episodes)==65
remote="""import pathlib,json,hashlib
root=pathlib.Path(CAPTURE_DATA_ROOT)
names=json.loads(input())
for name in names:
 p=root/name; a=p.stat(); h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 b=p.stat(); assert (a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns)
 print(json.dumps(dict(path=name,size=b.st_size,sha256=h.hexdigest())))
"""
# Pass names as literal script data, keeping the remote command free of filenames.
remote=remote.replace('CAPTURE_DATA_ROOT',repr(os.environ['PIPER_CAPTURE_DATA_ROOT']))
remote=remote.replace('names=json.loads(input())','names='+repr([e['file'] for e in episodes]))
lines=subprocess.check_output(['ssh','-o','BatchMode=yes',os.environ['PIPER_CAPTURE_SSH'],'python3 -'],input=remote.encode()).decode().splitlines()
hashes=[json.loads(x) for x in lines]
for r in hashes:
 p=src/r['path']; h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 assert p.stat().st_size==r['size'] and h.hexdigest()==r['sha256'],r['path']
(out/'provenance/source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
(out/'provenance/source_accepted_manifest.json').write_text(json.dumps({'episodes':episodes},ensure_ascii=False,indent=2)+'\n')
print('65 source files SHA256 verified',flush=True)
rows=[]; new_eps=[]; output_hashes=[]; preview=[]
for e in episodes:
 with h5py.File(src/e['file'],'r') as f:
  q=f['observations/qpos'][:]; t=f['timestamp'][:]; n=len(t)
  smooth=np.median(np.lib.stride_tricks.sliding_window_view(np.pad(q,((2,2),(0,0)),mode='edge'),5,axis=0),axis=-1)
  k=3; vel=np.abs(smooth[k:]-smooth[:-k])/(t[k:]-t[:-k])[:,None]
  active=np.flatnonzero((vel[:,:6].max(axis=1)>.03)|(vel[:,6]>.005))
  assert len(active),e['id']
  lo=int(active[0]); hi=int(active[-1]+k+1)
  anchor=f['timing/anchor_monotonic'][:]
  ct=f['control/t'][:]
  first=min(float(f['timing/control_monotonic'][lo]),float(f['timing/action_sent_monotonic'][lo]),float(anchor[lo]))
  last=max(float(f['timing/control_monotonic'][hi-1]),float(f['timing/action_sent_monotonic'][hi-1]),float(anchor[hi-1]))
  cl=max(0,int(np.searchsorted(ct,first,side='right')-1)); ch=min(len(ct),int(np.searchsorted(ct,last,side='right')))
  assert ch>cl
  info=dict(source_file=e['file'],source_frames=n,start_frame_inclusive=lo,end_frame_exclusive=hi,head_removed_frames=lo,tail_removed_frames=n-hi,output_frames=hi-lo,source_control_samples=len(ct),control_start_inclusive=cl,control_end_exclusive=ch,joint_speed_threshold_rad_s=.03,gripper_speed_threshold_m_s=.005,median_window_frames=5,speed_span_frames=3,timestamps_rebased=False)
  quality=copy.deepcopy(e['quality']); quality.update(source_frames=n,frames=hi-lo,metrics_scope='original untrimmed episode; frames updated after trimming',trimming=info)
  dst=out/e['file']
  with h5py.File(dst,'w') as g:
   for key,val in f.attrs.items():g.attrs[key]=val
   g.attrs['source_quality_json']=f.attrs.get('quality_json','{}')
   g.attrs['quality_json']=json.dumps(quality,ensure_ascii=False)
   g.attrs['trimming_json']=json.dumps(info)
   def transfer(name,obj):
    if isinstance(obj,h5py.Group):
     target=g.require_group(name)
    else:
     if name.startswith('control/'):
      assert obj.shape[0]==len(ct),(name,obj.shape)
      data=obj[cl:ch]
     elif obj.ndim and obj.shape[0]==n:data=obj[lo:hi]
     else:raise ValueError(('unclassified dataset',name,obj.shape))
     opts={}
     if obj.chunks:opts.update(chunks=tuple(min(a,b) for a,b in zip(obj.chunks,data.shape)),maxshape=obj.maxshape)
     if obj.compression:opts.update(compression=obj.compression,compression_opts=obj.compression_opts)
     target=g.create_dataset(name,data=data,dtype=obj.dtype,**opts)
    for key,val in obj.attrs.items():target.attrs[key]=val
   f.visititems(transfer)
  with h5py.File(dst,'r') as g:
   assert len(g['action'])==hi-lo
   def check(name,obj):
    if not isinstance(obj,h5py.Dataset):return
    start,end=(cl,ch) if name.startswith('control/') else (lo,hi)
    assert len(obj)==end-start,name
    # Read every output dataset and compare every value, including both JPEG streams.
    a=f[name][start:end]; b=obj[:]
    if h5py.check_dtype(vlen=obj.dtype) is not None:
     assert all(np.array_equal(x,y) for x,y in zip(a,b)),name
    else:assert np.array_equal(a,b,equal_nan=True),name
   g.visititems(check)
   for cam in ['cam0','cam1']:
    for idx in [0,len(g['timestamp'])-1]:
     im=Image.open(io.BytesIO(bytes(g['observations/images/'+cam][idx])));im.load()
   if e['id'] in [3,6,76,86]:
    panel=Image.new('RGB',(640,270),'white'); d=ImageDraw.Draw(panel)
    for j,idx in enumerate([0,len(g['timestamp'])-1]):
     im=Image.open(io.BytesIO(bytes(g['observations/images/cam0'][idx]))).resize((320,240))
     panel.paste(im,(j*320,30))
    d.text((5,5),f"Episode {e['id']} | first / last | cut {lo} + {n-hi} frames",fill='black');preview.append(panel)
  (out/'raw'/f"episode_{e['id']:06d}.quality.json").write_text(json.dumps(quality,ensure_ascii=False,indent=2)+'\n')
  ne=copy.deepcopy(e);ne.update(frames=hi-lo,quality=quality,trimming=info);new_eps.append(ne)
  row=dict(episode_id=e['id'],source_frames=n,kept_frames=hi-lo,head_removed_frames=lo,tail_removed_frames=n-hi,head_removed_seconds=lo/30,tail_removed_seconds=(n-hi)/30,control_samples_kept=ch-cl);rows.append(row)
  h=hashlib.sha256()
  with dst.open('rb') as fh:
   for c in iter(lambda:fh.read(8*1024*1024),b''):h.update(c)
  output_hashes.append(f"{h.hexdigest()}  {e['file']}\n")
  print(f"episode {e['id']:06d}: {n} -> {hi-lo}, removed {lo}+{n-hi}",flush=True)
(out/'manifest.json').write_text(json.dumps(dict(next_id=manifest['next_id'],episodes=new_eps),ensure_ascii=False,indent=2)+'\n')
with (out/'trim_report.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'SHA256SUMS').write_text(''.join(output_hashes))
if preview:
 sheet=Image.new('RGB',(640,270*len(preview)),'white')
 for i,p in enumerate(preview):sheet.paste(p,(0,i*270))
 sheet.save(out/'boundary_preview.jpg')
totals={k:sum(r[k] for r in rows) for k in ['source_frames','kept_frames','head_removed_frames','tail_removed_frames']}
totals.update(episodes=len(rows),source_sha256_verified=True,all_output_datasets_compared_to_source=True,both_cameras_boundary_jpegs_decoded=True)
(out/'verification.json').write_text(json.dumps(totals,indent=2)+'\n')
print(json.dumps(totals),flush=True)
