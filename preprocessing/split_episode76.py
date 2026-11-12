import os
import copy,csv,hashlib,json,pathlib
import cv2,h5py,numpy as np

root=pathlib.Path(os.environ['PIPER_DATA_WORKSPACE'])/'outputs/piper_65_trimmed'
backup=root.parent.parent/'work/episode76_before_split'
manifest=json.loads((root/'manifest.json').read_text());parent=next(e for e in manifest['episodes'] if e['id']==76)
parts=[]
with h5py.File(backup/'episode_000076.hdf5','r') as f:
 n=len(f['timestamp']);assert n==1692
 ct=f['control/t'][:];anchor=f['timing/anchor_monotonic'][:]
 for part,(eid,lo,hi) in enumerate([(76,0,752),(87,932,1692)],1):
  first=min(float(f['timing/control_monotonic'][lo]),float(f['timing/action_sent_monotonic'][lo]),float(anchor[lo]))
  last=max(float(f['timing/control_monotonic'][hi-1]),float(f['timing/action_sent_monotonic'][hi-1]),float(anchor[hi-1]))
  cl=max(0,int(np.searchsorted(ct,first,side='right')-1));ch=int(np.searchsorted(ct,last,side='right'))
  info=dict(source_episode_id=76,part=part,parent_trimmed_frames=n,parent_trimmed_start_frame_inclusive=lo,parent_trimmed_end_frame_exclusive=hi,original_source_start_frame_inclusive=19+lo,original_source_end_frame_exclusive=19+hi,reason='Two demonstrations separated by manual object reset; excluded parent frames [752,932).',timestamps_rebased=False)
  e=copy.deepcopy(parent);e.update(id=eid,file=f'raw/episode_{eid:06d}.hdf5',frames=hi-lo,source_episode_id=76,split_part=part,splitting=info)
  quality=copy.deepcopy(e['quality']);quality.update(frames=hi-lo,splitting=info);e['quality']=quality
  # Preserve the previous crop record as provenance; the effective original slice is explicit.
  e['trimming']=dict(parent_trimming=parent['trimming'],splitting=info,output_frames=hi-lo)
  e['quality']['trimming']=e['trimming']
  path=root/e['file']
  with h5py.File(path,'w') as g:
   for k,v in f.attrs.items():g.attrs[k]=v
   g.attrs['episode_id']=eid;g.attrs['source_episode_id']=76;g.attrs['split_part']=part
   g.attrs['trimming_json']=json.dumps(e['trimming']);g.attrs['splitting_json']=json.dumps(info);g.attrs['quality_json']=json.dumps(quality,ensure_ascii=False)
   def transfer(name,obj):
    if isinstance(obj,h5py.Group):target=g.require_group(name)
    else:
     a,b=(cl,ch) if name.startswith('control/') else (lo,hi)
     assert obj.shape[0]==(len(ct) if name.startswith('control/') else n),name
     data=obj[a:b];opts={}
     if obj.chunks:opts.update(chunks=tuple(min(x,y) for x,y in zip(obj.chunks,data.shape)),maxshape=obj.maxshape)
     if obj.compression:opts.update(compression=obj.compression,compression_opts=obj.compression_opts)
     target=g.create_dataset(name,data=data,dtype=obj.dtype,**opts)
    for k,v in obj.attrs.items():target.attrs[k]=v
   f.visititems(transfer)
  with h5py.File(path,'r') as g:
   def verify(name,obj):
    if not isinstance(obj,h5py.Dataset):return
    a,b=(cl,ch) if name.startswith('control/') else (lo,hi)
    x=f[name][a:b];y=obj[:];assert len(y)==b-a
    if h5py.check_dtype(vlen=obj.dtype) is not None:assert all(np.array_equal(u,v) for u,v in zip(x,y)),name
    else:assert np.array_equal(x,y,equal_nan=True),name
   g.visititems(verify)
  (root/'raw'/f'episode_{eid:06d}.quality.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2)+'\n')
  parts.append(e)
  print(eid,hi-lo,(hi-lo)/30,flush=True)
new=[]
for e in manifest['episodes']:
 new.extend(parts if e['id']==76 else [e])
manifest.update(episodes=new,next_id=88)
(root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(root/'provenance/episode76_split.json').write_text(json.dumps(dict(parent_episode_id=76,original_source_frames=1950,parent_trimmed_frames=1692,excluded_parent_frames=[752,932],excluded_seconds=6,parts=[e['splitting'] for e in parts],all_split_datasets_compared_to_parent=True),indent=2)+'\n')
# Keep the original 65-record trimming report and add explicit split columns.
with (root/'trim_report.csv').open() as f:
 reader=csv.DictReader(f);fields=reader.fieldnames+['split_part','parent_trimmed_start_frame','parent_trimmed_end_frame','split_gap_removed_frames'];rows=list(reader)
updated=[]
for row in rows:
 if row['episode_id']!='76':updated.append(row);continue
 for e in parts:
  r=copy.deepcopy(row);r.update(episode_id=e['id'],kept_frames=e['frames'],split_part=e['split_part'],parent_trimmed_start_frame=e['splitting']['parent_trimmed_start_frame_inclusive'],parent_trimmed_end_frame=e['splitting']['parent_trimmed_end_frame_exclusive'],split_gap_removed_frames=180 if e['split_part']==1 else 0)
  with h5py.File(root/e['file']) as f:r['control_samples_kept']=len(f['control/t'])
  updated.append(r)
with (root/'trim_report.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fields);w.writeheader();w.writerows(updated)
sha=[]
for e in new:
 h=hashlib.sha256()
 with (root/e['file']).open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 sha.append(f"{h.hexdigest()}  {e['file']}\n")
(root/'SHA256SUMS').write_text(''.join(sha))
verification=json.loads((root/'verification.json').read_text());verification.update(episodes=66,source_episodes=65,kept_frames=sum(e['frames'] for e in new),split_gap_removed_frames=180,split_output_datasets_compared_to_parent=True)
(root/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
# Reuse the existing exporter function, without rerunning the complete batch.
script=(root.parent.parent/'work/export_videos.py').read_text()
ns={};exec(script.split('with concurrent.futures.ThreadPoolExecutor')[0],ns)
records=json.loads((root/'videos/video_manifest.json').read_text());vparts=[ns['export'](e) for e in parts]
videos=[]
for r in records:videos.extend(vparts if r['episode_id']==76 else [r])
(root/'videos/video_manifest.json').write_text(json.dumps(videos,indent=2)+'\n')
for r in vparts:
 c=cv2.VideoCapture(str(root/'videos'/r['file']));assert int(c.get(cv2.CAP_PROP_FRAME_COUNT))==r['frames'];assert c.read()[0];c.set(cv2.CAP_PROP_POS_FRAMES,r['frames']-1);assert c.read()[0];c.release()
html=(root/'videos/index.html').read_text().replace('65 段','66 段')
old=next(r for r in records if r['episode_id']==76)
old_option=f'<option value="{old["file"]}">Episode {old["episode_id"]:06d} · {old["seconds"]:.1f} 秒</option>'
options=''.join(f'<option value="{r["file"]}">Episode {r["episode_id"]:06d}（原76第{i}段） · {r["seconds"]:.1f} 秒</option>' for i,r in enumerate(vparts,1))
assert old_option in html
(root/'videos/index.html').write_text(html.replace(old_option,options))
v=json.loads((root/'videos/verification.json').read_text());v['videos']=66;(root/'videos/verification.json').write_text(json.dumps(v,indent=2)+'\n')
# The old contact sheet showed the unsplit episode; regenerate its 76 row.
from PIL import Image,ImageDraw
sheet=Image.open(root/'boundary_preview.jpg').convert('RGB')
with h5py.File(root/parts[0]['file']) as f:
 import io
 panel=Image.new('RGB',(640,270),'white');d=ImageDraw.Draw(panel);d.text((5,5),'Episode 76 (part 1) | first / last | split at manual reset',fill='black')
 for j,i in enumerate([0,parts[0]['frames']-1]):panel.paste(Image.open(io.BytesIO(bytes(f['observations/images/cam0'][i]))).resize((320,240)),(320*j,30))
 sheet.paste(panel,(0,540));sheet.save(root/'boundary_preview.jpg')
readme=(root/'README.txt').read_text().replace('65 段','66 段').replace('只选来源 manifest.json 中 status=accepted 的 66 段，保留原 episode 编号。','来源是 65 段 status=accepted 记录，其中原第76段拆分为76和87，共66段。')
readme+='\n第76段拆分更新：原裁剪视频 [0,25.0667) 秒为第76段，[31.0667,56.4) 秒为第87段，去掉中间6秒人工重置过程。两个片段分别752帧和760帧。所有图像、动作、状态、时间记录同步裁剪并逐值校验。trim_report.csv 中两个片段共享来源统计，不应累加其 source_frames；实际总帧数见 verification.json。目录名沿用 piper_65_trimmed，当前实际66段。\n'
(root/'README.txt').write_text(readme)
print('DONE 66 episodes, split durations:',[r['seconds'] for r in vparts],flush=True)
