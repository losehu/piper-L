import os
import copy,csv,hashlib,json,pathlib,re,io
import cv2,h5py,numpy as np
from PIL import Image,ImageDraw

base=pathlib.Path(os.environ['PIPER_DATA_WORKSPACE']);root=base/'outputs/piper_65_trimmed'
manifest=json.loads((root/'manifest.json').read_text());e=next(e for e in manifest['episodes'] if e['id']==87)
lo=53
with h5py.File(base/'work/episode76_before_split/episode_000087_before_retrim.hdf5','r') as f:
 n=len(f['timestamp']);assert n==760
 ct=f['control/t'][:];anchor=f['timing/anchor_monotonic'][:]
 first=min(float(f['timing/control_monotonic'][lo]),float(f['timing/action_sent_monotonic'][lo]),float(anchor[lo]))
 cl=max(0,int(np.searchsorted(ct,first,side='right')-1))
 info=dict(source_episode_id=76,part=2,parent_trimmed_frames=1692,parent_trimmed_start_frame_inclusive=985,parent_trimmed_end_frame_exclusive=1692,original_source_start_frame_inclusive=1004,original_source_end_frame_exclusive=1711,reason='Removed manual reset, wrist-only adjustment and waiting; retain first reach toward objects.',timestamps_rebased=False,additional_head_removed_frames=53,additional_head_removed_seconds=53/30)
 e.update(frames=n-lo,splitting=info)
 e['trimming']['splitting']=info;e['trimming']['output_frames']=n-lo
 e['quality'].update(frames=n-lo,splitting=info,trimming=e['trimming'])
 with h5py.File(root/e['file'],'w') as g:
  for k,v in f.attrs.items():g.attrs[k]=v
  g.attrs['splitting_json']=json.dumps(info);g.attrs['trimming_json']=json.dumps(e['trimming']);g.attrs['quality_json']=json.dumps(e['quality'],ensure_ascii=False)
  def transfer(name,obj):
   if isinstance(obj,h5py.Group):target=g.require_group(name)
   else:
    start=cl if name.startswith('control/') else lo
    assert obj.shape[0]==(len(ct) if name.startswith('control/') else n)
    data=obj[start:];opts={}
    if obj.chunks:opts.update(chunks=tuple(min(a,b) for a,b in zip(obj.chunks,data.shape)),maxshape=obj.maxshape)
    if obj.compression:opts.update(compression=obj.compression,compression_opts=obj.compression_opts)
    target=g.create_dataset(name,data=data,dtype=obj.dtype,**opts)
   for k,v in obj.attrs.items():target.attrs[k]=v
  f.visititems(transfer)
 with h5py.File(root/e['file']) as g:
  def check(name,obj):
   if not isinstance(obj,h5py.Dataset):return
   start=cl if name.startswith('control/') else lo
   a=f[name][start:];b=obj[:]
   if h5py.check_dtype(vlen=obj.dtype) is not None:assert all(np.array_equal(u,v) for u,v in zip(a,b)),name
   else:assert np.array_equal(a,b,equal_nan=True),name
  g.visititems(check)
  nc=len(g['control/t'])
(root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(root/'raw/episode_000087.quality.json').write_text(json.dumps(e['quality'],ensure_ascii=False,indent=2)+'\n')
split=json.loads((root/'provenance/episode76_split.json').read_text());split.update(excluded_parent_frames=[752,985],excluded_seconds=233/30);split['parts'][1]=info
(root/'provenance/episode76_split.json').write_text(json.dumps(split,indent=2)+'\n')
with (root/'trim_report.csv').open() as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
for r in rows:
 if r['episode_id']=='76':r['split_gap_removed_frames']=233
 if r['episode_id']=='87':r.update(kept_frames=707,parent_trimmed_start_frame=985,control_samples_kept=nc)
with (root/'trim_report.csv').open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
h=hashlib.sha256()
with (root/e['file']).open('rb') as f:
 for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
lines=(root/'SHA256SUMS').read_text().splitlines();lines=[f"{h.hexdigest()}  {e['file']}" if line.endswith('  '+e['file']) else line for line in lines]
(root/'SHA256SUMS').write_text('\n'.join(lines)+'\n')
v=json.loads((root/'verification.json').read_text());v.update(kept_frames=sum(x['frames'] for x in manifest['episodes']),split_gap_removed_frames=233,episode87_additional_head_removed_frames=53)
(root/'verification.json').write_text(json.dumps(v,indent=2)+'\n')
script=(base/'work/export_videos.py').read_text();ns={};exec(script.split('with concurrent.futures.ThreadPoolExecutor')[0],ns)
record=ns['export'](e)
vm=json.loads((root/'videos/video_manifest.json').read_text());old=next(x for x in vm if x['episode_id']==87);vm=[record if x['episode_id']==87 else x for x in vm]
(root/'videos/video_manifest.json').write_text(json.dumps(vm,indent=2)+'\n')
c=cv2.VideoCapture(str(root/'videos'/record['file']));assert int(c.get(cv2.CAP_PROP_FRAME_COUNT))==707;assert c.read()[0];c.set(cv2.CAP_PROP_POS_FRAMES,706);assert c.read()[0];c.release()
html=(root/'videos/index.html').read_text();html=html.replace(f'（原76第2段） · {old["seconds"]:.1f} 秒',f'（原76第2段） · {record["seconds"]:.1f} 秒');(root/'videos/index.html').write_text(html)
readme=(root/'README.txt').read_text();readme=re.sub(r'第76段拆分更新：.*?(?=\n|$)','第76段拆分更新：原裁剪视频 [0,25.0667) 秒为第76段，[32.8333,56.4) 秒为第87段，去掉中间7.7667秒人工重置、腕部调整和等待。两个片段分别752帧和707帧。所有图像、动作、状态、时间记录同步裁剪并逐值校验。trim_report.csv 中两个片段共享来源统计，不应累加其 source_frames；实际总帧数见 verification.json。目录名沿用 piper_65_trimmed，当前实际66段。',readme)
(root/'README.txt').write_text(readme)
sheet=Image.new('RGB',(1280,530),'white');d=ImageDraw.Draw(sheet)
for row,eid in enumerate([76,87]):
 with h5py.File(root/f'raw/episode_{eid:06d}.hdf5') as f:
  frames=len(f['timestamp']);d.text((5,row*265+5),f'Episode {eid}: {frames/30:.3f}s | first / last',fill='black')
  for col,k in enumerate([0,frames-1]):
   for cam in range(2):
    im=Image.open(io.BytesIO(bytes(f[f'observations/images/cam{cam}'][k]))).resize((320,240));sheet.paste(im,(col*640+cam*320,row*265+25))
sheet.save(root/'episode76_split_preview.jpg')
print('DONE',record,flush=True)
