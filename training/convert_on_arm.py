"""Offline conversion only: reads existing files, never opens a robot/CAN device."""
import concurrent.futures,hashlib,json,os,pathlib
import av,cv2,h5py,numpy as np,pyarrow as pa,pyarrow.parquet as pq
from PIL import Image

ROOT=pathlib.Path(os.environ.get('PIPER_RUN_ROOT',str(pathlib.Path(__file__).resolve().parents[1]/'artifacts/training'))).expanduser().resolve()
LEROBOT_ROOT=pathlib.Path(os.environ.get('HF_LEROBOT_HOME',str(ROOT/'lerobot'))).expanduser().resolve()
SOURCE=pathlib.Path(os.environ.get('PIPER_SELECTED_DIR',str(ROOT/'selected'))).expanduser().resolve()
spec=json.loads(pathlib.Path(os.environ.get('PIPER_SPLIT_FILE',str(pathlib.Path(__file__).with_name('dataset_manifest.json')))).read_text())
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def stats(a):
 a=np.asarray(a,dtype=np.float64)
 return {**{k:np.atleast_1d(getattr(a,k)(axis=0)).tolist() for k in ['min','max','mean','std']},'count':[len(a)]}
def encode(dataset,path,start,end):
 path.parent.mkdir(parents=True,exist_ok=True)
 sums=np.zeros(3);sumsq=np.zeros(3);mins=np.ones(3);maxs=np.zeros(3);count=0
 with av.open(str(path),'w') as container:
  stream=container.add_stream('libx264',rate=30);stream.width=stream.height=224;stream.pix_fmt='yuv420p';stream.codec_context.thread_count=2;stream.options={'crf':'18','preset':'veryfast','g':'15'}
  for i in range(start,end):
   bgr=cv2.imdecode(dataset[i],cv2.IMREAD_COLOR)
   assert bgr is not None and bgr.shape==(480,640,3)
   rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
   # Same PIL bilinear resize and zero padding as OpenPI's resize_with_pad.
   resized=Image.fromarray(rgb).resize((224,168),resample=Image.BILINEAR)
   canvas=Image.new('RGB',(224,224),0);canvas.paste(resized,(0,28));rgb=np.asarray(canvas)
   x=rgb[::8,::8].reshape(-1,3).astype(np.float64)/255
   sums+=x.sum(0);sumsq+=(x*x).sum(0);mins=np.minimum(mins,x.min(0));maxs=np.maximum(maxs,x.max(0));count+=len(x)
   for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):container.mux(packet)
  for packet in stream.encode():container.mux(packet)
 with av.open(str(path)) as container:decoded=sum(1 for _ in container.decode(video=0))
 assert decoded==end-start,(str(path),decoded,end-start)
 mean=sums/count
 return {**{k:v.reshape(3,1,1).tolist() for k,v in dict(min=mins,max=maxs,mean=mean,std=np.sqrt(np.maximum(sumsq/count-mean*mean,0))).items()},'count':[end-start]}
def export_one(args):
 r,index,offset,output=args;a,b=r['start'],r['end'];n=b-a
 assert n==r['frames']
 with h5py.File(SOURCE/r['source'],'r') as f:
  assert f.attrs['complete'] and not f.attrs.get('diagnostic',False)
  state=np.asarray(f['observations/qpos'][a:b],np.float32);action=np.asarray(f['action'][a:b],np.float32)
  assert state.shape==action.shape==(n,7) and np.isfinite(state).all() and np.isfinite(action).all()
  cols={'observation.state':state,'action':action,'timestamp':np.arange(n,dtype=np.float32)/30,'frame_index':np.arange(n,dtype=np.int64),'episode_index':np.full(n,index,dtype=np.int64),'index':np.arange(offset,offset+n,dtype=np.int64),'task_index':np.zeros(n,dtype=np.int64)}
  arrays={k:pa.FixedSizeListArray.from_arrays(pa.array(v.ravel()),7) if v.ndim==2 else pa.array(v) for k,v in cols.items()}
  p=output/('data/chunk-000/episode_%06d.parquet'%index);p.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.table(arrays),p)
  st={k:stats(v) for k,v in cols.items()}
  for cam in ['cam0','cam1']:
   key='observation.images.'+cam;p=output/('videos/chunk-000/%s/episode_%06d.mp4'%(key,index));st[key]=encode(f['observations/images/'+cam],p,a,b)
 print('converted',r['partition'],r['id'],n,flush=True)
 return dict(episode_index=index,tasks=[spec['task']],length=n),dict(episode_index=index,stats=st),dict(r,episode_index=index)

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 LEROBOT_ROOT.mkdir(parents=True,exist_ok=True)
 cv2.setNumThreads(1)
 unique={r['source']:r['source_sha256'] for r in spec['records']}
 for rel,expected in unique.items():assert digest(SOURCE/rel)==expected,rel
 print('all original source hashes verified',flush=True)
 outputs=[]
 for partition in ['train','val']:
  target=LEROBOT_ROOT/'local'/('piper_grasp_'+partition);assert not target.exists(),str(target)
  output=target.with_name(target.name+'.partial');assert not output.exists();output.mkdir(parents=True)
  records=[r for r in spec['records'] if r['partition']==partition];jobs=[];total=0
  for i,r in enumerate(records):jobs.append((r,i,total,output));total+=r['frames']
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(export_one,jobs))
  features={k:dict(dtype='float32',shape=[7],names=['j1','j2','j3','j4','j5','j6','gripper_width']) for k in ['observation.state','action']}
  for cam in ['cam0','cam1']:features['observation.images.'+cam]=dict(dtype='video',shape=[224,224,3],names=['height','width','channel'],info={'video.height':224,'video.width':224,'video.codec':'h264','video.pix_fmt':'yuv420p','video.is_depth_map':False,'video.fps':30,'video.channels':3,'has_audio':False})
  for key in ['timestamp','frame_index','episode_index','index','task_index']:features[key]=dict(dtype='float32' if key=='timestamp' else 'int64',shape=[1],names=None)
  info=dict(codebase_version='v2.1',robot_type='piper_single_follower',total_episodes=len(records),total_frames=total,total_tasks=1,total_videos=len(records)*2,total_chunks=1,chunks_size=1000,fps=30,splits={'train':'0:%d'%len(records)},data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',features=features)
  meta=output/'meta';meta.mkdir();(meta/'info.json').write_text(json.dumps(info,indent=2))
  for filename,rows in [('episodes.jsonl',[v[0] for v in results]),('episodes_stats.jsonl',[v[1] for v in results]),('tasks.jsonl',[dict(task_index=0,task=spec['task'])])]:
   (meta/filename).write_text(''.join(json.dumps(r)+'\n' for r in rows))
  aggregate={}
  for key in features:
   ss=[v[1]['stats'][key] for v in results];weights=np.array([s['count'][0] for s in ss]);mean=sum(np.asarray(s['mean'])*w for s,w in zip(ss,weights))/weights.sum();var=sum((np.asarray(s['std'])**2+(np.asarray(s['mean'])-mean)**2)*w for s,w in zip(ss,weights))/weights.sum()
   aggregate[key]=dict(min=np.min([s['min'] for s in ss],axis=0).tolist(),max=np.max([s['max'] for s in ss],axis=0).tolist(),mean=mean.tolist(),std=np.sqrt(var).tolist(),count=[int(weights.sum())])
  (meta/'stats.json').write_text(json.dumps(aggregate));(output/'provenance.json').write_text(json.dumps([v[2] for v in results],indent=2));output.rename(target);outputs.append(str(target))
 lines=[]
 for p in sorted((LEROBOT_ROOT).rglob('*')):
  if p.is_file():lines.append('%s  %s\n'%(digest(p),p.relative_to(LEROBOT_ROOT)))
 (LEROBOT_ROOT/'SHA256SUMS').write_text(''.join(lines));(ROOT/'conversion_complete.json').write_text(json.dumps(dict(outputs=outputs,episodes=54,train=48,val=6,all_source_hashes_verified=True,all_video_frames_decoded=True),indent=2))
 print('COMPLETE',outputs,flush=True)
if __name__=='__main__':main()
