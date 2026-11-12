import os
import concurrent.futures, json, os, pathlib, subprocess, threading
import h5py, imageio_ffmpeg

root=pathlib.Path(os.environ['PIPER_DATA_WORKSPACE'])/'outputs/piper_65_trimmed'
out=root/'videos';out.mkdir(exist_ok=True)
episodes=json.loads((root/'manifest.json').read_text())['episodes']
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()

def export(e):
 target=out/f"episode_{e['id']:06d}.mp4"
 with h5py.File(root/e['file'],'r') as f:
  cams=[f['observations/images/'+c][:] for c in ['cam0','cam1']]
  fps=float(f.attrs['fps'])
 r,w=os.pipe()
 cmd=[ffmpeg,'-y','-hide_banner','-loglevel','error','-threads','1','-f','image2pipe','-framerate',str(fps),'-c:v','mjpeg','-i','pipe:0','-threads','1','-f','image2pipe','-framerate',str(fps),'-c:v','mjpeg','-i',f'pipe:{r}','-filter_complex_threads','1','-filter_complex','[0:v][1:v]hstack=inputs=2[v]','-map','[v]','-c:v','libx264','-preset','ultrafast','-crf','24','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(target)]
 p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE,pass_fds=(r,))
 os.close(r)
 errors=[]
 def feed(stream,frames):
  try:
   with stream:
    for frame in frames:stream.write(frame.tobytes())
  except Exception as exc:errors.append(str(exc))
 ts=[threading.Thread(target=feed,args=(p.stdin,cams[0])),threading.Thread(target=feed,args=(os.fdopen(w,'wb'),cams[1]))]
 for t in ts:t.start()
 err=p.stderr.read().decode()
 for t in ts:t.join()
 rc=p.wait()
 if rc or errors:raise RuntimeError((e['id'],rc,errors,err))
 print(f"episode {e['id']:06d}: {len(cams[0])} frames, {target.stat().st_size} bytes",flush=True)
 return dict(episode_id=e['id'],file=target.name,frames=len(cams[0]),fps=fps,seconds=len(cams[0])/fps,bytes=target.stat().st_size)

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 records=list(pool.map(export,episodes))
(out/'video_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
items=''.join(f'<option value="{r["file"]}">Episode {r["episode_id"]:06d} · {r["seconds"]:.1f} 秒</option>' for r in records)
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Piper 抓取数据视频</title>
<style>body{max-width:1280px;margin:32px auto;background:#17191d;color:#eee;font:18px sans-serif;padding:16px}video{width:100%;background:#000;margin-top:20px}select,button{font:inherit;padding:10px;margin-right:10px}p{color:#bbb}</style>
<h1>Piper 抓取数据 · 65 段裁剪版</h1><p>左侧 cam0，右侧 cam1。30 FPS 原速播放；仅裁剪首尾静止，中间停顿保留。</p>
<button id="prev">上一段</button><select id="pick">OPTIONS</select><button id="next">下一段</button>
<video id="player" controls preload="metadata" src="FIRST"></video>
<script>const pick=document.getElementById('pick'),player=document.getElementById('player');function load(){player.src=pick.value;player.load()}pick.onchange=load;document.getElementById('prev').onclick=()=>{pick.selectedIndex=Math.max(0,pick.selectedIndex-1);load()};document.getElementById('next').onclick=()=>{pick.selectedIndex=Math.min(pick.options.length-1,pick.selectedIndex+1);load()};</script></html>'''.replace('OPTIONS',items).replace('FIRST',records[0]['file'])
(out/'index.html').write_text(html)
print('DONE',len(records),sum(r['bytes'] for r in records),flush=True)
