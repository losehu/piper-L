import os
import copy,csv,hashlib,html,json,pathlib,shutil,statistics
base=pathlib.Path(os.environ['PIPER_DATA_WORKSPACE']);src=base/'outputs/piper_65_trimmed';out=base/'outputs/quality_review';clean=base/'outputs/piper_screened'
manifest=json.loads((src/'manifest.json').read_text());metrics=json.loads((out/'metrics.json').read_text());mm={r['id']:r for r in metrics}
manual={
13:('27–31 秒','人手在末次搬运时拉住/整理末端线缆，属于未记录在 action 中的外部介入。只能确认拉线，不能据此断言机器人本身抓取失败。',27),
16:('约 28.5–29.5 秒','任务完成后，人手进入盒内直接拿动西瓜。该尾段不能作为纯机器人示范；可通过额外尾裁挽救前半段，但本次未再改动来源。',28.5),
18:('约 20–24 秒','搬运西瓜前后出现人手拉线，训练观察中有外部介入，先不加入干净示范子集。',20)}
review={
6:('8–15 秒','瓶子抓取前多次对位和调整，整段 37.5 秒；建议与顺畅示范分开，是否属于必要对位需连续视频确认。',8),
7:('12–16、25–31 秒','橘子放置及香蕉抓放存在多次调整，整段 36.4 秒；有恢复/冗余动作嫌疑，不能只凭夹爪次数判失败。',12),
9:('4–6 秒','瓶子从直立变为横倒后再抓起，包含状态改变后的恢复操作；最终完成不等于适合干净示范集。',4),
10:('19–26 秒','西瓜搬运前后瓶子出现在桌面并发生位移，之后再对位抓瓶；包含明显额外整理/补抓过程，具体碰撞原因不作断言。',19),
15:('17–26 秒','先接近香蕉后改抓橘子，随后再次返回香蕉，存在额外选目标/对位过程；列为风格复核，不判定任务失败。',17),
17:('14–17 秒','先接近西瓜又转去抓瓶子，存在目标切换和额外绕行；列为风格复核，不判定任务失败。',14),
34:('3–8 秒','瓶子附近多次夹取/调整后才搬走，存在补抓嫌疑；最终物体入盒。',3),
37:('3–9 秒','瓶子在桌面改变位置后再次对位抓取，含额外调整过程；可作为恢复数据另存。',3),
39:('15–23 秒','瓶子附近持续多次接近和夹爪调整后才搬运，整段 32.7 秒；先放复核集。',15)}
decisions=[]
for e in manifest['episodes']:
 eid=e['id'];m=mm[eid]
 if eid in manual:status='exclude_from_clean';label='暂不入选：人工介入';span,reason,jump=manual[eid]
 elif eid in review:status='review';label='复核：调整/恢复动作';span,reason,jump=review[eid]
 else:status='candidate';label='保留候选';span='';reason='数值检查通过，抽帧未发现明显外部介入、漏放或额外重置；并非逐帧认证。';jump=0
 if eid==11:reason+=' 本段画面中主要是三件物品，未仅按固定四次抓放要求剔除。'
 decisions.append(dict(episode_id=eid,status=status,label=label,seconds=m['seconds'],evidence_time=span,reason=reason,jump_seconds=jump,file=e['file'],video=f'episode_{eid:06d}.mp4',visual_review='cam0_1Hz_detail' if (out/f'detail_{eid:06d}.jpg').exists() else 'cam0_8_uniform_frames',technical_metrics=m))
selected=[e for e in manifest['episodes'] if e['id'] not in manual and e['id'] not in review]
assert len(selected)==54
for folder in ['raw','videos']:(clean/folder).mkdir(parents=True,exist_ok=True)
source_hash={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (src/'SHA256SUMS').read_text().splitlines()}
sha=[]
for e in selected:
 rel=e['file'];dest=clean/rel;shutil.copy2(src/rel,dest)
 h=hashlib.sha256()
 with dest.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 assert h.hexdigest()==source_hash[rel],rel
 sha.append(f'{h.hexdigest()}  {rel}\n')
 q=pathlib.Path(rel).with_suffix('.quality.json');shutil.copy2(src/q,clean/q)
 video=f'videos/episode_{e["id"]:06d}.mp4';shutil.copy2(src/video,clean/video)
 print('selected',e['id'],flush=True)
newmanifest=copy.deepcopy(manifest);newmanifest['episodes']=copy.deepcopy(selected);newmanifest['quality_screening']=dict(source_episodes=66,candidates=54,manual_intervention_excluded=3,review_held_out=9,scope='Numerical checks on all samples; sparse visual inspection, not full video certification.',report='../quality_review/report.html')
for e in newmanifest['episodes']:e['screening_status']='candidate'
(clean/'manifest.json').write_text(json.dumps(newmanifest,ensure_ascii=False,indent=2)+'\n');(clean/'SHA256SUMS').write_text(''.join(sha))
v=json.loads((src/'videos/video_manifest.json').read_text());vids=[r for r in v if r['episode_id'] in {e['id'] for e in selected}]
(clean/'videos/video_manifest.json').write_text(json.dumps(vids,indent=2)+'\n')
orig_html=(src/'videos/index.html').read_text();options=''.join(f'<option value="{r["file"]}">Episode {r["episode_id"]:06d} · {r["seconds"]:.1f} 秒</option>' for r in vids)
import re
orig_html=re.sub(r'(<select id="pick">).*?(</select>)',lambda m:m.group(1)+options+m.group(2),orig_html,flags=re.S).replace('66 段','54 段候选').replace('Piper 抓取数据 · 54 段候选裁剪版','Piper 抓取数据 · 54 段筛选候选')
(clean/'videos/index.html').write_text(orig_html)
(out/'decisions.json').write_text(json.dumps(decisions,ensure_ascii=False,indent=2)+'\n')
fields=['episode_id','status','label','seconds','evidence_time','reason','visual_review']
with (out/'decisions.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(decisions)
for name,ids in [('candidate_ids',[e['id'] for e in selected]),('manual_intervention_ids',list(manual)),('review_ids',list(review))]:(out/(name+'.txt')).write_text('\n'.join(map(str,ids))+'\n')
total=sum(e['frames'] for e in selected)
summary=dict(source_segments=66,source_original_episodes=65,candidate_segments=54,candidate_frames=total,candidate_seconds=total/30,manual_intervention_ids=list(manual),review_ids=list(review),all_selected_hdf5_sha256_matches_source=True,source_files_changed=False,visual_scope='All 66 episodes: 8 uniformly sampled cam0 frames; 16 targeted episodes additionally sampled at 1 Hz; three intervention examples checked in both cameras. Not continuous playback or frame-by-frame review.',technical_findings=dict(nonfinite_episodes=0,nonmonotonic_timestamp_episodes=0,camera_frame_number_gaps_episodes=0,max_pair_skew_ms=max(r['camera_skew_max_ms'] for r in metrics),max_image_action_age_ms=max(r['action_age_max_ms'] for r in metrics)))
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');(clean/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
report=f'''筛选结果（2026-11-11）

原始 65 段通过记录经第76段拆分后为66段。本轮按较保守的干净示范标准：
- 54 段保留候选，另存 ../piper_screened。
- 3 段暂不纳入干净子集：13、16、18，有明确人手/拉线介入。
- 9 段复核：6、7、9、10、15、17、34、37、39，存在补抓/反复对位/目标切换等疑点。
候选子集共 {total} 帧、{total/30:.1f} 秒（{total/1800:.2f} 分钟）。原66段均保留未改。

检查范围与限制
所有数值样本检查有限值、时间顺序、相机帧号、图像/动作时间差；所有66段各看8张均匀取样图，重点16段进一步按1秒间隔看图，3个外部介入例子查看双相机原分辨率证据。不是逐帧审片，短暂掉落/人手可能漏检，54段应称保留候选，不能称全部已认证高质量。

数值结果
未发现非有限值、主时间戳倒退、相机帧号复用/倒退/跳号。最大双相机配对时间差 {summary['technical_findings']['max_pair_skew_ms']:.2f} ms，最大图像对应动作年龄 {summary['technical_findings']['max_image_action_age_ms']:.2f} ms。软阈值本身不证明动作质量。
action 与 qpos 是命令和测量反馈，存在控制跟踪滞后；最大差值没有直接当成坏数据阈值。夹爪开合次数也没有单独用于剔除，夹持不同物体宽度会影响迟滞计数。
早期前10段平均31.7秒、关节累计路径52.1 rad；后20段23.9秒、40.4 rad，表明效率/动作风格有变化，但任务布局不同，不能据此把早期数据一刀切掉。

外部介入
13：约27–31秒人手拉线；不能据此断言抓取失败，但不是纯机器人输入。
16：约28.5秒后人手拿动盒内西瓜，发生在任务末尾；进一步尾裁可能挽救，此轮先隔离。
18：约20–24秒人手拉线，搬运过程有外部作用。

恢复/调整数据不一定无价值。若目标是容错恢复，应单独保留并标记；此处仅从干净示范候选集中隔离。

report.html 可按类别过滤，并从疑点时间点播放视频；decisions.csv 为逐段决定；metrics.csv 为数值统计。所有时间指向当前裁剪版视频，编号87为原76的第二段。
'''
(out/'report.txt').write_text(report)
(clean/'README.txt').write_text('54 段筛选后候选数据\n\n'+report+'\n文件结构与来源相同。manifest.json 仅包含入选54段，raw/ 包含HDF5与质量说明，videos/index.html 可逐段看视频。所有入选HDF5均经过SHA-256拷贝校验。本子集独立复制，未修改或删除原66段数据。\n')
cards=[]
for d in decisions:
 eid=d['episode_id'];m=mm[eid];detail=f'detail_{eid:06d}.jpg';evidence=f'evidence_{eid:06d}.jpg'
 links=f'<a href="{detail}" target="_blank">1秒取样图</a>' if (out/detail).exists() else ''
 if (out/evidence).exists():links+=f' · <a href="{evidence}" target="_blank">双相机证据</a>'
 cards.append(f'''<article data-status="{d['status']}"><h3>第 {eid} 段 · {d['seconds']:.2f} 秒 <span class="{d['status']}">{d['label']}</span></h3><p>{html.escape(d['reason'])}</p><p>疑点：{html.escape(d['evidence_time']) or '未发现明显问题'}　{links}</p><button onclick="play('{d['video']}',{d['jump_seconds']},{eid})">从疑点播放</button> <a href="../piper_65_trimmed/videos/{d['video']}#t={d['jump_seconds']}" target="_blank">单独打开视频</a><small>相机差 max {m['camera_skew_max_ms']:.1f} ms · 动作年龄 max {m['action_age_max_ms']:.1f} ms · 关节误差 p95 {m['joint_error_p95_deg']:.1f}° · {d['visual_review']}</small></article>''')
overview=''.join(f'<a href="overview_{i:02d}.jpg" target="_blank">总览 {i}</a> ' for i in range(1,12))
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Piper 数据筛选</title><style>body{background:#16191f;color:#e5e9f0;max-width:1250px;margin:30px auto;padding:16px;font:16px/1.6 sans-serif}a{color:#7bc7ff}button,select{padding:9px;font:inherit;margin:4px}article{border:1px solid #414b5c;padding:14px;margin:14px 0;border-radius:10px}small{display:block;color:#aab7c9;margin-top:10px}.candidate{color:#8edbb0}.review{color:#ffd485}.exclude_from_clean{color:#ff9b9b}video{width:100%;max-height:470px;background:#000}#playerbox{position:sticky;top:0;background:#16191f;z-index:1;padding-bottom:10px}details{margin:18px 0}</style>
<h1>66 段重新筛选：54 候选 / 3 人工介入 / 9 复核</h1><p>按干净示范标准筛选，原66段未修改。候选仍需留意短暂失败：本轮是数值全查＋画面抽样，不是逐帧认证。</p><p><a href="../piper_screened/videos/index.html">看54段候选视频</a> · <a href="decisions.csv">逐段筛选表</a> · <a href="report.txt">方法与限制</a></p>
<details><summary>全部66段画面总览</summary>OVERVIEW</details><div id="playerbox"><strong id="playing">选一段开始复核</strong><video id="player" controls preload="none"></video></div>
<label>显示类别 <select id="filter" onchange="filter()"><option value="all">全部66段</option><option value="flagged" selected>12段隔离/复核</option><option value="exclude_from_clean">3段人工介入</option><option value="review">9段调整/恢复复核</option><option value="candidate">54段保留候选</option></select></label>CARDS
<script>function filter(){const v=document.getElementById('filter').value;document.querySelectorAll('article').forEach(a=>a.hidden=!(v==='all'||a.dataset.status===v||v==='flagged'&&a.dataset.status!=='candidate'))}function play(file,time,id){const p=document.getElementById('player');document.getElementById('playing').textContent='第 '+id+' 段';p.src='../piper_65_trimmed/videos/'+file;p.onloadedmetadata=()=>{p.currentTime=time;p.play()};p.load()}filter()</script></html>'''.replace('OVERVIEW',overview).replace('CARDS',''.join(cards))
(out/'report.html').write_text(page)
print(json.dumps(summary,ensure_ascii=False),flush=True)
