"""Single-GPU priority scheduling and continuous upstream RLT learning."""
import contextlib,dataclasses,fcntl,json,threading,time
import numpy as np
from rlt_piper.contract import CHUNK,normalize,upstream
from rlt_piper.checkpoint import current,publish
from rlt_piper.recording import transitions

KEYS=['z_rl','proprio','ref_chunk','action_chunk','rewards','done','next_z_rl','next_proprio','next_ref_chunk','source_chunk']

class PriorityGate:
    def __init__(self):self.condition=threading.Condition();self.busy=False;self.waiters=0
    @contextlib.contextmanager
    def inference(self):
        with self.condition:
            self.waiters+=1
            try:
                while self.busy:self.condition.wait()
                self.busy=True
            finally:self.waiters-=1
        try:yield
        finally:
            with self.condition:self.busy=False;self.condition.notify_all()
    @contextlib.contextmanager
    def learning(self):
        with self.condition:
            accepted=not self.busy and not self.waiters
            if accepted:self.busy=True
        try:yield accepted
        finally:
            if accepted:
                with self.condition:self.busy=False;self.condition.notify_all()

class Learner:
    def __init__(self,root,directory,policy,gate):
        import jax,jax.numpy as jnp
        from flax import serialization
        upstream(root)
        from rlt_online_rl.config import RLTOnlineRLConfig
        from rlt_online_rl.trainer import init_train_state,train_step
        self.jax,self.jnp,self.serialization=jax,jnp,serialization
        self.root,self.directory,self.policy,self.gate=root,directory,policy,gate
        self.lock=(directory/'learner.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.cfg=RLTOnlineRLConfig(z_dim=policy.meta['z_dim'],chunk_len=CHUNK,action_representation='delta_chunk',
                                  warmup_min_size=600,warmup_post_collect_updates=2000,fixed_std=.05)
        self.state,self.actor,self.critic=init_train_state(self.cfg,rng=jax.random.PRNGKey(42));self.settings=None
        self.seen=set();self.ready_at=None;self.persisted=0;self.rows=[];self.revisions={};self.cache={}
        path=current(directory)
        if (path/'learner.msgpack').exists():
            meta=json.loads((path/'actor.json').read_text())
            if meta['token_sha256']!=policy.meta['token_sha256']:raise RuntimeError('Learner token mismatch')
            self.state=serialization.from_bytes(self.state,(path/'learner.msgpack').read_bytes())
            self.settings=meta['execution_settings'];self.seen=set(meta.get('seen',[]));self.ready_at=meta.get('ready_at',600)
            self.persisted=int(self.state.global_step)
        self.train_step=train_step;self.stop_event=threading.Event();self.error=None;self.rng=np.random.default_rng(43)
        self.status=dict(status='waiting_for_replay',global_step=int(self.state.global_step),actor_version=policy.version)
        b=64;dummy={k:jnp.zeros(shape,dtype=dtype) for k,shape,dtype in [
            ('z_rl',(b,self.cfg.z_dim),jnp.float32),('proprio',(b,7),jnp.float32),('ref_chunk',(b,CHUNK,7),jnp.float32),
            ('action_chunk',(b,CHUNK,7),jnp.float32),('rewards',(b,CHUNK),jnp.float32),('done',(b,),jnp.float32),
            ('next_z_rl',(b,self.cfg.z_dim),jnp.float32),('next_proprio',(b,7),jnp.float32),
            ('next_ref_chunk',(b,CHUNK,7),jnp.float32),('source_chunk',(b,CHUNK),jnp.int32)]}
        # Compile AND block before server readiness. Discard synthetic state.
        warmed=train_step(self.state,dummy,actor=self.actor,critic=self.critic,rl_config=self.cfg,bc_weight=5.,q_weight=0.,delta_weight=0.)
        jax.block_until_ready(warmed)
    def scan(self):
        for path in sorted((self.root/'rlt_episodes').glob('episode_*')):
            manifest=path/'episode.json'
            if not manifest.exists():continue
            meta=json.loads(manifest.read_text());signature=(meta.get('revision'),manifest.stat().st_mtime_ns)
            if self.revisions.get(path)==signature:continue
            rows=transitions(path,self.policy.meta['token_sha256'])
            if rows:
                if self.settings is None:self.settings=meta['settings']
                if meta['settings']!=self.settings:raise RuntimeError('Replay RTC/filter settings mismatch')
            self.cache[path]=rows;self.revisions[path]=signature
        rows=[row for group in self.cache.values() for row in group]
        for i,row in enumerate(rows):self.seen.add(row.get('transition_id','legacy:%s'%i))
        # Bounded GPU sampling working set; disk journal retains complete history.
        self.rows=rows[-20000:]
    def batch(self):
        rows=self.rows;human=[r for r in rows if np.any(r['source_chunk']==2)]
        # Reserve half the batch for correction supervision when available.
        selected=[rows[i] for i in self.rng.integers(0,len(rows),size=64 if not human else 32)]
        if human:selected += [human[i] for i in self.rng.integers(0,len(human),size=32)]
        data={k:np.stack([r[k] for r in selected]) for k in KEYS}
        for key,state in [('ref_chunk','proprio'),('action_chunk','proprio'),('next_ref_chunk','next_proprio')]:
            data[key]=normalize(data[key],data[state],self.policy.norms['actions'])
        return {k:self.jnp.asarray(v,dtype=self.jnp.int32 if k=='source_chunk' else self.jnp.float32) for k,v in data.items()}
    def save(self):
        step=int(self.state.global_step)
        if not self.settings or step==self.persisted:return
        state=self.jax.tree.map(np.asarray,self.state);version=int(state.actor_version);ready=step>=2000
        payload=dict(version=version,actor_params=state.actor_params,rl_config=dataclasses.asdict(self.cfg),token_sha256=self.policy.meta['token_sha256'])
        info=dict(token_sha256=self.policy.meta['token_sha256'],actor_updates=version,global_step=step,
                  transitions=len(self.rows),execution_settings=self.settings,ready=ready,seen=sorted(self.seen),ready_at=self.ready_at,
                  episodes=[p.name for p in self.cache],stage='online_rl' if ready else 'bc_warmup',real_robot_evaluation='not yet validated')
        publish(self.directory,self.serialization.to_bytes(state),payload,info);self.persisted=step
    def start(self):
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def run(self):
        next_scan=0
        try:
            while not self.stop_event.wait(.05):
                now=time.monotonic()
                if now>=next_scan:self.scan();next_scan=now+2
                step=int(self.state.global_step)
                if len(self.rows)>=600 and self.ready_at is None:self.ready_at=len(self.seen)
                budget=0 if self.ready_at is None else 2000+max(0,len(self.seen)-self.ready_at)
                self.status=dict(status='training' if step<budget else 'waiting_for_replay',transitions=len(self.rows),
                                 human_transitions=sum(np.any(r['source_chunk']==2) for r in self.rows),global_step=step,
                                 actor_version=self.policy.version,ready=step>=2000)
                if len(self.rows)<600 or step>=budget:continue
                with self.gate.learning() as allowed:
                    if not allowed:continue
                    before=time.monotonic();batch=self.batch()
                    state,metrics=self.train_step(self.state,batch,actor=self.actor,critic=self.critic,rl_config=self.cfg,
                        bc_weight=5.,q_weight=0. if step<2000 else .1,delta_weight=0.)
                    self.jax.block_until_ready((state,metrics))
                    if not all(np.isfinite(float(v)) for v in metrics.values()):raise RuntimeError('Nonfinite learner metrics')
                    self.state=state;step=int(state.global_step)
                    if time.monotonic()-before>.05:raise RuntimeError('Learner step exceeded 50ms GPU budget; learning paused')
                    if step>=2000 and step%20==0 and self.policy.stage=='online':
                        payload=dict(version=int(state.actor_version),actor_params=state.actor_params,rl_config=dataclasses.asdict(self.cfg))
                        self.policy.pending=(payload,dict(execution_settings=self.settings))
                if step%100==0:self.save()
        except BaseException as e:
            self.error=repr(e);self.status=dict(status='learning_paused',error=self.error,global_step=int(self.state.global_step))
            print('RLT后台训练已暂停，当前策略继续：',self.error,flush=True)
        finally:
            try:self.scan();self.save()
            finally:self.lock.close()
    def close(self):
        self.stop_event.set()
        if hasattr(self,'thread'):
            self.thread.join(15)
            if self.thread.is_alive():raise RuntimeError('Learner shutdown timed out')
        else:self.lock.close()
