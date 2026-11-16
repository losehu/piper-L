"""Conservative mesh AABB geometry, OBB self-collision and checked joint paths."""
import json,math,time
from pathlib import Path
import numpy as np

def rotation(rpy):
    x,y,z=rpy;cx,sx=math.cos(x),math.sin(x);cy,sy=math.cos(y),math.sin(y);cz,sz=math.cos(z),math.sin(z)
    return np.array([[cz*cy,cz*sy*sx-sz*cx,cz*sy*cx+sz*sx],[sz*cy,sz*sy*sx+cz*cx,sz*sy*cx-cz*sx],[-sy,cy*sx,cy*cx]])

def transform(xyz,rpy):
    t=np.eye(4);t[:3,:3]=rotation(rpy);t[:3,3]=xyz;return t

def overlap(a,b,padding=0.):
    """Separating-axis theorem for oriented bounding boxes, including contact."""
    ca,ra,ha=a;cb,rb,hb=b;ha=ha+padding;hb=hb+padding
    r=ra.T@rb;t=ra.T@(cb-ca);ar=np.abs(r)+1e-10
    if np.any(np.abs(t)>ha+ar@hb):return False
    if np.any(np.abs(r.T@t)>hb+ar.T@ha):return False
    for i in range(3):
        for j in range(3):
            i1,i2=(i+1)%3,(i+2)%3;j1,j2=(j+1)%3,(j+2)%3
            if abs(t[i2]*r[i1,j]-t[i1]*r[i2,j])>ha[i1]*ar[i2,j]+ha[i2]*ar[i1,j]+hb[j1]*ar[i,j2]+hb[j2]*ar[i,j1]:return False
    return True

class Model:
    def __init__(self,path):
        self.spec=json.loads(Path(path).read_text())
        if self.spec.get('schema')!='piper-l-leader-envelope-v1':raise RuntimeError('Invalid leader collision model')
        self.cancelled=lambda:False
        self.joints=self.spec['joints'];self.links=self.spec['links'];self.margin=float(self.spec['margin_m'])
        if len(self.joints)!=6 or not .01<=self.margin<=.05:raise RuntimeError('Invalid leader kinematics/margin')
        self.lower=np.array([j['lower'] for j in self.joints]);self.upper=np.array([j['upper'] for j in self.joints])
        self.origins=[transform(j['xyz'],j['rpy']) for j in self.joints]
        self.boxes=[(np.array(b['min']),np.array(b['max'])) for b in self.links]
        self.tool=self.spec['tool'];self.reach=float(self.spec['sweep_radius_bound_m'])
        self.exclusions={tuple(sorted(pair)) for pair in self.spec.get('excluded_pairs',[])}
        if (5,7) in self.exclusions:
            # Prove the attachment half-space exclusion from the actual body
            # bounds. Rotation about joint6's z axis cannot change this z bound.
            import itertools
            lo,hi=self.boxes[5];points=np.array(list(itertools.product(*zip(lo,hi))))
            inv=np.linalg.inv(self.origins[5]);points=points@inv[:3,:3].T+inv[:3,3]
            if points[:,2].max()>1e-6 or self.tool['min'][2]<0:
                raise RuntimeError('Teaching tool / wrist half-space exclusion is not valid')
    def frames(self,q):
        poses=[np.eye(4)]
        for origin,angle in zip(self.origins,q):
            spin=np.eye(4);spin[:3,:3]=rotation([0,0,float(angle)]);poses.append(poses[-1]@origin@spin)
        return poses
    def geometry(self,q):
        frames=self.frames(q);boxes=[]
        for pose,(lo,hi) in zip(frames,self.boxes):
            boxes.append((pose[:3,:3]@((lo+hi)/2)+pose[:3,3],pose[:3,:3],(hi-lo)/2))
        pose=frames[-1];lo,hi=np.array(self.tool['min']),np.array(self.tool['max'])
        boxes.append((pose[:3,:3]@((lo+hi)/2)+pose[:3,3],pose[:3,:3],(hi-lo)/2))
        return boxes
    def collision(self,q,extra=0.):
        if self.cancelled():raise RuntimeError('Leader path check cancelled by stop')
        q=np.asarray(q)
        if q.shape!=(6,) or not np.isfinite(q).all() or np.any(q<self.lower) or np.any(q>self.upper):return 'joint limits'
        boxes=self.geometry(q)
        for i,a in enumerate(boxes):
            # Fixed base is mounted on the table; moving geometry must stay above it.
            if i and a[0][2]-np.abs(a[1][2])@a[2]<extra-.002:return 'mounting plane'
            for j in range(i+1,len(boxes)):
                if j==i+1 or (i,j) in self.exclusions:continue
                if overlap(a,boxes[j],self.margin+extra):return '%s/%s'%(self.spec['names'][i],self.spec['names'][j])
        return None
    def edge(self,a,b):
        a,b=np.asarray(a),np.asarray(b)
        if self.collision(a) or self.collision(b):return False
        def segment(left,right,depth):
            sweep=self.reach*float(np.sum(np.abs(right-left)))
            if self.collision(left,sweep) is None and self.collision(right,sweep) is None:return True
            if depth>=18:return False
            middle=(left+right)/2
            if self.collision(middle):return False
            return segment(left,middle,depth+1) and segment(middle,right,depth+1)
        return segment(a,b,0)
    def plan(self,start,goal,budget_s=4.):
        start,goal=np.asarray(start),np.asarray(goal)
        for name,q in [('start',start),('goal',goal)]:
            reason=self.collision(q)
            if reason:raise RuntimeError('Leader %s collision: %s'%(name,reason))
        if self.edge(start,goal):return [start.copy(),goal.copy()]
        # Bounded deterministic RRT; every extension and shortcut is swept-checked.
        rng=np.random.default_rng(81);nodes=[start.copy()];parents=[-1];until=time.monotonic()+budget_s
        while time.monotonic()<until and len(nodes)<1500:
            sample=goal if rng.random()<.25 else rng.uniform(self.lower,self.upper)
            distances=[np.linalg.norm(n-sample) for n in nodes];i=int(np.argmin(distances));delta=sample-nodes[i]
            norm=np.linalg.norm(delta);candidate=nodes[i]+delta*min(1.,np.radians(12)/max(norm,1e-9))
            if not self.edge(nodes[i],candidate):continue
            nodes.append(candidate);parents.append(i);k=len(nodes)-1
            if np.linalg.norm(candidate-goal)>np.radians(30) or not self.edge(candidate,goal):continue
            path=[goal.copy()]
            while k>=0:path.append(nodes[k]);k=parents[k]
            path.reverse();return path
        raise RuntimeError('No collision-free leader path found; follower remains held')
