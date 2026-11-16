"""Reproduce conservative leader geometry from pinned official checkouts."""
import argparse,struct,xml.etree.ElementTree as E,json,hashlib,math,subprocess
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--urdf-source',type=Path,required=True);p.add_argument('--aloha-source',type=Path,required=True);p.add_argument('--sim-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
source=args.urdf_source;master=args.aloha_source
def bounds(p):
 b=p.read_bytes();n=struct.unpack_from('<I',b,80)[0];lo=[math.inf]*3;hi=[-math.inf]*3
 if len(b)!=84+n*50:raise RuntimeError('Unknown STL encoding')
 for i in range(n):
  v=struct.unpack_from('<9f',b,84+50*i+12)
  for j in range(3):lo[j]=min(lo[j],*v[j::3]);hi[j]=max(hi[j],*v[j::3])
 return dict(min=lo,max=hi,mesh_sha256=hashlib.sha256(b).hexdigest())
r=E.parse(source/'piper_l/urdf/piper_l_description.urdf').getroot();joints=[]
for j in r.findall('joint'):
 if j.attrib['type']!='revolute':continue
 o=j.find('origin');limit=j.find('limit')
 joints.append(dict(xyz=list(map(float,o.attrib['xyz'].split())),rpy=list(map(float,o.attrib['rpy'].split())),lower=float(limit.attrib['lower']),upper=float(limit.attrib['upper'])))
links=[bounds(source/'piper_l/meshes'/f'{name}.stl') for name in ['base_link']+['link%d'%i for i in range(1,7)]]
# The official master mesh extends 130mm laterally. The 180mm radial bound
# covers its mesh corner, both jaws' full travel, and any flange clocking.
# 120mm axial bound exceeds the 77mm leader drawing and mesh jaw assembly.
meta=dict(schema='piper-l-leader-envelope-v1',joints=joints,links=links,names=['base']+['link%d'%i for i in range(1,7)]+['leader_tool'],
 tool=dict(min=[-.18,-.18,0.],max=[.18,.18,.12],construction='clocking-independent full-opening box enclosing official leader meshes; deliberately conservative'),
 margin_m=.015,sweep_radius_bound_m=1.5,excluded_pairs=[],
 upstream=dict(body=dict(url='https://github.com/agilexrobotics/agx_arm_urdf',commit='680fa925cd00db7e2628563d2edc5540963045de'),
 leader=dict(url='https://github.com/agilexrobotics/mobile_aloha_sim',commit='f799ae0192c0c2bbd502ec4d9bbec8c2633acd1a',
 meshes={f'link{i}':bounds(master/'aloha_new_description/meshes/piper_master_meshes'/f'link{i}.STL') for i in [6,7,8]})),
 validation='conservative geometry only; real installation and physical mode switching not commissioned')
srdf=args.sim_source/'Moveit2/piper_l_gripper_moveit_config/config/piper_l.srdf'
names=['base_link']+['link%d'%i for i in range(1,7)]
for row in E.parse(srdf).getroot().findall('disable_collisions'):
 a,b=row.get('link1'),row.get('link2')
 if a in names and b in names:meta['excluded_pairs'].append(sorted([names.index(a),names.index(b)]))
meta['excluded_pairs'].append([5,7])
meta['attachment_halfspace_exclusion']=dict(pair=[5,7],reason='Link5 lies entirely behind the J6 flange plane; official leader meshes, including full jaw travel, lie forward. Joint6 rotates about that plane normal. The connected flange contact is not a clearance violation.')
meta['upstream']['self_collision_matrix']=dict(url='https://github.com/agilexrobotics/agx_arm_sim',commit='f8cd8b147c75d59e14f90fb0646770eefa268ed0',path='Moveit2/piper_l_gripper_moveit_config/config/piper_l.srdf',sha256=hashlib.sha256(srdf.read_bytes()).hexdigest(),note='Only body-body pairs. No stock-gripper exclusions applied to enlarged teaching handle.')
for directory,key in [(source,'body'),(master,'leader'),(args.sim_source,'self_collision_matrix')]:
 revision=subprocess.check_output(['git','-C',str(directory),'rev-parse','HEAD'],text=True).strip()
 if revision!=meta['upstream'][key]['commit']:raise RuntimeError('Upstream revision differs from audited geometry: '+key)
path=args.output;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(meta,indent=2)+'\n');print(path)
