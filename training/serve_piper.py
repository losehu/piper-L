"""Serve a trained PiPER policy; this script does not control physical hardware."""
import argparse
import piper_config
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',default=8000,type=int);a=p.parse_args()
 cfg=piper_config.get_config()
 policy=policy_config.create_trained_policy(cfg,a.checkpoint,default_prompt=piper_config.PROMPT)
 websocket_policy_server.WebsocketPolicyServer(policy,host=a.host,port=a.port,metadata=cfg.policy_metadata).serve_forever()
