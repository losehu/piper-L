"""Single-H100 fixed inference worker with persistent HTTP/1.1 observations."""
import argparse
import json
import logging
from pathlib import Path
import queue
import os
import threading
import time
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import numpy as np
from .wire import decode,encode,MAX_BYTES,PREPROCESS,PROTOCOL

HERE=Path(__file__).resolve().parent


class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    disable_nagle_algorithm=True

    def setup(self):
        super().setup();self.connection.settimeout(65)
        with self.server.counter_lock:
            self.server.counter+=1;self.connection_id=self.server.counter

    def _send(self,status,result):
        raw=json.dumps(result,separators=(',',':'),allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(raw)))
        self.send_header('Connection','close' if self.close_connection else 'keep-alive')
        self.end_headers();self.wfile.write(raw)

    def do_GET(self):
        if self.path!='/health':return self._send(404,dict(ok=False))
        return self._send(200,dict(protocol=PROTOCOL,ok=True,model_loaded=True,warmed_up=True,
            checkpoint_step=2250,denoising_steps=5,compiled_action=True,robot_connected=False,
            action_horizon=32,action_rate_hz=30,model_gpu_resident=True,backend='bf16',
            preprocess=PREPROCESS,connection_id=self.connection_id,execution_worker='fixed main thread',
            transports=['application/json','application/vnd.piper.vpp2-observation']))

    def do_POST(self):
        try:
            if self.headers.get('Transfer-Encoding'):raise ValueError('Chunked transfer unsupported')
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=MAX_BYTES:raise ValueError('Invalid content length')
            start_read=time.perf_counter();raw=self.rfile.read(n);received=time.perf_counter()
            if len(raw)!=n:raise ValueError('Truncated request')
            if self.path=='/shutdown':
                job=SimpleNamespace(kind='stop',done=threading.Event());self.server.jobs.put(job)
                if not job.done.wait(30):raise RuntimeError('Shutdown timeout')
                return self._send(200,dict(ok=True))
            if self.path not in ('/infer','/echo'):return self._send(404,dict(ok=False))
            a,b,state,meta,codec,pixels=decode(raw,self.headers.get_content_type())
            decoded=time.perf_counter()
            if self.path=='/echo':
                result=dict(self.server.echo_result,benchmark_echo_only=True,inference_ms=0.)
            else:
                job=SimpleNamespace(kind='infer',args=(a,b,state),done=threading.Event(),result=None,error=None,queued=time.perf_counter())
                self.server.jobs.put(job)
                if not job.done.wait(90):raise RuntimeError('Inference timeout')
                if job.error:raise job.error
                result=job.result
            result.update(request_id=meta['request_id'],connection_id=self.connection_id,codec=codec,
                preprocess=PREPROCESS,pixel_sha256=pixels,
                server_body_read_ms=1000*(received-start_read),server_decode_ms=1000*(decoded-received),
                server_total_ms=1000*(time.perf_counter()-received))
            self._send(200,result)
        except (ValueError,KeyError,TypeError) as error:
            self.close_connection=True;self._send(400,dict(ok=False,error=str(error)))
        except (BrokenPipeError,ConnectionResetError):self.close_connection=True
        except Exception:
            logging.exception('Network benchmark failed');self.close_connection=True
            self._send(503,dict(ok=False,error='inference_failed'))


def serve(policy,echo_result,port):
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.daemon_threads=True
    server.jobs=queue.Queue();server.counter=0;server.counter_lock=threading.Lock();server.echo_result=echo_result
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    print('READY_NETWORK port=%d'%port,flush=True)
    try:
        while True:
            job=server.jobs.get()
            if job.kind=='stop':job.done.set();break
            try:
                begin=time.perf_counter();result=policy.infer_prepared(*job.args,seed=42)
                result.pop('normalized_action');result['fixed_worker_ms']=1000*(time.perf_counter()-begin)
                job.result=result
            except BaseException as error:job.error=error
            finally:job.done.set()
    finally:server.shutdown();server.server_close()


def main():
    import torch
    from vpp2_runtime.server import Policy
    parser=argparse.ArgumentParser()
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--sample',type=Path,required=True)
    parser.add_argument('--cuda-graphs',action='store_true')
    parser.add_argument('--compile-prefill',action='store_true')
    parser.add_argument('--port',type=int,default=19868);args=parser.parse_args()
    logging.basicConfig(level=logging.INFO);torch.set_num_threads(4)
    if torch.cuda.device_count()!=1:raise RuntimeError('Expose exactly one H100 with CUDA_VISIBLE_DEVICES')
    with np.load(args.sample,allow_pickle=False) as sample:
        images={key:np.array(sample[key]) for key in ('cam0','cam1')};state=np.array(sample['state'])
    raw,kind=encode(images,state,'warmup','json_keepalive');a,b,s,_,_,pixels=decode(raw,kind)
    start=time.perf_counter();policy=Policy(args.assets,device='cuda:0',denoising_steps=5,backend='bf16')
    load_seconds=time.perf_counter()-start
    policy.context=policy.context.to(policy.device,dtype=torch.bfloat16)
    policy.context_mask=policy.context_mask.to(policy.device)
    tensors=list(policy.model.named_parameters())+list(policy.model.named_buffers())
    if any(t.device.type!='cuda' for _,t in tensors):raise RuntimeError('Model is not fully GPU-resident')
    if any(t.dtype in (torch.int8,torch.uint8) for _,t in tensors):raise RuntimeError('Unexpected quantized model storage')
    options={'triton.cudagraphs':args.cuda_graphs,'compile_threads':2}
    compiled=torch.compile(policy.model.mot.forward_action_with_video_cache,fullgraph=False,dynamic=False,options=options)
    if args.compile_prefill:
        policy.model.mot.prefill_video_cache=torch.compile(policy.model.mot.prefill_video_cache,
            fullgraph=False,dynamic=False,options=options)
    calls=[0]
    def denoise(*args,**kwargs):calls[0]+=1;return compiled(*args,**kwargs)
    policy.model.mot.forward_action_with_video_cache=denoise
    original_infer=policy.infer_prepared
    def infer(*args,**kwargs):
        if torch.cuda.is_available() and cuda_graphs:torch.compiler.cudagraph_mark_step_begin()
        calls[0]=0;result=original_infer(*args,**kwargs)
        if calls[0]!=5:raise RuntimeError('Wrong actual denoising step count')
        result['actual_denoising_calls']=calls[0]
        return result
    cuda_graphs=args.cuda_graphs
    policy.infer_prepared=infer;warm=[]
    for i in range(8):
        start=time.perf_counter();result=policy.infer_prepared(a,b,s,seed=42)
        warm.append(dict(index=i,total_ms=1000*(time.perf_counter()-start),inference_ms=result['inference_ms']))
        print(json.dumps(warm[-1]),flush=True)
    reference=np.asarray(result['normalized_action'])
    changed_a=np.roll(a,2,axis=1).copy();changed_state=s.copy();changed_state[0]+=.05
    changed=policy.infer_prepared(changed_a,b,changed_state,seed=42)
    restored=policy.infer_prepared(a,b,s,seed=42)
    repeat_error=float(np.max(abs(np.asarray(restored['normalized_action'])-reference)))
    changed_error=float(np.max(abs(np.asarray(changed['normalized_action'])-reference)))
    if repeat_error>1e-5 or changed_error<1e-5:raise RuntimeError('Compiled graph input freshness/repeat check failed')
    policy.graph_validation=dict(repeat_max_normalized_difference=repeat_error,changed_input_difference=changed_error,
        cuda_graphs=args.cuda_graphs,compiled_prefill=args.compile_prefill)
    print(json.dumps(dict(status='ready',load_seconds=load_seconds,warmups=warm,pixel_sha256=pixels,
        checkpoint_step=2250,denoising_steps=5,model_gpu_resident=True,graph_validation=policy.graph_validation)),flush=True)
    state_path=os.environ.get('VPP2_SERVER_STATE')
    if state_path:Path(state_path).write_text(json.dumps(dict(status='ready',graph_validation=policy.graph_validation,warmups=warm),indent=2)+'\n')
    echo={k:v for k,v in result.items() if k!='normalized_action'}
    serve(policy,echo,args.port)


if __name__=='__main__':main()
