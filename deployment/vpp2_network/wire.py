"""Binary VPP2 observations preserving the legacy model input pixels exactly."""
import base64
import hashlib
import io
import json
import struct
import numpy as np
from PIL import Image

MAGIC=b'VPB2'
PROTOCOL='vpp2-piper-v1'
PREPROCESS='jpeg95-pil-bilinear224x168-pad28-v1'
MODES=('json_close','json_keepalive','jpeg_binary','png_prepared','webp_prepared')
MAX_BYTES=4*1024*1024


def jpeg_pair(images):
    result=[]
    for key in ('cam0','cam1'):
        array=np.asarray(images[key])
        if array.shape!=(480,640,3) or array.dtype!=np.uint8:raise ValueError('Expected640x480 RGB uint8')
        buffer=io.BytesIO();Image.fromarray(array).save(buffer,format='JPEG',quality=95)
        result.append(buffer.getvalue())
    return result


def prepare_jpegs(blobs):
    cameras=[]
    for blob in blobs:
        with Image.open(io.BytesIO(blob)) as image:
            if image.format!='JPEG' or image.size!=(640,480):raise ValueError('Expected640x480 JPEG')
            resized=image.convert('RGB').resize((224,168),Image.Resampling.BILINEAR)
            canvas=Image.new('RGB',(224,224),0);canvas.paste(resized,(0,28))
            cameras.append(np.array(canvas,dtype=np.uint8))
    return cameras


def pixel_hash(cameras):
    return hashlib.sha256(np.concatenate(cameras,axis=1).tobytes()).hexdigest()


def common(state,request_id):
    values=np.asarray(state,dtype=np.float32)
    if values.shape!=(7,) or not np.isfinite(values).all():raise ValueError('Invalid measured state')
    return dict(protocol=PROTOCOL,request_id=request_id,state_units='rad,m',camera_order=['cam0','cam1'],state=values.tolist(),seed=42)


def encode(images,state,request_id,mode):
    if mode not in MODES:raise ValueError('Unknown transport mode')
    meta=common(state,request_id);blobs=jpeg_pair(images)
    if mode.startswith('json_'):
        for key,blob in zip(('cam0','cam1'),blobs):meta[key+'_jpeg_b64']=base64.b64encode(blob).decode('ascii')
        return json.dumps(meta,allow_nan=False).encode(),'application/json'
    if mode=='jpeg_binary':meta.update(codec='jpeg_pair',preprocess=PREPROCESS)
    else:
        cameras=prepare_jpegs(blobs);joined=np.concatenate(cameras,axis=1);buffer=io.BytesIO()
        if mode=='png_prepared':
            Image.fromarray(joined).save(buffer,format='PNG',compress_level=3)
            codec='prepared_png'
        else:
            Image.fromarray(joined).save(buffer,format='WEBP',lossless=True,quality=0,method=2,exact=True)
            codec='prepared_webp_lossless'
        meta.update(codec=codec,preprocess=PREPROCESS,pixel_sha256=pixel_hash(cameras))
        blobs=[buffer.getvalue()]
    meta['lengths']=[len(b) for b in blobs]
    header=json.dumps(meta,separators=(',',':'),allow_nan=False).encode()
    return MAGIC+struct.pack('!I',len(header))+header+b''.join(blobs),'application/vnd.piper.vpp2-observation'


def validate_meta(meta):
    if meta.get('protocol')!=PROTOCOL or meta.get('state_units')!='rad,m' or meta.get('camera_order')!=['cam0','cam1'] or meta.get('seed')!=42:
        raise ValueError('Observation identity/units/order mismatch')
    if not isinstance(meta.get('request_id'),str) or not 0<len(meta['request_id'])<=128:raise ValueError('Invalid request id')
    state=np.asarray(meta.get('state'),dtype=np.float32)
    if state.shape!=(7,) or not np.isfinite(state).all():raise ValueError('Invalid measured state')
    if abs(state[:6]).max()>2*np.pi or not 0<=state[6]<=.1:raise ValueError('Gross state unit mismatch')
    return state


def decode(raw,content_type):
    if not 0<len(raw)<=MAX_BYTES:raise ValueError('Invalid observation size')
    if content_type=='application/json':
        meta=json.loads(raw);state=validate_meta(meta)
        blobs=[base64.b64decode(meta[key+'_jpeg_b64'],validate=True) for key in ('cam0','cam1')]
        cameras=prepare_jpegs(blobs);codec='json_jpeg_pair'
    else:
        if content_type!='application/vnd.piper.vpp2-observation' or len(raw)<8 or raw[:4]!=MAGIC:raise ValueError('Invalid binary protocol')
        count=struct.unpack('!I',raw[4:8])[0]
        if not 0<count<=16384 or 8+count>=len(raw):raise ValueError('Invalid metadata length')
        meta=json.loads(raw[8:8+count]);state=validate_meta(meta)
        if meta.get('preprocess')!=PREPROCESS:raise ValueError('Wrong preprocessing identity')
        lengths=meta.get('lengths');codec=meta.get('codec')
        if not isinstance(lengths,list) or not lengths or any(type(n) is not int or not 0<n<MAX_BYTES for n in lengths):raise ValueError('Invalid image lengths')
        if 8+count+sum(lengths)!=len(raw):raise ValueError('Truncated or extra image data')
        blobs=[];cursor=8+count
        for n in lengths:blobs.append(raw[cursor:cursor+n]);cursor+=n
        if codec=='jpeg_pair' and len(blobs)==2:cameras=prepare_jpegs(blobs)
        elif codec in ('prepared_png','prepared_webp_lossless') and len(blobs)==1:
            with Image.open(io.BytesIO(blobs[0])) as image:
                expected='PNG' if codec=='prepared_png' else 'WEBP'
                if image.format!=expected or image.size!=(448,224):raise ValueError('Wrong prepared image shape/codec')
                joined=np.array(image.convert('RGB'),dtype=np.uint8)
            cameras=[joined[:,:224].copy(),joined[:,224:].copy()]
            if pixel_hash(cameras)!=meta.get('pixel_sha256'):raise ValueError('Prepared pixels changed in transit')
        else:raise ValueError('Wrong codec or image count')
    return cameras[0],cameras[1],state,meta,codec,pixel_hash(cameras)
