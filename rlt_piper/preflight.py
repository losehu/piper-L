"""Read-only boot checks before model loading or robot connections."""
import json,subprocess
from pathlib import Path

def check_can(settings,leader_config):
    cfg=json.loads(Path(leader_config).read_text());problems=[]
    for channel,serial in [('can0',cfg['adapter_serials']['can0']),('can1',settings['adapter_serial'])]:
        props=subprocess.run(['udevadm','info','-q','property','-p','/sys/class/net/'+channel],capture_output=True,text=True,timeout=3)
        if props.returncode or 'ID_SERIAL='+serial not in props.stdout.splitlines():raise RuntimeError(channel+' USB-CAN适配器缺失或身份不符')
        link=subprocess.run(['ip','-j','-details','link','show',channel],capture_output=True,text=True,timeout=3)
        if link.returncode:raise RuntimeError(channel+' 网络接口不存在')
        row=json.loads(link.stdout)[0];speed=row.get('linkinfo',{}).get('info_data',{}).get('bittiming',{}).get('bitrate')
        if 'UP' not in row['flags'] or speed!=1000000:problems.append(channel+' 未启用 / 非1Mbps')
    if problems:raise RuntimeError('；'.join(problems)+'。请先在部署目录执行 sudo ./restore_rlt_can.sh；仅恢复网络接口，不会使能或移动机械臂。')
