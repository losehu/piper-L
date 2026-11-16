#!/usr/bin/env bash
# Only activate the two verified SocketCAN network links. Never sends arm frames.
set -euo pipefail
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
if [[ $EUID -ne 0 ]]; then echo '需要管理员认证：sudo ./restore_rlt_can.sh'; exit 1; fi
exec 8>/home/arm/piper_calibration/.can0.lock
flock -n 8 || { echo 'can0 被其他程序占用，未修改'; exit 1; }
exec 9>/home/arm/piper_calibration/.can1.lock
flock -n 9 || { echo 'can1 被其他程序占用，未修改'; exit 1; }
for task_channel in can0 can1; do
    case "$task_channel" in
        can0) task_serial=004D00264648571620303731;;
        can1) task_serial=0024001E4648570B20303731;;
    esac
    udevadm info -q property -p "/sys/class/net/$task_channel" | grep -Fx "ID_SERIAL_SHORT=$task_serial" >/dev/null || { echo "$task_channel 身份不匹配，未修改"; exit 1; }
done
for task_channel in can0 can1; do
    if ! ip -j link show "$task_channel" | python3 -c 'import json,sys;sys.exit(0 if "UP" in json.load(sys.stdin)[0]["flags"] else 1)'; then
        ip link set "$task_channel" type can bitrate 1000000
        ip link set "$task_channel" up
    fi
    ip -j -details link show "$task_channel" | python3 -c 'import json,sys;d=json.load(sys.stdin)[0];assert "UP" in d["flags"] and d["linkinfo"]["info_data"]["bittiming"]["bitrate"]==1000000;print(d["ifname"]+": UP / 1Mbps")'
done
echo 'CAN网络接口已恢复；没有发送电机使能、回零或运动指令。'
