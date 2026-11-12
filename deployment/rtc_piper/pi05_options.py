"""Explicit pi05 ablations; the default deployment stays at2500 with RTC."""
CHECKPOINTS={1500:(1500,'checkpoints/pi05_piper_1500_bf16'),
             2500:(2499,'checkpoints/pi05_piper_2500_bf16')}


def checkpoint_settings(settings,step=None):
    selected=dict(settings)
    if step is not None:
        if step not in CHECKPOINTS:raise ValueError('pi05 step must be1500 or2500')
        index,path=CHECKPOINTS[step]
        selected.update(model_step=step,checkpoint_index=index,checkpoint=path)
    expected=CHECKPOINTS.get(selected['model_step'])
    if expected!=(selected['checkpoint_index'],selected['checkpoint']):
        raise ValueError('pi05 checkpoint identity mismatch')
    return selected


def execution_settings(settings,step=None,no_smoothing=False,no_rtc=False):
    selected=checkpoint_settings(settings,step)
    if no_smoothing:selected['smoothing_enabled']=False
    if no_rtc:
        selected.update(execution_mode='sync_chunks',execute_horizon=50,request_timeout_s=2.,
                        settle_timeout_s=5.,policy_timeout_s=50/30+5+2+2)
    return selected
