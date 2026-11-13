"""Explicit pi05 ablations; the default deployment stays at2500 with RTC."""
import math

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
    # Keep pi05 tuning separate from the shared defaults used by VPP2.
    options=settings.get('pi05',{})
    if not isinstance(options,dict):raise ValueError('pi05 settings must be a mapping')
    steps=options.get('chunk_smoothing_steps',selected['chunk_smoothing_steps'])
    tau=options.get('command_filter_tau_s',selected['command_filter_tau_s'])
    if (isinstance(steps,bool) or not isinstance(steps,int)
            or not 2<=steps<=selected['execute_horizon']):
        raise ValueError('Invalid pi05 chunk_smoothing_steps')
    if (isinstance(tau,bool) or not isinstance(tau,(int,float))
            or not math.isfinite(tau) or not .01<=tau<=.2):
        raise ValueError('Invalid pi05 command_filter_tau_s')
    selected.update(chunk_smoothing_steps=steps,command_filter_tau_s=tau)
    if no_smoothing:selected['smoothing_enabled']=False
    if no_rtc:
        selected.update(execution_mode='sync_chunks',execute_horizon=50,request_timeout_s=2.,
                        settle_timeout_s=5.,policy_timeout_s=50/30+5+2+2)
    return selected
