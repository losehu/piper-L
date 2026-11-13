"""Native desktop console for Piper human-in-the-loop RLT. Opens idle."""
import argparse,fcntl,os,time
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from rlt_piper.ui_process import Process,Demo

BG='#eef2f6';INK='#172b43';MUTED='#67788c';TEAL='#087e76';BLUE='#315ed2';RED='#b42336'
PHASE={'idle':'待机','loading':'加载模型','checking':'离线检查','preparing':'机械臂准备 / 自动对齐','running':'执行中','stopping':'阻尼停止中','stopped':'已停止','finished':'本轮结束','error':'启动失败 / 运行异常'}

class App:
    def __init__(self,window,root,demo=False,engine=None):
        self.window=window;self.root=Path(root);self.demo=demo;self.engine=engine or (Demo(root) if demo else Process(root));self.closing=False
        self.last_images={};self.photos={};self.last_notice=None
        window.title('Piper-L · 人工在环训练'+(' [演示]' if demo else ''));window.geometry('1200x920');window.minsize(1080,850);window.configure(bg=BG)
        window.option_add('*Font',('Noto Sans CJK SC',10));window.protocol('WM_DELETE_WINDOW',self.close)
        style=ttk.Style(window);style.theme_use('clam');style.configure('TCombobox',padding=6)
        header=tk.Frame(window,bg=INK,height=76);header.pack(fill='x');header.pack_propagate(False)
        tk.Label(header,text='PIPER-L',font=('DejaVu Sans',21,'bold'),fg='white',bg=INK).pack(side='left',padx=(24,14))
        tk.Label(header,text='人工在环在线学习  /  π0.5 + RTC + 平滑',fg='#bdcddd',bg=INK,font=('Noto Sans CJK SC',12)).pack(side='left')
        self.badge=tk.Label(header,text='● 待机',bg='#2b4259',fg='white',padx=16,pady=8);self.badge.pack(side='right',padx=24)
        tk.Label(window,text='Space / Esc / Ctrl+C 独立阻尼急停     ·     I 接管/交还     ·     S 成功 / F 失败     ·     关闭窗口会先停止本窗口的任务',bg=BG,fg=MUTED,font=('Noto Sans CJK SC',9)).pack(side='bottom',pady=(4,12))
        strip=tk.Frame(window,bg=BG);strip.pack(fill='x',padx=20,pady=(16,12));self.metrics={}
        for key,title in [('source','控制来源'),('rtt','推理延迟'),('replay','有效回放'),('human','人工纠正'),('step','训练步数'),('actor','Actor 版本')]:
            card=tk.Frame(strip,bg='white',padx=14,pady=10);card.pack(side='left',fill='both',expand=True,padx=4)
            tk.Label(card,text=title,bg='white',fg=MUTED,font=('Noto Sans CJK SC',9)).pack(anchor='w')
            value=tk.Label(card,text='—',font=('DejaVu Sans',16,'bold'),bg='white',fg=INK);value.pack(anchor='w',pady=(4,0));self.metrics[key]=value
        body=tk.Frame(window,bg=BG);body.pack(fill='x',padx=24)
        controls=tk.Frame(body,bg='white',padx=18,pady=14,width=280,height=430);controls.pack(side='left',fill='y',padx=(0,14));controls.pack_propagate(False)
        tk.Label(controls,text='本轮控制',bg='white',fg=INK,font=('Noto Sans CJK SC',13,'bold')).pack(anchor='w')
        self.stage=tk.StringVar(value='在线训练');self.combo=ttk.Combobox(controls,textvariable=self.stage,state='readonly',values=['在线训练','仅基础策略采集','Actor 评估'],width=23);self.combo.pack(fill='x',pady=(10,5))
        self.learning=tk.BooleanVar(value=True);self.learnbox=tk.Checkbutton(controls,text='后台持续训练',variable=self.learning,bg='white',activebackground='white',fg=INK);self.learnbox.pack(anchor='w')
        self.start_button=self.button(controls,'启动本轮',TEAL,self.start);self.start_button.pack(fill='x',pady=(10,7))
        self.takeover_button=self.button(controls,'人工接管  I',BLUE,lambda:self.action('takeover'));self.takeover_button.pack(fill='x',pady=(0,7))
        row=tk.Frame(controls,bg='white');row.pack(fill='x')
        self.success=self.button(row,'成功  S','#25774c',lambda:self.action('success'));self.success.pack(side='left',expand=True,fill='x',padx=(0,4))
        self.failure=self.button(row,'失败  F','#a15a16',lambda:self.action('failure'));self.failure.pack(side='left',expand=True,fill='x',padx=(4,0))
        self.stop_button=self.button(controls,'■  阻尼急停',RED,self.emergency);self.stop_button.pack(fill='x',pady=(12,6))
        self.check_button=tk.Button(controls,text='只检查模型 · 不动臂',command=lambda:self.start(check=True),bd=0,bg='#e9eef5',fg=INK,pady=7,cursor='hand2');self.check_button.pack(fill='x',pady=(0,9))
        tk.Label(controls,text='启动会自动准备、对齐并执行。\n接管前主臂须已处于拖动模式。',justify='left',wraplength=240,bg='white',fg=MUTED,font=('Noto Sans CJK SC',9)).pack(anchor='w')
        cameras=tk.Frame(body,bg=BG);cameras.pack(side='left',expand=True,fill='both')
        self.camera_labels={}
        for key,title in [('cam0','CAM 01 · 场景相机'),('cam1','CAM 02 · 手腕相机')]:
            panel=tk.Frame(cameras,bg='white',padx=10,pady=8);panel.pack(side='left',expand=True,fill='both',padx=(0,8) if key=='cam0' else (0,0))
            tk.Label(panel,text=title,bg='white',fg=MUTED,font=('DejaVu Sans',10,'bold')).pack(anchor='w',pady=(0,8))
            canvas=tk.Label(panel,text='等待实时画面\n启动本轮后显示',bg='#dfe7ef',fg='#657c91',width=38,height=13,compound='center');canvas.pack(fill='both',expand=True)
            self.camera_labels[key]=canvas
        details=tk.Frame(window,bg=BG);details.pack(fill='x',padx=24,pady=(12,8))
        self.detail=tk.Label(details,text='机械臂未启动 · Corrector 关闭',bg=BG,fg=MUTED,anchor='w');self.detail.pack(fill='x')
        self.training=tk.Label(details,text='等待新回合。人工接管作为纠正示范，S/F 单独标记整轮结果。',bg=BG,fg=TEAL,anchor='w');self.training.pack(fill='x',pady=(4,0))
        logs=tk.Frame(window,bg='white',padx=12,pady=10);logs.pack(fill='both',expand=True,padx=24,pady=(0,12))
        tk.Label(logs,text='运行日志',bg='white',fg=INK,font=('Noto Sans CJK SC',11,'bold')).pack(anchor='w',pady=(0,6))
        self.log=ScrolledText(logs,height=6,bg='#172536',fg='#d4e0ed',insertbackground='white',font=('DejaVu Sans Mono',10),bd=0,padx=12,pady=10,state='disabled',wrap='word');self.log.pack(fill='both',expand=True)
        def stop_key(_):
            self.emergency();return 'break'
        # Widget bindings precede Tk Button's class binding: Space must never
        # activate a focused Success/Start button before the global stop sees it.
        for widget in [window,self.start_button,self.takeover_button,self.success,self.failure,self.stop_button,
                       self.check_button,self.learnbox,self.combo,self.log]:
            for key in ('<KeyPress-space>','<KeyRelease-space>','<Escape>','<Control-c>'):widget.bind(key,stop_key)
        self.append('界面已就绪，机械臂未启动。\n'+('演示模式：所有按钮仅操作模拟状态。\n' if demo else ''))
        self.tick()
    def button(self,parent,text,color,command):return tk.Button(parent,text=text,command=command,bg=color,fg='white',activebackground=color,activeforeground='white',bd=0,pady=9,cursor='hand2',font=('Noto Sans CJK SC',10,'bold'),disabledforeground='#acb9c5')
    def append(self,text):
        if not text:return
        bottom=self.log.yview()[1]>=.98
        self.log.configure(state='normal');self.log.insert('end',text)
        if int(self.log.index('end-1c').split('.')[0])>600:self.log.delete('1.0','100.0')
        self.log.configure(state='disabled')
        if bottom:self.log.see('end')
    def start(self,check=False):
        try:
            stage={'在线训练':'online','仅基础策略采集':'collect','Actor 评估':'eval'}[self.stage.get()]
            self.last_images={}
            for label in self.camera_labels.values():label.configure(image='',text='等待实时画面\n启动本轮后显示')
            self.engine.start(stage,learn=self.learning.get() and stage!='eval',check=check)
        except Exception as e:self.append('启动失败：%s\n'%e)
    def action(self,action):
        try:self.engine.send(action)
        except Exception as e:self.append('操作未执行：%s\n'%e)
    def emergency(self):
        try:self.engine.stop()
        except Exception as e:self.append('停止请求：%s\n'%e)
    def tick(self):
        state=self.engine.poll();self.append(self.engine.take_logs());phase=state.get('phase','idle');active=self.engine.active
        self.badge.configure(text='● '+PHASE.get(phase,phase),bg=RED if phase in ('error','stopping') else ('#0a756f' if phase=='running' else '#2b4259'))
        learner=state.get('learner',{});mode=state.get('mode');source=state.get('source')
        values=dict(source='人工接管' if mode=='human' else {'base':'π0.5','rlt':'RLT Actor','hold':'保持'}.get(source,'待机'),
            rtt=('%.0f ms'%state['rtt_ms']) if state.get('rtt_ms') is not None else '—',replay=str(learner.get('transitions','—')),
            human=str(learner.get('human_transitions','—')),step=str(learner.get('global_step','—')),actor=str(learner.get('actor_version','—')))
        for key,value in values.items():self.metrics[key].configure(text=value)
        enabled=active and phase=='running' and state.get('ready') and not state.get('stopped')
        self.start_button.configure(state='disabled' if active or self.closing else 'normal');self.check_button.configure(state='disabled' if active or self.closing else 'normal')
        self.combo.configure(state='disabled' if active else 'readonly');self.learnbox.configure(state='disabled' if active else 'normal')
        self.takeover_button.configure(text='交还策略  I' if mode=='human' else '人工接管  I',command=lambda:self.action('resume' if self.engine.state.get('mode')=='human' else 'takeover'),state='normal' if enabled and not state.get('switching') else 'disabled')
        for button in [self.success,self.failure]:button.configure(state='normal' if enabled else 'disabled')
        self.stop_button.configure(state='normal' if active else 'disabled')
        if 'joint_deg' in state:self.detail.configure(text='关节角度 °   '+ '   '.join('%6.1f'%v for v in state['joint_deg'])+'     夹爪 %.1f mm'%state.get('gripper_mm',0))
        elif phase=='idle':self.detail.configure(text='机械臂未启动 · Corrector 关闭')
        training={'waiting_for_replay':'等待有效回放','training':'后台训练中','learning_paused':'训练已暂停','off':'后台训练关闭'}.get(learner.get('status'),'等待训练状态')
        if active:self.training.configure(text=training+(' · BC 预热，当前由 π0.5 控制' if not learner.get('ready') and source=='base' else '')+(' · '+learner['error'] if learner.get('error') else ''),fg=RED if learner.get('error') else TEAL)
        elif state.get('outcome') is not None:self.training.configure(text='本轮已标记：'+('成功' if state['outcome']==1 else '失败')+'。重开下一轮即可继续采集和学习。')
        elif phase in ('stopped','error','finished'):self.training.configure(text=PHASE[phase]+'，请查看日志。')
        notice=state.get('notice')
        if notice and notice!=self.last_notice:self.append(notice+'\n');self.last_notice=notice
        for key,encoded in state.get('images',{}).items():
            if key in self.camera_labels and encoded!=self.last_images.get(key):
                try:
                    photo=tk.PhotoImage(data=encoded);self.photos[key]=photo;self.last_images[key]=encoded
                    self.camera_labels[key].configure(image=photo,text='',width=400,height=225)
                except tk.TclError:pass
        if self.closing and not active:self.engine.cleanup();self.window.destroy();return
        self.window.after(40,self.tick)
    def close(self):
        if self.engine.active:self.closing=True;self.append('正在阻尼停止并保存本轮数据，完成后关闭窗口。\n');self.emergency()
        else:self.engine.cleanup();self.window.destroy()

def main():
    p=argparse.ArgumentParser();p.add_argument('--demo',action='store_true');p.add_argument('--root',type=Path,default=Path(os.environ.get('PIPER_DEPLOY_ROOT',Path(__file__).resolve().parents[1])));a=p.parse_args()
    (a.root/'logs').mkdir(exist_ok=True)
    lock=(a.root/'logs/rlt_ui.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('RLT UI 已打开，请使用现有窗口。')
    window=tk.Tk();App(window,a.root,a.demo);window.mainloop()

if __name__=='__main__':main()
