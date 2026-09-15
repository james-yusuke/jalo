"""Bounded, resumable video adaptation and validation-only render calibration."""
from __future__ import annotations
import copy
import csv
import json
import os
import time
from pathlib import Path
import torch
from torch.utils.data import Subset
from .data import collate, model_inputs
from .engine import SampleStream, dataset_for, load_checkpoint
from .metrics import decode, InstanceMetrics
from .model import build_model
from .quality import VehicleQuality, calibration_rank
from .roi_loss import ROICriterion
from .runtime import digest, environment, memory_bytes, restore_rng, rng_state, seed_all, select_device, synchronize, write_json

THRESHOLDS=(.3,.4,.5,.6,.7)


def resume_configs_match(saved,current):
    """Changing the execution backend must not relax data/model/budget identity."""
    return {k:v for k,v in saved.items() if k!='device'}=={k:v for k,v in current.items() if k!='device'}


class PairedSampleStream(SampleStream):
    """Randomized image pairs keep every effective batch exactly half COCO/video."""
    def next(self):
        import random
        if self.size%2:raise ValueError('Paired domain stream needs an even sample count')
        if self.position==self.size:
            self.epoch+=1;self.position=0;self.order=None
        if self.order is None:
            pairs=list(range(self.size//2));random.Random(self.seed+self.epoch).shuffle(pairs)
            self.order=[2*i+domain for i in pairs for domain in (0,1)]
        index=self.order[self.position];self.position+=1;return index


def subset_prediction(prediction, threshold):
    keep=prediction['scores']>=threshold
    return {key:value[keep] for key,value in prediction.items()}


@torch.no_grad()
def evaluate_adaptation(model,dataset,device,thresholds=THRESHOLDS,foreground_only=True):
    was_training=model.training;model.eval();started=time.perf_counter();inference=0.
    quality={v:VehicleQuality() for v in thresholds}
    ap={v:InstanceMetrics(dataset.classes) for v in thresholds}
    from .loss import SetCriterion
    raw=InstanceMetrics(dataset.classes)
    criterion=(ROICriterion(model.num_classes) if getattr(model,'architecture',None)=='vehicle_roi_v2' else SetCriterion(model.num_classes)).to(device)
    loss=0.
    per_frame=[]
    try:
        for i in range(len(dataset)):
            batch=collate([dataset[i]]);inputs=model_inputs(batch,device);target=batch['target'][0]
            synchronize(device);then=time.perf_counter();out=model(**inputs);synchronize(device)
            inference+=time.perf_counter()-then
            loss+=float(criterion(out,[target])['total'])
            all_prediction=decode(out,[target['transform']])[0]
            raw.update(all_prediction,target)
            winners=out['logits'][0].detach().cpu().argmax(-1)<model.num_classes
            keep=winners[all_prediction['query_indices']] if foreground_only else torch.ones_like(all_prediction['scores'],dtype=torch.bool)
            prediction={k:v[keep] for k,v in all_prediction.items()}
            row={'image_id':target['image_id'],'source_time':target['time']}
            for threshold in thresholds:
                selected=subset_prediction(prediction,threshold)
                quality[threshold].update(selected,target);ap[threshold].update(selected,target)
                one=VehicleQuality();one.update(selected,target);row[str(threshold)]=one.compute()
            per_frame.append(row)
        calibration=[]
        for threshold in thresholds:
            item={'threshold':threshold,**quality[threshold].compute(),**ap[threshold].compute()}
            # InstanceMetrics also reports prediction_count; the painted count remains explicit.
            item['painted_instance_count']=quality[threshold].predictions
            calibration.append(item)
        selected=max(calibration,key=lambda q:calibration_rank(q,q['mask_mAP']))
        result={'raw_ap':raw.compute(),'calibration':calibration,'selected':selected,'per_frame':per_frame,
            'loss':loss/len(dataset),'evaluation_seconds':time.perf_counter()-started,'inference_seconds':inference,
            'inference_fps':len(dataset)/max(inference,1e-9),'active_parameters':model.active_parameter_count(),
            'device':str(device),**memory_bytes(device)}
        return result
    finally:model.train(was_training)


def report_files(path,result):
    write_json(path,result)
    rows=result['calibration']
    with Path(path).with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def evaluate_adaptation_checkpoint(model,checkpoint,path,config,split,device,output):
    if split not in ('train','val','test'):raise ValueError('Unknown adaptation split')
    if digest(config['manifest'])!=checkpoint['manifest_sha256']:
        raise ValueError('Evaluation annotation/split manifest differs from the trained checkpoint')
    config=copy.deepcopy(config);profile=checkpoint.get('render_settings')
    if profile:config['image_size']=profile['image_size']
    thresholds=THRESHOLDS if split=='val' else ((profile or {}).get('threshold',.3),)
    lock=Path(config['data_root'])/'final_test_lock.json'
    identity={'checkpoint_sha256':checkpoint['loaded_sha256'],'manifest_sha256':checkpoint['manifest_sha256'],
              'render_settings':profile}
    if split=='test':
        if not profile or not checkpoint.get('selection',{}).get('evaluated_on_val'):
            raise ValueError('Freeze validation-selected render settings before evaluating test')
        if lock.exists():
            previous=json.loads(lock.read_text())
            if previous['identity']!=identity:
                raise ValueError('Held-out test already consumed: do not tune or select another checkpoint on test')
            if previous.get('result') is None:
                raise RuntimeError('Test evaluation was interrupted; preserve this lock and audit before retrying')
            result=previous['result']
            if output:report_files(output,result)
            return result
        # Exclusive creation means simultaneous evaluations cannot consume test twice.
        with lock.open('x') as f:json.dump({'identity':identity,'result':None},f)
    result=evaluate_adaptation(model,dataset_for(config,split),device,thresholds)
    result.update(checkpoint=str(Path(path).resolve()),split=split,**identity)
    if split=='test':write_json(lock,{'identity':identity,'result':result})
    if output:report_files(output,result)
    return result


class ExperimentBudget:
    """Shared ledger includes all trial updates and intermediate evaluation wall time."""
    def __init__(self,path,max_seconds=28800,max_updates=50000):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.max_seconds=min(float(max_seconds),28800.);self.max_updates=min(int(max_updates),50000)
        self.state=json.loads(self.path.read_text()) if self.path.exists() else {'seconds':0.,'updates':0,'events':[]}
        self._wall_start=None;self._wall_initial_seconds=0.
    def start_wall_clock(self):
        """Include setup, checkpoint I/O and ledger writes during this active run."""
        if self._wall_start is not None:raise RuntimeError('Budget wall clock is already running')
        self._wall_initial_seconds=self.state['seconds'];self._wall_start=time.perf_counter()
    def elapsed_seconds(self):
        wall=0. if self._wall_start is None else self._wall_initial_seconds+time.perf_counter()-self._wall_start
        return max(self.state['seconds'],wall)
    def reconcile_wall_clock(self):
        missing=self.elapsed_seconds()-self.state['seconds']
        if missing>0:self.charge(missing,0,'logging_checkpoint_overhead')
    def remaining(self):return max(0.,self.max_seconds-self.elapsed_seconds())
    def can_start(self,reserve=15.):return self.remaining()>reserve and self.state['updates']<self.max_updates
    def charge(self,seconds,updates,phase):
        self.state['seconds']+=float(seconds);self.state['updates']+=updates
        self.state['events'].append({'phase':phase,'seconds':float(seconds),'updates':updates})
        write_json(self.path,self.state)


def train_adaptation(config,run_dir=None,resume=None):
    if (Path(config['data_root'])/'final_test_lock.json').exists():
        raise ValueError('Final test already consumed: this adaptation experiment is frozen; do not resume training or tune on test')
    config=copy.deepcopy(config);seed_all(config['seed']);device=select_device(config.get('device','auto'))
    run=Path(run_dir or config.get('run_dir','runs/vehicle_adapt_v2'));run.mkdir(parents=True,exist_ok=True)
    if any(run.iterdir()) and not resume:raise FileExistsError(f'{run} is not empty; use --resume or a new run')
    tc=config['train'];ledger=ExperimentBudget(config.get('budget_ledger','runs/vehicle_adaptation_budget.json'),
                                            tc.get('max_seconds',28800),tc.get('max_updates',50000))
    ledger.start_wall_clock()
    if tc['batch_size']*tc['accumulation']%2:
        raise ValueError('An even effective batch is required for equal COCO/video sampling')
    manifest=digest(config['manifest']);coco_sha=digest(config['coco_manifest'])
    ann=json.loads(Path(config['manifest']).read_text())['annotations_sha256']
    checkpoint=load_checkpoint(resume) if resume else None
    if checkpoint:
        if not resume_configs_match(checkpoint['config'],config) or checkpoint['manifest_sha256']!=manifest or checkpoint['coco_manifest_sha256']!=coco_sha:
            raise ValueError('Resume requires identical training config, annotations, split and COCO selection; only device may change')
        if checkpoint['config'].get('device')!=config.get('device'):
            write_json(run/f'device_resume_step{checkpoint["step"]}.json',{
                'checkpoint_sha256':checkpoint['loaded_sha256'],'saved_device':checkpoint['config'].get('device'),
                'requested_device':config.get('device'),'resolved_device':str(device),
                'note':'Model/optimizer/data/RNG state restored; numerical training across backends is not bitwise equivalent.'})
    model=build_model(config,'single',pretrained=False if checkpoint else None).to(device).train()
    if checkpoint:model.load_state_dict(checkpoint['model'])
    backbone=[];other=[]
    for name,p in model.named_parameters():
        (backbone if name.startswith('backbone.') and not any(s in name for s in ('project','fuse','pixel','refine')) else other).append(p)
    optimizer=torch.optim.AdamW([{'params':backbone,'lr':tc['backbone_lr']},{'params':other,'lr':tc['lr']}],weight_decay=tc['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=tc.get('max_updates',50000))
    step=0;phase='overfit';phase_seconds={p:0. for p in ('overfit','warmup','joint')};phase_steps={p:0 for p in phase_seconds}
    profile={'threshold':.3,'mask_threshold':.5,'foreground_only':True,'image_size':config['inference_size'],'alpha':.45}
    if config.get('mask_projection')=='native_roi':profile['mask_projection']='native_roi'
    best_rank=(-1.,-1.,-1.);selection={};stream=None;first_loss=None;last_loss=None
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler']);restore_rng(checkpoint['rng'])
        step=checkpoint['step'];phase=checkpoint['phase'];phase_seconds=checkpoint['phase_seconds'];phase_steps=checkpoint['phase_steps']
        best_rank=tuple(checkpoint['best_rank']);profile=checkpoint['render_settings'];selection=checkpoint['selection']
        first_loss=checkpoint['first_overfit_loss'];last_loss=checkpoint['last_overfit_loss']
    else:seed_all(config['seed']+10000)
    if not checkpoint and config.get('qa_checkpoint'):
        qa=load_checkpoint(config['qa_checkpoint'])
        video_manifest=json.loads(Path(config['manifest']).read_text())
        qa_manifest=Path(config['qa_checkpoint']).parent/'manifest.json'
        train_seconds={im['second'] for im in video_manifest['splits']['train']}
        if (qa.get('architecture')!='vehicle_roi_v2' or qa['config']['model']!=config['model'] or
            not qa.get('qa',{}).get('loss_decreased') or qa['qa']['source_sha256']!=video_manifest['source_sha256'] or
            not set(qa['qa']['image_seconds'])<=train_seconds or digest(qa_manifest)!=qa['qa']['annotation_sha256']):
            raise ValueError('QA initialization must be this architecture, source video and a successful overfit check')
        model.load_state_dict(qa['model']);optimizer.load_state_dict(qa['optimizer']);scheduler.load_state_dict(qa['scheduler'])
        restore_rng(qa['rng']);step=qa['step'];phase='warmup'
        phase_seconds['overfit']=qa['qa']['seconds'];phase_steps['overfit']=qa['step']
        first_loss=qa['qa']['first_loss'];last_loss=qa['qa']['last_loss']
        write_json(run/'initialization.json',{'checkpoint':config['qa_checkpoint'],'sha256':qa['loaded_sha256'],'qa':qa['qa']})
    criterion=ROICriterion(model.num_classes).to(device)
    write_json(run/'config.json',config);write_json(run/'environment.json',environment())
    (run/'manifest.json').write_bytes(Path(config['manifest']).read_bytes())
    stage_config=None;active_phase=None
    def save(name):
        state={'format_version':3,'task':model.task,'architecture':model.architecture,'classes':list(config['classes']),
            'variant':'single','config':config,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'rng':rng_state(),'stream':stream.state(),'step':step,'phase':phase,
            'phase_seconds':phase_seconds,'phase_steps':phase_steps,'best_rank':best_rank,'render_settings':profile,
            'selection':selection,'manifest_sha256':manifest,'annotations_sha256':ann,'coco_manifest_sha256':coco_sha,
            'budget':ledger.state,'environment':environment(),'first_overfit_loss':first_loss,'last_overfit_loss':last_loss}
        tmp=run/(name+'.tmp');torch.save(state,tmp);os.replace(tmp,run/name)
    print(f'Adaptation: device={device}, update={step}, shared remaining={ledger.remaining():.1f}s',flush=True)
    if resume and (run/'train.jsonl').exists():
        entries=[json.loads(line) for line in (run/'train.jsonl').read_text().splitlines()]
        abandoned=[row for row in entries if row['step']>step]
        if abandoned:
            with (run/'abandoned_updates.jsonl').open('a') as f:
                for row in abandoned:f.write(json.dumps(row)+'\n')
            (run/'train.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in entries if row['step']<=step))
    while ledger.can_start(tc.get('step_reserve_seconds',30)):
        # Completed stage checkpoints resume at the following stage, without an extra update.
        if phase=='overfit' and (phase_steps[phase]>=tc.get('overfit_updates',200) or phase_seconds[phase]>=1800):phase='warmup'
        if phase=='warmup' and (phase_steps[phase]>=tc.get('warmup_updates',1000) or phase_seconds[phase]>=3600):phase='joint'
        if active_phase!=phase:
            stage_config=copy.deepcopy(config)
            stage_config['image_size']=config['inference_size'] if phase=='joint' else config['initial_size']
            if phase=='overfit':
                stage_config['train']['flip_probability']=0.
                dataset=dataset_for(stage_config,'train',False)
                candidates=[i for i,im in enumerate(dataset.images) if im['annotations']]
                if len(candidates)<4:raise ValueError('Overfit check needs at least four annotated vehicle images')
                data=Subset(dataset,candidates[:4])
            else:data=dataset_for(stage_config,'train',True)
            stream_type=SampleStream if phase=='overfit' else PairedSampleStream
            stream=stream_type(len(data),config['seed'])
            if checkpoint and checkpoint['phase']==phase:
                stream=stream_type(len(data),config['seed'],**checkpoint['stream']);checkpoint=None
            active_phase=phase
        started=time.perf_counter();optimizer.zero_grad(set_to_none=True);losses={}
        for _ in range(tc['accumulation']):
            batch=collate([data[stream.next()] for _ in range(tc['batch_size'])]);out=model(**model_inputs(batch,device))
            if phase=='warmup':out['training_roi_logits']=model.roi_masks(out['mask_features'],[t['boxes'].to(device) for t in batch['target']])
            parts=criterion(out,batch['target'],phase='warmup' if phase=='warmup' else 'joint')
            if not torch.isfinite(parts['total']):raise FloatingPointError(f'Nonfinite loss at {phase}/{step}')
            (parts['total']/tc['accumulation']).backward()
            for key,value in parts.items():losses[key]=losses.get(key,0.)+float(value.detach())/tc['accumulation']
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),tc['clip_grad'],error_if_nonfinite=True)
        optimizer.step();scheduler.step();synchronize(device);step+=1;phase_steps[phase]+=1
        seconds=time.perf_counter()-started;phase_seconds[phase]+=seconds;ledger.charge(seconds,1,phase)
        if phase=='overfit':
            if first_loss is None:first_loss=losses['total']
            last_loss=losses['total']
        row={'step':step,'phase':phase,**losses,'gradient_norm':float(norm),'phase_seconds':phase_seconds[phase],
             'experiment_seconds':ledger.state['seconds']}
        with (run/'train.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if step==1 or step%10==0:print(f'{phase} {step}: loss={losses["total"]:.4f}, used={ledger.state["seconds"]:.1f}s',flush=True)
        transition=(phase=='overfit' and (phase_steps[phase]>=tc.get('overfit_updates',200) or phase_seconds[phase]>=1800)) or (
                    phase=='warmup' and (phase_steps[phase]>=tc.get('warmup_updates',1000) or phase_seconds[phase]>=3600))
        if phase=='warmup' and step%tc['validate_every']==0 and ledger.remaining()>120:
            before=time.perf_counter();state=rng_state()
            result=evaluate_adaptation(model,dataset_for(stage_config,'val'),device);restore_rng(state)
            seconds=time.perf_counter()-before;phase_seconds[phase]+=seconds;ledger.charge(seconds,0,'warmup_validation')
            result.update(step=step,phase=phase,image_size=stage_config['image_size'],eligible_for_selection=False)
            report_files(run/'warmup_metrics.json',result)
            with (run/'evaluations.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
            print(f'Warmup validation: mask AP={result["raw_ap"]["mask_mAP"]:.4f}',flush=True)
        if phase=='joint' and (step%tc['validate_every']==0 or not ledger.can_start(180)) and ledger.remaining()>120:
            before=time.perf_counter();state=rng_state()
            result=evaluate_adaptation(model,dataset_for(stage_config,'val'),device);restore_rng(state)
            seconds=time.perf_counter()-before;phase_seconds[phase]+=seconds;ledger.charge(seconds,0,'validation')
            chosen=result['selected'];rank=calibration_rank(chosen,chosen['mask_mAP'])
            result.update(step=step,experiment_seconds=ledger.state['seconds']);report_files(run/'latest_metrics.json',result)
            with (run/'evaluations.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
            if rank>best_rank:
                best_rank=rank;profile['threshold']=chosen['threshold']
                selection={'evaluated_on_val':True,'step':step,'metrics':chosen,'quality_pass':chosen['quality_pass']}
                save('best.pt');report_files(run/'best_metrics.json',result)
            print(f'Validation: precision={chosen["pixel_precision"]}, recall={chosen["vehicle_recall_ge32"]}, cabin={chosen["interior_false_paint_rate"]}, pass={chosen["quality_pass"]}',flush=True)
            if chosen['quality_pass'] and tc.get('stop_on_quality_pass',False):
                save('last.pt')
                ledger.reconcile_wall_clock()
                write_json(run/'status.json',{'step':step,'phase':phase,'stop_reason':'validation_quality_gate',
                    'experiment_seconds':ledger.state['seconds'],'experiment_updates':ledger.state['updates']})
                return run/'best.pt'
        if transition:
            if phase=='overfit':
                write_json(run/'overfit_check.json',{'first_loss':first_loss,'last_loss':last_loss,'updates':phase_steps[phase],
                    'seconds':phase_seconds[phase],'loss_decreased':last_loss<first_loss})
                if not last_loss<first_loss:
                    save('last.pt');raise RuntimeError('Overfit loss did not decrease; diagnose before further training')
            save(phase+'_complete.pt');save('last.pt');phase='warmup' if phase=='overfit' else 'joint'
            # Save after constructing the next phase stream, at the next update.
        if step%tc['checkpoint_every']==0 or not ledger.can_start(180):
            # A stage-transition checkpoint must describe its new data order accurately.
            if active_phase==phase:save('last.pt')
        if not ledger.can_start(tc.get('step_reserve_seconds',30)):
            if active_phase==phase:save('last.pt')
            break
        ledger.reconcile_wall_clock()
    if stream is not None and active_phase==phase:save('last.pt')
    ledger.reconcile_wall_clock()
    write_json(run/'status.json',{'step':step,'phase':phase,'phase_seconds':phase_seconds,'phase_steps':phase_steps,
        'experiment_seconds':ledger.state['seconds'],'experiment_updates':ledger.state['updates'],'stop_reason':'shared_budget',
        'selected_checkpoint_exists':(run/'best.pt').exists()})
    return run/('best.pt' if (run/'best.pt').exists() else 'last.pt')
