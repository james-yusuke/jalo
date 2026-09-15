"""Continuous tracker evaluation; labels are read only at the fixed reference times."""
import copy
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from .encoding import TimedVideoReader
from .geometry import Letterbox
from .mask_video import tracked_instances
from .quality import VehicleQuality, calibration_rank
from .runtime import synchronize
from .video import ClasswiseTracker


def rendered_prediction(instances, shape):
    """Quality applies the same confidence order and single-coverage rule as paint_masks."""
    return {'masks':torch.from_numpy(np.stack([i['mask'] for i in instances])) if instances else torch.zeros((0,*shape),dtype=torch.bool),
            'scores':torch.tensor([i['score'] for i in instances]),
            'labels':torch.tensor([i['class_id'] for i in instances],dtype=torch.long)}


@torch.no_grad()
def evaluate_sequence(model,dataset,device,thresholds=(.3,),ledger=None,deadline_reserve=0):
    """Update ByteTrack at every presentation frame, measure masks at labeled frames.

    Skipping local mask convolution on unlabeled frames cannot change the boxes or
    ByteTrack state. No reference annotations are passed into model.forward.
    """
    was_training=model.training;model.eval();started=time.perf_counter()
    quality={t:VehicleQuality(dataset.classes,'ByteTrack on every source-FPS presentation frame; painted masks on reference frames') for t in thresholds}
    by_source={}
    for index,item in enumerate(dataset.images):by_source.setdefault(item['source_id'],{})[item['frame_index']]=index
    sources={s['id']:s for s in dataset.manifest['sources']}
    frames=0;reference_frames=0;inference=0.;complete=True
    try:
        for sid,indices in by_source.items():
            source=sources[sid];fps=source['fps']
            capture=TimedVideoReader(source['path'])
            if abs(capture.fps-fps)>1e-4:
                capture.release();raise ValueError('Tracker and reference extraction presentation FPS disagree')
            tracker=ClasswiseTracker(fps,dataset.classes);last=max(indices)
            try:
                for frame_index in range(last+1):
                    if ledger and not ledger.can_evaluate(deadline_reserve+30):complete=False;break
                    ok,frame=capture.read()
                    if not ok:raise ValueError('Video ended before its annotated reference frames')
                    is_reference=frame_index in indices
                    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    transform=Letterbox.create(rgb,dataset.image_size)
                    image,padding=transform.image(rgb)
                    inputs={'images':image[None,None].expand(1,3,-1,-1,-1).to(device),
                            'padding':padding[None,None].expand(1,3,-1,-1).to(device),
                            'history_valid':torch.zeros(1,2,dtype=torch.bool,device=device),
                            'time_deltas':torch.tensor([[.2,.4]],device=device)}
                    synchronize(device);then=time.perf_counter()
                    if getattr(model,'architecture',None)=='vehicle_roi_v2':
                        out=model(**inputs,mask_query_threshold=min(thresholds) if is_reference else 1.)
                    else:out=model(**inputs)
                    synchronize(device);inference+=time.perf_counter()-then
                    # At non-reference frames no mask restoration is needed to update ByteTrack.
                    instances=tracked_instances(out,transform,tracker,frame_index/fps,dataset.classes,
                                                min(thresholds) if is_reference else 1.)
                    frames+=1
                    if is_reference:
                        target=dataset[indices[frame_index]]['target']
                        for threshold in thresholds:
                            chosen=[i for i in instances if i['score']>=threshold]
                            quality[threshold].update(rendered_prediction(chosen,frame.shape[:2]),target)
                        reference_frames+=1
                    if frames%300==0:print(f'Tracker evaluation: {sid} {frame_index/fps:.1f}s, {reference_frames} reference frames',flush=True)
                if not complete:break
            finally:capture.release()
        return {'complete':complete and reference_frames==len(dataset),'presentation_frames':frames,'reference_frames':reference_frames,
                'seconds':time.perf_counter()-started,'inference_seconds':inference,
                'inference_fps':frames/max(inference,1e-9),
                'calibration':[{'threshold':t,**q.compute()} for t,q in quality.items()]}
    finally:
        model.train(was_training)
        if ledger:ledger.reconcile_wall_clock()


def combine_calibration(single,sequence,minimum_targets=0):
    """Choose on validation only; fixed-threshold test calls contain one candidate."""
    seq={row['threshold']:row for row in sequence['calibration']};rows=[]
    for row in single['calibration']:
        actual=seq[row['threshold']]
        count=row['vehicle_count_ge32']+row['small_vehicle_count']
        sufficient=count>=minimum_targets
        item={**row,'single_frame_quality':copy.deepcopy(row),'rendered_quality':actual,
              'evaluation_complete':sequence['complete'],'sufficient_targets':sufficient,
              'quality_pass':bool(row['quality_pass'] and actual['quality_pass'] and sufficient and sequence['complete'])}
        rows.append(item)
    # Both stages must satisfy false-paint constraints. Min recall breaks feasible ties.
    def rank(row):
        raw=calibration_rank(row,row.get('mask_mAP'))
        actual=calibration_rank(row['rendered_quality'],row.get('mask_mAP'))
        feasible=raw[0] and actual[0] and row['evaluation_complete'] and row['sufficient_targets']
        return (int(feasible),min(raw[1],actual[1]),row.get('mask_mAP') or 0.)
    return {**single,'calibration':rows,'selected':max(rows,key=rank),'sequence':sequence,
            'minimum_reference_vehicles':minimum_targets}


def compare_final(checkpoint_path, baseline_path, config, device, output=None):
    """Consume the held-out source once for a validation-frozen model pair."""
    import json
    from .adapt import ExperimentBudget, EvaluationBudgetExceeded, evaluate_adaptation, report_files
    from .engine import checkpoint_model, dataset_for
    from .runtime import digest, write_json
    root=Path(config['data_root']);lock=root/'final_test_lock.json'
    model,cp=checkpoint_model(checkpoint_path,device)
    baseline,old=checkpoint_model(baseline_path,device)
    if config.get('initialization_sha256')!=old['loaded_sha256']:
        raise ValueError('Final comparison baseline must be the pinned public model')
    if digest(config['manifest'])!=cp['manifest_sha256']:
        raise ValueError('Final comparison references differ from training')
    frozen_path=Path(checkpoint_path).parent/'validation_frozen.json'
    frozen=json.loads(frozen_path.read_text())
    if (frozen['checkpoint_sha256']!=cp['loaded_sha256'] or frozen['baseline_sha256']!=old['loaded_sha256'] or
        frozen['manifest_sha256']!=cp['manifest_sha256'] or frozen['render_settings']!=cp['render_settings']):
        raise ValueError('Freeze the model pair and their validation-selected profiles first')
    identity={'checkpoint_sha256':cp['loaded_sha256'],'baseline_sha256':old['loaded_sha256'],
              'manifest_sha256':cp['manifest_sha256'],'render_settings':cp['render_settings'],
              'frozen_validation_sha256':digest(frozen_path)}
    if lock.exists():
        record=json.loads(lock.read_text())
        if record['identity']!=identity:raise ValueError('Held-out test already consumed for another model/profile')
        if record.get('result') is None:raise RuntimeError('Interrupted final comparison; audit the partial results and lock before any retry')
        if output:report_files(output,record['result'])
        return record['result']
    ledger=ExperimentBudget(config['budget_ledger'],config['train']['max_seconds'],config['train']['max_updates'])
    if ledger.remaining()<60:raise ValueError('Insufficient shared experiment budget for final comparison')
    with lock.open('x') as f:json.dump({'identity':identity,'result':None},f)
    ledger.start_wall_clock();results={}
    try:
        for name,current,profile in [('baseline',baseline,frozen['baseline_render_settings']),('improved',model,cp['render_settings'])]:
            dataset_config=copy.deepcopy(config);dataset_config['image_size']=profile['image_size']
            dataset=dataset_for(dataset_config,'test')
            try:
                raw=evaluate_adaptation(current,dataset,device,(profile['threshold'],),ledger=ledger)
            except EvaluationBudgetExceeded:
                results[name]={'calibration':[{'threshold':profile['threshold'],'quality_pass':False}],
                               'selected':{'quality_pass':False,'evaluation_complete':False},'stop_reason':'shared_budget'}
                write_json(root/f'final_test_{name}.json',results[name]);break
            sequence=evaluate_sequence(current,dataset,device,(profile['threshold'],),ledger=ledger,deadline_reserve=10)
            results[name]=combine_calibration(raw,sequence,minimum_targets=100)
            write_json(root/f'final_test_{name}.json',results[name])
            if not sequence['complete']:break
        if 'improved' in results:
            result={**results['improved'],'baseline':results['baseline'],'validation':frozen['validation'],
                    'baseline_validation':frozen['baseline_validation'],'split':'test',**identity}
        else:
            result={'calibration':[{'threshold':cp['render_settings']['threshold'],'quality_pass':False}],
                    'selected':{'quality_pass':False,'evaluation_complete':False},'partial_results':results,
                    'stop_reason':'shared_budget','validation':frozen['validation'],'split':'test',**identity}
        write_json(lock,{'identity':identity,'result':result})
        if output:report_files(output,result)
        return result
    finally:ledger.reconcile_wall_clock()


def freeze_validation(checkpoint_path,baseline_path,config,device,output=None):
    """Measure both detector and renderer on validation, then freeze a portable profile."""
    import json
    from .adapt import ExperimentBudget, EvaluationBudgetExceeded, evaluate_adaptation, report_files
    from .engine import checkpoint_model,dataset_for
    from .runtime import digest,write_json
    root=Path(config['data_root']);run=Path(checkpoint_path).parent
    if (root/'final_test_lock.json').exists():raise ValueError('Final test consumed; validation tuning is frozen')
    destination=run/'validated.pt';frozen_path=run/'validation_frozen.json'
    if destination.exists() or frozen_path.exists():raise FileExistsError('Validation selection already frozen in this run')
    model,cp=checkpoint_model(checkpoint_path,device);baseline,old=checkpoint_model(baseline_path,device)
    if old['loaded_sha256']!=config.get('initialization_sha256') or cp['manifest_sha256']!=digest(config['manifest']):
        raise ValueError('Validation requires the pinned public baseline and the training manifest')
    ledger=ExperimentBudget(config['budget_ledger'],config['train']['max_seconds'],config['train']['max_updates'])
    reserve=config['train'].get('final_evaluation_reserve_seconds',1800)
    if not ledger.can_evaluate(reserve+60):raise ValueError('No validation budget remains after reserving final evaluation')
    ledger.start_wall_clock();results={};profiles={}
    try:
        for name,current,size in [('improved',model,config['inference_size']),('baseline',baseline,old['config']['image_size'])]:
            cfg=copy.deepcopy(config);cfg['image_size']=size;dataset=dataset_for(cfg,'val')
            try:
                raw=evaluate_adaptation(current,dataset,device,ledger=ledger,deadline_reserve=reserve)
            except EvaluationBudgetExceeded:
                write_json(run/'validation_incomplete.json',{'partial_results':results,'interrupted_model':name,'reason':'shared_budget'})
                raise
            sequential=evaluate_sequence(current,dataset,device,thresholds=(.3,.4,.5,.6,.7),ledger=ledger,deadline_reserve=reserve+10)
            result=combine_calibration(raw,sequential);results[name]=result
            if not sequential['complete']:
                write_json(run/'validation_incomplete.json',results)
                raise RuntimeError('Validation incomplete within its shared budget; final test remains unopened')
            chosen=result['selected']
            profiles[name]={'threshold':chosen['threshold'],'mask_threshold':.5,'foreground_only':True,'image_size':size,'alpha':.45}
            if getattr(current,'mask_projection',None)=='native_roi':profiles[name]['mask_projection']='native_roi'
            report_files(run/f'{name}_validation.json',result)
        cp['render_settings']=profiles['improved']
        cp['selection']={'evaluated_on_val':True,'step':cp['step'],'metrics':results['improved']['selected'],
                         'quality_pass':results['improved']['selected']['quality_pass']}
        cp.pop('loaded_sha256',None)
        temporary=destination.with_suffix('.tmp');torch.save(cp,temporary);temporary.replace(destination)
        frozen={'checkpoint_sha256':digest(destination),'baseline_sha256':old['loaded_sha256'],
                'manifest_sha256':cp['manifest_sha256'],'render_settings':profiles['improved'],
                'baseline_render_settings':profiles['baseline'],'validation':results['improved']['selected'],
                'baseline_validation':results['baseline']['selected'],'test_predictions_seen':False}
        write_json(frozen_path,frozen)
        if output:report_files(output,results['improved'])
        return {'checkpoint':str(destination),'selection':str(frozen_path),**frozen}
    finally:ledger.reconcile_wall_clock()
