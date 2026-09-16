"""Initialize from the current ROI model or migrate historical global masks."""
import torch


def initialize_trained_roi(model, checkpoint):
    """Start a new experiment with every learned ROI tensor, without optimizer state."""
    if (getattr(model, 'architecture', None) != 'vehicle_roi_v2' or
            checkpoint.get('architecture') != 'vehicle_roi_v2' or
            checkpoint.get('task') != 'instance_segmentation' or
            checkpoint.get('variant') != 'single' or checkpoint.get('step', 0) < 1):
        raise ValueError('Initialization requires a trained single-frame ROI checkpoint')
    if checkpoint.get('classes') != ['car', 'truck', 'bus'] or model.num_classes != 3:
        raise ValueError('Initialization class order must be car, truck, bus')
    config = checkpoint['config']
    if (config['model']['heads'] != model.decoder[0].heads or
            config['model']['queries'] != model.queries or
            config.get('mask_projection', 'dense') != model.mask_projection or
            config.get('mask_objective', 'global') != model.mask_objective or
            config.get('foreground_supervision', 'feature') != model.foreground_supervision):
        raise ValueError('Initialization model semantics differ')
    source, target = checkpoint['model'], model.state_dict()
    if (set(source) != set(target) or any(source[name].shape != target[name].shape or
            not torch.isfinite(source[name]).all() for name in target)):
        raise ValueError('Incompatible trained ROI weights')
    # Learned box deltas and local heads must survive this transfer unchanged.
    model.load_state_dict(source, strict=True)
    return {'source_checkpoint_sha256': checkpoint.get('loaded_sha256'),
            'source_architecture': 'vehicle_roi_v2', 'target_architecture': 'vehicle_roi_v2',
            'source_step': checkpoint['step'],
            'source_manifest_sha256': checkpoint.get('manifest_sha256'),
            'transferred': [{'source': name, 'target': name, 'shape': list(value.shape),
                             'role': 'current trained ROI model'} for name, value in source.items()],
            'initialized': [], 'unused_source_keys': [],
            # All heads are learned already; there are no new modules to warm up.
            'warmup_frozen_targets': [], 'optimizer_restored': False,
            'transferred_parameters': sum(p.numel() for p in model.parameters()),
            'total_parameters': sum(p.numel() for p in model.parameters())}


def initialize_roi(model, checkpoint):
    """Copy learned features, never reinterpret absolute boxes as box deltas."""
    if (getattr(model, 'architecture', None) != 'vehicle_roi_v2' or
            checkpoint.get('architecture', 'jalo_v1') != 'jalo_v1' or
            checkpoint.get('task') != 'instance_segmentation' or
            checkpoint.get('variant') != 'single' or checkpoint.get('step', 0) < 1):
        raise ValueError('ROI initialization requires a trained single-frame JALO global-mask checkpoint')
    if checkpoint['classes'] != ['car', 'truck', 'bus'] or model.num_classes != 3:
        raise ValueError('Initialization class order must be car, truck, bus')
    source, target = checkpoint['model'], model.state_dict()
    mapping = {}
    for name in target:
        if name.startswith('backbone.'):
            mapping[name] = (name, 'backbone and pixel pyramid')
        elif name.startswith('decoder.'):
            _, layer, component, *tail = name.split('.')
            suffix = '.'.join(tail)
            if component in ('self_attention', 'norms', 'ffn'):
                mapping[name] = (name, component)
            elif component == 'cross_attention':
                mapping[name] = (f'decoder.{layer}.current_attention.{suffix}', 'current-frame cross attention')
            elif component == 'classifier':
                mapping[name] = (f'classifier.{suffix}', 'foreground/background classification')
            elif component == 'box' and tail[0] == '0':
                mapping[name] = (f'box_head.{suffix}', 'box hidden features; output is NOT transferred')
    # Validate the entire mapping before mutating any parameter.
    for name, (old, _) in mapping.items():
        if old not in source or source[old].shape != target[name].shape:
            raise ValueError(f'Incompatible transferred weight: {old} -> {name}')
        if not torch.isfinite(source[old]).all():
            raise ValueError(f'Nonfinite transferred weight: {old}')
    transferred = []
    for name, (old, role) in mapping.items():
        target[name] = source[old].detach().clone()
        transferred.append({'source': old, 'target': name, 'shape': list(target[name].shape), 'role': role})
    # Enforce identity refinements even if the caller constructed a modified model.
    for name in target:
        if name.startswith('decoder.') and '.box.2.' in name:
            target[name] = torch.zeros_like(target[name])
    model.load_state_dict(target, strict=True)
    parameters = dict(model.named_parameters())
    return {'source_checkpoint_sha256': checkpoint.get('loaded_sha256'),
            'source_architecture': 'jalo_v1', 'target_architecture': model.architecture,
            'transferred': transferred,
            'initialized': [{'target': n, 'shape': list(t.shape),
                             'method': 'zero box delta' if '.box.2.' in n else 'new module initialization'}
                            for n, t in target.items() if n not in mapping],
            'unused_source_keys': sorted(set(source) - {v[0] for v in mapping.values()}),
            'transferred_parameters': sum(parameters[n].numel() for n in mapping if n in parameters),
            'total_parameters': sum(p.numel() for p in parameters.values())}


def freeze_inherited(model, report, freeze):
    """Warmup freezes precisely the migrated parameters, including shared heads."""
    inherited = set(report.get('warmup_frozen_targets',[row['target'] for row in report.get('transferred', [])]))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not (freeze and name in inherited))


def continue_preliminary_roi(model, checkpoint):
    """Start a new annotation revision from learned local masks, preserving box deltas.

    This initializes weights only. Exact optimizer/data-order restoration remains
    the responsibility of --resume with its unchanged manifest.
    """
    import copy
    previous=checkpoint.get('initialization',{})
    if (not checkpoint.get('preliminary') or checkpoint.get('architecture')!='vehicle_roi_v2' or
        getattr(model,'architecture',None)!='vehicle_roi_v2' or checkpoint.get('step',0)<1 or
        checkpoint.get('task')!='instance_segmentation' or checkpoint.get('variant')!='single' or
        checkpoint.get('classes')!=['car','truck','bus'] or not previous.get('source_checkpoint_sha256')):
        raise ValueError('Continuation requires a trained preliminary ROI checkpoint with public-model lineage')
    cfg=checkpoint['config']
    if (cfg['model']['heads']!=model.decoder[0].heads or cfg['model']['queries']!=model.queries or
        cfg.get('mask_projection','dense')!=model.mask_projection or
        cfg.get('mask_objective','global')!=model.mask_objective or
        cfg.get('foreground_supervision','feature')!=model.foreground_supervision):
        raise ValueError('Continuation model semantics differ')
    source,target=checkpoint['model'],model.state_dict()
    if set(source)!=set(target) or any(source[n].shape!=target[n].shape or not torch.isfinite(source[n]).all() for n in target):
        raise ValueError('Incompatible continued ROI weights')
    model.load_state_dict(source,strict=True)
    frozen=previous.get('warmup_frozen_targets',[r['target'] for r in previous['transferred']])
    chain=copy.deepcopy(previous.get('continuation_chain',[]))
    chain.append({'checkpoint_sha256':checkpoint.get('loaded_sha256'),
                  'manifest_sha256':checkpoint['manifest_sha256'],'step':checkpoint['step']})
    return {'source_checkpoint_sha256':previous['source_checkpoint_sha256'],
            'parent_checkpoint_sha256':checkpoint.get('loaded_sha256'),
            'source_architecture':'vehicle_roi_v2','target_architecture':'vehicle_roi_v2',
            'transferred':[{'source':n,'target':n,'shape':list(t.shape),'role':'continued local model'} for n,t in source.items()],
            'initialized':[],'unused_source_keys':[],'warmup_frozen_targets':frozen,
            'continuation_chain':chain,'optimizer_restored':False,
            'transferred_parameters':sum(p.numel() for p in model.parameters()),
            'total_parameters':sum(p.numel() for p in model.parameters())}
