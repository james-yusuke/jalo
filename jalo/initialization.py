"""Explicit semantic migration from the public global-mask model to local masks."""
import torch


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
    inherited = {row['target'] for row in report.get('transferred', [])}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not (freeze and name in inherited))
