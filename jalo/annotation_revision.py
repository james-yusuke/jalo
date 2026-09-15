"""Explicitly migrate training state after a source-only reference correction.

Ordinary resume still requires an identical manifest. This separate operation
invalidates all model selection and refuses any image/split or test-lock change.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import torch
from .engine import load_checkpoint
from .runtime import digest, load_config, write_json


def assert_same_images(before, after):
    for key in ('dataset', 'task', 'classes', 'seed', 'source_sha256', 'intervals'):
        if before[key] != after[key]:
            raise ValueError(f'Annotation revision changed {key}')
    fields = ('id', 'second', 'split', 'frame_index', 'source_timestamp_seconds',
              'width', 'height', 'path', 'sha256')
    for split in ('train', 'val', 'test'):
        old = [{k: x[k] for k in fields} for x in before['splits'][split]]
        new = [{k: x[k] for k in fields} for x in after['splits'][split]]
        if old != new:
            raise ValueError(f'Annotation revision changed images/order/split: {split}')


def migrate_reference_checkpoint(checkpoint_path, config_path, output, reason):
    checkpoint = load_checkpoint(checkpoint_path)
    config = load_config(config_path)
    old_config = checkpoint['config']
    if checkpoint.get('architecture') != 'vehicle_roi_v2':
        raise ValueError('Reference migration supports the custom ROI architecture only')
    mutable = {'data_root', 'manifest', 'run_dir', 'annotation_revision'}
    if {k: v for k, v in old_config.items() if k not in mutable} != {
            k: v for k, v in config.items() if k not in mutable}:
        raise ValueError('Reference migration must preserve model and training settings')
    for cfg in (old_config, config):
        if (Path(cfg['data_root']) / 'final_test_lock.json').exists():
            raise ValueError('Cannot revise references after final test is consumed')
    if digest(old_config['manifest']) != checkpoint['manifest_sha256']:
        raise ValueError('Original reference manifest was modified')
    before = json.loads(Path(old_config['manifest']).read_text())
    after = json.loads(Path(config['manifest']).read_text())
    assert_same_images(before, after)
    if digest(Path(config['data_root']) / 'annotations.json') != after['annotations_sha256']:
        raise ValueError('Corrected annotations changed after prepare')
    audit = json.loads((Path(config['data_root']) / 'review_audit.json').read_text())
    if audit['manifest_sha256'] != digest(config['manifest']) or len(audit['frames']) != 169:
        raise ValueError('Corrected annotation review is incomplete or stale')
    lineage = {'reason': reason, 'source_checkpoint': str(Path(checkpoint_path).resolve()),
               'source_checkpoint_sha256': checkpoint['loaded_sha256'],
               'prior_manifest_sha256': checkpoint['manifest_sha256'],
               'manifest_sha256': digest(config['manifest']),
               'prior_annotations_sha256': before['annotations_sha256'],
               'annotations_sha256': after['annotations_sha256'],
               'preserved_step': checkpoint['step'], 'selection_invalidated': True,
               'test_predictions_used': False,
               'state_preserved': ['model', 'optimizer', 'scheduler', 'rng', 'stream',
                                   'step', 'phase', 'phase_seconds', 'phase_steps']}
    config['annotation_revision'] = lineage
    # Keep the on-disk config identical to the migrated checkpoint for strict resume.
    import yaml
    migrated = copy.copy(checkpoint)
    migrated.pop('loaded_sha256', None)
    migrated.update(config=config, manifest_sha256=lineage['manifest_sha256'],
                    annotations_sha256=lineage['annotations_sha256'], best_rank=(-1., -1., -1.),
                    selection={}, render_settings={'threshold': .3, 'mask_threshold': .5,
                        'foreground_only': True, 'image_size': config['inference_size'], 'alpha': .45})
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        torch.save(migrated, handle)
    Path(config_path).write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    write_json(output.parent / 'annotation_revision.json', lineage)
    return lineage
