"""Portable measurement provenance; an integrity check, not a security signature."""
import hashlib
import json
import copy


def inference_config(config):
    return {k:copy.deepcopy(config[k]) for k in ('task','architecture','classes','model','image_size',
                                               'mask_projection','mask_objective','foreground_supervision') if k in config}


def model_digest(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        value=tensor.detach().cpu().contiguous()
        h.update(json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode())
        h.update(value.reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def certificate(checkpoint, final_test):
    identity=final_test.get('identity',{})
    if (identity.get('checkpoint_sha256')!=checkpoint['loaded_sha256'] or
        identity.get('manifest_sha256')!=checkpoint.get('manifest_sha256') or
        identity.get('render_settings')!=checkpoint.get('render_settings')):
        raise ValueError('Final evaluation does not belong to this checkpoint and profile')
    result=final_test.get('result') or {}
    validation=result.get('validation',checkpoint.get('selection',{}).get('metrics',{}))
    test=result.get('selected',{})
    return {'format_version':1,'model_sha256':model_digest(checkpoint['model']),
            'inference_config':inference_config(checkpoint.get('config',{})),
            'manifest_sha256':checkpoint['manifest_sha256'],'render_settings':copy.deepcopy(checkpoint['render_settings']),
            'validation':copy.deepcopy(validation),'test':copy.deepcopy(test),
            'quality_pass':bool(validation.get('quality_pass') and test.get('quality_pass')),
            'reference_type':'AI-created, visually reviewed polygons; no independent human verification',
            'scope':'Only the named held-out daylight source; no safety or generalization certification'}


def validate_certificate(checkpoint, settings):
    cert=checkpoint.get('quality_certificate',{})
    if (cert.get('format_version')!=1 or cert.get('model_sha256')!=model_digest(checkpoint['model']) or
            cert.get('inference_config')!=inference_config(checkpoint.get('config',{})) or
            cert.get('manifest_sha256')!=checkpoint.get('manifest_sha256') or
            cert.get('render_settings')!=checkpoint.get('render_settings') or not cert.get('quality_pass') or
            not cert.get('validation',{}).get('quality_pass') or not cert.get('test',{}).get('quality_pass') or
            not cert.get('test',{}).get('sufficient_targets') or not cert.get('test',{}).get('evaluation_complete')):
        raise ValueError('Full export requires recorded passing validation and final evaluation; use --duration for a diagnostic preview')
    if any(settings[k]!=cert['render_settings'].get(k) for k in settings):
        raise ValueError('Full export requires validated render settings; use --duration for overridden settings')
