"""Write the 45 full-data ImageNet, SSv2 and COCO bbox LoRA configurations."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

MODELS_FILE = Path(__file__).resolve().parents[1] / 'configs' / 'figure3-lora-models.json'
WORKERS = {'imagenet': 10, 'ssv2': 10, 'coco': 6}


def write_configs(paths, out):
    if not isinstance(paths, dict) or paths.keys() != {'encoders', 'datasets'}:
        raise ValueError('paths must specify encoders and datasets')
    models = json.loads(MODELS_FILE.read_bytes())
    for key, required in [('encoders', models), ('datasets', WORKERS)]:
        bindings = paths[key]
        if not isinstance(bindings, dict) or bindings.keys() != required.keys():
            raise ValueError(f'{key} must specify exactly {", ".join(required)}')
        if any(not isinstance(p, str) or not Path(p).is_absolute() for p in bindings.values()):
            raise ValueError(f'{key} requires absolute paths on the training host')
    if out.exists() or out.is_symlink():
        raise FileExistsError(f'output already exists: {out}')

    # Publish only after all files have been written on the destination filesystem.
    with TemporaryDirectory(dir=out.parent) as temp:
        stage = Path(temp) / 'configs'
        stage.mkdir(mode=0o700)
        for model, spec in models.items():
            backbone = {
                'kind': model, 'arch': spec['arch'], 'img_size': spec['img_size'], 'patch_size': 16,
                'encoder': paths['encoders'][model],
            }
            for task, workers in WORKERS.items():
                base = {
                    'profile': 'b5x_lora_dpt_v2', 'device': 'cuda', 'track': 'lora',
                    'task': task, 'data_root': paths['datasets'][task], 'backbone': backbone,
                    'workers': workers, 'continuation': True, 'worker_lifecycle': 'reseed_each_epoch',
                }
                if task == 'ssv2':
                    base['micro_batch_size'] = 8
                if task in spec.get('schedules', {}):
                    base['schedule'] = spec['schedules'][task]
                for seed in range(3):
                    filename = stage / f'{task}-{model}-s{seed}.json'
                    filename.write_text(json.dumps(dict(base, seed=seed), indent=2) + '\n')
        stage.rename(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, required=True, help='JSON containing local path bindings')
    parser.add_argument('--out', type=Path, required=True, help='new configuration directory')
    args = parser.parse_args()
    try:
        write_configs(json.loads(args.paths.read_bytes()), args.out)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
