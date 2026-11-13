"""Bind the new report, App default and packaged frozen experiment to one config."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]


def configuration(root=ROOT):
    path=root/'artifacts/final_v4.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def default_folder(available,root=ROOT):
    final=configuration(root)
    preferred=str((root/final['primary_folder']).resolve()) if final else None
    return preferred if preferred in available else available[0]


def report_matches(folder,root=ROOT):
    final=configuration(root)
    return bool(final and Path(folder).resolve()==(root/final['primary_folder']).resolve())


def data_dictionary(folder,root=ROOT):
    local=Path(folder)/'DATA_DICTIONARY.md'
    return local if local.is_file() else root/'data/processed/DATA_DICTIONARY.md'


def showcase(folder,root=ROOT):
    if not report_matches(folder,root):return None
    final=configuration(root)
    if not final.get('showcase_path'):return None
    import hashlib
    path=root/final['showcase_path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=final['showcase_sha256']:
        raise ValueError('Report showcase changed after verification.')
    data=json.loads(path.read_text())
    if hashlib.sha256((Path(folder)/'primary_gnn.pt').read_bytes()).hexdigest()!=data['checkpoint_sha256']:
        raise ValueError('Report showcase belongs to another checkpoint.')
    return data
