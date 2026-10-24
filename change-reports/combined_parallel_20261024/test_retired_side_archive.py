import json
from pathlib import Path
import zipfile
import pytest
import retired_side_archive as a


def test_verified_lossless_archive_preserves_every_byte_before_duplicate_removal(tmp_path):
    working=tmp_path/'case15/working';folder=working/'load_complete';folder.mkdir(parents=True)
    inputs={'C2S_SS_simulation16.dynf':b'model-data'*500,'AM_case/snapshot.snp':bytes(range(256))*10,
            'AM_case/native.tmp':b'transient but preserved','AM_case/AspenAMLock.aml':b''}
    for name,data in inputs.items():
        path=folder/name;path.parent.mkdir(exist_ok=True);path.write_bytes(data)
    receipt=a.compress_verified(folder,tmp_path/'archive',prune=True,allowed_root=working)
    assert not folder.exists() and receipt['verified']
    with zipfile.ZipFile(receipt['archive']) as zipped:
        assert {name:zipped.read(name) for name in zipped.namelist()}==inputs
    assert Path(receipt['archive']).with_suffix('.pruned.json').exists()
    assert receipt['scientific_results_or_durable_inputs_removed']==0


def test_unsafe_folder_refuses_before_any_removal(tmp_path):
    folder=tmp_path/'unrelated';folder.mkdir();source=folder/'result.txt';source.write_text('keep')
    with pytest.raises(Exception):a.compress_verified(folder,tmp_path/'archive',prune=True,allowed_root=tmp_path/'case15/working')
    assert source.read_text()=='keep'
    with pytest.raises(RuntimeError,match='exact isolated'):a.compress_verified(folder,tmp_path/'archive',prune=True)


def pool_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'STATE',tmp_path)
    pool=tmp_path/'branches_0_test';pool.mkdir()
    decisions=[]
    for index in range(3):
        decisions.append(dict(channel='weights',step=index,batches={'0':{'branches':{
            'policy':{'native_identity':{'pid':100+index,'created':200+index}}}}}))
    (pool/'decisions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in decisions))
    for case in (15,17):
        child=pool/('case'+str(case));child.mkdir();events=[]
        if case==15:
            for index in range(3):
                folder=child/'working'/('load_'+str(index));folder.mkdir(parents=True)
                events.append(dict(event='opened_fresh_application',current={'pid':100+index,'created':200+index},
                                   model=str(folder/'C2S_SS_simulation16.dynf')))
        (child/'native_process_recovery.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in events))
    return pool


def test_two_newest_primary_comparisons_stay_unpacked(tmp_path,monkeypatch):
    pool=pool_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(a.p.h,'process',lambda pid:{'pid':pid,'alive':False,'created':None})
    candidates=a.eligible_pool(pool)
    assert len(candidates)==1 and candidates[0][0].name=='load_0'
    candidates[0][0].rmdir()
    assert a.eligible_pool(pool)==[]


def test_live_or_inaccessible_native_working_folders_are_ineligible(tmp_path,monkeypatch):
    pool=pool_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(a.p.h,'process',lambda pid:{'pid':pid,'alive':True,'created':200+pid-100})
    assert a.eligible_pool(pool)==[]
    monkeypatch.setattr(a.p.h,'process',lambda pid:{'pid':pid,'alive':None,'created':None})
    assert a.eligible_pool(pool)==[]
