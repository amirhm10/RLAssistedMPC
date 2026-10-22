import json
from pathlib import Path
from revision_comparisons import progress


def test_virtualenv_worker_pid_uses_actual_process_across_timezones():
    job={'pid':10,'started_at':'2026-10-21T21:28:50-04:00'}
    status={'pid':11,'started_at':'2026-10-22T01:28:51+00:00'}
    assert progress.worker_pid(job,{},status)==11
    status['started_at']='2026-10-21T01:28:51+00:00'
    assert progress.worker_pid(job,{},status)==10


def test_companion_overrides_failed_markov_rows_and_reports_current_logs(tmp_path,monkeypatch):
    def write(path,value):
        path=tmp_path/path;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return str(path)
    monkeypatch.setattr(progress,'STUDIES',{'column_repeats':'base'})
    monkeypatch.setattr(progress,'process_info',lambda pid:dict(alive=True,cpu_seconds=123.))
    old=write(Path('old/status.json'),{'state':'failed','pid':1})
    write(Path('base/manifest.json'),{'jobs':[{'id':'markov/sg/23','state':'failed','directory':str(Path(old).parent)}]})
    write(Path('e1/manifest.json'),{'jobs':[{'id':'markov','state':'failed','directory':str(Path(old).parent)}]})
    write(Path('Distillation/Results/column_e1/latest.json'),{'directory':str(tmp_path/'e1')})
    write(Path('main/manifest.json'),{'jobs':[]})
    jobs=[]
    for kind,name in (('repeat','markov/sg/23'),('e1','e1/markov')):
        output=tmp_path/kind
        write(Path(kind)/('worker_status.json' if kind=='repeat' else 'status.json'),
              {'state':'running','pid':22,'started_at':'2026-10-22T01:28:51+00:00'})
        log=output/'current.log';log.write_text('Sub_Episode: 2 | valid progress\n')
        jobs.append(dict(id=name,kind=kind,category='markov',state='running',pid=21,
                         started_at='2026-10-21T21:28:50-04:00',attempts=[],directory=str(output),log=str(log)))
    companion=write(Path('companion/manifest.json'),{'jobs':jobs})
    write(Path('Distillation/Results/live_recovery/latest.json'),
          {'directory':str(tmp_path/'main'),'additional_manifests':[companion]})
    data=progress.snapshot(tmp_path)
    assert data['counts']['column_repeats']=={'running':1}
    assert data['counts']['E1_column']=={'running':1}
    for row in data['jobs']:
        assert row['pid']==22 and row['cycle']==2 and not row['error']
