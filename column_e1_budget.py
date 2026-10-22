"""Isolated, versioned 20-decision column E1 protocol and checkpoint migration."""
import ast
from collections import Counter
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from unittest.mock import patch

from column_e1.runtime import ColumnStudySession, digest_state, integrity_payload
from column_e1.checkpoints import NativeCheckpointManager
from review_experiments.e1.study import WINDOWS, independent_rng, json_safe

PROTOCOL_ID='column-e1-20-clock-strata-v1'
DECISIONS=20
STRATA=(('early',0,40,10),('middle',40,100,5),('late',100,200,5))


def choose_steps(candidates,seed,*labels):
    """A nested, outcome-blind subset of the original prospective schedule."""
    import numpy as np
    candidates=np.asarray(sorted(set(map(int,candidates))),dtype=int)
    if len(candidates)<DECISIONS:raise ValueError('Insufficient original eligible samples')
    groups={name:candidates[(candidates%200>=lo)&(candidates%200<hi)] for name,lo,hi,_ in STRATA}
    quotas={name:min(target,len(groups[name])) for name,_,_,target in STRATA}
    missing=DECISIONS-sum(quotas.values())
    for name,_,_,_ in STRATA:
        extra=min(missing,len(groups[name])-quotas[name]);quotas[name]+=extra;missing-=extra
    if missing:raise ValueError('Cannot meet the reduced decision budget')
    selected=[];mapping={}
    for name,_,_,_ in STRATA:
        n=quotas[name]
        if not n:continue
        rng=independent_rng(seed,PROTOCOL_ID,*labels,name)
        # One point from each time-ordered bin prevents concentrating all
        # selected points in a small part of a learning phase.
        for pool in np.array_split(groups[name],n):
            step=int(rng.choice(pool));selected.append(step);mapping[step]=name
    return sorted(selected),mapping,dict(original_candidates={k:len(v) for k,v in groups.items()},quotas=quotas)


def migrate_session(session,environment,protocol_sha):
    """Change diagnostics only; preserve every advancing controller binding."""
    if session.all_records or any(session.counts.values()):
        raise RuntimeError('Migration requires the verified pre-capture checkpoint')
    import numpy as np
    substeps=int(environment.get('time_in_sub_episodes',environment.get('ctx',{}).get('time_in_sub_episodes',0)))
    if substeps!=400:raise RuntimeError('Unexpected scientific episode clock')
    schedules={};windows={};long_steps=set();snapshots=set();strata={};design={}
    for site in session.compiled.sites.values():
        existing=session.schedules[site.key]
        chosen=[]
        for window,first,last in WINDOWS:
            pool=[step for step in existing if (first-1)*substeps<=step<last*substeps]
            if len(pool)!=100:raise RuntimeError('Original 100-decision schedule differs')
            steps,mapping,allocation=choose_steps(pool,session.options.seed,session.compiled.category,site.channel,window)
            chosen.extend(steps)
            for step in steps:
                windows[(site.key,step)]=window;strata[(site.key,step)]=mapping[step]
            rng=independent_rng(session.options.seed,PROTOCOL_ID,session.compiled.category,site.channel,window,'exports')
            long_steps.update((site.key,int(step)) for step in rng.choice(steps,2,replace=False))
            snapshots.add((site.key,int(rng.choice(steps))))
            design[f'{site.key}/{window}']=dict(selected_steps=steps,**allocation)
        schedules[site.key]=set(chosen)
    session.options=replace(session.options,decisions_per_window=DECISIONS)
    session.schedules=schedules;session.window_for_step=windows
    session.long_steps=long_steps;session.snapshot_steps=snapshots
    session.budget_strata=strata;session.budget_protocol_sha256=protocol_sha
    session.budget_design=design
    from column_review.common import write
    write(session.output/'sampling.json',dict(protocol_id=PROTOCOL_ID,protocol_sha256=protocol_sha,
         options=json_safe(vars(session.options)),windows=WINDOWS,design=design,
         selected_steps={key:sorted(value) for key,value in schedules.items()},
         long_steps=sorted(long_steps),snapshot_steps=sorted(snapshots),
         sampling='Nested, outcome-blind clock strata and time bins; no observed-return selection',
         interpretation='Selected-condition diagnostics, not unweighted overall operating-time failure rates',
         strata_note='Early/middle/late refer to position in the 200-step reference block, not measured settling.'))
    return design


class BudgetSession(ColumnStudySession):
    def after(self,site,environment):
        import review_experiments.e1.study as study
        original=study.json_safe
        def tagged(value):
            if isinstance(value,dict) and 'gate_source_line' in value and 'metrics' in value:
                value=dict(value,budget_protocol_id=PROTOCOL_ID,
                           budget_protocol_sha256=self.budget_protocol_sha256,
                           sampling_stratum=self.budget_strata[(site.key,int(value['step']))])
            return original(value)
        with patch.object(study,'json_safe',tagged):
            return super().after(site,environment)


def manager_class(protocol_sha,job,*,validation_steps=0):
    """Permit exactly the signed old checkpoint, then seal the new protocol."""
    class BudgetManager(NativeCheckpointManager):
        def __init__(self,directory,sources,**kwargs):
            kwargs.update(every_cycles=1)
            if validation_steps:
                kwargs.update(technical_stop=validation_steps,technical_resume_steps=validation_steps)
            super().__init__(directory,sources,**kwargs)
            self.original_sources=dict(self.sources)
            self.sources=dict(self.sources,column_e1_budget_protocol=protocol_sha)
            if not self.has_checkpoint:raise RuntimeError('Budget worker requires a complete checkpoint')
        def restore(self,environment,session):
            from column_review.common import read,write
            index=read(self.index);new_sources=dict(self.sources)
            migrating=index['source_hashes']==self.original_sources
            if migrating:
                authorized=job['authorized_checkpoint']
                if (index['sha256'],index['completed_steps'])!=(authorized['sha256'],authorized['completed_steps']):
                    raise RuntimeError('Original checkpoint is not the one approved for migration')
                self.sources=self.original_sources
            elif index['source_hashes']!=new_sources:
                raise RuntimeError('Checkpoint belongs to another protocol')
            try:super().restore(environment,session)
            finally:self.sources=new_sources
            if migrating:
                before=digest_state(integrity_payload(environment,self.compiled))
                migrate_session(session,environment,protocol_sha)
                after=digest_state(integrity_payload(environment,self.compiled))
                if before!=after:raise RuntimeError('Protocol migration changed live learner/controller state')
                write(self.directory/'budget_migration.json',dict(protocol_id=PROTOCOL_ID,
                      protocol_sha256=protocol_sha,restored_steps=self.restored_step,
                      original_checkpoint_sha256=index['sha256'],full_python_state_before=before,
                      full_python_state_after=after,controller_unchanged=True))
            elif getattr(session,'budget_protocol_sha256',None)!=protocol_sha:
                raise RuntimeError('Restored session protocol does not match its checkpoint')
            session.source_hashes=new_sources
    return BudgetManager


def adapted_worker(function,*,validation_steps=0):
    """Retain the original worker and change only count/technical-stop checks."""
    source,start=inspect.getsourcelines(function)
    tree=ast.parse(textwrap.dedent(''.join(source)))
    expected=stops=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected' for t in node.targets):
            if not isinstance(node.value,ast.IfExp):raise RuntimeError('Unexpected completion check')
            node.value=ast.parse("(4 if job['category']=='combined' else 1)*4*job['options']['decisions_per_window']",mode='eval').body
            expected+=1
        if isinstance(node,ast.ExceptHandler) and isinstance(node.type,ast.Name) and node.type.id=='CheckpointStop':
            test=node.body[0]
            if not isinstance(test,ast.If) or not isinstance(test.body[0],ast.Raise):
                raise RuntimeError('Unexpected controlled-stop handler')
            test.test=ast.parse('not technical and not __budget_validation_steps',mode='eval').body
            stops+=1
    if (expected,stops)!=(1,1):raise RuntimeError('Worker adaptation contract changed')
    ast.fix_missing_locations(tree);ast.increment_lineno(tree,start-1)
    namespace=dict(function.__globals__,__budget_validation_steps=validation_steps)
    exec(compile(tree,inspect.getsourcefile(function),'exec'),namespace)
    return namespace[function.__name__]
