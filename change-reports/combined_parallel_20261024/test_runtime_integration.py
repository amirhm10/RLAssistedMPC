"""Generation-safe delegated ownership and conditional handoff refusal."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import parallel_branches as p
from runtime_integration import merge_owned,closing_stack,ManagedBranchPool
import recover_parallel as activation


def test_delegated_ownership_keeps_parent_and_both_reused_pid_generations():
    parent=[dict(pid=1,created=10,alive=True),dict(pid=2,created=20,alive=True)]
    children=[dict(pid=2,created=21,alive=True),dict(pid=3,created=30,alive=True)]
    before=deepcopy((parent,children))
    merged=merge_owned(parent,children,case=15,journal=Path('case15/owned.json'),worker_identity={'pid':9,'created':90})
    assert {(r['pid'],r['created']) for r in merged}=={(1,10),(2,20),(2,21),(3,30)}
    assert merged[2]['case']==15 and merged[2]['branch_worker']['pid']==9
    assert (parent,children)==before
    assert merge_owned(merged,children,case=15,journal=Path('case15/owned.json'),worker_identity={'pid':9,'created':90})==merged


def test_pool_retires_before_original_worker_final_ownership_check():
    events=[];pool=SimpleNamespace(close=lambda:events.append('pool'))
    with closing_stack(pool)() as stack:
        stack.callback(lambda:events.append('main'))
        stack.callback(lambda:events.append('branch'))
    events.append('final_ownership_check')
    assert events==['branch','main','pool','final_ownership_check']


def test_ownership_aggregation_refuses_an_unrelated_journal(tmp_path):
    pool=object.__new__(ManagedBranchPool)
    pool.template={'aggregate_journal':str(tmp_path/'foreign.json')};pool.next_owner_sync=0.
    with pytest.raises(RuntimeError,match='Only this combined'):pool.sync_owners(force=True)
    assert not (tmp_path/'foreign.json').exists()


def plan_fixture(tmp_path):
    plan=dict(category='combined',case=16,seed=7,expected_matched_pairs=320,
        decisions_per_window=20,main_training_cycles=300,
        authorization='user_approved_verified_parallel_migration',net_saving_after_development_and_replay_hours=4.)
    for key in ('pilot_acceptance','transport_acceptance'):
        path=tmp_path/(key+'.json');path.write_text(json.dumps({'passed':True,'enough_time_saved':True,'ready_for_conditional_migration':True}))
        plan[key]={'path':str(path),'sha256':p.h.digest(path)}
    return plan


def test_conditional_restart_refuses_unproven_or_changed_receipts(tmp_path):
    plan=plan_fixture(tmp_path);activation.scope(plan)
    plan['pilot_acceptance']['sha256']='wrong'
    with pytest.raises(RuntimeError,match='changed'):activation.scope(plan)


def test_conditional_restart_refuses_insufficient_net_saving(tmp_path):
    plan=plan_fixture(tmp_path);plan['net_saving_after_development_and_replay_hours']=2.99
    with pytest.raises(RuntimeError,match='three net hours'):activation.scope(plan)


def test_conditional_restart_never_duplicates_the_live_combined_worker(tmp_path):
    plan=plan_fixture(tmp_path);plan['failed_worker']={'pid':15940,'created':134373287822018420,'alive':True}
    with pytest.raises(RuntimeError,match='live, reused or inaccessible'):
        activation.r.start_guard(plan,identity=lambda pid:dict(pid=pid,alive=True,created=134373287822018420),
                                 resources=lambda:{'headroom_bytes':100*2**30},applications=[],status={'pid':15940})
