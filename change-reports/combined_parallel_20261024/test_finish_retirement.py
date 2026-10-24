import pytest
from finish_retirement import server_rows


def test_console_hosts_are_inspected_but_never_signaled():
    rows=[{'pid':1,'name':'sim_server.exe'},{'pid':2,'name':'conhost.exe'},
          {'pid':3,'name':'am_task_server.exe'},{'pid':4,'name':'conhost.exe'}]
    servers,ignored=server_rows(rows)
    assert [r['pid'] for r in servers]==[1,3]
    assert [r['pid'] for r in ignored]==[2,4]


def test_unknown_descendants_refuse_all_cleanup():
    with pytest.raises(RuntimeError,match='untouched'):server_rows([{'name':'other_training.exe','pid':99}])
