"""Bounded same-model retries for the sealed loader's document-open fault."""
from contextlib import contextmanager
import inspect
import time
from unittest.mock import patch

SERVER_FAULT=-2147417851


def open_line(loader):
    lines,start=inspect.getsourcelines(loader)
    matches=[start+i for i,line in enumerate(lines) if 'plant.ad.openDocument(' in line]
    if len(matches)!=1:raise RuntimeError('Sealed native loader document-open contract differs')
    return matches[0]


def is_open_fault(error,loader,line):
    code=getattr(error,'hresult',error.args[0] if error.args else None)
    if code!=SERVER_FAULT:return False
    frame=error.__traceback__
    while frame:
        if frame.tb_frame.f_code is loader.__code__ and frame.tb_lineno==line:return True
        frame=frame.tb_next
    return False


def retry(loader,plant,metadata,output,owned,record,*,sleeper=time.sleep,limit=3):
    if limit!=3:raise ValueError('Only three bounded document-open attempts are permitted')
    line=open_line(loader)
    for attempt in range(1,limit+1):
        try:
            native=loader(plant,metadata,output)
        except Exception as error:
            if not is_open_fault(error,loader,line):raise
            # The generation-safe helper must confirm this exact disposable
            # application belongs to the current worker before any retry.
            identity=owned(plant,output)
            record(output,dict(event='document_open_retry' if attempt<limit else 'document_open_retry_exhausted',
                attempt=attempt,limit=limit,hresult=SERVER_FAULT,identity=identity,
                native_model_sha256=metadata['project_file_sha256'],native_snapshot_sha256=metadata['file_sha256'],
                scientific_state_or_bounds_changed=False))
            if attempt==limit:raise
            sleeper(2*attempt)
            # The original loader rehashes/copies the exact same saved model
            # and snapshot, retires only its owned failed application and
            # creates another owned process. It performs no advancing step.
            continue
        if attempt>1:
            record(output,dict(event='document_open_retry_recovered',attempt=attempt,
                native_model_sha256=metadata['project_file_sha256'],native_snapshot_sha256=metadata['file_sha256']))
        return native


@contextmanager
def installed(fresh):
    original=fresh.load_project
    def loading(plant,metadata,output):
        return retry(original,plant,metadata,output,fresh.owned_identity,fresh.append)
    with patch.object(fresh,'load_project',loading):yield
