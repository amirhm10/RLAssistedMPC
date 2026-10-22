"""Full training checkpoints for isolated recovery workers, without new E1 sampling."""
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import time
import uuid
from unittest.mock import patch


class TrainingSession:
    def __init__(self, compiled, output):
        self.compiled, self.output = compiled, Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        self.started=time.perf_counter()
        self.all_records=[]
        self.active_key=self.native=None
        self.method_restore={}
    def wants(self,*args): return False
    def is_active(self,*args): return False
    def before(self,*args): pass
    def after(self,*args): pass
    def finish(self,*args): pass
    def detach(self): pass


@contextmanager
def repeat_checkpoints(job_path, *, stop_after=None):
    import column_e1_recovery
    column_e1_recovery.install()
    from column_review.common import read,digest,write
    from column_e1.checkpoints import NativeCheckpointManager
    from revision_recovery.checkpoints import install_runner
    from systems.distillation.plant import DistillationColumnAspen
    job_path=Path(job_path);job=read(job_path);out=job_path.parent
    root=Path(__file__).parent
    sources={name:digest(root/name) for name in ('review_training_recovery.py','review_numeric_recovery.py','column_e1_recovery.py')}
    sources.update(read(Path(job['batch_directory'])/'manifest.json')['source_hashes'])
    manager=NativeCheckpointManager(out/'training_checkpoints',sources,technical_stop=stop_after)
    original_init,original_close=DistillationColumnAspen.__init__,DistillationColumnAspen.close
    def initialize(plant,path,*args,**kwargs):
        if Path(path).resolve()!=Path(job['aspen_path']).resolve():
            raise RuntimeError('Recovery worker requested an unassigned Aspen case')
        folder=out/('native_'+uuid.uuid4().hex[:8]);folder.mkdir()
        model=folder/Path(path).name
        shutil.copyfile(path,model)
        write(out/'native_working_case.json',dict(canonical_case=str(path),private_model=str(model),source_sha256=digest(path)))
        return original_init(plant,str(model),*args,**kwargs)
    def close(plant,snaps_path=None,prefix='snp'):
        return original_close(plant,None,prefix)
    def factory(compiled,args,kwargs):
        return TrainingSession(compiled,out/'checkpoint_session')
    with patch.object(DistillationColumnAspen,'__init__',initialize),patch.object(DistillationColumnAspen,'close',close), \
         install_runner(job['category'],factory,manager):
        yield manager


def full_policy_manager():
    from recover_polymer_threshold import PolicyExportManager
    from revision_recovery.checkpoints import CheckpointManager
    from column_review.common import digest
    root=Path(__file__).parent
    class FullPolicyManager(PolicyExportManager):
        def __init__(self,directory,metadata):
            super().__init__(directory,metadata)
            sources=dict(metadata['source_hashes'])
            sources.update({name:digest(root/name) for name in ('review_training_recovery.py','review_numeric_recovery.py')})
            self.full=CheckpointManager(Path(directory).parent/'training_checkpoints',sources)
            self.full.compiled=self._compiled
        @property
        def compiled(self):return self._compiled
        @compiled.setter
        def compiled(self,value):
            self._compiled=value
            if hasattr(self,'full'):self.full.compiled=value
        @property
        def has_checkpoint(self):return hasattr(self,'full') and self.full.has_checkpoint
        def save(self,environment,session):
            super().save(environment,session)
            self.full.save(environment,session)
        def restore(self,environment,session):
            return self.full.restore(environment,session)
    return FullPolicyManager
