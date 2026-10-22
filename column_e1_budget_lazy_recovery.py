"""Initialize a pre-gate checkpoint's prospective schedule before migration."""
from unittest.mock import patch


def migration_function(original, archived_sampling):
    def migrate(session, environment, protocol_sha):
        initialized = not session.schedules
        if initialized:
            if session.options.decisions_per_window != 100:
                raise RuntimeError('Lazy migration needs the original 100-point options')
            session._initialize(environment)
            actual = {key: sorted(value) for key, value in session.schedules.items()}
            if actual != archived_sampling['selected_steps']:
                raise RuntimeError('Rebuilt prospective schedule differs from its preserved original')
        session.budget_lazy_initialized = initialized
        return original(session, environment, protocol_sha)
    return migrate


def manager_factory(original_factory, original_migrate, archived_sampling, repair_sha):
    def factory(protocol_sha, job, *, validation_steps=0):
        base = original_factory(protocol_sha, job, validation_steps=validation_steps)
        class LazyManager(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.sources = dict(self.sources, column_e1_budget_lazy_protocol=repair_sha)

            def restore(self, environment, session):
                import column_e1_budget as budget
                from column_review.common import read, write
                migrate = migration_function(original_migrate, archived_sampling)
                with patch.object(budget, 'migrate_session', migrate):
                    super().restore(environment, session)
                path = self.directory/'budget_migration.json'
                receipt = read(path)
                receipt.update(lazy_recovery_protocol_sha256=repair_sha,
                               original_schedule_rebuilt_and_verified=getattr(session, 'budget_lazy_initialized', False),
                               checkpoint_sources=self.sources)
                write(path, receipt)
        return LazyManager
    return factory
