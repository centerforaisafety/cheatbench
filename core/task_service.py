"""Optional task-owned host services, scoped to one runner invocation."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path


def parse_service(root: Path, raw) -> dict | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) - {'module', 'config'}:
        raise ValueError('task.yaml service must contain only module and config')
    module = raw.get('module')
    if not isinstance(module, str) or not module:
        raise ValueError('task.yaml service.module must name a Python file')
    path = (root / module).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or path.suffix != '.py':
        raise ValueError('task.yaml service.module must be a Python file inside the task')
    config = raw.get('config', {})
    if not isinstance(config, dict):
        raise ValueError('task.yaml service.config must be a mapping')
    return {'module': module, 'config': dict(config)}


@contextmanager
def open_service(task, *, rows, out_dir, load_model):
    """The hook returns a context manager; it owns startup and guaranteed cleanup.

    Hooks may update the selected in-memory rows' staged-file mappings. They must
    not rewrite the shared dataset. load_model resolves the run's models.yaml.
    """
    if task.service is None:
        yield
        return
    path = task.root / task.service['module']
    spec = importlib.util.spec_from_file_location(f'task_service_{task.name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'start', None)):
        raise ValueError(f'{path}: service must define start(...)')
    with module.start(task=task, rows=rows, out_dir=out_dir,
                      config=task.service['config'], load_model=load_model) as service:
        yield service
