"""Streamed guard for unexpected task reviewer failures (stdlib only)."""


def install_grader_guard(source, namespace):
    import functools
    import inspect
    import json
    import sys

    def report(exc):
        print('\x1eRHGRADER_FATAL ' + json.dumps({
            'error': f'{type(exc).__name__}: {exc}'}), file=sys.stderr, flush=True)

    def guarded(fn):
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_call(*args, **kwargs):
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    report(exc)
                    raise
            return async_call

        @functools.wraps(fn)
        def call(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                report(exc)
                raise
        return call

    try:
        exec(compile(source, '<task-grader>', 'exec'), namespace)
    except Exception as exc:
        report(exc)
        raise
    if 'make_tool' in namespace:
        original = namespace['make_tool']

        @guarded
        def make_tool(*args, **kwargs):
            state, call = original(*args, **kwargs)
            return state, guarded(call)
        namespace['make_tool'] = make_tool
    if 'make_extra_tools' in namespace:
        extra_factory = namespace['make_extra_tools']

        @guarded
        def make_extra_tools(*args, **kwargs):
            return [dict(spec, call=guarded(spec['call']))
                    for spec in extra_factory(*args, **kwargs)]
        namespace['make_extra_tools'] = make_extra_tools
    if 'make_grader' in namespace:
        namespace['make_grader'] = guarded(namespace['make_grader'])
