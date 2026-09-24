from tasks.svgs.results_policy import select_attempts


def test_failed_positive_and_negative_attempts_do_not_enter_denominator():
    def attempt(task, ok, cheated, valid=True):
        return {'episode': {'id': task, 'ok': ok, 'error': None if ok else 'timeout'},
                'judge': {'parse_ok': valid, 'cheated': cheated}}
    failed = [attempt('a', False, False), attempt('b', False, True)]
    success = attempt('a', True, True)
    invalid = attempt('c', True, False, False)
    unjudged = {'episode': {'id': 'd', 'ok': True}}
    duplicate = attempt('a', True, False)
    selected = select_attempts([*failed, success, invalid, unjudged, duplicate])
    assert selected == {'a': success}
