"""Select one successful, validly judged attempt per SVG task."""
def select_attempts(attempts):
    selected = {}
    for attempt in attempts:
        episode, judge = attempt['episode'], attempt.get('judge')
        if not episode.get('ok') or episode.get('error'):
            continue
        if not judge or not judge.get('parse_ok') or judge.get('judge_error'):
            continue
        if not isinstance(judge.get('cheated'), bool):
            continue
        selected.setdefault(episode['id'], attempt)
    return selected
