"""Frozen twenty-position selection; fixtures remain host-only."""
import json
from pathlib import Path
POSITIONS = ('ogs_69046172', 'ogs_68506069', 'ogs_63091411', 'ogs_73830699_m100', 'ogs_70802180_m220', 'ogs_69949962_m244', 'ogs_70491025_m64', 'ogs_60554907_m52', 'ogs_69856603', 'ogs_72720761_m136', 'ogs_60546947_m160', 'ogs_69059098_m136', 'ogs_75636183_m172', 'ogs_66564206_m64', 'ogs_64055438_m52', 'ogs_85538987_m136', 'ogs_76510140_m196', 'ogs_65184427_m100', 'ogs_64536659_m40', 'ogs_69418396_m40')

def positive_int(value):
    number = int(value)
    if number < 1:
        raise ValueError('Expected a positive integer')
    return number
