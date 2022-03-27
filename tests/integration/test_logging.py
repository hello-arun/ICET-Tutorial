"""
Test logging module in icet, e.g. output to file etc.
"""

import logging
import tempfile
from io import StringIO
from ase.build import bulk
from icet import ClusterSpace
from icet.input_output.logging_tools import set_log_config, logger, formatter


# Test logger
# --------------
structure = bulk('Al')
cutoffs = [4.0]
symbols = ['Al', 'Ti']

for log_level in [logging.DEBUG, logging.INFO, logging.WARNING]:

    # Log ClusterSpace output to StringIO stream
    for handler in logger.handlers:
        logger.removeHandler(handler)

    stream = StringIO()
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    stream_handler.setLevel(logging.INFO)

    cs = ClusterSpace(structure, cutoffs, symbols)
    stream_handler.flush()
    lines1 = stream.getvalue().split('\n')[:-1]  # remove last blank line

    # Log ClusterSpace output to file
    for handler in logger.handlers:
        logger.removeHandler(handler)

    logfile = tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False)
    set_log_config(filename=logfile.name, level=logging.INFO)

    cs = ClusterSpace(structure, cutoffs, symbols)
    logfile.seek(0)
    lines2 = [m.replace('\n', '') for m in logfile.readlines()]

    # assert lines1 (from stringIO stream) and lines (from file stream) are equal
    assert len(lines1) == len(lines2)
    for l1, l2 in zip(lines1, lines2):
        assert l1 == l2
