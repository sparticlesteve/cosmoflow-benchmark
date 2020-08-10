"""
Utilities for MLPerf logging.
Depends on the mlperf_logging package at
https://github.com/mlperf/logging
"""

# System
import os

# Externals
from mlperf_logging import mllog

def configure_mllogger(log_dir):
    """Setup the MLPerf logger"""
    mllog.config(filename=os.path.join(log_dir, 'mlperf.log'))
    return mllog.get_mllogger()

def log_submission_info(benchmark='cosmoflow',
                        org='UNDEFINED',
                        division='UNDEFINED',
                        status='UNDEFINED',
                        platform='UNDEFINED'):
    """Log general MLPerf submission details from config"""
    mllogger = mllog.get_mllogger()
    mllogger.event(key=mllog.constants.SUBMISSION_BENCHMARK, value=benchmark)
    mllogger.event(key=mllog.constants.SUBMISSION_ORG, value=org)
    mllogger.event(key=mllog.constants.SUBMISSION_DIVISION, value=division)
    mllogger.event(key=mllog.constants.SUBMISSION_STATUS, value=status)
    mllogger.event(key=mllog.constants.SUBMISSION_PLATFORM, value=platform)
