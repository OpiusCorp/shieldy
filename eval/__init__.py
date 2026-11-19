"""Detector evaluation harness package."""

import os

# Ensure matplotlib and tensorflow behave cleanly and don't stall on font cache
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
