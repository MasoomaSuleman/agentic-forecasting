---
name: code-analysis-playbook
description: >-
  How to use the code sandbox to diagnose USD/CAD return history before forecasting.
---

# Code-analysis playbook

All numerical data are in the JSON payload; there are no disk files or network access.

Parse once:
import io
import pandas as pd
df = pd.read_csv(io.StringIO(payload["target_history_csv"]))

Compute recent trend/returns, recent standard deviation, and min/max moves. Use
these diagnostics to calibrate uncertainty and avoid implausible point forecasts.
Positive DEXCAUS return means USD strengthened against CAD.
