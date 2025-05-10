"""
DEA (Data Envelopment Analysis) implementation for mobile plan ranking.
"""

from modules.dea.dea_scipy import run_scipy_dea
from modules.dea.dea_run import run_dea_analysis
from modules.dea import calculate_rankings_with_dea, calculate_rankings_with_ties
