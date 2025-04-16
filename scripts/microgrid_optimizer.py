"""
This script optimizes the microgrid network using PyPSA.
"""

import os
import pypsa
from helpers import create_logger

if __name__ == "__main__":
    if "snakemake" not in globals():
        from helpers import mock_snakemake

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        snakemake = mock_snakemake("microgrid_optimizer", configfiles=["configs/microgrid_ALLbutStore_1N_cycling.yaml"])
    
    logger = create_logger("microgrid_optimizer", logfile=snakemake.log[0])
    
    logger.info(f"Loading PyPSA network")
    n = pypsa.Network(snakemake.input[0])

    logger.info(f"Optimizing PyPSA network")
    n.optimize(
        solver_name=snakemake.params.solver_name,
        solver_options=snakemake.params.solver_options,
    )

    logger.info(f"Saving PyPSA network")
    n.export_to_netcdf(snakemake.output[0])
    