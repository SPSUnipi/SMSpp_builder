"""
This script aims at verifying the dispatch of the SMS++ project with respect to PyPSA.
"""

import os
import pypsa
import re
import pandas as pd
import numpy as np
from helpers import create_logger

def get_marginal_costs(n):
    marg_cost = pd.concat(
        [
            n.generators_t.p.mul(n.snapshot_weightings.objective, axis=0).sum().mul(n.generators.marginal_cost),
            n.storage_units_t.p.mul(n.snapshot_weightings.objective, axis=0).sum().mul(n.storage_units.marginal_cost),
            n.links_t.p0.mul(n.snapshot_weightings.objective, axis=0).sum().mul(n.links.marginal_cost),
            n.stores_t.p.mul(n.snapshot_weightings.objective, axis=0).sum().mul(n.stores.marginal_cost),
        ],
    )
    return marg_cost

def get_capital_costs(n):
    cap_cost = pd.concat(
        [
            n.generators.eval("p_nom_opt * capital_cost"),
            n.storage_units.eval("p_nom_opt * capital_cost"),
            n.links.eval("p_nom_opt * capital_cost"),
            n.lines.eval("s_nom_opt * capital_cost"),
            n.stores.eval("e_nom_opt * capital_cost"),
        ]
    )
    return cap_cost

if __name__ == "__main__":
    if "snakemake" not in globals():
        from helpers import mock_snakemake

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        snakemake = mock_snakemake("verify_investment", configfiles=["configs/ALLbuthydrostore_1N.yaml"])
    
    logger = create_logger("verify_investment", logfile=snakemake.log[0])
    
    tolerances = snakemake.params.tolerances
    
    # Read PyPSA and get marginal costs
    n = pypsa.Network(snakemake.input.pypsa_network)
    obj = n.objective

    # Read SMS++ output and get objective value
    with open(snakemake.input.smspp_log, "r") as f:
        smspp_log = f.read()
    
    res = re.search("Solution value: (.*)\n", smspp_log)
    smspp_obj = float(res.group(1).replace("\r", ""))

    # check tolerances
    err_relative = (smspp_obj - obj)/(obj + 1e-6)
    err_absolute = (smspp_obj - obj)

    # Print results
    logger.info("SMS++ obj                             : %.6f" % smspp_obj)
    logger.info("PyPSA obj                             : %.6f" % obj)
    logger.info("Relative difference SMS++ - PyPSA [%%]: %.5f" % (100*err_relative))
    logger.info("Absolute difference SMS++ - PyPSA [€] : %.5f" % (err_absolute))

    error_flag_rel = False
    error_flag_abs = False
    # test tolerances
    if np.abs(err_relative) > tolerances["relative"]:
        logger.error("Relative error is too high")
        error_flag_rel = True
    if np.abs(err_absolute) > tolerances["absolute"]:
        logger.error("Absolute error is too high")
        error_flag_abs = True
    
    error_flag = error_flag_abs and error_flag_rel

    if error_flag:
        raise Exception("Verification failed")
    else:
        logger.info("Verification successful")