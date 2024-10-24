"""
This file aims at posing the key functions to build a sample microgrid model using PyPSA,
to be used as a reference for the SMS++ project.

The input is a configuration file and the script returns an optimized PyPSA model.
"""

import pypsa
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
import netCDF4 as nc
import pandas as pd
import numpy as np
import os
from helpers import create_logger

NC_DOUBLE = "f8"
NP_DOUBLE = np.float64
NC_UINT = "u4"
NP_UINT = np.uint32

def _type_0D_or_1D(value):
    if isinstance(value, pd.Series) or isinstance(value, list) or isinstance(value, np.ndarray):
        return ("TimeHorizon",)
    else:
        return ()

def create_variable(b, name, value, dtype=NC_DOUBLE, dim=None):
    """
    Create a variable in the block

    Parameters
    ----------
    b : netCDF4.Group
        The block where the variable will be created
    name : str
        The name of the variable
    value : int or float or np.ndarray or pd.Series
        The value of the variable
    dtype : (optional) str
        The data type of the variable, default double
    dim : (optional) tuple
        The dimensions of the variable, default None
    """
    if dim is None:
        dim = _type_0D_or_1D(value)
    var = b.createVariable(name, dtype, dim)
    var[:] = value
    return var

def create_dimension(b, name, value):
    """
    Create a dimension in the block

    Parameters
    ----------
    b : netCDF4.Group
        The block where the dimension will be created
    name : str
        The name of the dimension
    value : int
        The value of the dimension
    """
    b.createDimension(name, value)


def get_paramer_as_dense(n, component, field, weights=True):
    """
    Get the parameters of a component as a dense DataFrame

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network
    component : str
        The component to get the parameters from
    field : str
        The field to get the parameters from

    Returns
    -------
    pd.DataFrame
        The parameters of the component as a dense DataFrame
    """
    sns = n.snapshots
    if not n.investment_period_weightings.empty:  # TODO: check with different version
        periods = sns.unique("period")
        period_weighting = n.investment_period_weightings.objective[periods]
    weighting = n.snapshot_weightings.objective
    if not n.investment_period_weightings.empty:
        weighting = weighting.mul(period_weighting, level=0).loc[sns]
    else:
        weighting = weighting.loc[sns]
    field_val = get_as_dense(n, component, field, sns)
    if weights:
        field_val = field_val.mul(weighting, axis=0)
    return field_val

def create_smspp_file(fp, attribute=1):
    ds = nc.Dataset(fp, "w")
    ds.setncattr("SMS++_file_type", attribute)  # Set file type to 1 for problem file
    return ds

def add_master(ds, type, name="Block_0", n_timesteps=0, n_units=0, id=None):
    mb = ds.createGroup(name)  # Create the master block
    if id is not None:
        mb.id = "0"
    mb.type = type  # mandatory attribute for all blocks
    create_dimension(mb, "TimeHorizon", n_timesteps)  # Create the time horizon dimension
    create_dimension(mb, "NumberUnits", n_units)  # Create the number of units
    create_dimension(mb, "NumberElectricalGenerators", n_units)  # TODO: to change
    return mb

def add_network(
        b,
        n_nodes,
        n_lines=0,
        start_line=None,
        end_line=None,
        min_power_flow=None,
        max_power_flow=None,
        susceptance=None,
    ):
    """
    Add the network to the block
    """
    # TODO: currently is single-node only
    if n_nodes > 1:
        raise NotImplementedError("Only single-node networks are supported")
    nb = b.createGroup("NetworkData")
    create_dimension(nb, "NumberNodes", 1)
    return nb

def add_demand(
        b,
        demand,
    ):
    """
    Add the demand to the block
    """
    active_demand = b.createVariable("ActivePowerDemand", NC_DOUBLE, ("TimeHorizon",)) #("NumberNodes", "TimeHorizon"))
    active_demand[:] = demand.values.transpose()  # indexing between python and SMSpp is different: transpose
    return active_demand

def get_thermal_blocks(n, id_initial, res_carrier):
    """
    Get the parameters of the thermal generators

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network
    id_initial : int
        The initial id for the thermal generators
    res_carrier : list
        The list of renewable carriers

    Returns
    -------
    list
        The list of dictionaries with the parameters of the thermal blocks
    """
    thermal_generators = n.generators[~n.generators.index.isin(res_carrier)]

    id_thermal = id_initial

    marginal_cost = get_paramer_as_dense(n, "Generator", "marginal_cost")
    p_min_pu = get_paramer_as_dense(n, "Generator", "p_min_pu", weights=False)
    p_max_pu = get_paramer_as_dense(n, "Generator", "p_max_pu", weights=False)

    tub_blocks = []
    for (idx_name, row) in thermal_generators.iterrows():
        tub_blocks.append(
            {
                "id": id_thermal,
                "block_type": "ThermalUnitBlock",
                "MinPower": (row.p_nom_opt * p_min_pu.loc[:, idx_name]).values,
                "MaxPower": (row.p_nom_opt * p_max_pu.loc[:, idx_name]).values,
                "StartUpCost": 0.0,
                "LinearTerm": marginal_cost.loc[:, idx_name].values,
                "ConstantTerm": 0.0,
                "MinUpTime": 0.0,
                "MinDownTime": 0.0,
                "InitialPower": 0.0, #n.loads_t.p_set.iloc[0, id_initial],
                "InitUpDownTime": 1.0,
                "InertiaCommitment": 1.0,
            }
        )
        id_thermal += 1
    return tub_blocks

def add_unit_block(
        b,
        id,
        block_type,
        **kwargs,
    ):
    """
    Add a unit block to the block

    Parameters
    ----------
    b : netCDF4.Group
        The block where the unit block will be created
    id : int
        The id of the unit block
    block_type : str
        The type of the unit block
    kwargs : dict
        The parameters of the unit block
    """
    tub = b.createGroup(f"UnitBlock_{id}")
    tub.id = str(id)
    tub.type = block_type

    for key, value in kwargs.items():
        create_variable(tub, key, value)

    return tub

# def get_renewable_blocks():



#     renewable_generators = n.generators[n.generators.index.isin(renewable_carriers)]

# id_ren = id_thermal

# if not renewable_generators.empty:
#     for (idx_name, row) in renewable_generators.iterrows():

#         tiub = master.createGroup(f"UnitBlock_{id_ren}")
#         tiub.id = str(id_ren)
#         tiub.type = "IntermittentUnitBlock"

#         n_max_power = n.generators_t.p_max_pu.loc[:, idx_name] * row.p_nom_opt
        

#         # max power
#         max_power = tiub.createVariable("MaxPower", NC_DOUBLE, ("TimeHorizon",))
#         max_power[:] = n_max_power

#         # # max capacity
#         # max_capacity = tiub.createVariable("MaxCapacity", NC_DOUBLE)
#         # max_capacity[:] = row[1].p_nom_max 

#         id_ren += 1
    


if __name__ == "__main__":
    if "snakemake" not in globals():
        from helpers import mock_snakemake

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        snakemake = mock_snakemake("smspp_dispatch_builder")
    
    logger = create_logger("smspp_dispatch_builder", logfile=snakemake.log[0])

    res_carriers = snakemake.params.renewable_carriers
    
    # Read PyPSA
    n = pypsa.Network(snakemake.input[0])
    n_timesteps = len(n.snapshots)
    n_units = len(n.generators) + len(n.storage_units)  # excluding links for now

    unit_count = 0

    # Initialize SMSpp file
    ds = create_smspp_file(snakemake.output[0])

    try:
    
        # Create master block as UCBlock for dispatching purposes
        mb = add_master(ds, "UCBlock", n_timesteps=n_timesteps, n_units=n_units)

        # Add network data to the master block
        add_network(mb, n.buses.shape[0])

        # Add demand data to the master block
        add_demand(mb, n.loads_t.p_set)

        # Add thermal units to the master block
        tub_blocks = get_thermal_blocks(n, unit_count, res_carriers)
        for tub_block in tub_blocks:
            add_unit_block(mb, **tub_block)
        unit_count += len(tub_blocks)
    except Exception as e:
        logger.error(e)
    finally:
        ds.close()

