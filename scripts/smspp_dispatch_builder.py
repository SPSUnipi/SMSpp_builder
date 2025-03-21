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

DIMENSION_KWARGS = [
    "TimeHorizon",
    "NumberUnits",
    "NumberElectricalGenerators",
    "NumberNodes",
    "NumberArcs",
    "NumberReservoirs",
    "TotalNumberPieces",
]

NC_DOUBLE = "f8"
NP_DOUBLE = np.float64
NC_UINT = "u4"
NP_UINT = np.uint32

def map_variable_type(name, value):
    if isinstance(value, pd.Series) or isinstance(value, list) or isinstance(value, np.ndarray):
        if name == "hydro":
            return ("NumberArcs",)
        else:
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
        dim = map_variable_type(name, value)
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

def add_master(ds, type, name="Block_0", n_timesteps=0, n_generators=0, n_elec_gens=None, id=None):
    mb = ds.createGroup(name)  # Create the master block
    if id is not None:
        mb.id = "0"
    if n_elec_gens is None:
        n_elec_gens = n_generators
    mb.type = type  # mandatory attribute for all blocks
    create_dimension(mb, "TimeHorizon", n_timesteps)  # Create the time horizon dimension
    create_dimension(mb, "NumberUnits", n_generators)  # Create the number of units
    create_dimension(mb, "NumberElectricalGenerators", n_elec_gens)  # Create number of electrical generators
    return mb

def get_bus_idx(n, bus_series, dtype="uint32"):
    """
    Returns the numeric index of the bus in the network n for each element of the bus_series.
    """
    return bus_series.map(n.buses.index.get_loc).astype(dtype)

def add_network(
        mb,
        n,
        bub_carriers,
        hub_carriers,
    ):

    # ndg = mb.createGroup("NetworkData")
    # ndg.createDimension("NumberNodes", len(n.buses))  # Number of nodes

    mb.createDimension("NumberNodes", len(n.buses))  # Number of nodes

    if len(n.buses) > 1:
        # NOTE: here we assume first generators are added, then storage units etc.
        all_generators = list(n.generators.bus.str[4:].astype(int).values)
        battery_units = list(n.storage_units[n.storage_units.index.isin(bub_carriers)].bus.str[4:].astype(int).values)
        battery_units += list(n.stores[n.stores.index.isin(bub_carriers)].bus.str[4:].astype(int).values)
        hydro_units = n.storage_units[n.storage_units.index.isin(hub_carriers)]
        all_generators += battery_units
        all_generators += list(np.repeat(hydro_units.bus.str[4:].astype(int).values, 3))  # each hydro unit has 3 arcs
        all_generators = [x for x in all_generators if x is not None]

        mb.createDimension("NumberLines", len(n.lines)+len(n.links)+len(n.transformers))  # Number of lines

        # generators' node
        generator_node = mb.createVariable("GeneratorNode", NC_UINT, ("NumberElectricalGenerators",))
        generator_node[:] = np.array(all_generators, dtype=NP_UINT)


        # start lines
        start_line = mb.createVariable("StartLine", NC_UINT, ("NumberLines",))
        start_line[:] = np.concatenate([
            get_bus_idx(n, n.lines.bus0).values,
            get_bus_idx(n, n.links.bus0).values,
            get_bus_idx(n, n.transformers.bus0).values,
        ])
        # end lines
        end_line = mb.createVariable("EndLine", NC_UINT, ("NumberLines",))
        end_line[:] = np.concatenate([
            get_bus_idx(n, n.lines.bus1).values,
            get_bus_idx(n, n.links.bus1).values,
            get_bus_idx(n, n.transformers.bus1).values,
        ])
        # Min power flow
        min_power_flow = mb.createVariable("MinPowerFlow", NC_DOUBLE, ("NumberLines",))
        min_power_flow[:] = np.concatenate([
            - n.lines.s_nom_opt.values,
            n.links.p_nom_opt.values * n.links.p_min_pu.values,
            - n.transformers.s_nom_opt,
        ])
        
        # Max power flow
        max_power_flow = mb.createVariable("MaxPowerFlow", NC_DOUBLE, ("NumberLines",))
        max_power_flow[:] = np.concatenate([
            n.lines.s_nom_opt.values,
            n.links.p_nom_opt.values * n.links.p_max_pu.values,
            n.transformers.s_nom_opt.values,
        ])

        # Susceptance
        susceptance = mb.createVariable("LineSusceptance", NC_DOUBLE, ("NumberLines",))
        if (n.lines.x != 0.).any():
            # TODO: to revise to support susceptance; as develop_AC_HVDC_mode PR is merged, this should be feasible
            logger.warning(
                f"Non-null line susceptance is not yet supported for lines: {n.lines[n.lines.x != 0.].index}\n"
                "Setting susceptance to 0.0"
            )
        susceptance[:] = 0.0


def add_demand(
        b,
        n,
    ):
    """
    Add the demand to the block
    """
    demand = n.loads_t.p_set.rename(columns=n.loads.bus)
    demand_matrix = demand.T.reindex(n.buses.index).fillna(0.)
    active_demand = b.createVariable("ActivePowerDemand", NC_DOUBLE, ("NumberNodes","TimeHorizon",)) #("NumberNodes", "TimeHorizon"))
    active_demand[:] = demand_matrix.values  # indexing between python and SMSpp is different: transpose
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
        dimension_kwargs=DIMENSION_KWARGS,
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
        if key in dimension_kwargs:
            create_dimension(tub, key, value)
        else:
            create_variable(tub, key, value)

    return tub

def get_renewable_blocks(n, id_initial, res_carrier):
    """
    Get the parameters of the renewable generators
    
    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network
    id_initial : int
        The initial id for the renewable generators
    res_carrier : list
        The list of renewable carriers
    """
    renewable_generators = n.generators[n.generators.index.isin(res_carrier)]

    id_renewable = id_initial

    p_max_pu = get_paramer_as_dense(n, "Generator", "p_max_pu", weights=False)

    rub_blocks = []
    for (idx_name, row) in renewable_generators.iterrows():
        rub_blocks.append(
            {
                "id": id_renewable,
                "block_type": "IntermittentUnitBlock",
                "MinPower": 0.0,
                "MaxPower": (row.p_nom_opt * p_max_pu.loc[:, idx_name]).values,
            }
        )
        id_renewable += 1
    return rub_blocks


def get_battery_blocks(n, id_initial, bub_carriers):
    """
    Get the parameters of the battery units

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network
    id_initial : int
        The initial id for the battery units
    bub_carriers : list
        The list of battery carriers

    Returns
    -------
    list
        The list of dictionaries with the parameters of the battery blocks
    """
    # TODO: extend to stores too
    battery_units = n.storage_units[n.storage_units.index.isin(bub_carriers)]

    id_battery = id_initial

    p_min_pu = get_paramer_as_dense(n, "StorageUnit", "p_min_pu", weights=False)
    p_max_pu = get_paramer_as_dense(n, "StorageUnit", "p_max_pu", weights=False)

    bub_blocks = []
    for (idx_name, row) in battery_units.iterrows():
        # when cycling, set negative initial storage
        init_store = row.state_of_charge_initial * row.p_nom_opt * row.max_hours
        if row.cyclic_state_of_charge:
            init_store = -1.
        
        bub_blocks.append(
            {
                "id": id_battery,
                "block_type": "BatteryUnitBlock",
                "MinPower": (row.p_nom_opt * p_min_pu.loc[:, idx_name]).values,
                "MaxPower": (row.p_nom_opt * p_max_pu.loc[:, idx_name]).values,
                "MinStorage": 0.0,
                "MaxStorage": row.p_nom_opt * row.max_hours,
                "InitialStorage": init_store,
                "StoringBatteryRho": row.efficiency_store,
                "ExtractingBatteryRho": row.efficiency_dispatch,
            }
        )
        id_battery += 1

    battery_units = n.stores[n.stores.index.isin(bub_carriers)]

    e_min_pu = get_paramer_as_dense(n, "Store", "e_min_pu", weights=False)
    e_max_pu = get_paramer_as_dense(n, "Store", "e_max_pu", weights=False)


    for (idx_name, row) in battery_units.iterrows():
        bub_blocks.append(
            {
                "id": id_battery,
                "block_type": "BatteryUnitBlock",
                "MinPower": (row.e_nom_opt * e_min_pu.loc[:, idx_name]).values * 10,
                "MaxPower": (row.e_nom_opt * e_max_pu.loc[:, idx_name]).values * 10,
                "MinStorage": 0.0,
                "MaxStorage": row.e_nom_opt,
                "InitialStorage": row.e_initial,
                "StoringBatteryRho": 1.0,
                "ExtractingBatteryRho": 1.0,
            }
        )
        id_battery += 1
    return bub_blocks


# def get_hydro_blocks(n, id_initial, hub_carriers):
#     """
#     Get the parameters of the hydro units

#     Parameters
#     ----------
#     n : pypsa.Network
#         The PyPSA network
#     id_initial : int
#         The initial id for the hydro units
#     hub_carriers : list
#         The list of hydro carriers

#     Returns
#     -------
#     list
#         The list of dictionaries with the parameters of the hydro blocks
#     """
#     hydro_systems = n.storage_units[n.storage_units.index.isin(hub_carriers)]

#     id_hydro = id_initial

#     N_ARCS = 3

#     p_min_pu = get_paramer_as_dense(n, "StorageUnit", "p_min_pu", weights=False)
#     p_max_pu = get_paramer_as_dense(n, "StorageUnit", "p_max_pu", weights=False)
#     inflow = get_paramer_as_dense(n, "StorageUnit", "inflow", weights=False)

#     hub_blocks = []
#     for (idx_name, row) in hydro_systems.iterrows():
#         hub_blocks.append(
#             {
#                 "id": id_hydro,
#                 "block_type": "HydroUnitBlock",
#                 "NumberReservoirs": 1,
#                 "NumberArcs": N_ARCS,
#                 "TotalNumberPieces": N_ARCS,
#                 "StartArc": np.full((N_ARCS,), 0, dtype=NP_UINT),
#                 "EndArc": np.full((N_ARCS,), 1, dtype=NP_UINT),
#                 "MinPower": np.array([0.0, 0.0, row.p_nom_opt * row.p_min_pu], dtype=NP_DOUBLE),
#                 "MaxPower": np.array([row.p_nom_opt * row.p_max_pu, 0.0, 0.0], dtype=NP_DOUBLE),
#                 "MinFlow": np.array([0.0, 0.0, 1.5*row.p_nom_opt * row.p_min_pu], dtype=NP_DOUBLE),
#                 "MaxFlow": np.array([1.5*row.p_nom_opt * row.p_max_pu, 1.5*row.p_nom_opt * row.p_max_pu, 0.0], dtype=NP_DOUBLE),
#                 "Inflows": np.array([inflow.loc[:, idx_name].values]),
#                 "MinVolumetric": 0.0,
#                 "MaxVolumetric": row.p_nom_opt * row.max_hours,
#                 "InitialVolumetric": row.state_of_charge_initial * row.p_nom_opt * row.max_hours,
#                 "LinearTerm": row.efficiency_store,
#                 "ConstantTerm": row.efficiency_dispatch,
#                 "NumberPieces": np.full((N_ARCS,), 1, dtype=NP_UINT),
#             }
#         )
#         id_hydro += 1
#     return hub_blocks

def add_hydro_unit_blocks(mb, n, unit_count, hub_carriers):
    """
    Add the hydro units to the master block.
    This is a raw basic implementation.
    
    Parameters
    ----------
    mb : netCDF4.Group
        The master block
    n : pypsa.Network
        The PyPSA network
    unit_count : int
        The current count of units
    hub_carriers : list
        The list of hydro carriers
    """

    hydro_systems = n.storage_units.loc[n.storage_units.index.isin(hub_carriers)]

    id_hydro = unit_count

    if not hydro_systems.empty:
        for (idx_name, row) in hydro_systems.iterrows():

            tiub = mb.createGroup(f"UnitBlock_{id_hydro}")
            tiub.id = str(id_hydro)
            tiub.type = "HydroUnitBlock"

            tiub.createDimension("NumberReservoirs", 1)  # optional, the number of reservoirs
            N_ARCS = 3  # First arc: production, second arc spillage, third arc pumping
            tiub.createDimension("NumberArcs", N_ARCS)  # optional, the number of arcs connecting the reservoirs
            # No NumberIntervals
            
            MAX_FLOW = 100*n.storage_units_t.inflow.loc[:, idx_name].max()
            P_MAX = row.p_nom_opt * row.p_max_pu
            P_MIN = row.p_nom_opt * row.p_min_pu

            # StartArc
            start_arc = tiub.createVariable("StartArc", NC_UINT, ("NumberArcs",))
            start_arc[:] = np.array([0, 0, 0], dtype=NP_UINT)
            # start_arc[:] = np.array([0, 0, 1], dtype=NP_UINT)

            # EndArc
            end_arc = tiub.createVariable("EndArc", NC_UINT, ("NumberArcs",))
            end_arc[:] = np.array([1, 1, 1], dtype=NP_UINT)
            # end_arc[:] = np.array([1, 1, 0], dtype=NP_UINT)

            # MaxPower
            max_power = tiub.createVariable("MaxPower", NC_DOUBLE, ("NumberArcs",)) #, ("NumberArcs",)) #, ("TimeHorizon",)) #"NumberArcs"))
            max_power[:] = np.array([P_MAX, 0., 0.], dtype=NP_DOUBLE)

            # MinPower
            min_power = tiub.createVariable("MinPower", NC_DOUBLE, ("NumberArcs",)) #, ("NumberArcs",)) #, ("TimeHorizon",)) #"NumberArcs"))
            min_power[:] = np.array([0., 0., P_MIN], dtype=NP_DOUBLE)

            # MinFlow
            min_flow = tiub.createVariable("MinFlow", NC_DOUBLE, ("NumberArcs",)) #, ("TimeHorizon",))
            min_flow[:] = np.array([0., 0., -MAX_FLOW], dtype=NP_DOUBLE)

            # MaxFlow
            max_flow = tiub.createVariable("MaxFlow", NC_DOUBLE, ("NumberArcs",)) #, ("TimeHorizon",))
            max_flow[:] = np.array([P_MAX * 100., MAX_FLOW, 0.], dtype=NP_DOUBLE)
            
            # MinVolumetric
            min_volumetric = tiub.createVariable("MinVolumetric", NC_DOUBLE) #, ("TimeHorizon",))
            min_volumetric[:] = 0.0

            # MaxVolumetric
            max_volumetric = tiub.createVariable("MaxVolumetric", NC_DOUBLE)
            max_volumetric[:] = row.p_nom_opt * row.max_hours

            
            # Inflows
            inflows = tiub.createVariable("Inflows", NC_DOUBLE, ("NumberReservoirs", "TimeHorizon")) #,"NumberReservoirs",))  #"NumberReservoirs", 
            inflows[:] = np.array([n.storage_units_t.inflow.loc[:, idx_name]])

            # InitialVolumetric
            initial_volumetric = tiub.createVariable("InitialVolumetric", NC_DOUBLE) #, ("NumberReservoirs",))
            # when cycling set negative initial storage            
            if row.cyclic_state_of_charge:
                initial_volumetric[:] = -1.
            else:
                initial_volumetric[:] = row.state_of_charge_initial * row.max_hours * row.p_nom_opt

            # NumberPieces
            pieces = np.full((N_ARCS,), 1, dtype=NP_UINT)
            number_pieces = tiub.createVariable("NumberPieces", NC_UINT, ("NumberArcs",))
            number_pieces[:] = pieces

            # TotalNumberPieces
            tiub.createDimension("TotalNumberPieces", pieces.sum())

            # LinearTerm
            linear_term = tiub.createVariable("LinearTerm", NC_DOUBLE, ("TotalNumberPieces",))
            # linear_term[:] = np.array([1/n.storage_units.loc[idx_name, "efficiency_dispatch"], 0., n.storage_units.loc[idx_name, "efficiency_store"]], dtype=NP_DOUBLE)
            linear_term[:] = np.array([1/n.storage_units.loc[idx_name, "efficiency_dispatch"], 0., n.storage_units.loc[idx_name, "efficiency_store"]], dtype=NP_DOUBLE)

            # ConstTerm
            const_term = tiub.createVariable("ConstantTerm", NC_DOUBLE, ("TotalNumberPieces",))
            const_term[:] = np.full((N_ARCS,), 0.0, dtype=NP_DOUBLE)

            id_hydro += 1

if __name__ == "__main__":
    if "snakemake" not in globals():
        from helpers import mock_snakemake

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        snakemake = mock_snakemake("smspp_dispatch_builder", configfiles=["configs/microgrid_Tr_2N.yaml"])
    
    logger = create_logger("smspp_dispatch_builder", logfile=snakemake.log[0])

    block_config = snakemake.params.block_config
    res_carriers = block_config["intermittent_unit_block_carriers"]
    bub_carriers = block_config["battery_unit_block_carriers"]
    hub_carriers = block_config["hydro_unit_block_carriers"]
    
    # Read PyPSA
    n = pypsa.Network(snakemake.input[0])
    n_timesteps = len(n.snapshots)
    n_generators = len(n.generators) + len(n.storage_units) + len(n.stores) # excluding links for now

    n_hydro = hydro_systems = n.storage_units.loc[n.storage_units.index.isin(hub_carriers)].shape[0]

    n_elec_gens = n_generators + 2 * n_hydro

    unit_count = 0

    # Initialize SMSpp file
    ds = create_smspp_file(snakemake.output[0])

    try:
    
        # Create master block as UCBlock for dispatching purposes
        mb = add_master(ds, "UCBlock", n_timesteps=n_timesteps, n_generators=n_generators, n_elec_gens=n_elec_gens)

        # Add network data to the master block
        ndg = add_network(mb, n, bub_carriers, hub_carriers)

        # Add demand data to the network block
        add_demand(mb, n)

        # Add thermal units to the master block
        tub_blocks = get_thermal_blocks(n, unit_count, res_carriers)
        for tub_block in tub_blocks:
            add_unit_block(mb, **tub_block)
        unit_count += len(tub_blocks)

        # Add renewable units to the master block
        rub_blocks = get_renewable_blocks(n, unit_count, res_carriers)
        for rub_block in rub_blocks:
            add_unit_block(mb, **rub_block)
        unit_count += len(rub_blocks)

        # Add battery units [only storage units for now]
        bub_blocks = get_battery_blocks(n, unit_count, bub_carriers)
        for bub_block in bub_blocks:
            add_unit_block(mb, **bub_block)
        unit_count += len(bub_blocks)

        # Add hydro units
        add_hydro_unit_blocks(mb, n, unit_count, hub_carriers)
        
    except Exception as e:
        logger.error(e)
        raise e
    finally:
        ds.close()

