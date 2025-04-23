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

from smspp_dispatch_builder import (
    create_variable,
    create_dimension,
    map_variable_type,
    NC_DOUBLE,
    NP_DOUBLE,
    NC_UINT,
    NP_UINT,
    NC_BYTE,
    NP_BYTE,
    DIMENSION_KWARGS,
    create_smspp_file,
    get_paramer_as_dense,
    add_master,
    get_bus_idx,
    add_demand,
    add_unit_block,
)

OBJ_ORDER=["Generator", "Store", "StorageUnit", "Link", "Line"]
NETWORK_ASSETS = ["Line", "Link"]

CARRIER_ORDER=["diesel", "pv", "wind", "curtailment", "battery", "hydro"]

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
            -n.lines.s_max_pu.values,
            n.links.p_min_pu.values,
            -n.transformers.s_max_pu.values,
        ])
        
        # Max power flow
        max_power_flow = mb.createVariable("MaxPowerFlow", NC_DOUBLE, ("NumberLines",))
        max_power_flow[:] = np.concatenate([
            n.lines.s_max_pu.values,
            n.links.p_max_pu.values,
            n.transformers.s_max_pu.values,
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


def get_thermal_blocks(n, id_initial, ther_carriers):
    """
    Get the parameters of the thermal generators

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network
    id_initial : int
        The initial id for the thermal generators
    ther_carriers : list
        The list of renewable carriers

    Returns
    -------
    list
        The list of dictionaries with the parameters of the thermal blocks
    """
    thermal_generators = n.generators[n.generators.carrier.isin(ther_carriers)]

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
                "MinPower": p_min_pu.loc[:, idx_name].values,
                "MaxPower": p_max_pu.loc[:, idx_name].values,
                "StartUpCost": 0.0,
                "LinearTerm": marginal_cost.loc[:, idx_name].values,
                "ConstantTerm": 0.0,
                "MinUpTime": 0.0,
                "MinDownTime": 0.0,
                "InitialPower": 1.0, #n.loads_t.p_set.iloc[0, id_initial],
                "InitUpDownTime": 1.0,
                "InertiaCommitment": 1.0,
            }
        )
        id_thermal += 1
    return tub_blocks

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
    renewable_generators = n.generators[n.generators.carrier.isin(res_carrier)]

    id_renewable = id_initial

    p_max_pu = get_paramer_as_dense(n, "Generator", "p_max_pu", weights=False)

    rub_blocks = []
    for (idx_name, row) in renewable_generators.iterrows():
        rub_blocks.append(
            {
                "id": id_renewable,
                "block_type": "IntermittentUnitBlock",
                "MinPower": 0.0,
                "MaxPower": p_max_pu.loc[:, idx_name].values,
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
                "MinPower": p_min_pu.loc[:, idx_name].values,
                "MaxPower": p_max_pu.loc[:, idx_name].values,
                "MinStorage": 0.0,
                "MaxStorage": row.max_hours,
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
                "MinPower": e_min_pu.loc[:, idx_name].values * 10,
                "MaxPower": e_max_pu.loc[:, idx_name].values * 10,
                "MinStorage": 0.0,
                "MaxStorage": row.e_nom_opt,
                "InitialStorage": row.e_initial,
                "StoringBatteryRho": 1.0,
                "ExtractingBatteryRho": 1.0,
            }
        )
        id_battery += 1
    return bub_blocks

def get_slack_blocks(n, id_initial, slack_carrier):
    """
    Get the parameters of the slack generators
    
    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network
    id_initial : int
        The initial id for the slack generators
    slack_carrier : list
        The list of slack carriers
    """
    slack_generators = n.generators[n.generators.carrier.isin(slack_carrier)]

    id_slack = id_initial

    marginal_cost = get_paramer_as_dense(n, "Generator", "marginal_cost")

    sub_blocks = []
    for (idx_name, row) in slack_generators.iterrows():
        sub_blocks.append(
            {
                "id": id_slack,
                "block_type": "SlackUnitBlock",
                # "MinPower": 0.0,
                "MaxPower": np.repeat(row.p_nom_opt, n_timesteps),
                "ActivePowerCost": marginal_cost.loc[:, idx_name].values,
            }
        )
        id_slack += 1
    return sub_blocks


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
            N_ARCS = 2  # First arc: production, second arc spillage, third arc pumping
            tiub.createDimension("NumberArcs", N_ARCS)  # optional, the number of arcs connecting the reservoirs
            # No NumberIntervals
            
            MAX_FLOW = 100*n.storage_units_t.inflow.loc[:, idx_name].max()
            P_MAX = row.p_max_pu
            P_MIN = row.p_min_pu

            # StartArc
            start_arc = tiub.createVariable("StartArc", NC_UINT, ("NumberArcs",))
            start_arc[:] = np.array([0, 0], dtype=NP_UINT)
            # start_arc[:] = np.array([0, 0, 1], dtype=NP_UINT)

            # EndArc
            end_arc = tiub.createVariable("EndArc", NC_UINT, ("NumberArcs",))
            end_arc[:] = np.array([1, 1], dtype=NP_UINT)
            # end_arc[:] = np.array([1, 1, 0], dtype=NP_UINT)

            # MaxPower
            max_power = tiub.createVariable("MaxPower", NC_DOUBLE, ("NumberArcs",)) #, ("NumberArcs",)) #, ("TimeHorizon",)) #"NumberArcs"))
            max_power[:] = np.array([P_MAX, 0.], dtype=NP_DOUBLE)

            # MinPower
            min_power = tiub.createVariable("MinPower", NC_DOUBLE, ("NumberArcs",)) #, ("NumberArcs",)) #, ("TimeHorizon",)) #"NumberArcs"))
            min_power[:] = np.array([0., P_MIN], dtype=NP_DOUBLE)

            # MinFlow
            min_flow = tiub.createVariable("MinFlow", NC_DOUBLE, ("NumberArcs",)) #, ("TimeHorizon",))
            min_flow[:] = np.array([0., -MAX_FLOW], dtype=NP_DOUBLE)

            # MaxFlow
            max_flow = tiub.createVariable("MaxFlow", NC_DOUBLE, ("NumberArcs",)) #, ("TimeHorizon",))
            max_flow[:] = np.array([MAX_FLOW, 0.], dtype=NP_DOUBLE)
            
            # MinVolumetric
            min_volumetric = tiub.createVariable("MinVolumetric", NC_DOUBLE) #, ("TimeHorizon",))
            min_volumetric[:] = 0.0

            # MaxVolumetric
            max_volumetric = tiub.createVariable("MaxVolumetric", NC_DOUBLE)
            max_volumetric[:] = row.max_hours

            
            # Inflows
            inflows = tiub.createVariable("Inflows", NC_DOUBLE, ("NumberReservoirs", "TimeHorizon")) #,"NumberReservoirs",))  #"NumberReservoirs", 
            inflows[:] = np.array([n.storage_units_t.inflow.loc[:, idx_name]])

            # InitialVolumetric
            initial_volumetric = tiub.createVariable("InitialVolumetric", NC_DOUBLE) #, ("NumberReservoirs",))
            # when cycling set negative initial storage            
            if row.cyclic_state_of_charge:
                initial_volumetric[:] = -1.
            else:
                initial_volumetric[:] = row.state_of_charge_initial * row.max_hours

            # NumberPieces
            pieces = np.full((N_ARCS,), 1, dtype=NP_UINT)
            number_pieces = tiub.createVariable("NumberPieces", NC_UINT, ("NumberArcs",))
            number_pieces[:] = pieces

            # TotalNumberPieces
            tiub.createDimension("TotalNumberPieces", pieces.sum())

            # LinearTerm
            linear_term = tiub.createVariable("LinearTerm", NC_DOUBLE, ("TotalNumberPieces",))
            # linear_term[:] = np.array([1/n.storage_units.loc[idx_name, "efficiency_dispatch"], 0., n.storage_units.loc[idx_name, "efficiency_store"]], dtype=NP_DOUBLE)
            linear_term[:] = np.array([n.storage_units.loc[idx_name, "efficiency_dispatch"], 1/n.storage_units.loc[idx_name, "efficiency_store"]], dtype=NP_DOUBLE)

            # ConstTerm
            const_term = tiub.createVariable("ConstantTerm", NC_DOUBLE, ("TotalNumberPieces",))
            const_term[:] = np.full((N_ARCS,), 0.0, dtype=NP_DOUBLE)

            id_hydro += 1

# get the nominal name
def nom_obj(obj):
    if obj.lower() == "line":
        return "s_nom"
    elif obj.lower() == "store":
        return "e_nom"
    else:
        return "p_nom"

# get list of component by type
def get_param_list(dict_objs, col, max_val=1e6):
    """
    Get the list of parameters for each object in the network
    """
    lvals = []
    for obj in OBJ_ORDER:
        if col == "type":
            lvals += [obj for i in range(len(dict_objs[obj].index))]
        elif col == "id": # get the id starting from the first object
            lvals += list(dict_objs[obj].index)
        elif col == "smspp_asset": # get the id starting from the first object
            lvals += list(dict_objs[obj].loc[:, "smspp_asset"].values)
        else:
            df_col = nom_obj(obj) + col[3:] if col.startswith("nom_") else col
            lvals += list(dict_objs[obj][df_col].values)
    if isinstance(lvals[0], float) or isinstance(lvals[0], int):
        lvals = [np.clip(val, -max_val, max_val) for val in lvals]
    return lvals

def get_extendable_dict(n, exclude_carriers=[]):
    def _sort_order(x):
        carrier_sort = {carrier: id for id, carrier in enumerate(CARRIER_ORDER)}
        if x in carrier_sort.keys():
            return carrier_sort[x]
        else:
            return max(carrier_sort.values()) + 1
    
    # get the extendable objects
    dict_extendable = {
        obj: (
            n.df(obj)
            .sort_values(by=["carrier"], key=lambda x: x.map(lambda y: _sort_order(y)))
            .reset_index()
            .rename(columns={obj: "name"})
            .query(f"carrier not in {exclude_carriers}")
            .query(f"{nom_obj(obj)}_extendable == True")
            .assign(smspp_asset=lambda x: (1 if obj in NETWORK_ASSETS else 0))
        )
        for obj in OBJ_ORDER
    }

    # network assets
    pre_nid = len(n.df(NETWORK_ASSETS[0]))
    for nobj in NETWORK_ASSETS[1:]:
        dict_extendable[nobj].index = dict_extendable[nobj].index + pre_nid
        pre_nid += len(dict_extendable[nobj].index)
    
    # other assets
    UNIT_ASSETS = [obj for obj in OBJ_ORDER if obj not in NETWORK_ASSETS]
    pre_uid = len(n.df(UNIT_ASSETS[0]))
    for uobj in UNIT_ASSETS[1:]:
        dict_extendable[uobj].index = dict_extendable[uobj].index + pre_uid
        pre_uid += len(dict_extendable[uobj].index)

    # Number of extendable assets
    n_extendable = sum(len(df.index) for df in dict_extendable.values())

    return dict_extendable, n_extendable

def add_investment_block(ds, n, exclude_carriers=[]):
    """
    Add the investment block to the dataset
    Parameters
    ----------
    ds : netCDF4.Dataset
        The dataset
    n : pypsa.Network
        The PyPSA network
    """
    dict_extendable, n_extendable = get_extendable_dict(n, exclude_carriers)

    b = ds.createGroup("InvestmentBlock")  # Create the first main block

    # master.id = "0"  # mandatory attribute for all blocks
    b.type = "InvestmentBlock"  # mandatory attribute for all blocks

    # num of extendables
    b.createDimension("NumAssets", n_extendable)

    # assets
    assets = b.createVariable("Assets", NC_UINT, ("NumAssets",))
    assets[:] = get_param_list(dict_extendable, "id")

    # investment cost
    cost = b.createVariable("Cost", NC_DOUBLE, ("NumAssets",))
    cost[:] = get_param_list(dict_extendable, "capital_cost")

    # Lower bound
    lb = b.createVariable("LowerBound", NC_DOUBLE, ("NumAssets",))
    lb[:] = np.full((n_extendable,), 1e-3, dtype=NP_DOUBLE)
    # lb[:] = get_param_list(dict_extendable, "nom_min")

    # Upper bound
    ub = b.createVariable("UpperBound", NC_DOUBLE, ("NumAssets",))
    ub[:] = get_param_list(dict_extendable, "nom_max")

    # Installed Capacity
    ic = b.createVariable("InstalledCapacity", NC_DOUBLE, ("NumAssets",))
    ic[:] = np.full((n_extendable,), 0., dtype=NP_DOUBLE)

    # asset type
    asset_type = b.createVariable("AssetType", NC_BYTE, ("NumAssets",))
    asset_type[:] = get_param_list(dict_extendable, "smspp_asset")

    return b


if __name__ == "__main__":
    if "snakemake" not in globals():
        from helpers import mock_snakemake

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        snakemake = mock_snakemake("smspp_investment_builder", configfiles=["configs/ALLbuthydrostore_1N.yaml"])
    
    logger = create_logger("smspp_investment_builder", logfile=snakemake.log[0])

    block_config = snakemake.params.block_config
    res_carriers = block_config["intermittent_unit_block_carriers"]
    ther_carriers = block_config["thermal_unit_block_carriers"]
    bub_carriers = block_config["battery_unit_block_carriers"]
    hub_carriers = block_config["hydro_unit_block_carriers"]
    sub_carriers = block_config["slack_unit_block_carriers"]
    
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
        # Add investment block
        b = add_investment_block(ds, n, sub_carriers)
    
        # Create master block as UCBlock for dispatching purposes
        mb = add_master(b, "UCBlock", name="InnerBlock", n_timesteps=n_timesteps, n_generators=n_generators, n_elec_gens=n_elec_gens)

        # Add network data to the master block
        ndg = add_network(mb, n, bub_carriers, hub_carriers)

        # Add demand data to the network block
        add_demand(mb, n)

        # Add thermal units to the master block
        tub_blocks = get_thermal_blocks(n, unit_count, ther_carriers)
        for tub_block in tub_blocks:
            add_unit_block(mb, **tub_block)
        unit_count += len(tub_blocks)

        # Add renewable units to the master block
        rub_blocks = get_renewable_blocks(n, unit_count, res_carriers)
        for rub_block in rub_blocks:
            add_unit_block(mb, **rub_block)
        unit_count += len(rub_blocks)

        # Add battery units [only storage units for now]
        sub_blocks = get_slack_blocks(n, unit_count, sub_carriers)
        for sub_block in sub_blocks:
            add_unit_block(mb, **sub_block)
        unit_count += len(sub_blocks)

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

