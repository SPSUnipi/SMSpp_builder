# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 08:12:37 2024

@author: aless
"""

import pandas as pd
import pypsa

class Transformation:
    """
    Transformation class for converting the components of a PyPSA energy network into unit blocks.
    In particular, these are ready to be implemented in SMS++

    The class takes as input a PyPSA network.
    It reads the specified network components and converts them into a dictionary of unit blocks (`unitblocks`).
    
    Attributes:
    ----------
    unitblocks : dict
        Dictionary that holds the parameters for each unit block, organized by network components.
    
    IntermittentUnitBlock_parameters : dict
        Parameters for an IntermittentUnitBlock, like solar and wind turbines.
        The values set to a float number are absent in Pypsa, while lambda functions are used to get data from
        Pypa DataFrames
    
    ThermalUnitBlock_parameters : dict
        Parameters for a ThermalUnitBlock
    """

    def __init__(self, n):
        """
        Initializes the Transformation class.

        Parameters:
        ----------
        
        n : PyPSA Network
            PyPSA energy network object containing components such as generators and storage units.
            
        Methods:
        ----------
        init : Start the workflow of the class
        
        """
        
        # Attribute for unit blocks
        self.unitblocks = dict()
        
        # Parameters for intermittent units
        self.IntermittentUnitBlock_parameters = {
            "Gamma": 0.0,
            "Kappa": 0.0,
            "MaxPower": lambda p_nom_opt, p_max_pu: p_nom_opt * p_max_pu,
            "MinPower": lambda p_nom_opt, p_min_pu: p_nom_opt * p_min_pu,
            "IntertiaPower": 0.0
        }
        
        # Parameters for thermal units
        self.ThermalUnitBlock_parameters = {
            "InitUpDownTime": lambda up_time_before: up_time_before,
            "MinUpTime": lambda min_up_time: min_up_time,
            "MinDownTime": lambda min_down_time: min_down_time,
            "DeltaRampUp": lambda ramp_limit_up: ramp_limit_up,
            "DeltaRampDown": lambda ramp_limit_down: ramp_limit_down,
            "MinPower": lambda p_nom_opt, p_min_pu: p_nom_opt * p_min_pu,
            "MaxPower": lambda p_nom_opt, p_max_pu: p_nom_opt * p_max_pu,
            "PrimaryRho": 0.0,
            "SecondaryRho": 0.0,
            "Availability": 0.0,
            "QuadTerm": lambda marginal_cost_quadratic: marginal_cost_quadratic,
            "LinearTerm": lambda marginal_cost: marginal_cost,
            "ConstTerm": 0.0,
            "StartUpCost": lambda start_up_cost: start_up_cost,
            "InitialPower": lambda p: p[0],
            "FixedConsumption": 0.0,
            "InertiaCommitment": 1.0
        }
        
        # Initialize with the parser and network
        self.init(n)
        
    def init(self, n):
        """
        Initialization method describing the workflow of the class.

        Parameters:
        ----------

        n : PyPSA Network
            PyPSA network object for the energy network.
            
        Methods : iterate_components
            It iterates over all the components to convert them into UnitBlocks
        """
        self.iterate_components(n)
        
        """
        Reads the parameter file using the path specified in the parser.

        Parameters:
        ----------
        parser : object
            Object containing paths for the input files.

        Sets:
        --------
        self.parameters : dict
            Dictionary of DataFrames containing parameters from different sheets in the Excel file.
        """
        self.parameters = pd.read_excel(f"{parser.input_data_path}/{parser.input_name_parameters}", sheet_name=None)
        
    def add_UnitBlock(self, attr_name, components_df, components_t):
        """
        Adds a unit block to the `unitblocks` dictionary for a given component.

        Parameters:
        ----------
        attr_name : str
            Attribute name containing the unit block parameters (Intermittent or Thermal).
        
        components_df : DataFrame
            DataFrame containing information for a single component.
            For example, n.generators.loc['wind']

        components_t : DataFrame
            Temporal DataFrame (e.g., snapshot) for the component.
            For example, n.generators_t

        Sets:
        --------
        self.unitblocks[components_df.name] : dict
            Dictionary of transformed parameters for the component.
        """
        converted_dict = {}
        if hasattr(self, attr_name):
            unitblock_parameters = getattr(self, attr_name)
        else:
            print("Block not yet implemented") # TODO: Replace with logger
        
        for key, func in unitblock_parameters.items():
            if callable(func):
                # Extract parameter names from the function
                param_names = func.__code__.co_varnames[:func.__code__.co_argcount]
                args = []
                
                for param in param_names:
                    if param in components_t:
                        df = components_t[param]
                        if not df.empty and components_df.name in df.columns:
                            args.append(df[components_df.name].values)
                        else:
                            args.append(components_df.get(param, 1))
                    else:
                        args.append(components_df.get(param, 1))
                
                # Apply function to the parameters
                converted_dict[key] = func(*args)
            else:
                converted_dict[key] = func
                
        self.unitblocks[components_df.name] = converted_dict
        
    def iterate_components(self, n):
        """
        Iterates over the network components and adds them as unit blocks.

        Parameters:
        ----------
        n : PyPSA Network
            PyPSA network object containing components to iterate over.
            
        Methods: add_UnitBlock
            Method to convert the DataFrame and get a UnitBlock
        
        Adds:
        ---------
        The components to the `unitblocks` dictionary, with distinct attributes for intermittent and thermal units.
        """
        renewable_carriers = ['solar', 'PV', 'wind']
        
        for components in n.iterate_components(["Generator", "StorageUnit"]):
            components_df = components.df            # DataFrame containing all components of that type
            component_type = f"{components_df.index.name.lower()}s_t"
            for component in components_df.index:
                if any(carrier in component for carrier in renewable_carriers):
                    attr_name = "IntermittentUnitBlock_parameters"
                else:
                    attr_name = "ThermalUnitBlock_parameters"
                
                self.add_UnitBlock(attr_name, components_df.loc[component], getattr(n, component_type, None))

                    
        

                        