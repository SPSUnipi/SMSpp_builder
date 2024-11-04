# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:07:31 2024

@author: aless
"""

import pandas as pd
import numpy as np
import logging
import pypsa

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -10s %(funcName) '
              '-10s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logging.getLogger("Article").setLevel(logging.WARNING)

#%% Definition of the network
class NetworkDefinition():
    
    def __init__(self, parser):
        self.parser = parser
        
        self.init()
        
    def init(self):
        self.n = pypsa.Network()
        
        self.define_snapshots()
        
        all_sheets = self.read_excel_components()
        self.add_all_components(all_sheets)
        
        self.add_costs_components()
        self.add_demand()
        self.add_renewables()
        
        
    def define_snapshots(self):
        self.n.snapshots = range(0, self.parser.n_snapshots)
        self.n.snapshot_weightings.objective = self.parser.weight
    
    def read_excel_components(self):
        # I read all the Excel sheets
        file_path = f"{self.parser.input_data_path}/{self.parser.input_name_components}"
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        return all_sheets
    
    def add_all_components(self, all_sheets):
        # I iterate for all the excel files
        for sheet_name, data in all_sheets.items():
                self.add_component(self.n, sheet_name, data)
        
    # Function to add components (buses, links, etc.)
    def add_component(self, network, component_type, data):
        for _, row in data.iterrows():
            # I extract the name
            name = row[data.columns[0]]  # First column should be name! Otherwise you have to change the code
    
            # I create a dict with populated parameters, except for name
            params = {col: row[col] for col in data.columns if col != 'name' and pd.notna(row[col])}
            
            # I add the component to the network
            network.add(component_type, name, **params)
            
    
    # Method to add the cost of components
    def add_costs_components(self):
        file_path = f"{self.parser.input_data_path}/{self.parser.input_name_costs}"
        costs = pd.read_excel(file_path, index_col=0)
        
        for components in self.n.iterate_components(["Generator", "StorageUnit"]): 
            components_df = components.df            # DataFrame che contiene tutti i componenti di quel tipo
            for component in components_df.index:
                components_df.loc[component, 'capital_cost'] = costs.at[component, 'Capital cost [€/MW]']
                components_df.loc[component, 'marginal_cost'] = costs.at[component, 'Marginal cost [€/MWh]']
        
    
    # Method to add the demand of profiles
    def add_demand(self):
        file_path = f"{self.parser.input_data_path}/{self.parser.input_name_demand}"
        
        # file name of the pv data
        df_demand_day = pd.read_csv(file_path)
        df_demand_day["hour"] = range(0, 24)

        n_days = int(len(self.n.snapshots)/24)

        for load in self.n.loads.index:
            df_demand_year = np.random.normal(
                np.tile(df_demand_day["demand"], n_days),
                np.tile(df_demand_day["standard_deviation"], n_days),
            )
            self.n.loads_t.p_set[load] = df_demand_year
                    
    # Method to add the per unit power of renewables
    def add_renewables(self):
        file_path_PV = f"{self.parser.input_data_path}/{self.parser.input_name_pv}"
        df_pv = pd.read_csv(file_path_PV, skiprows=3, nrows=len(self.n.snapshots))  # skip the first 3 rows
        
        file_path_wind = f"{self.parser.input_data_path}/{self.parser.input_name_wind}"
        df_wind = pd.read_csv(file_path_wind, skiprows=3, nrows=len(self.n.snapshots))  # skip the first 3 rows
        
        for generator in self.n.generators.index:
            if 'solar' in generator.lower() or 'pv' in generator.lower():
                self.n.generators_t.p_max_pu[generator] = df_pv['electricity']
            elif 'wind' in generator.lower():
                self.n.generators_t.p_max_pu[generator] = df_wind['electricity']
                
        
        
                    
