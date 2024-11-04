# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 08:12:37 2024

@author: aless
"""

import pandas as pd
import pypsa
from pypsa.descriptors import get_switchable_as_dense as get_as_dense

class Transformation():
    
    def __init__(self, parser, n):
        
        self.unitblocks = dict()
        
        self.IntermittentUnitBlock_parameters = {
            "Gamma": 0.0,
            "Kappa": 0.0,
            "MaxPower": lambda p_nom_opt, p_max_pu: p_nom_opt * p_max_pu, # Ricordati di ottenerli come get_as_dense, tipo get_as_dense(network, "Generator", "p_max_pu", network.snapshots)
            "MinPower": lambda p_nom_opt, p_min_pu: p_nom_opt * p_min_pu,
            "IntertiaPower": 0.0
            }
        
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
        
        self.init(parser, n)
        
        
    def init(self, parser, n):
        self.read_file(parser)
        
        self.iterate_components(n)
        
        
    def read_file(self, parser):
        self.parameters = pd.read_excel(f"{parser.input_data_path}/{parser.input_name_parameters}", sheet_name=None) 
        
    def add_IntermittentUnitBlock(self, components_df, components_t, name, n):
        converted_dict = {}
        
        for key, func in self.IntermittentUnitBlock_parameters.items():
            if callable(func):
                # Estrai i nomi dei parametri dalla funzione
                param_names = func.__code__.co_varnames[:func.__code__.co_argcount]
                
                # Valori dei parametri per chiamare la funzione
                args = []
                
                for param in param_names:
                    # Verifica se `param` esiste come chiave in `components_t`
                    if param in components_t:
                        df = components_t[param]
                        
                        # Controlla che il DataFrame non sia vuoto e che contenga l'indice richiesto
                        if not df.empty and components_df.index.name in df.columns:
                            args.append(df.loc[components_df.index.name].values[0])
                        else:
                            # Se le condizioni non sono soddisfatte, prendi il valore da `components_df`
                            args.append(components_df.get(param, 1))
                    else:
                        # Se `param` non esiste in `components_t`, prendi il valore da `components_df`
                        args.append(components_df.get(param, 1))

                # Calcola il risultato con i parametri estratti
                converted_dict[key] = func(*args)
            else:
                # Copia direttamente i valori non funzione
                converted_dict[key] = func
                
        self.unitblocks[components_df.index[0]] = converted_dict
            
    
    def add_ThermalUnitBlock(self, components_df, n):
        pass
        
        
    def iterate_components(self, n):
        renewable_carriers = ['solar', 'PV', 'wind']
        
        for components in n.iterate_components(["Generator", "StorageUnit"]):
            components_df = components.df            # DataFrame che contiene tutti i componenti di quel tipo
            for component in components_df.index:
                if any(carrier in component for carrier in renewable_carriers):
                    self.add_IntermittentUnitBlock(components_df, n.generators_t, components.name, n)
                else:
                    self.add_ThermalUnitBlock(components_df, n)
                    
                        