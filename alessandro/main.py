# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:14:38 2024

@author: aless
"""

from config.config import Config
from network_definition import NetworkDefinition
from transformation import Transformation

config = Config()
nd = NetworkDefinition(config)

network = nd.n.copy()
network.optimize(solver_name='gurobi')

transformation = Transformation(config, network)