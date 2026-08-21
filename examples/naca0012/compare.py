#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 08:02:10 2026

@author: vtpasquale
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


feature = pd.read_csv(os.path.join("adapt_check","adapt_history.csv"), index_col=False)
feature.columns = feature.columns.str.strip()

goal = pd.read_csv(os.path.join("adjoint_adapt","adapt_history.csv"), index_col=False)
goal.columns = goal.columns.str.strip()

plt.figure(1)
plt.semilogx(feature.nNodes, feature.cl,'-o', 
             goal.nNodes, goal.cl,'.-')
plt.legend(['Feature Adapt',
            'Goal Adapt'])
plt.title('Circulation Lift')

plt.figure(2)
plt.semilogx(feature.nNodes, feature.clp,'-o', 
             goal.nNodes, goal.clp,'.-')
plt.legend(['Feature Adapt',
            'Goal Adapt'])
plt.title('Pressure Lift')
