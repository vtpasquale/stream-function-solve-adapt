#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:07:18 2026

@author: vtpasquale
"""

import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("adapt_history.csv", index_col=False)
data.columns = data.columns.str.strip()

plt.figure(1)
plt.semilogx(data.nNodes, data.cl,'-o')

