# -*- coding: utf-8 -*-
"""
Created on 2019-08-22 09:37:36

@author: fmiorell
"""

#  This script registers the "turbo" colormap to matplotlib, and the reversed version as "turbo_r"
#  Reference:  https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html

import numpy as np
import matplotlib.pyplot as plt

from utils import add_turbo





if __name__=='__main__':

    XX, YY = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
    ZZ = np.sqrt(XX**2 + YY**2)
    add_turbo(plt)

    plt.figure()
    plt.imshow(ZZ, cmap='turbo')
    plt.colorbar()

    plt.figure()
    plt.imshow(ZZ, cmap='turbo_r')
    plt.colorbar()
    
    plt.show()


