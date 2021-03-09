# !/usr/bin/env python
"""
A live per-process RAM usage pie chart in python
"""

from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from rambles.core import LivePieChart

# todo: argparse
pie = LivePieChart()
#                            func              init           
ani = FuncAnimation(pie.fig, pie.update, None, pie.update,
                    save_count=0, interval=1000, blit=False)

plt.show()