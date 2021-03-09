# !/usr/bin/env python
"""
A live per-process RAM usage pie chart in python
"""

from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from ramble.core import LivePieChart

# todo: argparse
pie = LivePieChart()
#                            func              init           
ani = FuncAnimation(pie.fig, pie.update, 100, pie.update,
                    save_count=0, interval=250, blit=False)
# ani.save('~/work/rambles/example.gif',
#          fps=4)
plt.show()