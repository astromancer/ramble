# /usr/bin/python3
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from .core import LivePieChart

# todo: argparse
pie = LivePieChart()
#                            func              init           
ani = FuncAnimation(pie.fig, pie.update, None, pie.update,
                    save_count=0, interval=5, blit=True)

plt.show()