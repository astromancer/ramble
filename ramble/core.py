import time
import logging
# import functools as ftl

import numpy as np
from matplotlib import pyplot as plt
# from matplotlib.colors import ListedColormap
from matplotlib.patches import Wedge, Rectangle

from ps_mem import get_memory_usage, human, cmd_with_count
from psutil import virtual_memory

# from recipes.logging import LoggingMixin
from recipes import cosort


# setup logging
logging.basicConfig()
logger = logging.getLogger('rambles')  # get_module_logger()
logger.setLevel(logging.INFO)


TOTAL_RAM_KB = virtual_memory().total / 1024
THRESHOLD = 0.01    # 1%
# processes with fractional memory usage  below `THRESHOLD` not displayed

COLORS = plt.get_cmap('Set1').colors[::-1]

# FIXME: blitting of text that extends off-axes!
# FIXME: text ha needs to change when process gains / looses enough memory to
# change quadrants in the chart
# TODO: print total / used / free in human
# TODO mpl bug report. FuncAnimation doesn't actually work with generators
# dispite claims to the contrary in docs
# TODO: don't rotate wedge label if there is enough space to display it horizontally
# FIXME: Free / used percentages wrong (sometimes!?)
# FIXME: overlapping texts


def get_data():
    # get memory usage data
    sorted_cmds, shareds, counts, total, swaps, total_swap = \
        get_memory_usage(None, False, False)

    # '%.1f%% (%s)' % (p, human((p / 100) * total))

    # get fractional usage
    cmds, usage = np.array(sorted_cmds).T
    frac = usage.astype(float) / total
    # heap small ones together
    large = (frac >= THRESHOLD)
    #counts =  np.r_[counts[large], large.sum()]
    frac = np.r_[frac[large], frac[~large].sum()]
    names = np.append(cmds[large], 'others')
    # counts = {nm: counts[nm] for nm in cmds[large]}
    # counts['others'] = len(cmds) - large.sum()
    counts = [*map(counts.get, cmds[large]), len(cmds) - large.sum()]
    # counts =  np.r_[counts[large], large.sum()]
    data = dict(zip(names, zip(counts, frac)))

    return data, total


# class HackedAnimation():
#     def _blit_draw(self, artists):


class LivePieChart:  # (LoggingMixin):
    """A per-process RAM usage pie chart"""

    logger = logger

    def __init__(self, wedgeprops=None, labelprops=None, ptextprops=None,
                 label_space=0.025, colors=COLORS):
        #
        self.fig, axes = plt.subplots(
            2, 1,
            figsize=(8.75, 7.35),
            tight_layout=True,
            gridspec_kw=dict(height_ratios=(6, 1)))
        #
        self.fig.canvas.manager.set_window_title(
            'Ramble: Per Process Memory Usage')

        # setup axes
        self.ax, self.ax1 = ax, ax1 = axes
        self.colours = COLORS
        # plt.rcParams['axes.prop_cycle'].by_key()['color']
        # ax.set_aspect('equal')
        # make the axes limits larger so blitting works
        # (text needs to be inside axes!)
        self.fig.set_facecolor('0.5')
        ax.set(aspect='equal',
               facecolor='none',
               xlim=(-1.5, 1.5),
               ylim=(-1.2, 1.2))
        ax.set_axis_off()
        ax1.set(xticks=[], yticks=[], facecolor='none')

        # init texts
        self.text_used = ax1.text(0, 0, '', ha='center', va='center')
        self.text_free = ax1.text(0, 0, '', ha='center', va='center')
        self.text_time = ax.figure.text(0, 1, '', ha='left', va='top',
                                        fontweight='bold')
        self.text_total = ax.figure.text(0, 0.98,
                                         f'TOTAL RAM: {human(TOTAL_RAM_KB)}',
                                         ha='left', va='top',
                                         fontweight='bold')
        #  transform=ax.figure.transFigure)
        self.texts = self.text_used, self.text_free, self.text_time
        self.label_space = float(label_space)

        # make state
        self.data, self.total = get_data()
        self.art = {}
        self.wedgeprops = wedgeprops or {}
        self.labelprops = labelprops or {}
        self.ptextprops = ptextprops or {}

        #  make pie chart
        self.update_wedges()

    def update_wedges(self):
        #
        # n = len(self.data)
        self.logger.debug('Updating wedges')
        m = len(self.colours)
        counts, fracs = zip(*self.data.values())
        marks = np.cumsum((0, ) + fracs)

        current = set(self.art)  # set of process names
        # wedge, rect, label, ptext = self.art[cmd]

        i = 0
        for i, cmd in enumerate(self.data):
            current -= {cmd}
            thetas = marks[[i, i+1]]
            frac = self.data[cmd][1]
            x = thetas * self.total / TOTAL_RAM_KB
            if cmd in self.art:
                self.update_wedge(cmd, thetas, frac)
            else:
                # new
                colour = self.colours[i % m]
                wedge = self.get_wedge(thetas, cmd, colour)

                rect = Rectangle((x[0], 0), x.ptp(), 1, color=colour)
                self.ax1.add_patch(rect)
                texts = self.get_texts(cmd, np.pi * thetas.sum(),
                                       self.labelprops, self.ptextprops)
                self.art[cmd] = (wedge, rect, *texts)
                self.logger.debug(f'Created new wedge for {cmd}')

        self.logger.debug(f'Updated {i} wedges')

        # update free RAM text
        self.text_used.set_text(f'used: {x[1]:.1%}')
        self.text_used.set_position([0.5 * x[1], 0.5])
        self.text_free.set_text(f'free: {1 - x[1]:.1%}')
        self.text_free.set_position([0.5 * (1 + x[1]), 0.5])

        # update timestamp
        self.text_time.set_text(time.strftime('%c', time.localtime()))

        #  remove stale wedges
        for cmd in current:
            for art in self.art.pop(cmd):
                art.remove()

        if current:
            self.logger.debug(f'Removed {len(current)} wedges')

    def get_wedge(self, angles, cmd, facecolor, **wedgeprops):
        # theta2 = (theta1 + frac)  # if counterclock else (theta1 - frac)

        theta1, theta2 = angles
        # frac = theta2 - theta1
        count = self.data[cmd][0]
        wedge = Wedge((0, 0), 1,  # radius,
                      360. * min(theta1, theta2),
                      360. * max(theta1, theta2),
                      facecolor=facecolor,
                      clip_on=False,
                      label=f'{cmd} ({count}:i)')

        # wedge.set(**wedgeprops)
        self.ax.add_patch(wedge)
        return wedge

    # def get_rectangle(self, x, colour):

    def get_texts(self, cmd, theta, labelprops, ptextprops):

        txt = self.ax.text(0, 0, '',
                           clip_on=False,
                           size=plt.rcParams['xtick.labelsize'])
        txt.set(**labelprops)

        ptxt = self.ax.text(0, 0, '',
                            clip_on=False,
                            ha='center',
                            va='center')
        ptxt.set(**ptextprops)

        # update
        self.update_text(txt, ptxt, cmd, theta)
        return txt, ptxt

    def update_text(self, label, ptext, cmd, theta):
        #
        count, frac = self.data[cmd]

        # update label
        xy = np.array([np.cos(theta), np.sin(theta)]) * (1. + self.label_space)
        theta = np.degrees(theta)
        label.set(position=xy, text=cmd_with_count(cmd, count),
                  ha=('left' if xy[0] > 0 else 'right'),
                  va=('baseline' if (abs(theta - 90) < 5) else 'center'))

        # update ptext
        ptext.set(text=self.format_percent(frac),
                  position=(0.6 * xy),
                  rotation=(theta + 180 * int((theta > 90) & (theta < 270))))

    def update_wedge(self, cmd, angles, frac):
        """
        Update wedge and corresponding rectangle patches

        Parameters
        ----------
        cmd : [type]
            [description]
        angles : [type]
            [description]
        frac : [type]
            [description]
        """
        wedge, rect, label, ptext = self.art[cmd]

        wedge.set_theta1(360 * angles[0])
        wedge.set_theta2(360 * angles[1])

        x = angles * self.total / TOTAL_RAM_KB
        rect.set_xy((x[0], 0))
        rect.set_width(x.ptp())

        # update label position
        theta = angles.mean() * 2 * np.pi
        self.update_text(label, ptext, cmd, theta)

    def update_colours(self):
        wedges = next(zip(*self.art.values()))
        thetas, colours = zip(*[(w.theta1, w.get_facecolor()) for w in wedges])
        thetas, colours, wedges = cosort(thetas, colours, wedges)
        # check if two of the same colour next to each other
        if (np.diff([*colours, colours[0]], 1, 0) == 0).any():
            # re-assign colours
            n = len(self.colours)
            for i, w in enumerate(wedges):
                w.set_facecolor(self.colours[i % n])

    def format_percent(self, p):
        # percentage string formatter
        return f'{p:.1%} ({human(p * self.total)})'

    def get_artists(self, _=None):
        #
        self.logger.debug('Updating')
        self.data, self.total = get_data()
        self.update_wedges()
        self.update_colours()

        for art in self.art.values():
            yield from iter(art)

        yield from iter(self.texts)

    def update(self, _=None):
        return [*self.get_artists()]


# LivePieChart.logger.setLevel(logging.INFO)
