import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import matplotlib.pyplot
import numpy


class TsGym(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(np.array([0.0]), np.array([100]), dtype=np.float32)
        self.is_done = False
        pass

    def step(self, action):
        ob = self._get_state()
        reward = 0.12
        return ob, reward, self.is_done, {}

    def reset(self):
        return self._get_state()

    def render(self, mode='rgb_array', close=False):
        figure = matplotlib.pyplot.figure()
        plot = figure.add_subplot(111)

        x = numpy.arange(0, 100, 0.1)
        y = numpy.sin(x) / x
        plot.plot(x, y)
        rgb_array = self._fig2data(figure)
        return rgb_array

    def _fig2data(self, fig):
        # draw the renderer
        fig.canvas.draw()

        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = numpy.fromstring(fig.canvas.tostring_argb(), dtype=numpy.uint8)
        buf.shape = (w, h, 4)

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = numpy.roll(buf, 3, axis=2)
        return buf

    def _get_state(self):
        ob = [12]
        return ob
