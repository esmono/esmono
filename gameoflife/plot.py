import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d

import argparse
import time


class PetriDish(object):
    def __init__(self, size, i_number, seed='Random'):
        if seed == 'Random':
            self.state = np.random.randint(2, size=size)
        else:
            self.state = seed
        self.engine = GameOfLife(self, size)
        self.size = size
        self.i_number = i_number

        self.fig, self.ax = plt.subplots()
        self.fig.set_figheight(self.size[0]/10)
        self.fig.set_figwidth(self.size[1]/10)
        self.ax.pcolormesh(self.state, vmin=0, vmax=2, cmap=plt.cm.hot)

    def update(self, i):
        self.state = self.engine.apply_rules()
        pc = self.ax.pcolormesh(self.state, vmin=0, vmax=2, cmap=plt.cm.viridis)
        return pc,

    def animate(self):
        plt.axis('off')
        plt.tight_layout(pad=0)

        im_ani = animation.FuncAnimation(self.fig, self.update, frames=self.i_number,
                                         interval=200, repeat_delay=1000, blit=True)
        writer_gif = animation.PillowWriter(fps=3)
        im_ani.save('game-of-life.gif', writer=writer_gif)


class GameOfLife(object):
    '''The universe of the Game of Life is an infinite two-dimensional orthogonal grid of square cells,
    each of which is in one of two possible states, live or dead. Every cell interacts with its eight
    neighbours, which are the cells that are directly horizontally, vertically, or diagonally adjacent.
    At each step in time, the following transitions occur:

    1. Any live cell with fewer than two live neighbours dies (referred to as underpopulation or exposure[1]).
    2. Any live cell with more than three live neighbours dies (referred to as overpopulation or overcrowding).
    3. Any live cell with two or three live neighbours lives, unchanged, to the next generation.
    4. Any dead cell with exactly three live neighbours will come to life.

    The initial pattern constitutes the 'seed' of the system. The first generation is created by
    applying the above rules simultaneously to every cell in the seed — births and deaths happen
    simultaneously, and the discrete moment at which this happens is sometimes called a tick.
    (In other words, each generation is a pure function of the one before.) The rules continue to be
    applied repeatedly to create further generations.

    https://www.conwaylife.com/wiki/Conway%27s_Game_of_Life
    '''

    def __init__(self, petri_dish, size):
        self.kernel = [[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]]
        self.size = size
        self.state = petri_dish.state

    def apply_rules(self):
        neighborhood = convolve2d(self.state, self.kernel, 'same')
        birth = (neighborhood == 3) & (self.state == 0)
        survive = ((neighborhood == 2) | (neighborhood == 3)) & (self.state == 1)
        result_state = np.zeros(shape=self.size, dtype=int)
        result_state[birth | survive] = 1
        self.state = result_state
        return result_state


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('-h', '--height', help='Board Height', default=25)
    ap.add_argument('-w', '--width', help='Board Width', default=100)
    ap.add_argument('-i', '--inumber', help='Number of iterations', default=30)
    args = vars(ap.parse_args())

    height = int(args['height'])
    width = int(args['width'])
    i_number = args['inumber']
    petri_dish = PetriDish((height, width), i_number)
    petri_dish.animate()

if __name__ == '__main__':
    main()
