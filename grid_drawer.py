import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class Square:

    def __init__(self, position, width=10):
        assert len(position) == 3, "argument 'position' must be a (x, y) tuple"
        self.position = position
        self.width = width

    @staticmethod
    def draw(self):
        tile = np.zeros(shape=(self.width, self.width, 3))
        tile[:, :, 0].fill(0.25)
        tile[:, :, 1].fill(0.25)
        tile[:, :, 2].fill(1.0)
        return tile


class Grid:

    def __init__(self, layer, width, height):
        self.width = width
        self.height = height
        self.grid = []
        for i in range(width):
            row = []
            for j in range(height):
                row.append(Square(position=(layer, i, j)))
            self.grid.append(row)


class AgentSprite:

    def __init__(self, rect_width, num_layers):
        self.position = (0, 0, 0)
        self.rect_width = rect_width
        self.num_layers = num_layers
        self.neighborhood = [(1, 0, 0)]

    def move(self, position):
        self.position = position
        self.neighborhood = []
        if self.position[0] < self.num_layers - 1:
            self.neighborhood.append((self.position[0] + 1, int(self.position[1]/2), int(self.position[2]/2)))
        if self.position[0] > 0:
            self.neighborhood.append((self.position[0] - 1, 2 * self.position[1], 2 * self.position[2]))
            self.neighborhood.append((self.position[0] - 1, 2 * self.position[1] + 1, 2 * self.position[2]))
            self.neighborhood.append((self.position[0] - 1, 2 * self.position[1], 2 * self.position[2] + 1))
            self.neighborhood.append((self.position[0] - 1, 2 * self.position[1] + 1, 2 * self.position[2] + 1))

    def draw(self):
        return patches.Circle(xy=(self.position[1]*self.rect_width + int(self.rect_width/2),
                                  self.position[2]*self.rect_width + int(self.rect_width/2)),
                              radius=int(self.rect_width/2) - 1, color='r')


class Drawer:

    def __init__(self, agent, num_layers, tile_width):
        self.agent = agent
        # Knowing that it's a dyadic decomposition, there's no need to specify the shape of every layer
        self.num_layers = num_layers
        self.grids = {}
        self.tile_width = tile_width
        self.fig, self.ax = plt.subplots(1, self.num_layers, figsize=(10, 4), gridspec_kw={'width_ratios': [4, 2, 1, 0.5, 0.25]})
        # Show the env name in the window title
        self.fig.canvas.set_window_title('Dyadic conv')
        self.fig.canvas.draw_idle()
        self.imshow_obj = []
        layer = 0
        for i in range(self.num_layers - 1, -1, -1):
            num_tiles = pow(2, i)
            self.grids[num_tiles] = Grid(layer, num_tiles, num_tiles)
            layer += 1
        index = 0
        for key in self.grids.keys():
            img = self.render_grid(key=key)
            self.ax[index].imshow(img)
            index += 1
        self.backgrounds = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.ax]

        # Turn off x/y axis numbering/ticks
        for ax in self.ax:
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            _ = ax.set_xticklabels([])
            _ = ax.set_yticklabels([])
            # Flag indicating the window was closed
            self.closed = False

            def close_handler(evt):
                self.closed = True

        self.fig.canvas.mpl_connect('close_event', close_handler)
        self.env = {
            'grids': self.grids,
            'agent': self.agent
        }
        plt.show(block=False)

    def render(self, agent):
        for i in range(self.num_layers):
            self.ax[i].patches = []
        for pos in agent.neighborhood:
            self.ax[pos[0]].add_patch(patches.Rectangle((pos[1]*self.tile_width + 1, pos[2]*self.tile_width + 1),
                                                        width=self.tile_width - 2, height=self.tile_width - 2,
                                                        color=(0.25, 0.25, 1.0)))
        for bg in self.backgrounds:
            self.fig.canvas.restore_region(bg)
        self.ax[agent.position[0]].add_patch(agent.draw())
        for ax in self.ax:
            self.fig.canvas.blit(ax.bbox)

        plt.pause(0.5)

    def render_grid(self, key):
        img = np.zeros((key * self.tile_width + 1, key * self.tile_width + 1, 3))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if i % 10 == 0 or j % 10 == 0:
                    img[i, j, :] = 1.0
        img[img.shape[0] - 1, :, :] = 1.0
        img[:, img.shape[1] - 1, :] = 1.0

        return img

    def draw_square(self):
        tile = np.zeros(shape=(self.tile_width, self.tile_width, 3))
        tile[:, :, 0].fill(0.25)
        tile[:, :, 1].fill(0.25)
        tile[:, :, 2].fill(1.0)
        return tile


"""agent = AgentSprite(rect_width=10, num_layers=5)
drawer = Drawer(agent, num_layers=5, tile_width=10)
num_layers = 5
max_tiles = pow(2, num_layers - 1)
for k in range(100):
    start = time.time()
    drawer.render()
    next_pos = random.sample(agent.neighborhood, k=1)
    agent.move(next_pos[0])
    end = time.time()
    sys.stdout.write('\r{pos} - Rendering time: {t}'.format(pos=next_pos[0], t=end - start))
    sys.stdout.flush()"""