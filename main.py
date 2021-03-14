import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import deque
import random
import getopt


class PathFinder:
    def __init__(self, maze_data, step):
        self.x_size = len(maze_data)
        self.y_size = len(maze_data[0]) - 1
        self.maze_map = np.zeros((self.x_size, self.y_size), dtype=int)
        self.render_map = np.zeros((self.x_size, self.y_size))
        self.step = step

        for i in range(len(maze_data)):
            for j in range(len(maze_data[i]) - 1):
                if maze_data[i][j] == 'X':
                    # create map
                    self.maze_map[i][j] = 1
                    self.render_map[i][j] = 1

    def to_coord(self, vec):
        return vec['y'] * self.x_size + vec['x']

    def to_vec(self, coord):
        return {'x': coord % self.x_size, 'y': coord // self.x_size}

    @staticmethod
    def man_dist(x, y):
        res_x = x['x'] - y['x']
        res_y = x['y'] - y['y']
        return abs(res_x) + abs(res_y)

    def set_node_color(self, node, color):
        node = self.to_vec(node)
        self.render_map[node['x'], node['y']] = color

    def set_endpoint_colors(self, end_node, start_node):
        self.set_node_color(end_node, 0.6)
        self.set_node_color(start_node, 0.8)

    def render(self):
        plt.imshow(self.render_map, cmap='Accent')
        plt.show()

    def reconstruct_path(self, end_node, parent, start_node):
        current_path = end_node
        cnt = 0
        while current_path in parent:
            print(self.to_vec(current_path))
            if current_path != end_node and current_path != start_node:
                self.set_node_color(current_path, 0.5)
            current_path = parent[current_path]
            cnt = cnt + 1
        print("Path reconstructed.")
        print("Path length:", cnt)

    def random_search(self, start_node, end_node):
        closed_set = set()
        open_set = list()
        parent = dict()
        num_expanded = 0

        self.set_endpoint_colors(end_node, start_node)
        open_set.append(start_node)

        while len(open_set) > 0:
            current = int(random.choice(open_set))

            if current == end_node:
                break
            if current != end_node and current != start_node:
                self.set_node_color(current, 0.3)
            if self.step:
                self.render()
            for nextVec in self.get_neighbours(self.maze_map, self.to_vec(current)):
                next_node = self.to_coord(nextVec)
                if next_node not in closed_set and next_node not in open_set:
                    open_set.append(next_node)
                    parent[next_node] = current
                    num_expanded += 1
            open_set.remove(current)
            closed_set.add(current)

        self.render()
        self.reconstruct_path(end_node, parent, start_node)
        print("Number expanded:", num_expanded)
        self.render()

    def astar(self, start_node, end_node):
        closed_set = set()
        open_set = set()
        parent = dict()
        g_scores = dict()
        f_scores = dict()
        num_expanded = 0

        self.set_endpoint_colors(end_node, start_node)
        open_set.add(start_node)

        for i in range(self.x_size * self.y_size):
            g_scores[i] = sys.maxsize
            f_scores[i] = sys.maxsize

        g_scores[start_node] = 0
        f_scores[end_node] = self.man_dist(self.to_vec(start_node), self.to_vec(end_node))

        while len(open_set) > 0:
            lowest = sys.maxsize
            curr = 0

            for key in open_set:
                if f_scores[key] <= lowest:
                    lowest = f_scores[key]
                    curr = key

            if curr == end_node:
                break
            if curr != end_node and curr != start_node:
                self.set_node_color(curr, 0.3)

            if self.step:
                self.render()

            open_set.remove(curr)
            closed_set.add(curr)

            for nextVec in self.get_neighbours(self.maze_map, self.to_vec(curr)):
                next_node = self.to_coord(nextVec)
                if next_node == end_node:
                    parent[end_node] = curr
                if next_node in closed_set:
                    continue
                g_score = g_scores[curr] + 1

                if next_node not in open_set:
                    open_set.add(next_node)
                    num_expanded += 1

                if g_score < g_scores[next_node]:
                    parent[next_node] = curr
                    g_scores[next_node] = g_score
                    f_scores[next_node] = g_scores[next_node] + self.man_dist(self.to_vec(next_node), self.to_vec(end_node))

        self.render()
        self.reconstruct_path(end_node, parent, start_node)
        print("Number expanded:", num_expanded)
        self.render()

    def greedy_search(self, start_node, end_node):
        open_set = set()
        closed_set = set()
        parent = dict()
        num_expanded = 0

        self.set_endpoint_colors(end_node, start_node)
        open_set.add(start_node)

        while len(open_set) > 0:
            current = open_set.pop()

            if current == end_node:
                break
            if current != start_node:
                self.set_node_color(current, 0.3)
            if self.step:
                self.render()

            for nextVec in self.get_neighbours(self.maze_map, self.to_vec(current)):
                next_node = self.to_coord(nextVec)
                if next_node not in closed_set and next_node not in open_set:
                    open_set.add(next_node)
                    parent[next_node] = current
                    num_expanded += 1

            closed_set.add(current)

        self.render()
        self.reconstruct_path(end_node, parent, start_node)
        print("Number expanded:", num_expanded)
        self.render()

    def bfs(self, start_node, end_node):
        qu = deque()
        parent = dict()
        visited = set()
        num_expanded = 0

        self.set_endpoint_colors(end_node, start_node)
        qu.append(start_node)
        visited.add(start_node)

        while len(qu) > 0:
            current = qu.popleft()

            if current != end_node and current != start_node:
                self.set_node_color(current, 0.3)
            if self.step:
                self.render()
            if current == end_node:
                print("Path found k")
                break

            ngb = self.get_neighbours(self.maze_map, self.to_vec(current))
            for nextVec in ngb:
                next_node = self.to_coord(nextVec)

                if next_node not in visited:
                    visited.add(next_node)
                    parent[next_node] = current
                    qu.append(next_node)
                    num_expanded += 1

        self.render()
        self.reconstruct_path(end_node, parent, start_node)
        print("Number expanded:", num_expanded)
        self.render()

    def dfs(self, start_node, end_node):
        stack = []
        parent = dict()
        visited = set()
        num_expanded = 0

        self.set_endpoint_colors(end_node, start_node)
        stack.append(start_node)
        visited.add(start_node)

        while len(stack) > 0:
            current = stack.pop()

            if current == end_node:
                print("Path found")
                break

            ngb = self.get_neighbours(self.maze_map, self.to_vec(current))
            for nextVec in ngb:
                neighbour = self.to_coord(nextVec)

                if neighbour != end_node and neighbour != start_node:
                    self.set_node_color(neighbour, 0.3)

                if self.step:
                    self.render()

                if neighbour not in visited:
                    visited.add(neighbour)
                    parent[neighbour] = current
                    stack.append(neighbour)
                    num_expanded += 1

        self.render()
        self.reconstruct_path(end_node, parent, start_node)
        print("Number expanded:", num_expanded)
        self.render()

    def get_neighbours(self, map, node):
        ngb = []
        if node['x'] - 1 >= 0:
            if map[node['x'] - 1][node['y']] == 0:
                ngb.append({'x': node['x'] - 1, 'y': node['y']})
        if node['x'] + 1 < len(map) - 1:
            if map[node['x'] + 1][node['y']] == 0:
                ngb.append({'x': node['x'] + 1, 'y': node['y']})
        if node['y'] - 1 >= 0:
            if map[node['x']][node['y'] - 1] == 0:
                ngb.append({'x': node['x'], 'y': node['y'] - 1})
        if node['y'] + 1 < len(map[0]):
            if map[node['x']][node['y'] + 1] == 0:
                ngb.append({'x': node['x'], 'y':  node['y'] + 1})

        return ngb


def main(argv):
    inputfile = ''
    alg = "b"
    step = False

    try:
        opts, args = getopt.getopt(argv, "a:si:", ["alg=", "step", "ifile="])
    except getopt.GetoptError:
        print
        'Invalid arguments.'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-s':
            step = True
        elif opt in ("-a", "--alg"):
            alg = arg
        elif opt in ("-i", "--ifile"):
            inputfile = arg

    #filename = "dataset/72.txt"
    filename = inputfile
    with open(filename) as f:
        file_content = f.readlines()

    #parse and set maze start/end
    maze_data = file_content[0:-2]
    maze_start = np.array(file_content[-2].replace(",", "").split()[1:], dtype='int')
    maze_end = np.array(file_content[-1].replace(",", "").split()[1:], dtype='int')

    start_vec = {'x': maze_start[1], 'y': maze_start[0]}
    end_vec = {'x': maze_end[1], 'y': maze_end[0]}

    solver = PathFinder(maze_data, step)
    start_node = solver.to_coord(start_vec)
    end_node = solver.to_coord(end_vec)

    if alg == "dfs" or alg == "d":
        solver.dfs(start_node, end_node)
    elif alg == "bfs" or alg == "b":
        solver.bfs(start_node, end_node)
    elif alg == "astar" or alg == "a":
        solver.astar(start_node, end_node)
    elif alg == "grd" or alg == "g":
        solver.greedy_search(start_node, end_node)
    elif alg == "rnd" or alg == "r":
        solver.random_search(start_node, end_node)
    else:
        print("Invalid option alg:", alg)

    print("END")


if __name__ == "__main__":
    main(sys.argv[1:])
