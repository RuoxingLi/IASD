# import networkx as nx
from _collections import defaultdict
import struct


class ksp():
    def __init__(self, connectmatrix):
        self.cm = connectmatrix
        # Loading data
        N = 5
        NODE_NUM = len(self.cm)
        # Candidate_Paths[i][j][k]:the k-th path from i to j
        self.Candidate_Paths = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
        fp = open('ger17_Src_Dst_Paths_Cost.dat', 'rb')
        # NODE_NUM*NODE_NUM import precalculated paths (in terms of path_links)
        for ii in range(1, NODE_NUM * NODE_NUM + 1):
            #    temp_path = []
            if ii % NODE_NUM == 0:
                i = ii // NODE_NUM
                j = (ii % NODE_NUM) + NODE_NUM
            else:
                i = (ii // NODE_NUM) + 1
                j = ii % NODE_NUM

            temp_num = []
            for tt in range(N):
                temp_num += list(struct.unpack("i" * 1, fp.read(4 * 1)))  # temp_num[0]: the node-num of path k

            # note, if there are less than N paths for certain src-dest pairs,
            # then the last a few values of temp_num equate to '0'
            if i != j:
                for k in range(N):
                    temp_path = list(struct.unpack("i" * temp_num[k], fp.read(4 * temp_num[k])))
                    self.Candidate_Paths[i - 1][j - 1][k] = [i - 1 for i in temp_path]
        fp.close()

    def k_shortest_paths(self, src, dst, k):
        paths = [self.Candidate_Paths[src][dst][i] for i in range(k)]
        paths_len = []
        for path in paths:
            path_len = 0
            for a, b in zip(path[:-1], path[1:]):
                path_len += self.cm[a][b]
            paths_len.append(path_len)
        return paths, paths_len

    # def __init__(self, connectmatrix):
    #     self.cm = connectmatrix
    #     self.G = nx.Graph()
    #     self.__constructG__()
    #
    # def __constructG__(self):
    #     for node in range(len(self.cm)):
    #         self.G.add_node(node)
    #     for nodei in range(len(self.cm)):
    #         for nodej in range(len(self.cm)):
    #             if self.cm[nodei][nodej] > 0:
    #                 self.G.add_edge(nodei, nodej)
    #                 self.G[nodei][nodej]['weight'] = float(self.cm[nodei][nodej])
    #
    # def temp_graph(self):
    #     temp_graph = nx.Graph()
    #     for node in range(len(self.cm)):
    #         temp_graph.add_node(node)
    #     for nodei in range(len(self.cm)):
    #         for nodej in range(len(self.cm)):
    #             if self.cm[nodei][nodej] > 0:
    #                 temp_graph.add_edge(nodei, nodej)
    #                 temp_graph[nodei][nodej]['weight'] = float(self.cm[nodei][nodej])
    #     return temp_graph
    #
    # def k_shortest_paths(self, source, target, k=2, weight='weight'):
    #     # G is a networkx graph.
    #     # source and target are the labels for the source and target of the path.
    #     # k is the amount of desired paths.
    #     # weight = 'weight' assumes a weighed graph. If this is undesired, use weight = None.
    #
    #     A = [nx.dijkstra_path(self.G, source, target, weight='weight')]
    #     A_len = [sum([self.G[A[0][l]][A[0][l + 1]]['weight'] for l in range(len(A[0]) - 1)])]
    #     B = []
    #
    #     for i in range(1, k):
    #         for j in range(0, len(A[-1]) - 1):
    #             # Gcopy = cp.deepcopy(self.G)
    #             Gcopy = self.temp_graph()
    #             spurnode = A[-1][j]
    #             rootpath = A[-1][:j + 1]
    #             for path in A:
    #                 if rootpath == path[0:j + 1]:  # and len(path) > j?
    #                     if Gcopy.has_edge(path[j], path[j + 1]):
    #                         Gcopy.remove_edge(path[j], path[j + 1])
    #                     if Gcopy.has_edge(path[j + 1], path[j]):
    #                         Gcopy.remove_edge(path[j + 1], path[j])
    #             for n in rootpath:
    #                 if n != spurnode:
    #                     Gcopy.remove_node(n)
    #             try:
    #                 spurpath = nx.dijkstra_path(Gcopy, spurnode, target, weight='weight')
    #                 totalpath = rootpath + spurpath[1:]
    #                 if totalpath not in B:
    #                     B += [totalpath]
    #             except nx.NetworkXNoPath:
    #                 continue
    #         if len(B) == 0:
    #             break
    #         lenB = [sum([self.G[path[l]][path[l + 1]]['weight'] for l in range(len(path) - 1)]) for path in B]
    #         B = [p for _, p in sorted(zip(lenB, B))]
    #         A.append(B[0])
    #         A_len.append(sorted(lenB)[0])
    #         B.remove(B[0])
    #
    #     return A, A_len


# if __name__ == "__main__":
#
#     # Loading data
#     N = 10
#     ksp_test = ksp(ConnectMatrix)
#     NODE_NUM = len(ConnectMatrix)
#     # Candidate_Paths[i][j][k]:the k-th path from i to j
#     Candidate_Paths = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
#     fp = open('Src_Dst_Paths.dat', 'rb')
#     # NODE_NUM*NODE_NUM import precalculated paths (in terms of path_links)
#     for ii in range(1, NODE_NUM * NODE_NUM + 1):
#         #    temp_path = []
#         if ii % NODE_NUM == 0:
#             i = ii // NODE_NUM
#             j = (ii % NODE_NUM) + NODE_NUM
#         else:
#             i = (ii // NODE_NUM) + 1
#             j = ii % NODE_NUM
#
#         temp_num = []
#         for tt in range(N):
#             temp_num += list(struct.unpack("i" * 1, fp.read(4 * 1)))  # temp_num[0]: the node-num of path k
#
#         # note, if there are less than N paths for certain src-dest pairs,
#         # then the last a few values of temp_num equate to '0'
#         if i != j:
#             for k in range(N):
#                 temp_path = list(struct.unpack("i" * temp_num[k], fp.read(4 * temp_num[k])))
#                 Candidate_Paths[i - 1][j - 1][k] = [i - 1 for i in temp_path]
#     fp.close()
#
#
#     #
#     def ksp_get(src, dst, k):
#         paths = [Candidate_Paths[src][dst][i] for i in range(k)]
#         paths_len = []
#         for path in paths:
#             path_len = 0
#             for a, b in zip(path[:-1], path[1:]):
#                 path_len += ConnectMatrix[a][b]
#             paths_len.append(path_len)
#         return paths, paths_len
#
#
#     for i in range(14):
#         for j in range(14):
#             if i != j:
#                 a = ksp_get(i, j, 5)
#                 b = ksp_test.k_shortest_paths(i, j, 5)
#                 for ii in range(5):
#                     if a[1][ii] != b[1][ii]:
#                         print("error")

if __name__ == "__main__":
    cm = [[0, 354, 158, 0, 121, 294, 0, 0, 0, 0, 0, 0, 0, 211, 0, 0, 267],
          [354, 0, 0, 0, 0, 0, 0, 0, 224, 0, 0, 84, 0, 0, 0, 190, 396],
          [158, 0, 0, 0, 114, 306, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 160, 0, 0, 0, 0, 0, 0, 0, 0, 313, 0, 0, 0],
          [121, 0, 114, 160, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [294, 0, 306, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 173],
          [0, 0, 0, 0, 0, 0, 0, 156, 169, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 156, 0, 0, 92, 0, 0, 0, 0, 0, 0, 0],
          [0, 224, 0, 0, 0, 0, 169, 0, 0, 210, 0, 0, 0, 0, 0, 0, 282],
          [0, 0, 0, 0, 0, 0, 0, 92, 210, 0, 79, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 0, 70, 0, 0, 0, 0, 0],
          [0, 84, 0, 0, 0, 0, 0, 0, 0, 0, 70, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 34, 0, 0],
          [211, 0, 0, 313, 0, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0, 94, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 39, 0],
          [0, 190, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 94, 39, 0, 0],
          [267, 396, 0, 0, 0, 173, 0, 0, 282, 0, 0, 0, 0, 0, 0, 0, 0]]

    ksp_test = ksp(cm)
    print("ok")
