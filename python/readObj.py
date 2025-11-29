import igl
import numpy as np
from scipy.spatial.transform import Rotation
divide = 3
class File:
    def __init__(self, filename, subfilename = None):
        self.subfilename = subfilename
        self.filename = filename
        self.F = None
        self.folder = None
        self.count = 0
        self.count_particle = 0
        self.ev = None
        self.fe = None
        self.ef = None
        self.L = None
        self.dbla = None
        self.height = 0
        self.F_second = None
        self.F_third = None
    def readObj(self):
        V, self.F = igl.read_triangle_mesh(self.filename)
        # scale based on the bounding box
        # rotate 90 degree in x axis
        # V = V @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        # V = V @ np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        # V = V @ np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        scale = 1.0
        if self.subfilename is None:
            # V = V @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            max_coord = V.max(axis=0)
            min_coord = V.min(axis=0)
            scale = max(max_coord - min_coord)
            mean_point = (max_coord + min_coord)/2
            # mean_point should be [0.5, 0.5, 0.5]
            V = V - mean_point

            V = V/scale * 0.7
            V += 0.5
            # reduce y
            # V[:,1]

        self.ev, self.fe, self.ef = igl.edge_topology(V, self.F)
        self.L = igl.edge_lengths(V, self.ev)


        N = igl.per_face_normals(V, self.F, np.array([1.0,1.0,1.0]))
        # N = igl.per_vertex_normals(V, self.F)
        # get the center of the face

        C = np.mean(V[self.F], axis=1)
        # C = V

        dblA = igl.doublearea(V, self.F)
        self.dbla = igl.doublearea(V, self.F)
        # get height
        self.height = np.max(V[:, 1]) - np.min(V[:, 1])

        # dblA to vertex. vertex area is sum of the adjacent face area/3
        # vertex_area = np.zeros(V.shape[0])
        # for i in range(self.F.shape[0]):
        #     for j in range(3):
        #         vertex_area[self.F[i,j]] += dblA[i]/3
        # vertex_area = vertex_area/vertex_area.sum()*vertex_area.size
        if self.subfilename is not None:
            print("subfilename is not None")
            V_sub, self.F = igl.read_triangle_mesh(self.subfilename)
            # V_sub = V_sub - mean_point + 0.5
            #
            # V_sub = V_sub / scale / 1.2
            # V_sub += 0.5
            # # reduce y
            # V_sub[:, 1] -= 0.2
            V = V_sub
        #normalizing by average area
        dblA = dblA/dblA.sum()*dblA.size
        return C, N, dblA, V, scale



    def readObj_original(self):
        C, N, dbla, V, scale = self.readObj()
        # make rotation matrix that N becomes [0, 0, 1]
        rotation_matrices = []
        for i in range(N.shape[0]):
            axis = np.cross(N[i], np.array([0, 0, 1]))
            if np.allclose(axis, np.zeros(3)):
                R_matrix = np.eye(3)
            else:
                angle = np.arccos(np.dot(N[i], np.array([0, 0, 1])))
                axis = axis/np.linalg.norm(axis)
                r = Rotation.from_rotvec(axis*angle)
                R_matrix = r.as_matrix()
            rotation_matrices.append(R_matrix)
        rotation_matrices = np.array(rotation_matrices)
        # get the rotation matrix
        return C, N, dbla, V, scale, rotation_matrices





    def writeObj_csv(self, V, iteration):
        # make a folder
        filename = self.filename.split("/")[-1]
        filename = filename.split(".")[0]

        # get the date
        import datetime
        import os
        if self.count == 0:
            now = datetime.datetime.now()
            subfolder = "../results/"
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            file_count = 0
            folder = subfolder + "/" + filename + "_" + str(file_count)
            while os.path.exists(folder):
                file_count += 1
                folder = subfolder + "/" + filename + "_" + str(file_count)
            if not os.path.exists(folder):
                os.makedirs(folder)
            self.folder = folder
        filename = self.folder + "/" + str(iteration) + ".obj"
        print("writing to ", filename)
        igl.write_triangle_mesh(filename, V, self.F)
        self.count += 1