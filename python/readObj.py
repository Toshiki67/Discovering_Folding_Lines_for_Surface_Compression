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
        self.csv_text = "iteration,area_ratio,max_area_stretch,min_area_stretch,max_edge_stretch,min_edge_stretch,compression_ratio\n";
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

    def readObjline(self):
        V, F = igl.read_triangle_mesh(self.filename)
        # scale based on the bounding box
        # rotate 90 degree in x axis
        # V = V @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

        max_coord = V.max(axis=0)
        min_coord = V.min(axis=0)
        scale = max(max_coord - min_coord)
        mean_point = (max_coord + min_coord) / 2
        # mean_point should be [0.5, 0.5, 0.5]
        V = V - mean_point + 0.5

        V = V/scale/1.2
        V += 0.5
        # reduce y
        # V[:,1] -= 0.2

        E = igl.edges(F)
        # get the center of the edge
        # divide an edge into 4 and get 3 points

        C = np.mean(V[E], axis=1)
        C = np.zeros((E.shape[0]*divide, 3))
        for i in range(E.shape[0]):
            C[i*divide:(i+1)*divide] = np.linspace(V[E[i,0]], V[E[i,1]], divide+2)[1:-1]




        # print(C.shape)
        # normalized edge direction
        # N = V[E[:,1]] - V[E[:,0]]
        N = np.zeros((E.shape[0]*divide, 3))
        for i in range(E.shape[0]):
            N[i*divide:(i+1)*divide] = np.concatenate([[V[E[i,1]] - V[E[i,0]]]]*divide, axis=0)
        # duplicate for dividing edge
        # N = np.concatenate([N]*divide, axis=0)
        # N = igl.per_vertex_normals(V, F)
        # get the center of the face

        self.ev, self.fe, self.ef = igl.edge_topology(V, F)
        # get the cross product of face normal and edge direction
        E_1 = V[self.ev[:,1]] - V[self.ev[:,0]]
        E_1 = E_1/np.linalg.norm(E_1, axis=1)[:,None]
        FN = igl.per_face_normals(V, F, np.array([1.0,1.0,1.0]))
        #normalize
        FN = FN/np.linalg.norm(FN, axis=1)[:,None]
        # if ef element is -1 then use different column ef element
        for i in range(self.ef.shape[1]):
            mask = self.ef[:, i] == -1
            self.ef[mask, i] = self.ef[mask, 1 - i]

        # get the cross product of face normal and edge direction if ef[:, 0] is -1 then use ef[:, 1]
        N_0 = np.cross(FN[self.ef[:,0]], E_1)
        N_1 = np.cross(FN[self.ef[:,1]], E_1)

        N_0 = np.zeros((E.shape[0]*divide, 3))
        N_1 = np.zeros((E.shape[0]*divide, 3))
        for i in range(E.shape[0]):
            N_0[i*divide:(i+1)*divide] = np.concatenate([[np.cross(FN[self.ef[i,0]], E_1[i])]]*divide, axis=0)
            N_1[i*divide:(i+1)*divide] = np.concatenate([[np.cross(FN[self.ef[i,1]], E_1[i])]]*divide, axis=0)

        # length of the edge
        dblA = np.linalg.norm(N, axis=1)
        # if dbla[i] < 1e-10 then, dbla[i] make 1 and remember the index
        mask = dblA < 1e-10
        # print(mask)
        dblA[mask] = 1


        N = N/dblA[:,None]

        # dbla[i] whose i is remembered index should be 0
        dblA[mask] = 0

        # dblA to vertex. vertex area is sum of the adjacent face area/3
        # vertex_area = np.zeros(V.shape[0])
        # for i in range(F.shape[0]):
        #     for j in range(3):
        #         vertex_area[F[i,j]] += dblA[i]/3
        # vertex_area = vertex_area/vertex_area.sum()*vertex_area.size

        #normalizing by average area
        dblA = dblA/dblA.sum()*dblA.size

        return C, N, dblA, V, N_0, N_1

    def writeObj(self, V):
        # make a folder
        filename = self.filename.split("/")[-1]
        filename = filename.split(".")[0]

        # get the date
        import datetime
        import os
        if self.count == 0:
            now = datetime.datetime.now()
            date = now.strftime("%m_%d")
            subfolder = "../results/"+date
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
        filename = self.folder + "/" + str(self.count) + ".obj"
        igl.write_triangle_mesh(filename, V, self.F)
        self.count += 1

    def writeObj_csv(self, V, iteration):
        # make a folder
        filename = self.filename.split("/")[-1]
        filename = filename.split(".")[0]

        # get the date
        import datetime
        import os
        if self.count == 0:
            now = datetime.datetime.now()
            date = now.strftime("%m_%d")
            subfolder = "../results/" + date
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
        igl.write_triangle_mesh(filename, V, self.F)
        current_L = igl.edge_lengths(V, self.ev)
        # get the area ratio
        current_dbla = igl.doublearea(V, self.F)
        stretch = self.dbla/current_dbla
        max_area_stretch = np.max(stretch)
        min_area_stretch = np.min(stretch)
        edge_stretch = self.L/current_L
        max_edge_stretch = np.max(edge_stretch)
        min_edge_stretch = np.min(edge_stretch)
        stretch_ratio = np.sum(current_dbla)/np.sum(self.dbla)

        height = np.max(V[:, 1]) - np.min(V[:, 1])

        self.csv_text += str(iteration) + ","
        self.csv_text += str(stretch_ratio) + ","
        self.csv_text += str(max_area_stretch) + ","
        self.csv_text += str(min_area_stretch) + ","
        self.csv_text += str(max_edge_stretch) + ","
        self.csv_text += str(min_edge_stretch) + ","
        self.csv_text += str(height/self.height) + "\n"
        # save csv
        csv_filename = self.folder + "/" + "result.csv"
        with open(csv_filename, "w") as f:
            f.write(self.csv_text)
            f.close()


        self.count += 1
    def writeParticle(self, V, dm, iteration):
        # make a folder
        filename = self.filename.split("/")[-1]
        filename = filename.split(".")[0]
        filename = self.folder + "/particle_" + str(iteration) + ".obj"
        F_sub = np.array([[0, 1, 2]])
        igl.write_triangle_mesh(filename, V, F_sub)
        filename = self.folder + "/dm_" + str(iteration) + ".csv"
        np.savetxt(filename, dm, delimiter=",")
        self.count_particle += 1

    def write_time(self, time, iteration):
        # make a folder
        filename = self.filename.split("/")[-1]
        filename = filename.split(".")[0]
        filename = self.folder + "/time_" + str(iteration) + ".txt"
        with open(filename, "w") as f:
            f.write(str(time))
            f.close()
    def writeParticle_dg(self, F_dg_np, Sig_np, F_Dp_np, F_cauchy_stress_np, F_Q_np, F_R_np, F_G_np, F_u_p_np, F_sig_p_np, F_v_r_p_np, F_rho_p_np, F_f_e_rho_np, iteration):
        # make a folder
        filename = self.filename.split("/")[-1]
        filename = filename.split(".")[0]
        F_dg_np = F_dg_np.reshape(-1, 9)
        filename = self.folder + "/F_dg_" + str(iteration) + ".csv"
        np.savetxt(filename, F_dg_np, delimiter=",")
        Sig_np = Sig_np.reshape(-1, 9)
        filename = self.folder + "/Sig_" + str(iteration) + ".csv"
        np.savetxt(filename, Sig_np, delimiter=",")
        filename = self.folder + "/Dp_" + str(iteration) + ".csv"
        np.savetxt(filename, F_Dp_np, delimiter=",")
        F_cauchy_stress_np = F_cauchy_stress_np.reshape(-1, 9)
        filename = self.folder + "/cauchy_stress_" + str(iteration) + ".csv"
        np.savetxt(filename, F_cauchy_stress_np, delimiter=",")
        F_Q_np = F_Q_np.reshape(-1, 9)
        filename = self.folder + "/Q_" + str(iteration) + ".csv"
        np.savetxt(filename, F_Q_np, delimiter=",")
        F_R_np = F_R_np.reshape(-1, 9)
        filename = self.folder + "/R_" + str(iteration) + ".csv"
        np.savetxt(filename, F_R_np, delimiter=",")
        F_G_np = F_G_np.reshape(-1, 9)
        filename = self.folder + "/G_" + str(iteration) + ".csv"
        np.savetxt(filename, F_G_np, delimiter=",")
        F_u_p_np = F_u_p_np.reshape(-1, 9)
        filename = self.folder + "/u_p_" + str(iteration) + ".csv"
        np.savetxt(filename, F_u_p_np, delimiter=",")
        F_sig_p_np = F_sig_p_np.reshape(-1, 9)
        filename = self.folder + "/sig_p_" + str(iteration) + ".csv"
        np.savetxt(filename, F_sig_p_np, delimiter=",")
        F_v_r_p_np = F_v_r_p_np.reshape(-1, 9)
        filename = self.folder + "/v_r_p_" + str(iteration) + ".csv"
        np.savetxt(filename, F_v_r_p_np, delimiter=",")
        F_rho_p_np = F_rho_p_np.reshape(-1, 9)
        filename = self.folder + "/rho_p_" + str(iteration) + ".csv"
        np.savetxt(filename, F_rho_p_np, delimiter=",")
        F_f_e_rho_np = F_f_e_rho_np.reshape(-1, 9)
        filename = self.folder + "/f_e_rho_" + str(iteration) + ".csv"
        np.savetxt(filename, F_f_e_rho_np, delimiter=",")
        self.count_particle += 1

    def get_dihedral_angle(self, V):
        V = V.astype(np.float64)
        E_1 = V[self.ev[:, 1]] - V[self.ev[:, 0]]
        E_1 = E_1 / np.linalg.norm(E_1, axis=1)[:, None]
        FN = igl.per_face_normals(V, self.F, np.array([1.0, 1.0, 1.0]))
        # normalize
        FN = FN / np.linalg.norm(FN, axis=1)[:, None]
        # if ef element is -1 then use different column ef element
        for i in range(self.ef.shape[1]):
            mask = self.ef[:, i] == -1
            self.ef[mask, i] = self.ef[mask, 1 - i]

        # get the cross product of face normal and edge direction if ef[:, 0] is -1 then use ef[:, 1]
        N_0 = np.cross(FN[self.ef[:, 0]], E_1)
        N_1 = np.cross(FN[self.ef[:, 1]], E_1)
        # normalize
        N_0 = N_0 / np.linalg.norm(N_0, axis=1)[:, None]
        N_1 = N_1 / np.linalg.norm(N_1, axis=1)[:, None]
        # inner product N_0 and N_1
        cos_theta = np.sum(N_0*N_1, axis=1)
        cos_theta_div = np.zeros((cos_theta.shape[0]*divide))
        for i in range(cos_theta.shape[0]):
            cos_theta_div[i*divide:(i+1)*divide] = np.concatenate([[cos_theta[i]]*divide], axis=0)
            print(cos_theta[i])
        return cos_theta_div



    def writeV(self, V_second, V_third, iteration):
        # make a folder
        filename = self.filename.split("/")[-1]
        filename = filename.split(".")[0]

        # get the date
        import datetime
        import os
        if self.count == 0:
            now = datetime.datetime.now()
            date = now.strftime("%m_%d")
            subfolder = "../results/" + date
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
        filename = self.folder + "/" + str(iteration) + "_lv1.obj"
        igl.write_triangle_mesh(filename, V_second, self.F_second)
        filename = self.folder + "/" + str(iteration) + "_lv2.obj"
        igl.write_triangle_mesh(filename, V_third, self.F_third)


if "__main__" == __name__:
    filename = "../model/cone.obj"
    C, N, dblA= readObj(filename)