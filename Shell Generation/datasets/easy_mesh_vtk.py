import sys
import numpy as np
import vtk
from .Wrapping_Python_vtk_util_numpy_support import *
#import Wrapping_Python_vtk_util_numpy_support
from .Wrapping_Python_vtk_util_numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
#from PIL import Image
import math
#import matplotlib.pyplot as plt

class Easy_Mesh(object):
    def __init__(self, filename = None, warning=False):
        #initialize
        self.warning = warning
        self.reader = None
        self.vtkPolyData = None
        self.cells = np.array([])
        self.cell_ids = np.array([])
        self.points = np.array([])
        self.point_attributes = dict()
        self.cell_attributes = dict()
        self.filename = filename
        if self.filename != None:
            if self.filename[-3:].lower() == 'vtp':
                self.read_vtp(self.filename)
            elif self.filename[-3:].lower() == 'stl':
                self.read_stl(self.filename)
            elif self.filename[-3:].lower() == 'obj':
                self.read_obj(self.filename)
            else:
                if self.warning:
                    print('Not support file type')


    def get_mesh_data_from_vtkPolyData(self):
        data = self.vtkPolyData

        n_triangles = data.GetNumberOfCells()
        #print(n_triangles)
        n_points = data.GetNumberOfPoints()
        mesh_triangles = np.zeros([n_triangles, 9], dtype='float32')
        mesh_triangle_ids = np.zeros([n_triangles, 3], dtype='int32')
        mesh_points = np.zeros([n_points, 3], dtype='float32')

        for i in range(n_triangles):
            mesh_triangles[i][0], mesh_triangles[i][1], mesh_triangles[i][2] = data.GetPoint(data.GetCell(i).GetPointId(0))
            mesh_triangles[i][3], mesh_triangles[i][4], mesh_triangles[i][5] = data.GetPoint(data.GetCell(i).GetPointId(1))
            mesh_triangles[i][6], mesh_triangles[i][7], mesh_triangles[i][8] = data.GetPoint(data.GetCell(i).GetPointId(2))
            mesh_triangle_ids[i][0] = data.GetCell(i).GetPointId(0)
            mesh_triangle_ids[i][1] = data.GetCell(i).GetPointId(1)
            mesh_triangle_ids[i][2] = data.GetCell(i).GetPointId(2)

        for i in range(n_points):
            mesh_points[i][0], mesh_points[i][1], mesh_points[i][2] = data.GetPoint(i)

        self.cells = mesh_triangles
        self.cell_ids = mesh_triangle_ids
        self.points = mesh_points

        #read point arrays
        for i_attribute in range(self.vtkPolyData.GetPointData().GetNumberOfArrays()):
#            print(self.vtkPolyData.GetPointData().GetArrayName(i_attribute))
#            print(self.vtkPolyData.GetPointData().GetArray(i_attribute).GetNumberOfComponents())
            self.load_point_attributes(self.vtkPolyData.GetPointData().GetArrayName(i_attribute), self.vtkPolyData.GetPointData().GetArray(i_attribute).GetNumberOfComponents())

        #read cell arrays
        for i_attribute in range(self.vtkPolyData.GetCellData().GetNumberOfArrays()):
#            print(self.vtkPolyData.GetCellData().GetArrayName(i_attribute))
#            print(self.vtkPolyData.GetCellData().GetArray(i_attribute).GetNumberOfComponents())
            self.load_cell_attributes(self.vtkPolyData.GetCellData().GetArrayName(i_attribute), self.vtkPolyData.GetCellData().GetArray(i_attribute).GetNumberOfComponents())


    def read_stl(self, stl_filename):
        '''
        update
            self.filename
            self.reader
            self.vtkPolyData
            self.cells
            self.cell_ids
            self.points
            self.cell_attributes
            self.point_attributes
        '''
#        self.filename = stl_filename
        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_filename)
        reader.Update()
        self.reader = reader

        data = reader.GetOutput()
        self.vtkPolyData = data
        self.get_mesh_data_from_vtkPolyData()


    def read_obj(self, obj_filename):
        '''
        update
            self.filename
            self.reader
            self.vtkPolyData
            self.cells
            self.cell_ids
            self.points
            self.cell_attributes
            self.point_attributes
        '''
#        self.filename = obj_filename
        reader = vtk.vtkOBJReader()
        reader.SetFileName(obj_filename)
        reader.Update()
        self.reader = reader

        data = reader.GetOutput()
        self.vtkPolyData = data
        self.get_mesh_data_from_vtkPolyData()


    def read_vtp(self, vtp_filename):
        '''
        update
            self.filename
            self.reader
            self.vtkPolyData
            self.cells
            self.cell_ids
            self.points
            self.cell_attributes
            self.point_attributes
        '''
#        self.filename = vtp_filename
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(vtp_filename)
        reader.Update()
        self.reader = reader

        data = reader.GetOutput()
        self.vtkPolyData = data
        self.get_mesh_data_from_vtkPolyData()


    def load_point_attributes(self, attribute_name, dim):
        self.point_attributes[attribute_name] = np.zeros([self.points.shape[0], dim])
        try:
            if dim == 1:
                for i in range(self.points.shape[0]):
                    self.point_attributes[attribute_name][i, 0] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetValue(i)
            elif dim == 2:
                for i in range(self.points.shape[0]):
                    self.point_attributes[attribute_name][i, 0] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetComponent(i, 0)
                    self.point_attributes[attribute_name][i, 1] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetComponent(i, 1)
            elif dim == 3:
                for i in range(self.points.shape[0]):
                    self.point_attributes[attribute_name][i, 0] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetComponent(i, 0)
                    self.point_attributes[attribute_name][i, 1] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetComponent(i, 1)
                    self.point_attributes[attribute_name][i, 2] = self.vtkPolyData.GetPointData().GetArray(attribute_name).GetComponent(i, 2)
        except:
            if self.warning:
                print('No cell attribute named "{0}" in file: {1}'.format(attribute_name, self.filename))


    def get_point_curvatures(self, method='mean'):
        curv = vtk.vtkCurvatures()
        curv.SetInputData(self.vtkPolyData)
        if method == 'mean':
            curv.SetCurvatureTypeToMean()
        elif method == 'max':
            curv.SetCurvatureTypeToMaximum()
        elif method == 'min':
            curv.SetCurvatureTypeToMinimum()
        elif method == 'Gaussian':
            curv.SetCurvatureTypeToGaussian()
        else:
            curv.SetCurvatureTypeToMean()
        curv.Update()

        n_points = self.vtkPolyData.GetNumberOfPoints()
        self.point_attributes['Curvature'] = np.zeros([n_points, 1])
        for i in range(n_points):
            self.point_attributes['Curvature'][i] = curv.GetOutput().GetPointData().GetArray(0).GetValue(i)


    def get_cell_curvatures(self, method='mean'):
        self.get_point_curvatures(method=method)
        self.cell_attributes['Curvature'] = np.zeros([self.cells.shape[0], 1])

        # optimized way
        tmp_cell_curvts = self.point_attributes['Curvature'][self.cell_ids].squeeze()
        self.cell_attributes['Curvature'] = np.mean(tmp_cell_curvts, axis=-1).reshape([tmp_cell_curvts.shape[0], 1])



    def load_cell_attributes(self, attribute_name, dim):
        self.cell_attributes[attribute_name] = np.zeros([self.cells.shape[0], dim])
        try:
            if dim == 1:
                for i in range(self.cells.shape[0]):
                    self.cell_attributes[attribute_name][i, 0] = self.vtkPolyData.GetCellData().GetArray(attribute_name).GetValue(i)
            elif dim == 2:
                for i in range(self.cells.shape[0]):
                    self.cell_attributes[attribute_name][i, 0] = self.vtkPolyData.GetCellData().GetArray(attribute_name).GetComponent(i, 0)
                    self.cell_attributes[attribute_name][i, 1] = self.vtkPolyData.GetCellData().GetArray(attribute_name).GetComponent(i, 1)
            elif dim == 3:
                for i in range(self.cells.shape[0]):
                    self.cell_attributes[attribute_name][i, 0] = self.vtkPolyData.GetCellData().GetArray(attribute_name).GetComponent(i, 0)
                    self.cell_attributes[attribute_name][i, 1] = self.vtkPolyData.GetCellData().GetArray(attribute_name).GetComponent(i, 1)
                    self.cell_attributes[attribute_name][i, 2] = self.vtkPolyData.GetCellData().GetArray(attribute_name).GetComponent(i, 2)
        except:
            if self.warning:
                print('No cell attribute named "{0}" in file: {1}'.format(attribute_name, self.filename))

    def set_cell_labels(self, label_dict, tol=0.01):#0.01
        '''
        update:
            self.cell_attributes['Label']
        '''
        from scipy.spatial import distance_matrix
        self.cell_attributes['Label'] = np.zeros([self.cell_ids.shape[0], 1])

        cell_centers = (self.cells[:, 0:3] + self.cells[:, 3:6] + self.cells[:, 6:9]) / 3.0
        for i_label in label_dict:
            i_label_cell_centers = (label_dict[i_label][:, 0:3] + label_dict[i_label][:, 3:6] + label_dict[i_label][:, 6:9]) / 3.0
            D = distance_matrix(cell_centers, i_label_cell_centers)

            if len(np.argwhere(D<=tol)) > i_label_cell_centers.shape[0]:
                print(i_label_cell_centers.shape[0])
                print(len(np.argwhere(D <= tol)))
                sys.exit('tolerance ({0}) is too large, please adjust.'.format(tol))

            elif len(np.argwhere(D<=tol)) < i_label_cell_centers.shape[0]:
                print(i_label_cell_centers.shape[0])
                print(len(np.argwhere(D<=tol)))
                sys.exit('tolerance ({0}) is too small, please adjust.'.format(tol))
            else:
                for i in range(i_label_cell_centers.shape[0]):
                    label_id = np.argwhere(D<=tol)[i][0]
                    self.cell_attributes['Label'][label_id, 0] = int(i_label)
    def set_cell_labels_map(self, label_dict, ratio,tol=0.01):#0.01
        '''
        update:
            self.cell_attributes['Label']
        '''
        from scipy.spatial import distance_matrix
        ratio=0
        self.cell_attributes['Label'] = np.zeros([self.cell_ids.shape[0], 1])
        self.cell_attributes['Distance'] = np.ones([self.cell_ids.shape[0], 1])
        cell_centers = (self.cells[:, 0:3] + self.cells[:, 3:6] + self.cells[:, 6:9]) / 3.0
        for i_label in label_dict:
            i_label_cell_centers = (label_dict[i_label][:, 0:3] + label_dict[i_label][:, 3:6] + label_dict[i_label][:, 6:9]) / 3.0
            D = distance_matrix(cell_centers, i_label_cell_centers)
            Shortest_D = np.min(D, axis=1)
            #compute the shortest distance
            shortest_index = sorted(range(len(Shortest_D)), key=lambda k: Shortest_D[k])
            K_shortest_index = shortest_index[0:math.ceil(i_label_cell_centers.shape[0]*(1-ratio))]
            #print(Shortest_D[K_shortest_index])
            Shortest_D.sort()
            #print(Shortest_D)
            # Update the label if you find the better label with the shortest path

            for i,label_id in enumerate(K_shortest_index):
                 if self.cell_attributes['Distance'][label_id, 0] >= Shortest_D[i] and (Shortest_D[i]<tol):
                    self.cell_attributes['Label'][label_id, 0] = int(i_label)+1
                    self.cell_attributes['Distance'][label_id, 0] = Shortest_D[i]


    def Depth_Img(self):
        cell_centers = (self.cells[:, 0:3] + self.cells[:, 3:6] + self.cells[:, 6:9]) / 3.0
        max_z = np.int32(max(cell_centers[:, 2]))
        depth_z =  max_z - cell_centers[:,2]
        cell_centers= np.int32(cell_centers)
        IMG_z = np.zeros([100,100])
        depth_z = (255 * (depth_z - np.min(depth_z)) / np.ptp(depth_z)).astype(int)
        IMG_z[cell_centers[:,0],cell_centers[:,1]] = depth_z
        # Normalised [0,255] as integer: don't forget the parenthesis before astype(int)
        im_z = Image.fromarray(IMG_z)
        plt.imshow(im_z, cmap=plt.get_cmap('gray'))

        plt.savefig('lena_greyscale.png')
        plt.show()
    def Sub_divided_mesh(self,ref_labels,idx,Mesh_name,tol):
        Part_mesh = Easy_Mesh()
        Output_path = './Sliced_Lower_refined'
        import os
        from scipy.spatial import distance_matrix
        Jaw_idx = np.where(self.cell_attributes['Label'] == 0)[0]  # extract the target label
        mesh_jaw = Easy_Mesh()
        mesh_jaw.cells = self.cells[Jaw_idx]
        mesh_jaw.update_cell_ids_and_points()
        Jaw_cell_center = (self.cells[Jaw_idx, 0:3] + self.cells[Jaw_idx, 3:6] + self.cells[Jaw_idx, 6:9]) / 3.0
        i_tooth_idx = [idx for idx, e in enumerate(self.cell_attributes['Label']) if e in ref_labels]
        i_tooth_idx_mid = [idx for idx, e in enumerate(self.cell_attributes['Label']) if e in ref_labels[1:-1]]

        # i_tooth_idx = np.where(self.cell_attributes['Label'] == i_label)[0]  # extract the target label
        i_label_cell_centers = (self.cells[i_tooth_idx_mid, 0:3] + self.cells[i_tooth_idx_mid, 3:6] + self.cells[
                                                                                                      i_tooth_idx_mid,
                                                                                                      6:9]) / 3.0
        D = distance_matrix(Jaw_cell_center, i_label_cell_centers)
        # i_tooth_idx=np.arange(3000)
        Shortest_D = np.min(D, axis=1)
        # compute the shortest distance
        shortest_index = sorted(range(len(Shortest_D)), key=lambda k: Shortest_D[k])
        K_shortest_index = shortest_index[0:math.ceil(i_label_cell_centers.shape[0])]
        # print(Shortest_D[K_shortest_index])
        Shortest_D.sort()
        idx_max = np.max(np.where(Shortest_D < tol), axis=1)
        # ComplementElements_r = listComplementElements(np.arange(len(self.cells)), shortest_index[0:idx_max[0]])
        # Part_mesh.cells = np.delete(self.cells, ComplementElements_r, 0)
        Part_mesh.cells = mesh_jaw.cells[shortest_index[0:idx_max[0]]]
        Part_mesh.cells = np.append(Part_mesh.cells, self.cells[i_tooth_idx], axis=0)
        Part_mesh.update_cell_ids_and_points()
        Part_mesh.cell_attributes['Label'] = np.zeros([len(i_tooth_idx) + idx_max[0], 1], dtype=np.int32)
        # Part_mesh.cell_attributes['Label'][:] = 0#np.zeros([len(shortest_index[0:idx_max]), 1], dtype=np.int32)  # create cell array
        # add lables and teeth

        Part_mesh.cell_attributes['Label'][idx_max[0]:] = self.cell_attributes['Label'][i_tooth_idx]
        Part_mesh.to_vtp(os.path.join(Output_path, 'Part_{}_size_{}_{}'.format(idx, len(ref_labels), Mesh_name)))




    def get_cell_edges(self):
        '''
        update:
            self.cell_attributes['Edge']
        '''
        self.cell_attributes['Edge'] = np.zeros([self.cell_ids.shape[0], 3])

        for i_count in range(self.cell_ids.shape[0]):
            v1 = self.points[self.cell_ids[i_count, 0], :] - self.points[self.cell_ids[i_count, 1], :]
            v2 = self.points[self.cell_ids[i_count, 1], :] - self.points[self.cell_ids[i_count, 2], :]
            v3 = self.points[self.cell_ids[i_count, 0], :] - self.points[self.cell_ids[i_count, 2], :]
            self.cell_attributes['Edge'][i_count, 0] = np.linalg.norm(v1)
            self.cell_attributes['Edge'][i_count, 1] = np.linalg.norm(v2)
            self.cell_attributes['Edge'][i_count, 2] = np.linalg.norm(v3)


    def get_cell_normals(self):
        data = self.vtkPolyData
        n_triangles = data.GetNumberOfCells()
        #normal
        v1 = np.zeros([n_triangles, 3], dtype='float32')
        v2 = np.zeros([n_triangles, 3], dtype='float32')
        v1[:, 0] = self.cells[:, 0] - self.cells[:, 3]
        v1[:, 1] = self.cells[:, 1] - self.cells[:, 4]
        v1[:, 2] = self.cells[:, 2] - self.cells[:, 5]
        v2[:, 0] = self.cells[:, 3] - self.cells[:, 6]
        v2[:, 1] = self.cells[:, 4] - self.cells[:, 7]
        v2[:, 2] = self.cells[:, 5] - self.cells[:, 8]
        mesh_normals = np.cross(v1, v2)
        mesh_normal_length = np.linalg.norm(mesh_normals, axis=1)
        mesh_normals[:, 0] /= mesh_normal_length[:]
        mesh_normals[:, 1] /= mesh_normal_length[:]
        mesh_normals[:, 2] /= mesh_normal_length[:]
        self.cell_attributes['Normal'] = mesh_normals


    def compute_guassian_heatmap(self, landmark, sigma = 10.0, height = 1.0):
        '''
        inputs:
            landmark: np.array [1, 3]
            sigma (default=10.0)
            height (default=1.0)
        update:
            self.cell_attributes['heatmap']
        '''
        cell_centers = (self.cells[:, 0:3] + self.cells[:, 3:6] + self.cells[:, 6:9]) / 3.0
        heatmap = np.zeros([cell_centers.shape[0], 1])

        for i_cell in range(len(cell_centers)):
            delx = cell_centers[i_cell, 0] - landmark[0]
            dely = cell_centers[i_cell, 1] - landmark[1]
            delz = cell_centers[i_cell, 2] - landmark[2]
            heatmap[i_cell, 0] = height*math.exp(-1*(delx*delx+dely*dely+delz*delz)/2.0/sigma/sigma)
        self.cell_attributes['Heatmap'] = heatmap


    def compute_displacement_map(self, landmark):
        '''
        inputs:
            landmark: np.array [1, 3]
        update:
            self.cell_attributes['Displacement map']
        '''
        cell_centers = (self.cells[:, 0:3] + self.cells[:, 3:6] + self.cells[:, 6:9]) / 3.0
        displacement_map = np.zeros([cell_centers.shape[0], 3])

        for i_cell in range(len(cell_centers)):
            delx = cell_centers[i_cell, 0] - landmark[0]
            dely = cell_centers[i_cell, 1] - landmark[1]
            delz = cell_centers[i_cell, 2] - landmark[2]
            displacement_map[i_cell, 0] = delx
            displacement_map[i_cell, 1] = dely
            displacement_map[i_cell, 2] = delz
        self.cell_attributes['Displacement_map'] = displacement_map


    def compute_cell_attributes_by_svm(self, given_cells, given_cell_attributes, attribute_name, refine=True):
        '''
        inputs:
            given_cells: [n, 9] numpy array
            given_cell_attributes: [n, 1] numpy array
        update:
            self.cell_attributes[attribute_name]
        '''
        from sklearn import svm
        if given_cell_attributes.shape[1] == 1:
            self.cell_attributes[attribute_name] = np.zeros([self.cells.shape[0], 1])
            if refine:
                clf = svm.SVC(probability=True)
            else:
                clf = svm.SVC()
            clf.fit(given_cells, given_cell_attributes.ravel())
            self.cell_attributes[attribute_name][:, 0] = clf.predict(self.cells)

            if refine:
                self.cell_attributes[attribute_name+'_proba'] = clf.predict_proba(self.cells)
                self.graph_cut_refinement(self.cell_attributes[attribute_name+'_proba'])
        else:
            if self.warning:
                print('Only support 1D attribute')

    def compute_cell_attributes_by_knn(self, given_cells, given_cell_attributes, attribute_name, k=3, refine=True):
        '''
        inputs:
            given_cells: [n, 9] numpy array
            given_cell_attributes: [n, 1] numpy array
        update:
            self.cell_attributes[attribute_name]
        '''
        from sklearn.neighbors import KNeighborsClassifier
        if given_cell_attributes.shape[1] == 1:
            self.cell_attributes[attribute_name] = np.zeros([self.cells.shape[0], 1])
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(given_cells, given_cell_attributes.ravel())
            self.cell_attributes[attribute_name][:, 0] = neigh.predict(self.cells)
            self.cell_attributes[attribute_name+'_proba'] = neigh.predict_proba(self.cells)

            if refine:
                self.graph_cut_refinement(self.cell_attributes[attribute_name+'_proba'])
        else:
            if self.warning:
                print('Only support 1D attribute')

    def graph_cut_refinement(self, patch_prob_output):
        from pygco import cut_from_graph
        round_factor = 100
        patch_prob_output[patch_prob_output<1.0e-6] = 1.0e-6

        # unaries
        unaries = -round_factor * np.log10(patch_prob_output)
        unaries = unaries.astype(np.int32)
        unaries = unaries.reshape(-1, patch_prob_output.shape[1])

        # parawise
        pairwise = (1 - np.eye(patch_prob_output.shape[1], dtype=np.int32))

        #edges
        self.get_cell_normals()
        normals = self.cell_attributes['Normal'][:]
        cells = self.cells[:]
        cell_ids = self.cell_ids[:]
        barycenters = (cells[:, 0:3] + cells[:, 3:6] + cells[:, 6:9])/3.0

        lambda_c = 30
        edges = np.empty([1, 3], order='C')
        for i_node in range(cells.shape[0]):
            # Find neighbors
            nei = np.sum(np.isin(cell_ids, cell_ids[i_node, :]), axis=1)
            nei_id = np.where(nei==2)
            for i_nei in nei_id[0][:]:
                if i_node < i_nei:
                    cos_theta = np.dot(normals[i_node, 0:3], normals[i_nei, 0:3])/np.linalg.norm(normals[i_node, 0:3])/np.linalg.norm(normals[i_nei, 0:3])
                    if cos_theta >= 1.0:
                        cos_theta = 0.9999
                    theta = np.arccos(cos_theta)
                    phi = np.linalg.norm(barycenters[i_node, :] - barycenters[i_nei, :])
                    if theta > np.pi/2.0:
                        edges = np.concatenate((edges, np.array([i_node, i_nei, -math.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
                    else:
                        beta = 1 + np.linalg.norm(np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]))
                        edges = np.concatenate((edges, np.array([i_node, i_nei, -beta*math.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
        edges = np.delete(edges, 0, 0)
        edges[:, 2] *= lambda_c*round_factor
        edges = edges.astype(np.int32)

        refine_labels = cut_from_graph(edges, unaries, pairwise)
        refine_labels = refine_labels.reshape([-1, 1])

        # output refined result
        self.cell_attributes['Label'] = refine_labels


    def update_cell_ids_and_points(self):
        '''
        call when self.cells is modified
        update
            self.cell_ids
            self.points
        '''
        rdt_points = self.cells.reshape([int(self.cells.shape[0]*3), 3])
        self.points, idx = np.unique(rdt_points, return_inverse=True, axis=0)
        self.cell_ids = idx.reshape([-1, 3])

        if self.warning:
            print('Warning! self.cell_attributes are reset and need to be updated!')
        self.cell_attributes = dict() #reset
        self.point_attributes = dict() #reset
        self.update_vtkPolyData()


    def update_vtkPolyData(self):
        '''
        call this function when manipulating self.cells, self.cell_ids, or self.points
        '''
        vtkPolyData = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()

        points.SetData(numpy_to_vtk(self.points))
        cells.SetCells(len(self.cell_ids),
                       numpy_to_vtkIdTypeArray(np.hstack((np.ones(len(self.cell_ids))[:, None] * 3,
                                                          self.cell_ids)).astype(np.int64).ravel(),
                                               deep=1))
        vtkPolyData.SetPoints(points)
        vtkPolyData.SetPolys(cells)

        #update point_attributes
        for i_key in self.point_attributes.keys():
            point_attribute = vtk.vtkDoubleArray()
            point_attribute.SetName(i_key);
            if self.point_attributes[i_key].shape[1] == 1:
                point_attribute.SetNumberOfComponents(self.point_attributes[i_key].shape[1])
                for i_attribute in self.point_attributes[i_key]:
                    point_attribute.InsertNextTuple1(i_attribute)
                vtkPolyData.GetPointData().AddArray(point_attribute)
#                vtkPolyData.GetPointData().SetScalars(cell_attribute)
            elif self.point_attributes[i_key].shape[1] == 2:
                point_attribute.SetNumberOfComponents(self.point_attributes[i_key].shape[1])
                for i_attribute in self.point_attributes[i_key]:
                    point_attribute.InsertNextTuple2(i_attribute[0], i_attribute[1])
                vtkPolyData.GetPointData().AddArray(point_attribute)
#                vtkPolyData.GetPointData().SetVectors(cell_attribute)
            elif self.point_attributes[i_key].shape[1] == 3:
                point_attribute.SetNumberOfComponents(self.point_attributes[i_key].shape[1])
                for i_attribute in self.point_attributes[i_key]:
                    point_attribute.InsertNextTuple3(i_attribute[0], i_attribute[1], i_attribute[2])
                vtkPolyData.GetPointData().AddArray(point_attribute)
#                vtkPolyData.GetPointData().SetVectors(cell_attribute)
            else:
                if self.warning:
                    print('Check attribute dimension, only support 1D, 2D, and 3D now')

        #update cell_attributes
        for i_key in self.cell_attributes.keys():
            cell_attribute = vtk.vtkDoubleArray()
            cell_attribute.SetName(i_key);
            if self.cell_attributes[i_key].shape[1] == 1:
                cell_attribute.SetNumberOfComponents(self.cell_attributes[i_key].shape[1])
                for i_attribute in self.cell_attributes[i_key]:
                    cell_attribute.InsertNextTuple1(i_attribute)
                vtkPolyData.GetCellData().AddArray(cell_attribute)
#                vtkPolyData.GetCellData().SetScalars(cell_attribute)
            elif self.cell_attributes[i_key].shape[1] == 2:
                cell_attribute.SetNumberOfComponents(self.cell_attributes[i_key].shape[1])
                for i_attribute in self.cell_attributes[i_key]:
                    cell_attribute.InsertNextTuple2(i_attribute[0], i_attribute[1])
                vtkPolyData.GetCellData().AddArray(cell_attribute)
#                vtkPolyData.GetCellData().SetVectors(cell_attribute)
            elif self.cell_attributes[i_key].shape[1] == 3:
                cell_attribute.SetNumberOfComponents(self.cell_attributes[i_key].shape[1])
                for i_attribute in self.cell_attributes[i_key]:
                    cell_attribute.InsertNextTuple3(i_attribute[0], i_attribute[1], i_attribute[2])
                vtkPolyData.GetCellData().AddArray(cell_attribute)
#                vtkPolyData.GetCellData().SetVectors(cell_attribute)
            else:
                if self.warning:
                    print('Check attribute dimension, only support 1D, 2D, and 3D now')

        vtkPolyData.Modified()
        self.vtkPolyData = vtkPolyData


    def extract_largest_region(self):
        connect = vtk.vtkPolyDataConnectivityFilter()
        connect.SetInputData(self.vtkPolyData)
        connect.SetExtractionModeToLargestRegion()
        connect.Update()

        self.vtkPolyData = connect.GetOutput()
        self.get_mesh_data_from_vtkPolyData()
        if self.warning:
            print('Warning! self.cell_attributes are reset and need to be updated!')
        self.cell_attributes = dict() #reset
        self.point_attributes = dict() #reset


    def mesh_decimation(self, reduction_rate, original_label_status=True):
        # check mesh has label attribute or not
        if original_label_status:
            original_cells = self.cells.copy()
            original_labels = self.cell_attributes['Label'].copy()

        decimate_reader = vtk.vtkQuadricDecimation()
        decimate_reader.SetInputData(self.vtkPolyData)
        decimate_reader.SetTargetReduction(reduction_rate)
        decimate_reader.VolumePreservationOn()
        decimate_reader.Update()
        self.vtkPolyData = decimate_reader.GetOutput()
        self.get_mesh_data_from_vtkPolyData()
        if self.warning:
            print('Warning! self.cell_attributes are reset and need to be updated!')
        self.cell_attributes = dict() #reset
        self.point_attributes = dict() #reset

        if original_label_status:
            self.compute_cell_attributes_by_svm(original_cells, original_labels, 'Label')


    def mesh_subdivision(self, num_subdivisions, method='loop', original_label_status=False):
        if method == 'loop':
            subdivision_reader = vtk.vtkLoopSubdivisionFilter()
        elif method == 'butterfly':
            subdivision_reader = vtk.vtkButterflySubdivisionFilter()
        else:
            if self.warning:
                print('Not a valid subdivision method')

        # check mesh has label attribute or not
        if original_label_status:
            original_cells = self.cells.copy()
            original_labels = self.cell_attributes['Label'].copy()

        subdivision_reader.SetInputData(self.vtkPolyData)
        subdivision_reader.SetNumberOfSubdivisions(num_subdivisions)
        subdivision_reader.Update()
        self.vtkPolyData = subdivision_reader.GetOutput()
        self.get_mesh_data_from_vtkPolyData()
        if self.warning:
            print('Warning! self.cell_attributes are reset and need to be updated!')
        self.cell_attributes = dict() #reset
        self.point_attributes = dict() #reset

        if original_label_status:
            self.compute_cell_attributes_by_svm(original_cells, original_labels, 'Label')


    def mesh_transform(self, vtk_matrix):
        Trans = vtk.vtkTransform()
        Trans.SetMatrix(vtk_matrix)

        TransFilter = vtk.vtkTransformPolyDataFilter()
        TransFilter.SetTransform(Trans)
        TransFilter.SetInputData(self.vtkPolyData)
        TransFilter.Update()

        self.vtkPolyData = TransFilter.GetOutput()
        self.get_mesh_data_from_vtkPolyData()


    def mesh_reflection(self, ref_axis='x'):
        '''
        This function is only for tooth arch model,
        it will flip the label (n=15 so far) as well.
        input:
            ref_axis: 'x'/'y'/'z'
        '''
        RefFilter = vtk.vtkReflectionFilter()
        if ref_axis == 'x':
            RefFilter.SetPlaneToX()
        elif ref_axis == 'y':
            RefFilter.SetPlaneToY()
        elif ref_axis == 'z':
            RefFilter.SetPlaneToZ()
        else:
            if self.warning:
                print('Invalid ref_axis!')

        RefFilter.CopyInputOff()
        RefFilter.SetInputData(self.vtkPolyData)
        RefFilter.Update()

        self.vtkPolyData = RefFilter.GetOutput()
        self.get_mesh_data_from_vtkPolyData()

        original_cell_labels = np.copy(self.cell_attributes['Label']) # add original cell label back
        # for permanent teeth
        for i in range(1,17):
            if len(original_cell_labels==i) > 0:
                self.cell_attributes['Label'][original_cell_labels==i] = 17-i #1 -> 14, 2 -> 13, ..., 14 -> 1
        #For the preparatin
        self.cell_attributes['Label'][original_cell_labels == 17] = 17
        # for primary teeth
        #for i in range(15, 25):
           # if len(original_cell_labels==i) > 0:
               # self.cell_attributes['Label'][original_cell_labels==i] = 39-i #15 -> 24, 16 -> 23, ..., 24 -> 15


    def get_boundary_points(self):
        '''
        output: boundary_points [n, 3] nparray
        '''
        featureEdges = vtk.vtkFeatureEdges()
        featureEdges.SetInputData(self.vtkPolyData)
        featureEdges.BoundaryEdgesOn()
        featureEdges.FeatureEdgesOff()
        featureEdges.ManifoldEdgesOff()
        featureEdges.NonManifoldEdgesOff()
        featureEdges.Update()

        num_bps = featureEdges.GetOutput().GetNumberOfPoints()
        boundary_points = np.zeros([num_bps, 3], dtype='float32')
        for i in range(num_bps):
            boundary_points[i][0], boundary_points[i][1], boundary_points[i][2] = featureEdges.GetOutput().GetPoint(i)

        return boundary_points


    def to_vtp(self, vtp_filename):
        self.update_vtkPolyData()

        if vtk.VTK_MAJOR_VERSION <= 5:
            self.vtkPolyData.Update()

        writer = vtk.vtkXMLPolyDataWriter();
        writer.SetFileName("{0}".format(vtp_filename));
        if vtk.VTK_MAJOR_VERSION <= 5:
            writer.SetInput(self.vtkPolyData)
        else:
            writer.SetInputData(self.vtkPolyData)
        writer.Write()


    def to_obj(self, obj_filename):
        with open(obj_filename, 'w') as f:
            for i_point in self.points:
                f.write("v {} {} {}\n".format(i_point[0], i_point[1], i_point[2]))

            for i_label in np.unique(self.cell_attributes['Label']):
                f.write("g mmGroup{}\n".format(int(i_label)))
                label_cell_ids = np.where(self.cell_attributes['Label']==i_label)[0]
                for i_label_cell_id in label_cell_ids:
                    i_cell = self.cell_ids[i_label_cell_id]
                    f.write("f {}//{} {}//{} {}//{}\n".format(i_cell[0]+1, i_cell[0]+1, i_cell[1]+1, i_cell[1]+1, i_cell[2]+1, i_cell[2]+1))




#------------------------------------------------------------------------------
def GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                               translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                               scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2]):
    '''
    get transformation matrix (4*4)

    return: vtkMatrix4x4
    '''
    Trans = vtk.vtkTransform()

    ry_flag = np.random.randint(0,2) #if 0, no rotate
    rx_flag = np.random.randint(0,2) #if 0, no rotate
    rz_flag = np.random.randint(0,2) #if 0, no rotate
    if ry_flag == 1:
        # rotate along Yth axis
        Trans.RotateY(np.random.uniform(rotate_Y[0], rotate_Y[1]))
    if rx_flag == 1:
        # rotate along Xth axis
        Trans.RotateX(np.random.uniform(rotate_X[0], rotate_X[1]))
    if rz_flag == 1:
        # rotate along Zth axis
        Trans.RotateZ(np.random.uniform(rotate_Z[0], rotate_Z[1]))

    trans_flag = np.random.randint(0,2) #if 0, no translate
    if trans_flag == 1:
        Trans.Translate([np.random.uniform(translate_X[0], translate_X[1]),
                         np.random.uniform(translate_Y[0], translate_Y[1]),
                         np.random.uniform(translate_Z[0], translate_Z[1])])

    scale_flag = np.random.randint(0,2)
    if scale_flag == 1:
        Trans.Scale([np.random.uniform(scale_X[0], scale_X[1]),
                     np.random.uniform(scale_Y[0], scale_Y[1]),
                     np.random.uniform(scale_Z[0], scale_Z[1])])

    matrix = Trans.GetMatrix()

    return matrix


def listComplementElements(list1, list2):
    storeResults = []
    for num in list1:
        if num not in list2:  # this will essentially iterate your list behind the scenes
            storeResults.append(num)
    return storeResults
