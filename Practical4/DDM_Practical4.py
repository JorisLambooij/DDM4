# File created on: 2018-05-08 15:24:47.265919
#
# IMPORTANT:
# ----------
# - Before making a new Practical always make sure you have the latest version of the addon!
# - Delete the old versions of the files in your user folder for practicals you want to update (make a copy of your work!).
# - For the assignment description and to download the addon, see the course website at: http://www.cs.uu.nl/docs/vakken/ddm/
# - Mail any bugs and inconsistencies you see to: uuinfoddm@gmail.com
# - Do not modify any of the signatures provided.
# - You may add as many functions to the assignment as you see fit but try to avoid global variables.

import ddm
import bpy
import math
from mathutils import Vector, Matrix

# Place additional imports here

class Mesh():

    # Construct a new mesh according to some data
    def __init__(self, vertices, faces):
    
        # The vertices are stored as a list of vectors
        self.vertices = vertices
        
        # The faces are stored as a list of triplets of vertex indices
        self.faces = faces
    
        # The uv-coordinates are stored as a list of vectors, with their indices corresponding to the indices of the vertices (there is exactly one uv-coordinate per vertex)
        self.uv_coordinates = []
        for vertex in vertices:
            self.uv_coordinates.append(Vector( (vertex[0], vertex[2]) ))
    
        self.build_edge_list()
    
    # This function builds an edge list from the faces of the current mesh and stores it internally.
    # Make sure that each edge is unique. Remember that order does NOT matter, e.g. (1, 0) is the same edge as (0, 1).
    # The indices of the edges should correspond to the locations of the weights in your weight calculation.
    # All subsequent calls that do something with edges should return their indices. Getting the actual edge can then be done by calling get_edge(index).
    def build_edge_list(self):
    
        self.edges = []
        for triangle in self.faces:
            for i in range(3):
                j = i + 1
                if j == 3:
                    j = 0
                edge = (triangle[i], triangle[j])
                edge_r = (triangle[j], triangle[i])
                if edge not in self.edges and edge_r not in self.edges:
                    self.edges.append( edge )
        
    # ACCESSORS
    
    def get_vertices(self):
        return self.vertices
        
    def get_vertex(self, index):
        return self.vertices[index]
    
    def get_edges(self):
        return self.edges
        
    def get_edge(self, index):
        return self.edges[index]
    
    def get_faces(self):
        return self.faces
        
    def get_face(self, index):
        return self.faces[index]
        
    def get_uv_coordinates(self):
        return self.uv_coordinates
        
    def get_uv_coordinate(self, index):
        return self.uv_coordinates[index]
    
    # Returns the list of vertex coordinates belonging to a face.
    def get_face_vertices(self, face):
        return [ self.get_vertex(face[0]), self.get_vertex(face[1]), self.get_vertex(face[2]) ]
    
    # Looks up the edges belonging to this face in the edge list and returns their INDICES (not value). Make sure that each edge is unique (recall that (1, 0) == (0, 1)). These should match the order of your weights.
    def get_face_edges(self, face):
        indices = []
        for i in range (3):
            j = i + 1
            if j == 3:
                j = 0
            edge = (face[i], face[j])
            edge_r = (face[j], face[i])
            
            try:
                indices.append (self.edges.index(edge))
            except:
                indices.append (self.edges.index(edge_r))
            
        return indices
    
    # Returns the vertex coordinates of the vertices of the given edge (a pair of vertex indices e.g. (0,1) ) 
    def get_edge_vertices(self, edge):
        return [ self.get_vertex(edge[0]), self.get_vertex(edge[1])]
    
    # Returns the flap of the given edge belonging to edge_index, that is two faces connected to the edge 1 for a boundary edge, 2 for internal edges
    def get_flaps(self, edge_index):
        flap = []
        def face_contains_edge(face):
            for i in range (3):
                j = i + 1
                if j == 3:
                    j = 0
                f_edge = (face[i], face[j])
                f_edge_r = (face[j], face[i])
                
                edge_vs = self.edges[edge_index]
                if edge_vs == f_edge or edge_vs == f_edge_r:
                    return True
            return False

        for face in self.faces:
            if face_contains_edge(face):
                flap.append(face)
        # Watch out: edges might be on the boundary
        
        return flap
        
    # Returns the length of the given edge with edge_index
    def get_edge_length(self, edge_index):
        
        edge = self.edges[edge_index]
        
        p1 = self.vertices[edge[0]]
        p2 = self.vertices[edge[1]]
        e = p1 - p2
        return (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]) ** 0.5
        
    # Returns whether the edge has two adjacent faces
    def is_boundary_edge(self, edge_index):
        flap = self.get_flaps(edge_index)
        return len(flap) == 2
    
    # Returns the boundary of the mesh by returning the indices of the edges (from the internal edge list) that lie around the boundary.
    def boundary_edges(self):
        
        boundary = []
        for edge in self.edges:
            if is_boundary_edge:
                if edge[0] not in boundary:
                    boundary.append (edge[0])
                if edge[1] not in boundary:
                    boundary.append (edge[1])
                
        return boundary
        
    # Place any other accessors you need here
    
# This function is called when the DDM operator is selected in Blender.
def DDM_Practical4(context):
    M = get_mesh()
    weights = cotan_weights(M)
    #weights = uniform_weights(M)

    convex = Convex_Boundary_Method(M, weights, 0.5)
    show_mesh(convex, "convex")
    
    lcsm = LSCM(get_mesh())
    show_mesh(lcsm, "LSCM")
    
    # TODO: show_mesh on a copy of the active mesh with uniform UV coordinates, call this mesh "Uniform"
    
    # TODO: show_mesh on a copy of the active mesh with cot UV coordinates, call this mesh "Cot"
    
    # TODO: show_mesh on a copy of the active mesh with boundary free UV coordinates, call this mesh "LSCM"
    

# You may place extra functions here

# Slices a list of triplets and returns two lists of triplets based on a list of fixed columns
# For example if you have a set of triplets T from a matrix with 8 columns, and the fixed columns are [2, 4, 7]
# then all triplets that appear in column [1, 3, 5, 6] are put into "right_triplets" and all triplets that appear in column [2, 4, 7] are put into the set "left_triplets"
def slice_triplets(triplets, fixed_colums):

    left_triplets = []
    right_triplets = []

    # First find the complement of the fixed column set, by finding the maximum column number that appear in the triplets
    max_column = 0
    
    for triplet in triplets:
        if (triplet[1] > max_column):
            max_column = triplet[1]
    
    # and constructing the other columns from those
    other_columns = [x for x in range(0, max_column + 1) if x not in fixed_colums]
    
    # Now split the triplets
    for triplet in triplets:
    
        if (triplet[1] in fixed_colums):
            new_column_index = fixed_colums.index(triplet[1])
            left_triplets.append( (triplet[0], new_column_index, triplet[2]) )
        else:
            new_column_index = other_columns.index(triplet[1])
            right_triplets.append( (triplet[0], new_column_index, triplet[2]) )
            
    return (left_triplets, right_triplets)

# Returns the weights for each edge of mesh M.
# It therefore consists of a list of real numbers such that the index matches the index of the edge list in M.
def cotan_weights(M):
    
    # TODO: implement yourself
    weights = []
    for edgeIndex in range(len(M.edges)):
        flaps = M.get_flaps(edgeIndex)
        if len(flaps) == 1:
            weights.append(0.0)
        else:
            w = 0.0
            edges = M.get_edge_vertices(M.edges[edgeIndex])
            for flap in flaps:
                for vertex in [n for n in M.get_face_vertices(flap) if n not in edges]:
                    w += 1 / math.tan((vertex - edges[0]).angle(vertex - edges[1]))
            w /= 2
            weights.append(w)
    
    return weights
    
# Same as above but for uniform weights
def uniform_weights(M):

    # TODO: implement yourself
    weights = []
    for edgeIndex in range(len(M.edges)):
        if M.is_boundary_edge(edgeIndex) == 1:
            weights.append(0)
        else:
            weights.append(1)
    return weights
    
# Given a set of weights, return M with the uv-coordinates set according to the passed weights
def Convex_Boundary_Method(M, weights, r):
    #Construct the sparse diagonal matrix of weights
    print(len(weights))
    for w_i in range(len(weights)):
        weights[w_i] = (w_i, w_i, weights[w_i])
    W = ddm.Sparse_Matrix(weights, len(weights), len(weights))

    #find boundary verts and create a dictionary for each vertex which vertices in the boundary it connects to
    boundary_length = 0.0
    boundary_verts = set()
    boundary_edges = dict()
    for edgeIndex in range(len(M.edges)):
        if len(M.get_flaps(edgeIndex)) == 1:
            if(M.edges[edgeIndex][0] in boundary_edges):
                boundary_edges[M.edges[edgeIndex][0]].append(M.edges[edgeIndex][1])
            else:
                boundary_edges[M.edges[edgeIndex][0]] = [M.edges[edgeIndex][1]]
            if(M.edges[edgeIndex][1] in boundary_edges):
                boundary_edges[M.edges[edgeIndex][1]].append(M.edges[edgeIndex][0])
            else:
                boundary_edges[M.edges[edgeIndex][1]] = [M.edges[edgeIndex][0]]
            boundary_length += (M.get_vertex(M.edges[edgeIndex][0]) - M.get_vertex(M.edges[edgeIndex][1])).length
            boundary_verts.add(M.edges[edgeIndex][0])
            boundary_verts.add(M.edges[edgeIndex][1])

    #Construct d0 sparse matrix
    d0_list = []
    for e_i in range(len(M.edges)):
        d0_list.append( (e_i, M.edges[e_i][0], -1) )
        d0_list.append( (e_i, M.edges[e_i][1], 1) )

    #find columns to splice
    columns = []
    for vert in range(len(M.get_vertices())):
        if vert in boundary_verts:
            columns.append(vert)

    #create d0I and d0B
    d0B_list, d0I_list = slice_triplets(d0_list, columns)
    d0I_min_list = [(a,b, -c) for (a,b,c) in d0I_list]

    d0B = ddm.Sparse_Matrix(d0B_list, len(M.get_edges()), len(boundary_verts))
    d0I = ddm.Sparse_Matrix(d0I_list, len(M.get_edges()), len(M.get_vertices()) - len(boundary_verts))
    d0I_neg = ddm.Sparse_Matrix(d0I_min_list, len(M.get_edges()), len(M.get_vertices()) - len(boundary_verts))

    #Construct the boundary uv's. Everything is stored by index of the vertices
    boundary_vertex_to_uv = dict()
    firstLoop = True
    source_vert = list(boundary_edges)[0]
    prev_angle = 0
    target_vert = -1
    prev_vert = boundary_edges[source_vert][1]
    while firstLoop or source_vert != list(boundary_edges)[0]:
        if firstLoop:
            firstLoop = False
        if boundary_edges[source_vert][0] == prev_vert:
            target_vert = boundary_edges[source_vert][1]
        else:
            target_vert = boundary_edges[source_vert][0]
        length = (M.get_vertex(target_vert) - M.get_vertex(source_vert)).length
        angle = math.pi * 2.0 * length / boundary_length
        boundary_vertex_to_uv[source_vert] = Vector( (0.5 + r * math.cos(prev_angle), 0.5 + r * math.sin(prev_angle)) )
        prev_angle += angle
        prev_vert = source_vert
        source_vert = target_vert
    
    #construct uB and vB
    uB = []
    vB = []
    for v_i in range(len(M.get_vertices())):
        if v_i in boundary_verts:
            uB.append(boundary_vertex_to_uv[v_i].x)
            vB.append(boundary_vertex_to_uv[v_i].y)

    lhs = d0I.transposed() * W * d0I
    rhs = (d0I_neg.transposed() * W * d0B * uB)

    lhs.Cholesky()
    #solve for u
    uI = lhs.solve(rhs)

    rhs = (d0I_neg.transposed() * W * d0B * vB)
    #solved for v
    vI = lhs.solve(rhs)
    i_count = 0
    b_count = 0
    for i in range(len(M.get_vertices())):
        if i in boundary_verts:
            M.uv_coordinates[i] = Vector((uB[b_count], vB[b_count]))
            b_count += 1
        else:
            M.uv_coordinates[i] = Vector((uI[i_count], vI[i_count]))
            i_count += 1
    
    print(len(M.get_uv_coordinates()))
    print(len(M.get_vertices()))

    return M

# Using Least Squares Conformal Mapping, calculate the uv-coordinates of a given mesh M and return M with those uv-coordinates applied
def LSCM(M):
    def length(i1, i2):
        e = M.vertices[i1] - M.vertices[i2]
        return (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]) ** 0.5
        
    def angle(i1, i2, i3):
        e1 = M.vertices[i1] - M.vertices[i2]
        e2 = M.vertices[i3] - M.vertices[i2]
        cos = e1 * e2 / (length(i2, i1) * length(i3, i2))
        return math.acos(cos)
        
    def function_per_angle(angle_index):
        prev = angle_index - 1
        next  = angle_index + 1
        if next == 3:
            next = 0
        if prev == -1:
            prev = 2
        l_ij = length(triangle[prev], triangle[angle_index])
        l_jk = length(triangle[angle_index], triangle[next])
        angle_ijk = angle(triangle[prev], triangle[angle_index], triangle[next])
        (sin, cos) = (math.cos(angle_ijk), math.sin(angle_ijk))
        R = Matrix([ (cos, sin), (-sin, cos) ])
        M_ijk = l_ij / l_jk * R
        return M_ijk
        
    A_list = [] 
    def insertMatrix(params):
        #(x, y) = topleft pivot
        (x, y, matrix) = params
        A_list.append( (x + 0, y + 0, matrix[0][0]) )
        A_list.append( (x + 0, y + 1, matrix[0][1]) )
        A_list.append( (x + 1, y + 0, matrix[1][0]) )
        A_list.append( (x + 1, y + 1, matrix[1][1]) )
        return
    
    I = Matrix([ (1, 0), (0, 1) ])
    T = len(M.get_faces())
    for t in range(T):
        triangle = M.get_faces()[t]
        
        insertMatrix( (6 * t + 0, 2 * triangle[0], I) )
        insertMatrix( (6 * t + 2, 2 * triangle[0], -1 * function_per_angle(2)) )
        insertMatrix( (6 * t + 4, 2 * triangle[0], function_per_angle(0) - I) )
        
        insertMatrix( (6 * t + 0, 2 * triangle[1], function_per_angle(1) - I) )
        insertMatrix( (6 * t + 2, 2 * triangle[1], I) )
        insertMatrix( (6 * t + 4, 2 * triangle[1], -1 * function_per_angle(0)) )
        
        insertMatrix( (6 * t + 0, 2 * triangle[2], -1 * function_per_angle(1)) )
        insertMatrix( (6 * t + 2, 2 * triangle[2], function_per_angle(2) - I) )
        insertMatrix( (6 * t + 4, 2 * triangle[2], I) )
    
    V = len(M.get_vertices())
    #A = ddm.Sparse_Matrix( A_list, 6 * T, 2 * V )
    
    #############################################################################################################
    #############################################################################################################
    
    #find boundary verts and create a dictionary for each vertex which vertices in the boundary it connects to
    boundary_verts = set()
    boundary_edges = dict()
    for edgeIndex in range(len(M.edges)):
        if len(M.get_flaps(edgeIndex)) == 1:
            boundary_verts.add(M.edges[edgeIndex][0])
            boundary_verts.add(M.edges[edgeIndex][1])
    
    b = len(boundary_verts)
    b_verts_sorted = sorted(boundary_verts)
    columns = [b_verts_sorted[0], b_verts_sorted[math.floor(b / 2)] ]
    #columns = [b_verts_sorted[0], b_verts_sorted[ math.floor(b / 4) ], b_verts_sorted[ math.floor(b / 2) ], b_verts_sorted[ math.floor(3 * b / 4) ] ]
    
    #create d0I and d0B
    d0B_list, d0I_list = slice_triplets(A_list, columns)
    d0I_min_list = [(a,b, -c) for (a,b,c) in d0I_list]
    
    # create d0 matrices
    d0B = ddm.Sparse_Matrix(d0B_list, 6 * T, len(columns))
    d0I = ddm.Sparse_Matrix(d0I_list, 6 * T, 2 * V - len(columns))
    d0I_neg = ddm.Sparse_Matrix(d0I_min_list, 6 * T, 2 * V - len(columns))
    
    uvB = [0, 0, 1, 1]
    #vB = [0, 1]
    
    lhs = d0I.transposed() * d0I
    rhs = (d0I_neg.transposed() * d0B * uvB)
    
    #lhs.Cholesky()
    
    #solve for uv
    #uvI = lhs.solve(rhs)

    #print (uvI)
    #rhs = (d0I_neg.transposed() * d0B * vB)
    #solve for v
    #vI = lhs.solve(rhs)
    
    #reunite I and B
    i_count = 0
    b_count = 0
    for i in range(0):
        if i in columns:
            if b_count < len(uvB):
                M.uv_coordinates[i] = Vector((uvB[b_count], uvB[b_count + 1]))
                b_count += 2
        else:
            print (i_count, "/", len(uvI))
            if i_count < len(uvI):
                uv_co = Vector((uvI[i_count], uvI[i_count + 1]))
                M.uv_coordinates[i] = uv_co
                i_count += 2
    
    
    
    #############################################################################################################
    #############################################################################################################
    
    print ("LSCM done!")
    return M
    
# Builds a Mesh class object from the active object in the scene.
# in essence this function extracts data from the scene and returns it as a (simpler) Mesh class, triangulated where nessecary.
def get_mesh():
    obj = bpy.context.selected_objects[0]
    print (obj.name)
    verts = obj.data.vertices
    polys = obj.data.polygons
    
    vertices = []
    for vertex in verts:
        vertices.append(vertex.co)
        
    faces = []
    for poly in polys:
        faces.append( poly.vertices )
        
    return Mesh(vertices, faces)
    
# Given a Mesh class M, create a new object with name in the scene with the data from M
def show_mesh(M, name):
    
    me = bpy.data.meshes.new("Mesh")
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.objects.link(ob)
    
    edges = M.get_edges()
    faces = M.get_faces()
    verts = M.get_vertices()

    me.uv_textures.new("uv_test")
    me.from_pydata(verts, edges, faces)

    vert_uvs = [vec.to_tuple() for vec in M.get_uv_coordinates()]

    uv_layer = me.uv_layers[-1].data
    vert_loops = {}
    for l in me.loops:
        vert_loops.setdefault(l.vertex_index, []).append(l.index)
    for i, coord in enumerate(vert_uvs):
    # For every loop of a vertex
        for li in vert_loops[i]:
            uv_layer[li].uv = coord
    return
    
    pass