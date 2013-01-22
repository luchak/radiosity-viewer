#!/usr/bin/env python

import os
import random

import numpy

def Load(filename):
  def NextNonCommentLine(iterable):
    try:
      while True:
        line = iterable.next().strip()
        if line != '' and line[0] != '#':
          return line
    except StopIteration:
      return None

  def LoadOBJ(iterable):
    vertices = []
    faces = []
    line = NextNonCommentLine(iterable)
    while line:
      if line[0:2] == "v ":
        vertices.append([float(x) for x in line[2:].split()[:3]])
      elif line[0] == "f":
        face = [int((x.split("/"))[0])-1 for x in line[2:].split()]
        if len(face) == 4:
	  # Break rectangles into triangles
          faces.append(face[0:3])
          faces.append([face[2], face[3], face[0]])
        else:
          faces.append(face)
      line = NextNonCommentLine(iterable)

    return vertices, faces

  def LoadOFF(iterable):
    first_line = NextNonCommentLine(iterable)
    assert("OFF" == first_line or "COFF" == first_line)
    num_vertices, num_faces, num_edges = [int(x) for x in NextNonCommentLine(iterable).split()]

    vertices = []
    for i in xrange(num_vertices):
      vertices.append([float(x) for x in NextNonCommentLine(iterable).split()])

    faces = []
    for i in xrange(num_faces):
      faces.append([int(x) for x in NextNonCommentLine(iterable).split()[1:4]])

    # don't bother with edges

    return vertices, faces

  def LoadMesh(iterable):
    assert(NextNonCommentLine(iterable).startswith('MeshVersionFormatted'))
    assert(NextNonCommentLine(iterable) == 'Dimension')
    assert(NextNonCommentLine(iterable) == '3')

    assert(NextNonCommentLine(iterable) == 'Vertices')
    num_vertices = int(NextNonCommentLine(iterable))
    vertices = []
    for i in xrange(num_vertices):
      vertices.append([float(x) for x in NextNonCommentLine(iterable).split()[0:3]])

    assert(NextNonCommentLine(iterable) == 'Triangles')
    num_faces = int(NextNonCommentLine(iterable))
    faces = []
    for i in xrange(num_faces):
      faces.append([int(x) - 1 for x in NextNonCommentLine(iterable).split()[0:3]])

    return vertices, faces

  def LoadSMesh(iterable):
    num_vertices = int(NextNonCommentLine(iterable).split()[0])
    vertices = []
    for i in xrange(num_vertices):
      vertices.append([float(x) for x in NextNonCommentLine(iterable).split()[1:4]])

    num_faces = int(NextNonCommentLine(iterable).split()[0])
    faces = []
    for i in xrange(num_faces):
      faces.append([int(x) for x in NextNonCommentLine(iterable).split()[1:4]])

    return vertices, faces

  def LoadSTL(iterable):
    assert(NextNonCommentLine(iterable).split()[0] == 'solid')
    
    vertices = []
    faces = []
    vertex_index = {}

    while True:
      tokens = NextNonCommentLine(iterable).split()
      if tokens[0] == 'endsolid':
        break
      assert(tokens[0] == 'facet')
      assert(NextNonCommentLine(iterable) == 'outer loop')

      face_vertices = []
      for i in xrange(3):
        line = NextNonCommentLine(iterable)
        tokens = line.split()
        assert tokens[0] == 'vertex'

        if line not in vertex_index:
          vertex_index[line] = len(vertices)
          vertices.append([float(x) for x in tokens[1:4]])
        face_vertices.append(vertex_index[line])
      faces.append(face_vertices)

      assert(NextNonCommentLine(iterable) == 'endloop')
      assert(NextNonCommentLine(iterable) == 'endfacet')

    return vertices, faces

  loaders = {'.obj': LoadOBJ,
             '.off': LoadOFF,
             '.mesh': LoadMesh,
             '.smesh': LoadSMesh,
             '.stl': LoadSTL,
             }

  with open(filename, 'r') as mesh_lines:
    ext = os.path.splitext(filename)[1]
    return TriangleMesh(*(loaders[ext](mesh_lines)))

def BlendMeshes(meshes, coefs):
  vertex_arrays = numpy.array([numpy.array(mesh.vertices) for mesh in meshes])
  vertices = numpy.tensordot(vertex_arrays, coefs, [[0,], [0,]])

  return TriangleMesh(list(vertices), list(meshes[0].faces))

def MakeRegularGrid(width, height):
  vertices = []
  faces = []
  for i in range(width):
    for j in range(height):
      vertex_id = len(vertices)
      vertices.append([float(i), float(j), 0.0])
      if i > 0 and j > 0:
        faces.append([vertex_id - width - 1, vertex_id - width, vertex_id])
        faces.append([vertex_id, vertex_id - 1, vertex_id - width - 1])
  return TriangleMesh(vertices, faces)

      

class TriangleMesh(object):
  def __init__(self, vertices=None, faces=None, holes=None):
    self.vertices = numpy.array(vertices if vertices is not None else [])
    self.faces = numpy.array(faces if faces is not None else [])
    self.holes = numpy.array(holes if holes is not None else [])

    self.vertex_face_map = None

  def Copy(self):
    return TriangleMesh(self.vertices, self.faces, self.holes)

  def SaveToSMesh(self, filename):
    with open(filename, 'w') as output:
      output.write('%d 3 0 1\n' % len(self.vertices))
      for i, p in enumerate(self.vertices):
        output.write('%d %06f %06f %06f 1\n' % (i, p[0], p[1], p[2]))

      output.write('%d 0\n' % len(self.faces))
      for i, f in enumerate(self.faces):
        output.write('3 %d %d %d\n' % tuple(f))
        
      output.write('%d\n' % len(self.holes))
      for i, h in enumerate(self.holes):
        output.write('%d %06f %06f %06f\n' % (i, h[0], h[1], h[2]))

  def SaveToOBJ(self, filename):
    with open(filename, 'w') as output:
      for p in self.vertices:
        output.write('v %06f %06f %06f\n' % tuple(p))
      for f in self.faces:
        output.write('f %d %d %d\n' % tuple([i+1 for i in f]))
    

  def SaveToOFF(self, filename):
    with open(filename, 'w') as output:
      output.write('OFF\n')
      output.write('%d %d 0\n' % (len(self.vertices), len(self.faces)))

      for p in self.vertices:
        output.write('%06f %06f %06f\n' % tuple(p))

      for f in self.faces:
        output.write('3 %d %d %d\n' % tuple(f))

  def Save(self, filename):
    save_functions = {'.off': self.SaveToOFF,
                      '.obj': self.SaveToOBJ,
                      '.smesh': self.SaveToSMesh,
                      }
    save_functions[os.path.splitext(filename)[1]](filename)

  def FaceCentroid(self, face):
    return numpy.sum(self.vertices[self.faces[face]], axis=0)

  def FaceCentroids(self):
    return numpy.array([numpy.sum(self.vertices[face], axis=0) for face in self.faces])

  def AddBox(self, min_corner, max_corner):
    start_vertex_index = len(self.vertices)

    # Generate all corner vertices
    for i in xrange(8):
       self.vertices.append(
           [(max_corner[axis] if (i & (1 << axis)) else min_corner[axis])
             for axis in xrange(3)])

    def AddBoxFace(offsets):
      self.faces.append([start_vertex_index + offset for offset in offsets])
    # Generate faces (normals point outwards)
    AddBoxFace([0, 4, 2])
    AddBoxFace([2, 4, 6])
    AddBoxFace([1, 3, 5])
    AddBoxFace([3, 7, 5])

    AddBoxFace([0, 1, 4])
    AddBoxFace([1, 5, 4])
    AddBoxFace([2, 6, 3])
    AddBoxFace([3, 6, 7])

    AddBoxFace([0, 2, 1])
    AddBoxFace([4, 5, 6])
    AddBoxFace([1, 2, 3])
    AddBoxFace([5, 7, 6])

  def BoundingBoxCorners(self):
    mins = [1e10] * 3
    maxes = [-1e10] * 3
    for v in self.vertices:
      for i in xrange(3):
        mins[i] = min(mins[i], v[i])
        maxes[i] = max(maxes[i], v[i])
    return mins, maxes

  def AreaLengthFaceNormals(self):
    vertices = numpy.asarray(self.vertices)
    faces = numpy.asarray(self.faces)

    e1 = vertices[faces[:,1]] - vertices[faces[:,0]]
    e2 = vertices[faces[:,2]] - vertices[faces[:,0]]

    return numpy.cross(e1, e2) / 2

  def TransformVertices(self, fn):
    for i, p in enumerate(self.vertices):
      self.vertices[i] = fn(p)

  def UniformScale(self, factor):
    self.TransformVertices(lambda x: [component * factor for component in x])

  def Translate(self, vector):
    self.TransformVertices(lambda x: [x[i] + vector[i] for i in xrange(len(x))])

  def Bound(self, bound_size) :
    bb_min, bb_max = self.BoundingBoxCorners()
    translation = (bb_max+bb_min) * (-0.5)
    diff = bb_max-bb_min
    scale = max(diff)
    scale = bound_size/scale
    self.Translate(translation)
    self.UniformScale(scale)

  def SampleNewVerticesOnSurface(self, additional_vertices, noise=0.0):
    vertices = numpy.array(self.vertices)

    def FaceArea(face):
      a = vertices[face[2]] - vertices[face[0]]
      b = vertices[face[1]] - vertices[face[0]]
      result = 0.5 * numpy.linalg.norm(numpy.cross(a, b))
      return result

    face_areas = [FaceArea(face) for face in self.faces]
    print min(face_areas)
    print max(face_areas)
    total_area = sum(face_areas)
    vertex_rate = additional_vertices / total_area

    new_vertices = []
    for i, face in enumerate(self.faces):
      num_new_vertices = int(vertex_rate * face_areas[i])
      a = vertices[face[2]] - vertices[face[0]]
      b = vertices[face[1]] - vertices[face[0]]
      for j in xrange(num_new_vertices):
        ar = random.random()
        br = random.random()
        if ar + br >= 1.0:
          ar = 1.0 - ar
          br = 1.0 - br
        if noise > 0.0:
          new_vertices.append(ar*a + br*b + vertices[face[0]] + numpy.random.normal(0.0, noise, (3,)))
        else:
          new_vertices.append(ar*a + br*b + vertices[face[0]])

    self.vertices = list(self.vertices)
    self.vertices.extend(new_vertices)
  
  # def RemoveFace(self, face_index_or_tuple):
  #   """Removes the specified face. The argument can either be a
  #   single numerical face index or a three-tuple of vertex
  #   indices (order independent)."""
  #   
  #   n_faces_before = len(self.faces)
  #   if type(face_index_or_tuple) == tuple:
  #     face_to_remove = tuple(sorted(face_index_or_tuple))
  #     filter_func = lambda face: tuple(sorted(face)) != face_to_remove
  #     self.faces = filter(filter_func, self.faces)
  #   else:
  #     face_idx = face_index_or_tuple
  #     self.faces[face_idx:face_idx+1] = []
  #   n_faces_after = len(self.faces)
  #   
  #   print 'Removed face %s. Faces: %i -> %i.' % \
  #     (face_index_or_tuple, n_faces_before, n_faces_after)
      
  def RotateQuad(self, v1, v2, v3, v4):
    """Vertices should be specified in counter-clockwise order.
    Rotates a face as follows.
    
    v1------v4         v1------v4
    |     / |          | \     |
    |   /   |     =>   |   \   |
    | /     |          |     \ |
    v2------v3         v2------v3
    
    or the other way around.
    """
    # compute a map of the faces
    face_map = dict((tuple(sorted(face)),ii)
      for (ii,face) in enumerate(self.faces))

    print 'Rotating %i %i %i %i...' % (v1, v2, v3, v4)    
    try:
      face_1 = tuple(sorted((v1, v2, v4)))
      face_2 = tuple(sorted((v2, v3, v4)))
      index_1 = face_map[face_1]
      index_2 = face_map[face_2]
      self.faces[index_1] = [v1, v2, v3]
      self.faces[index_2] = [v1, v3, v4]
    except KeyError:
      face_1 = tuple(sorted((v1, v2, v3)))
      face_2 = tuple(sorted((v1, v3, v4)))
      index_1 = face_map[face_1]
      index_2 = face_map[face_2]
      self.faces[index_1] = [v1, v2, v4]
      self.faces[index_2] = [v2, v3, v4]
      
  def FlattenPyramid3(self, v1, v2, v3, c):
    """Mesh xform:
     v1------+     v1
     | \     |      | \
     |   c   |  =>  |   \
     | /   \ |      |     \
     v2------v3     v2------v3
    """
    face_map = dict((tuple(sorted(face)),ii)
     for (ii,face) in enumerate(self.faces))
    print 'Flattening %i %i %i (%i)...' % (v1, v2, v3, c)
    face_1 = tuple(sorted((v1, v2, c )))
    face_2 = tuple(sorted((v2, v3, c )))
    face_3 = tuple(sorted((v3, v1, c )))
    index_1 = face_map[face_1]
    index_2 = face_map[face_2]
    index_3 = face_map[face_3]
    self.faces[index_1] = [v1, v2, v3]
    self.faces[index_2] = [ 0,  0,  0]
    self.faces[index_3] = [ 0,  0,  0]
    
  def FlattenPyramid4(self, v1, v2, v3, v4, c):
    """Mesh xform:
     v1------v4     v1------v4
     | \   / |      |     / | 
     |   c   |  =>  |   /   | 
     | /   \ |      | /     | 
     v2------v3     v2------v3
    """
    face_map = dict((tuple(sorted(face)),ii)
      for (ii,face) in enumerate(self.faces))
    print 'Flattening %i %i %i %i (%i)...' % (v1, v2, v3, v4, c)
    face_1 = tuple(sorted((v1, v2, c )))
    face_2 = tuple(sorted((v2, v3, c )))
    face_3 = tuple(sorted((v3, v4, c )))
    face_4 = tuple(sorted((v4, v1, c )))
    index_1 = face_map[face_1]
    index_2 = face_map[face_2]
    index_3 = face_map[face_3]
    index_4 = face_map[face_4]
    self.faces[index_1] = [v1, v2, v4]
    self.faces[index_2] = [v2, v3, v4]
    self.faces[index_3] = [ 0,  0,  0]
    self.faces[index_4] = [ 0,  0,  0]
    
  def SmoothVertex(self, vv):
    """Places the vertex at the average of its one-ring."""
    print 'Smoothing vertex %i.' % vv
    one_ring = reduce(set.union, (set(face) for face in self.faces if vv in face))
    one_ring -= set((vv,))
    one_ring_vertices = numpy.array(self.vertices)[list(one_ring)]
    new_vertex = one_ring_vertices.sum(axis=0) / len(one_ring)
    self.vertices[vv] = map(float, new_vertex)
    
  def IsNormalized(self):
      """
      A mesh is "normalized" (my term ~ Adrien) iff there are no
      orphan vertices, i.e. every vertex in at least one traingle
      AND There are no edges which are [0, 0, 0], i.e. deleted
      """
      if [0, 0, 0] in self.faces:
        # condition 2 failed
        return False
      vertices_on_faces = set(numpy.array(self.faces).flat)
      all_vertices = set(xrange(len(self.vertices)))
      if vertices_on_faces != all_vertices:
        # condition 1 failed
        return False
      return True

  def Normalize(self):
      """Renormalizes this mesh so that two conditions hold:
      
      (1) There are no orphan vertices.
      (2) There are no edges which are [0, 0, 0], i.e. deleted
  
      See: TriangleMesh.IsNormalized()
      """
      if self.IsNormalized():
        print 'Triangle mesh already normalized.'
        return # nothing to do
        
      # collect some statistics
      old_n_vertices = len(self.vertices)
      old_n_faces = len(self.faces)
        
      # filter out all deteled faces
      self.faces = [face for face in self.faces if face != [0, 0, 0]]
        
      # dictionaries betwen old a new vertices
      vertex_map = list(set(numpy.array(self.faces).flat))
      vertex_map_inv = dict((vv,ii) for (ii,vv) in enumerate(vertex_map))
      vertex_map_inv_func = lambda old_indx: vertex_map_inv[old_indx]
      
      # now do the mapping
      self.vertices = [self.vertices[old_indx] for old_indx in vertex_map]
      self.faces = [map(vertex_map_inv_func, face) for face in self.faces]
      
      # print out summary
      new_n_vertices = len(self.vertices)
      new_n_faces = len(self.faces)
      print 'Renormalized mesh.'
      print 'Vertices: %i -> %i' % (old_n_vertices, new_n_vertices)
      print 'Faces: %i -> %i' % (old_n_faces, new_n_faces)
      assert self.IsNormalized()

  def VertexFaces(self, vertex):
    if self.vertex_face_map is None:
      self.vertex_face_map = [set() for v in self.vertices]
      for face, vertices in self.faces:
        for vertex in vertices:
          self.vertex_face_map[vertex].add(face)

    return self.vertex_face_map[vertex]

def ReplaceOBJVertices(in_path, out_path, new_vertices):
  out_lines = []
  num_vertices = 0
  with open(in_path, 'r') as infile:
    for line in infile:
      line = line.strip()
      if line.startswith('v '):
        out_lines.append('v %f %f %f' % tuple(new_vertices[num_vertices]))
        num_vertices += 1
      else:
        out_lines.append(line)

  if num_vertices != len(new_vertices):
    raise ValueError('Number of vertices do not match')

  with open(out_path, 'w') as outfile:
    for line in out_lines:
      outfile.write(line + '\n')
