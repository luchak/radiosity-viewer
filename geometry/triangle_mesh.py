#!/usr/bin/env python

import collections

import numpy

def LoadMeshFromOBJFile(filename):
  return TriangleMesh(LoadOBJFromFile(filename))

def LoadOBJFromFile(filename):
  with open(filename, 'r') as fd:
    return LoadOBJ(fd)

def LoadOBJ(fd):
  vertices = []
  vertex_normals = []
  vertex_texture_coords = []
  faces = []
  face_vertex_normals = []
  face_vertex_texture_coords = []
  groups = collections.defaultdict(list)
  materials = collections.defaultdict(list)
  mtl_file = None

  current_group = None
  current_material = None

  def AddVertex(tokens):
    # No length check: 4 elements is valid, and Blender for some reason outputs
    # 6.
    vertices.append([float(x) for x in tokens[:3]])

  def AddVertexNormal(tokens):
    assert len(tokens) == 3
    vertex_normals.append([float(x) for x in tokens])

  def AddVertexTexture(tokens):
    # w coordinate allowed?
    assert len(tokens) == 2
    vertex_texture_coords.append([float(x) for x in tokens])

  def AddTriangle(tokens):
    assert len(tokens) == 3
    split_tokens = [token.split('/') for token in tokens]
    num_components = len(split_tokens[0])
    assert all([len(split_token) == num_components for split_token in split_tokens])
    
    if current_group is not None:
      groups[current_group].append(len(faces))
    if current_material is not None:
      materials[current_material].append(len(faces))

    faces.append([int(split_token[0]) - 1 for split_token in split_tokens])

    has_texture = (num_components >= 2 and split_tokens[0][1] != '')
    has_normal = (num_components == 3)
    if has_texture:
      face_vertex_texture_coords.append([int(split_token[1]) - 1 for split_token in split_tokens])
    if has_normal:
      face_vertex_normals.append([int(split_token[2]) - 1 for split_token in split_tokens])

  for line in fd:
    line = line.strip()
    if len(line) == 0 or line[0] == '#':
      continue

    tokens = line.split()

    line_type_tag = tokens[0]
    line_data = tokens[1:]
    if line_type_tag == 'v':
      AddVertex(line_data)
    elif line_type_tag == 'vn':
      AddVertexNormal(line_data)
    elif line_type_tag == 'vt':
      AddVertexTexture(line_data)
    elif line_type_tag == 'f':
      # Break up non-triangle polygons into triangle fans
      for i in range(2, len(line_data)):
        AddTriangle([line_data[0], line_data[i-1], line_data[i]])
    elif line_type_tag == 'g':
      current_group = line_data[0]
    elif line_type_tag == 'usemtl':
      current_material = line_data[0]
    elif line_type_tag == 'mtllib':
      assert mtl_file == None
      mtl_file = line_data[0]

  result = {}
  result['vertices'] = vertices
  result['vertex_normals'] = vertex_normals
  result['vertex_texture_coords'] = vertex_texture_coords
  result['faces'] = faces
  result['face_vertex_normals'] = face_vertex_normals
  result['face_vertex_texture_coords'] = face_vertex_texture_coords
  if len(groups) > 0:
    result['groups'] = groups
  if len(materials) > 0:
    result['materials'] = materials
  if mtl_file is not None:
    result['mtl_file'] = mtl_file

  return result

class TriangleMesh(object):
  def __init__(self, attribs):
    # required
    self.vertices = numpy.asarray(attribs['vertices'])
    self.faces = numpy.asarray(attribs['faces'])

    # could be empty
    self.vertex_normals = numpy.asarray(attribs['vertex_normals'])
    self.face_vertex_normals = numpy.asarray(attribs['face_vertex_normals'])

    self.materials = {}
    if 'materials' in attribs:
      self.materials = dict((name, set(faces)) for name, faces in attribs['materials'].iteritems())
    default_material_faces = set(range(len(self.faces)))
    for key in self.materials:
      default_material_faces -= self.materials[key]
    if len(default_material_faces) > 0:
      self.materials[''] = default_material_faces

    self._cached_computed_face_normals = None

  def SetMaterialForFaces(self, faces, material_name):
    if len(faces) == 0:
      return
    faces = set(faces)
    for key in self.materials:
      self.materials[key] -= faces
    if material_name not in self.materials:
      self.materials[material_name] = set()
    self.materials[material_name] |= faces

  def HasVertexNormals(self):
    return len(self.vertex_normals) > 0

  def ComputedFaceNormals(self):
    if self._cached_computed_face_normals is None:
      e1 = self.vertices[self.faces[:,1]] - self.vertices[self.faces[:,0]]
      e2 = self.vertices[self.faces[:,2]] - self.vertices[self.faces[:,0]]

      area_normals = numpy.cross(e1, e2)
      area_normals_length = numpy.sum(area_normals * area_normals, axis=1) ** 0.5
      self._cached_computed_face_normals = area_normals / area_normals_length[:,numpy.newaxis]

    return self._cached_computed_face_normals

  def AABB(self):
    return numpy.array((numpy.min(self.vertices, axis=0),
                        numpy.max(self.vertices, axis=0)))

  def TrianglesIntersectedByRay(self, ray):
    # see http://graphics.stanford.edu/courses/cs348b-04/rayhomo.pdf
    r1 = numpy.repeat(ray[1][numpy.newaxis,:], len(self.faces), axis=0) * 1e-4
    r0 = numpy.repeat(ray[0][numpy.newaxis,:], len(self.faces), axis=0)
    r1 += r0
    p0 = self.vertices[self.faces[:,0]]
    p1 = self.vertices[self.faces[:,1]]
    p2 = self.vertices[self.faces[:,2]]

    def TetVolumesFromPoints(p0, p1, p2, p3):
      return numpy.sum((p1 - p0) * numpy.cross(p2 - p0, p3 - p0), axis=1) * (1.0 / 6.0)

    u0 = TetVolumesFromPoints(r0, r1, p1, p2)
    u1 = TetVolumesFromPoints(r0, r1, p2, p0)
    u2 = TetVolumesFromPoints(r0, r1, p0, p1)

    in_triangle_front = numpy.logical_and(u0 <= 0.0, numpy.logical_and(u1 <= 0.0, u2 <= 0.0))
    in_triangle_back = numpy.logical_and(u0 >= 0.0, numpy.logical_and(u1 >= 0.0, u2 >= 0.0))

    in_triangle = numpy.logical_or(in_triangle_front, in_triangle_back)

    intersected_faces = numpy.arange(len(self.faces))[in_triangle]

    ps0 = self.vertices[self.faces[intersected_faces, 0]]
    ps1 = self.vertices[self.faces[intersected_faces, 1]]
    ps2 = self.vertices[self.faces[intersected_faces, 2]]
    rs0 = r0[intersected_faces]
    rs1 = r1[intersected_faces]

    # s0 negative: past end
    # s1 negative: before start
    s0 = TetVolumesFromPoints(ps0, ps1, ps2, rs1)
    s1 = TetVolumesFromPoints(ps2, ps1, ps0, rs0)

    for i, s0i, s1i in zip(range(len(s0)), s0, s1):
      if in_triangle_front[intersected_faces[i]]:
        s0i = -s0i
        s1i = -s1i
      print s0i, s1i

    return intersected_faces
    
