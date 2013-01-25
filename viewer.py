#!/usr/bin/env python

import sys

import numpy

import OpenGL.GL as GL

import geometry.triangle_mesh as triangle_mesh
import util.gl.viewer_app as viewer_app

USE_PRECOMPUTED_NORMALS = False

def DrawMesh(mesh):
  GL.glEnable(GL.GL_LIGHTING)
  GL.glEnable(GL.GL_LIGHT0)
  for material, faces in mesh.materials.iteritems():
    faces = numpy.array(list(faces))
    if material == 'highlighted':
      diffuse = numpy.array([1.0, 1.0, 0.0, 1.0])
      GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_DIFFUSE, diffuse)
      specular = numpy.array([0.0, 0.0, 0.0, 1.0])
      GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, specular)
      ambient = numpy.array([1.0, 1.0, 0.1, 1.0])
      GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT, ambient)
    else:
      diffuse = numpy.array([0.8, 0.8, 0.8, 1.0])
      GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_DIFFUSE, diffuse)
      specular = numpy.array([0.0, 0.0, 0.0, 1.0])
      GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, specular)
      ambient = numpy.array([0.1, 0.1, 0.1, 1.0])
      GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT, ambient)
    GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
    GL.glVertexPointerf(mesh.vertices[mesh.faces[faces].flatten()])
    GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
    if USE_PRECOMPUTED_NORMALS and mesh.HasVertexNormals():
      GL.glNormalPointerf(mesh.vertex_normals[mesh.face_vertex_normals[faces]])
    else:
      GL.glNormalPointerf(numpy.repeat(mesh.ComputedFaceNormals()[faces], 3, axis=0))
    GL.glDrawElementsui(GL.GL_TRIANGLES, numpy.arange(3*len(faces)))

def DrawRays(rays):
  GL.glDisable(GL.GL_LIGHTING)
  ray_head_color = numpy.array([0.2, 0.2, 1.0, 1.0])
  ray_tail_color = numpy.array([0.0, 0.0, 0.3, 1.0])
  GL.glBegin(GL.GL_LINES)
  for ray in rays:
    GL.glColor4fv(ray_head_color)
    GL.glVertex3fv(ray[0])
    GL.glColor4fv(ray_tail_color)
    GL.glVertex3fv(ray[0] + ray[1])
  GL.glEnd()

def main(argv):
  mesh = triangle_mesh.LoadMeshFromOBJFile(argv[1])
  AABB = mesh.AABB()
  mesh.vertices -= (AABB[0] + AABB[1]) / 2.0
  axis_sizes = AABB[1] - AABB[0]
  mesh.vertices *= 5.0 / max(axis_sizes)

  rays = []

  def Init():
    GL.glHint(GL.GL_PERSPECTIVE_CORRECTION_HINT, GL.GL_NICEST)

  def Update(dt):
    pass

  def Display():
    DrawMesh(mesh)
    DrawRays(rays)

  app = viewer_app.ViewerApp(
      init_callback = Init,
      update_callback = Update,
      display_callback = Display,
      frame_rate = 60,
      )

  def Key(key, x, y):
    if (key == 'r'):
      ray = app.CastRayThroughWindowCoordinate(x, y)
      rays.append(app.CastRayThroughWindowCoordinate(x, y))
      closest_triangle = mesh.ClosestTriangleRayIntersection(ray)
      if closest_triangle is not None:
        mesh.SetMaterialForFaces([closest_triangle], 'highlighted')

  app.key_callback = Key

  app.Run()
  return 0

if __name__ == '__main__':
  sys.exit(main(sys.argv))
