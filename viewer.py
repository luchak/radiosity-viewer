#!/usr/bin/env python

import sys

import numpy

import OpenGL.GL as GL

import geometry.triangle_mesh as triangle_mesh
import util.gl.viewer_app as viewer_app

def DrawMesh(mesh):
  diffuse = numpy.array([0.8, 0.8, 0.8, 1.0])
  GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_DIFFUSE, diffuse)
  specular = numpy.array([0.0, 0.0, 0.0, 1.0])
  GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, specular)
  ambient = numpy.array([0.1, 0.1, 0.1, 1.0])
  GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT, ambient)
  GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
  GL.glVertexPointerf(mesh.vertices[mesh.faces.flatten()])
  GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
  if mesh.HasVertexNormals():
    GL.glNormalPointerf(mesh.vertex_normals[mesh.face_vertex_normals])
  else:
    GL.glNormalPointerf(numpy.repeat(mesh.ComputedFaceNormals(), 3, axis=0))
  GL.glDrawElementsui(GL.GL_TRIANGLES, numpy.arange(3*len(mesh.faces)))


def main(argv):
  mesh = triangle_mesh.LoadMeshFromOBJFile(argv[1])
  AABB = mesh.AABB()
  mesh.vertices -= (AABB[0] + AABB[1]) / 2.0
  axis_sizes = AABB[1] - AABB[0]
  mesh.vertices *= 5.0 / max(axis_sizes)

  def Init():
    GL.glEnable(GL.GL_LIGHTING)
    GL.glEnable(GL.GL_LIGHT0)
    GL.glHint(GL.GL_PERSPECTIVE_CORRECTION_HINT, GL.GL_NICEST)

  def Update(dt):
    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glRotatef(1.0, 0.0, 1.0, 0.0)

  def Display():
    DrawMesh(mesh)

  app = viewer_app.ViewerApp(
      init_callback = Init,
      update_callback = Update,
      display_callback = Display,
      frame_rate = 60,
      )

  app.Run()
  return 0

if __name__ == '__main__':
  sys.exit(main(sys.argv))
