#!/usr/bin/env python

import sys

import numpy

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT

import geometry.triangle_mesh as triangle_mesh

mesh = None
indices = None

def DrawMesh(mesh):
  GL.glEnableClientState(GL.GL_VERTEX_ARRAY);
  GL.glVertexPointerf(mesh.vertices[mesh.faces.flatten()])
  GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
  if mesh.HasVertexNormals():
    GL.glNormalPointerf(mesh.vertex_normals[mesh.face_vertex_normals])
  else:
    GL.glNormalPointerf(numpy.repeat(mesh.ComputedFaceNormals(), 3, axis=0))
  GL.glDrawElementsui(GL.GL_TRIANGLES, numpy.arange(3*len(mesh.faces)))


def RunAtInterval(interval_ms, function):
  def Runner(value):
    GLUT.glutTimerFunc(interval_ms, Runner, 0)
    function(16.0)
  GLUT.glutTimerFunc(interval_ms, Runner, 0)

def Reshape(w, h):
  GL.glViewport(0, 0, w, h)
  GL.glMatrixMode(GL.GL_PROJECTION)
  GL.glLoadIdentity()
  GLU.gluPerspective(45.0, float(w)/float(h), 1.0, 100.0)

def Init():
  GL.glEnable(GL.GL_DEPTH_TEST)
  GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
  GL.glClearColor(0.0,0.0,0.0,1.0)
  GL.glColor3f(1.0,1.0,1.0)
  GL.glPointSize(4.0)
  Reshape(640, 480)
  GL.glMatrixMode(GL.GL_MODELVIEW)
  GL.glLoadIdentity()
  GL.glEnable(GL.GL_LIGHTING)
  GL.glEnable(GL.GL_LIGHT0)
  GL.glHint(GL.GL_PERSPECTIVE_CORRECTION_HINT, GL.GL_NICEST)
  GLU.gluLookAt(0.0, 0.0, 10.0,
                0.0, 0.0, 0.0,
                0.0, 1.0, 0.0)

def Update(dt):
  GL.glMatrixMode(GL.GL_MODELVIEW)
  GL.glRotatef(1.0, 0.0, 1.0, 0.0)
  GLUT.glutPostRedisplay()

def Display():
  GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
  DrawMesh(mesh)
  GLUT.glutSwapBuffers()

def Mouse(button, state, x, y):
  print 'MOUSE', button, state, x, y

def Motion(x, y):
  print 'MOTION', x, y

def PassiveMotion(x, y):
  print 'PASSIVE', x, y

def Key(key, x, y):
  if key == 'q':
    sys.exit(0)

def main(argv):
  GLUT.glutInit()
  GLUT.glutInitWindowSize(640, 480)
  GLUT.glutCreateWindow('viewer')
  GLUT.glutInitDisplayMode(GLUT.GLUT_DOUBLE | GLUT.GLUT_RGB)
  GLUT.glutDisplayFunc(Display)
  GLUT.glutKeyboardFunc(Key)
  GLUT.glutReshapeFunc(Reshape)
  GLUT.glutMouseFunc(Mouse)
  GLUT.glutMotionFunc(Motion)
  GLUT.glutPassiveMotionFunc(PassiveMotion)
  global mesh
  mesh = triangle_mesh.LoadMeshFromOBJFile(argv[1])
  mesh.vertices -= numpy.mean(mesh.vertices, axis=0)
  axis_sizes = numpy.array(mesh.AABB()[1]) - numpy.array(mesh.AABB()[0])
  mesh.vertices *= 5.0 / max(axis_sizes)
  RunAtInterval(16, Update)
  Init()
  GLUT.glutMainLoop()
  return 0

if __name__ == '__main__':
  sys.exit(main(sys.argv))
