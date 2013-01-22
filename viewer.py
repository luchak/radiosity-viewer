#!/usr/bin/env python

import sys

import numpy

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT

import geometry.triangle_mesh as triangle_mesh

colors = None
mesh = None

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
  GLU.gluLookAt(0.0, 0.0, 10.0,
                0.0, 0.0, 0.0,
                0.0, 1.0, 0.0)

def Update(dt):
  GL.glMatrixMode(GL.GL_MODELVIEW)
  GL.glRotatef(1.0, 0.0, 1.0, 0.0)
  GLUT.glutPostRedisplay()

def Display():
  points = mesh.vertices
  indices = mesh.faces.flatten()
  GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
  GL.glEnableClientState(GL.GL_VERTEX_ARRAY);
  GL.glVertexPointerf(points)
  GL.glEnableClientState(GL.GL_COLOR_ARRAY);
  GL.glColorPointerf(colors)
  GL.glDrawElementsui(GL.GL_TRIANGLES, indices)
  GLUT.glutSwapBuffers()

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
  global mesh
  mesh = triangle_mesh.Load(argv[1])
  mesh.vertices -= numpy.mean(mesh.vertices, axis=0)
  mesh.vertices *= 20.0
  global colors
  colors = numpy.random.random(mesh.vertices.shape)
  RunAtInterval(16, Update)
  Init()
  GLUT.glutMainLoop()
  return 0

if __name__ == '__main__':
  sys.exit(main(sys.argv))
