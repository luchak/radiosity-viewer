#!/usr/bin/env python

import sys

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT

def _EmptyOneArgCallback(arg):
  pass

def _EmptyZeroArgCallback():
  pass

class ViewerApp(object):
  def __init__(
      self,
      update_callback = _EmptyOneArgCallback,
      display_callback = _EmptyZeroArgCallback,
      init_callback = _EmptyZeroArgCallback,
      window_name = 'viewer',
      window_size = (640, 480),
      frame_rate = 60,
      clear_color = (0.0, 0.0, 0.0, 1.0),
      field_of_view = 45.0,
      ):
    self.update_callback = update_callback
    self.display_callback = display_callback

    self.ms_per_frame = int(1000.0 / frame_rate)
    self.field_of_view = field_of_view

    GLUT.glutInit()
    GLUT.glutInitWindowSize(*window_size)
    GLUT.glutInitDisplayMode(GLUT.GLUT_DOUBLE | GLUT.GLUT_RGB | GLUT.GLUT_DEPTH)
    GLUT.glutCreateWindow(window_name)
    GLUT.glutDisplayFunc(self.Display)
    GLUT.glutKeyboardFunc(self.Key)
    GLUT.glutReshapeFunc(self.Reshape)
    GLUT.glutMouseFunc(self.Mouse)
    GLUT.glutMotionFunc(self.Motion)
    GLUT.glutPassiveMotionFunc(self.PassiveMotion)

    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
    GL.glClearColor(*clear_color)
    self.Reshape(*window_size)
    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glLoadIdentity()
    GLU.gluLookAt(0.0, 0.0, 10.0,
                  0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0)

    init_callback()

  def CurrentTime(self):
    return GLUT.glutGet(GLUT.GLUT_ELAPSED_TIME)

  def Run(self):
    self.last_update_time = self.CurrentTime()
    GLUT.glutTimerFunc(self.ms_per_frame, self._TimerCallback, 0)
    GLUT.glutMainLoop()

  def _TimerCallback(self, ignored):
    GLUT.glutTimerFunc(self.ms_per_frame, self._TimerCallback, 0)
    self.Update()

  def Reshape(self, w, h):
    GL.glViewport(0, 0, w, h)
    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glLoadIdentity()
    GLU.gluPerspective(self.field_of_view, float(w)/float(h), 1.0, 100.0)

  def Display(self):
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    self.display_callback()
    GLUT.glutSwapBuffers()

  def Update(self):
    now = self.CurrentTime()
    elapsed_ms = now - self.last_update_time
    self.update_callback(elapsed_ms / 1000.0)
    self.last_update_time = now
    GLUT.glutPostRedisplay()

  def Key(self, key, x, y):
    if key == 'q':
      sys.exit(0)

  def Mouse(self, button, state, x, y):
    print 'MOUSE', button, state, x, y

  def Motion(self, x, y):
    print 'MOTION', x, y

  def PassiveMotion(self, x, y):
    print 'PASSIVE', x, y
