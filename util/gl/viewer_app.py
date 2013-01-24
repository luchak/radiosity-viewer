#!/usr/bin/env python

import math
import sys

import numpy

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

    self.camera_theta = 0.0
    self.camera_phi = 0.0
    self.camera_r = 10.0
    self.camera_center = numpy.array((0.0, 0.0, 0.0))

    self.mouse_drag_mode = None

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

  def UpdateCameraRotationMatrix(self):
    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glPushMatrix()
    GL.glLoadIdentity()
    GL.glRotatef(self.camera_theta * 180.0 / math.pi, 1.0, 0.0, 0.0)
    GL.glRotatef(self.camera_phi * 180.0 / math.pi, 0.0, 1.0, 0.0)
    self.rotation_matrix = GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX)[:3,:3]
    GL.glPopMatrix()

  def Display(self):
    # Reset camera
    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glLoadIdentity()

    GL.glLoadIdentity()
    GLU.gluLookAt(0.0, 0.0, self.camera_r,
                  0.0, 0.0, self.camera_r - 1.0,
                  0.0, 1.0, 0.0)

    # Translate then rotate: correct camera viewpoint, but rotation around
    # object origin
    # Rotate then translate: wrong camera viewpoint, but rotation around camera
    # origin
    #
    # We deal with this by accumulating camera translation vectors when we
    # process mouse motion pan events, and doing rotate-then-translate.
    GL.glRotatef(self.camera_theta * 180.0 / math.pi, 1.0, 0.0, 0.0)
    GL.glRotatef(self.camera_phi * 180.0 / math.pi, 0.0, 1.0, 0.0)
    GL.glTranslatef(-self.camera_center[0], -self.camera_center[1], -self.camera_center[2])


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
    if button != 0:
      return

    if state == 0:
      self.mouse_last_drag_pos = (x, y)
      modifiers = GLUT.glutGetModifiers()
      if modifiers & GLUT.GLUT_ACTIVE_ALT:
        self.mouse_drag_mode = 'dolly'
      elif modifiers & GLUT.GLUT_ACTIVE_CTRL:
        self.mouse_drag_mode = 'pan'
      else:
        self.mouse_drag_mode = 'orbit'

    if state == 1:
      self.mouse_last_drag_pos = None
      self.mouse_drag_mode = None

  def Motion(self, x, y):
    dx = x - self.mouse_last_drag_pos[0]
    dy = y - self.mouse_last_drag_pos[1]
    self.mouse_last_drag_pos = (x, y)

    if self.mouse_drag_mode == 'orbit':
      self.camera_phi += dx * math.pi / 180.0
      self.camera_theta += dy * math.pi / 180.0
      self.camera_theta = min(self.camera_theta, math.pi/2 - 1e-8)
      self.camera_theta = max(self.camera_theta, -math.pi/2 + 1e-8)
      # Future pan movements will need to know which way the camera is oriented
      # in order to correctly accumulate camera translation.
      self.UpdateCameraRotationMatrix()
    elif self.mouse_drag_mode == 'dolly':
      # Dolly does not change the rotation point. To make dolly change the
      # rotation point (rotation point stays fixed distance in front of
      # camera), use a trick like below in pan, but with a vector purely in z.
      DOLLY_SCALE_FACTOR = 5e-2
      self.camera_r += dy * DOLLY_SCALE_FACTOR
    elif self.mouse_drag_mode == 'pan':
      PAN_SCALE_FACTOR = 2e-2
      self.camera_center += numpy.dot(self.rotation_matrix,
                                      PAN_SCALE_FACTOR * numpy.array((-dx, dy, 0)))

  def PassiveMotion(self, x, y):
    pass
