#!/usr/bin/env python
# coding: utf-8

# # Not your grandmother’s toolbox– the Robotics Toolbox reinvented for python
# ### Peter Corke and Jesse Haviland
# 
# This is the code for the examples in the paper published at ICRA2021.
# 

# In[1]:


from math import pi
import numpy as np

# display result of assignments
get_ipython().magic(u"config ZMQInteractiveShell.ast_node_interactivity = 'last_expr_or_assign'")
# make NumPy display a bit nicer
np.set_printoptions(linewidth=100, formatter={'float': lambda x: f"{x:10.4g}" if abs(x) > 1e-10 else f"{0:8.4g}"})
# make cells nice and wide
from IPython.display import display, HTML
display(HTML(data="""
<style>
    div#notebook-container    { width: 95%; }
    div#menubar-container     { width: 65%; }
    div#maintoolbar-container { width: 99%; }
</style>
"""))
get_ipython().magic(u'matplotlib notebook')


# # III.SPATIAL MATHEMATICS

# In[2]:


from spatialmath.base import *
T = transl(0.5, 0.0, 0.0) @ rpy2tr(0.1, 0.2, 0.3, order='xyz') @ trotx(-90, 'deg')


# In[3]:


from spatialmath import *
T = SE3(0.5, 0.0, 0.0) * SE3.RPY([0.1, 0.2, 0.3], order='xyz') * SE3.Rx(-90, unit='deg')


# In[4]:


T.eul()


# In[5]:


T.R


# In[6]:


T.plot(color='red', label='2')


# In[7]:


UnitQuaternion.Rx(0.3)
UnitQuaternion.AngVec(0.3, [1, 0, 0])


# In[8]:


R = SE3.Rx(np.linspace(0, pi/2, num=100))
len(R)


# # IV. ROBOTICS TOOLBOX
# ## A. Robot models

# In[9]:


from roboticstoolbox import *
# robot length values (metres)
d1 = 0.352
a1 = 0.070
a2 = 0.360
d4 = 0.380
d6 = 0.065;


# In[10]:


robot = DHRobot([
  RevoluteDH(d=d1, a=a1, alpha=-pi/2), 
  RevoluteDH(a=a2), 
  RevoluteDH(alpha=pi/2),
  ], name="my IRB140")


# In[11]:


puma = models.DH.Puma560()


# In[12]:


T = puma.fkine([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


# In[13]:


sol = puma.ikine_LM(T)


# In[14]:


puma.plot(sol.q);


# In[15]:


puma.ikine_a(T, config="lun")


# In[16]:


from roboticstoolbox import ETS as ET


# In[17]:


# Puma dimensions (m), see RVC2 Fig. 7.4 for details
l1 = 0.672
l2 = -0.2337
l3 = 0.4318
l4 = 0.0203
l5 = 0.0837
l6 = 0.4318;


# In[18]:


e = ET.tz(l1) * ET.rz() * ET.ty(l2) * ET.ry()     * ET.tz(l3) * ET.tx(l4) * ET.ty(l5) * ET.ry()     * ET.tz(l6) * ET.rz() * ET.ry() * ET.rz()


# In[19]:


robot = ERobot(e)
print(robot)


# In[20]:


panda = models.URDF.Panda()
print(panda)


# ## B. Trajectories

# In[25]:


traj = jtraj(puma.qz, puma.qr, 100)
qplot(traj.q)


# In[26]:


t = np.arange(0, 2, 0.010)
T0 = SE3(0.6, -0.5, 0.3)


# In[27]:


T1 = SE3(0.4, 0.5, 0.2)


# In[28]:


Ts = ctraj(T0, T1, t)
len(Ts)


# In[29]:


sol = puma.ikine_LM(Ts)
sol.q.shape


# ## C. Symbolic manipulation

# In[30]:


import spatialmath.base.symbolic as sym
phi, theta, psi = sym.symbol('φ, ϴ, ψ')
rpy2r(phi, theta, psi)


# In[31]:


q = sym.symbol("q_:6") # q = (q_1, q_2, ... q_5)
T = puma.fkine(q);


# In[32]:


puma = models.DH.Puma560(symbolic=True)
T = puma.fkine(q)
T.t[0]


# In[33]:


puma = models.DH.Puma560(symbolic=False)
J = puma.jacob0(puma.qn)


# In[34]:


J = puma.jacobe(puma.qn)


# ## D. Differential kinematics

# In[35]:


J = puma.jacob0(puma.qr)


# In[36]:


np.linalg.matrix_rank(J)


# In[37]:


jsingu(J)


# In[38]:


H = panda.hessian0(panda.qz)
H.shape


# In[39]:


puma.manipulability(puma.qn)


# In[40]:


puma.manipulability(puma.qn, method="asada")


# In[41]:


puma.manipulability(puma.qn, axes="trans")


# In[42]:


panda.jacobm(panda.qr)


# ## E. Dynamics

# In[43]:


tau = puma.rne(puma.qn, np.zeros((6,)), np.zeros((6,)))


# In[44]:


J = puma.inertia(puma.qn)


# In[45]:


C = puma.coriolis(puma.qn, 0.1 * np.ones((6,)))


# In[46]:


g = puma.gravload(puma.qn)


# In[47]:


qdd = puma.accel(puma.qn, tau, np.zeros((6,)))


# # V. NEW CAPABILITY
# ## B. Collision checking

# In[48]:


from spatialgeometry import Box
obstacle = Box([1, 1, 1], base=SE3(1, 0, 0)) 
iscollision = panda.collided(panda.qr, obstacle) # boolean
iscollision = panda.links[0].collided(obstacle)


# In[49]:


d, p1, p2 = panda.closest_point(panda.qr, obstacle)
print(d, p1, p2)
d, p1, p2 = panda.links[0].closest_point(obstacle)
print(d, p1, p2)


# ## C. Interfaces

# In[50]:


panda.plot(panda.qr, block=False);


# In[ ]:


from roboticstoolbox.backends.swift import Swift
backend = Swift()
backend.launch()   # create graphical world
backend.add(panda) # add robot to the world
panda.q = panda.qr        # update the robot
backend.step()    # display the world

