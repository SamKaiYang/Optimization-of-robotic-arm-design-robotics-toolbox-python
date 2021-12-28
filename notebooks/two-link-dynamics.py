#!/usr/bin/env python
# coding: utf-8

# # Dynamics of 2-link manipulator
# 
# ![Spong book](figs/spong-2link.png)
# 
# Reference:
# 
# Robot Modeling and Control, Spong, Vidyasagar, Hutchinson, Wiley 2006
# 
# We will choose the parameters to be

# In[1]:


from roboticstoolbox import DHRobot, RevoluteDH, ERobot, ELink, ETS
from math import pi, sin, cos
import numpy as np
np.set_printoptions(linewidth=100, suppress=True)



# link 1
m1 = 1
l1 = 1
lc1 = 0.5

# link 2
m2 = 1
l2 = 1
lc2 = 0.5

# joint configurations
q1 = [0, 0]        # o-o- arm horizontal
q2 = [0, pi/2]    # o-o| upper arm horizontal, lower arm vertical
q3 = [pi/2, 0]    # o|o| arm vertical
q4 = [pi/2, -pi/2] # o|o- upper arm upward, lower arm horizontal
qq = [q1, q2, q3, q4]
z = [0, 0]

# global parameters
g = 9.81


# ## Denavit-Hartenberg notation

# In[2]:


L1 = RevoluteDH(a=l1, m=m1, r=[-lc1, 0, 0])
L2 = RevoluteDH(a=l2, m=m2, r=[-lc2, 0, 0])
robot = DHRobot([L1, L2], gravity=[0, g, 0])


# The DH notation only allows rotation about the z-axis, which is out of the page in the Figure above.  Gravity acts downward so we set the gravity acceleration to be `[0, g, 0]` - accelerating upward at `g` provides the weight force experienced on Earth.

# In[3]:


print(robot)
print(robot.dyntable())


# In[4]:


for q in qq:
    tau = robot.gravload(q)
    print(tau)


# Using equations (7.85) and (7.86) we can write an expression for gravity torque derived analytically using Euler-Lagrange (rather than Newton-Euler) approach

# In[5]:


def EL_grav(q): 
    return [(m1 * lc1 + m2 * l1) * g * cos(q[0]) + m2 * lc2 * g * cos(q[0] + q[1]), m2 * lc2 * g * cos(q[0] + q[1])]


# In[6]:


for q in qq:
    tau_rne = robot.gravload(q)
    tau_el = EL_grav(q)
    print(tau_rne, 'vs', tau_el)


# In fact we could use symbolic values of the constants above, repeat the above process, and generate the symbolic equations for gravity load.

# Now we can explore the velocity coupling terms.  To isolate them we will set gravity and joint acceleration to zero

# In[7]:


robot.gravity = [0, 0, 0]
robot.rne(q2, [1, 1], z)


# Using equations on page 261 and (7.87) we can write the velocity dependent terms as

# In[8]:


def EL_velocity(q, qd):
    h = -m2 * l1 * lc2 * sin(q[1])
    c121 = h
    c211 = c121
    c221 = h
    c112 = -h
    c122 = 0
    c222 = 0
    return [c121 * qd[0] * qd[1] + c211 * qd[1] * qd[0] + c221 * qd[1] ** 2, c112 * qd[0] ** 2]


# In[9]:


qd = [1, 1]
for q in qq:
    tau_rne = robot.rne(q, qd, z)
    tau_el = EL_velocity(q, qd)
    print(tau_rne, 'vs', tau_el)


# In[10]:


from spatialmath.base import symbol

# link 1
m1 = symbol('m_1')
l1 = symbol('l_1')
lc1 = symbol('l_c_1')

# link 2
m2 = symbol('m_2')
l2 = symbol('l_2')
lc2 = symbol('l_c_2')

g = symbol('g')

L1 = RevoluteDH(a=l1, m=m1, r=[-lc1, 0, 0])
L2 = RevoluteDH(a=l2, m=m2, r=[-lc2, 0, 0])
robot = DHRobot([L1, L2], gravity=[0, g, 0], symbolic=True)


q = symbol('q_:2')
qd = symbol('qd_:2')
robot.rne_python(q, qd, [0, 0])


# ## ERobot notation

# We will redefine the parameters of the model to numerical values

# In[14]:


# link 1
m1 = 1
l1 = 1
lc1 = 0.5

# link 2
m2 = 1
l2 = 1
lc2 = 0.5

# joint configurations
q1 = [0, 0]        # o-o- arm horizontal
q2 = [0, pi/2]    # o-o| upper arm horizontal, lower arm vertical
q3 = [pi/2, 0]    # o|o| arm vertical
q4 = [pi/2, -pi/2] # o|o- upper arm upward, lower arm horizontal
qq = [q1, q2, q3, q4]
z = [0, 0]

# global parameters
g = 9.81


# In[15]:


L1 = ELink(ets=ETS.rz(), m=1, r=[0.5, 0, 0], name='L1')
L2 = ELink(ets=ETS.tx(1) * ETS.rz(), m=1, r=[0.5, 0, 0], parent=L1, name='L2')
robot = ERobot([L1, L2], gravity=[0, g, 0])


# In[18]:


for q in qq:
    tau_rne = robot.gravload(q)
    tau_el = EL_grav(q)
    print(tau_rne, 'vs', tau_el)


# In[17]:





# In[ ]:





# In[ ]:




