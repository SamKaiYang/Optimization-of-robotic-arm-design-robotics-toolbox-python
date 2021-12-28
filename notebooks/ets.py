#!/usr/bin/env python
# coding: utf-8

# # Introduction to Elementary Transform Sequences (ETS)
# Peter Corke

# In[1]:


import roboticstoolbox as rtb
import numpy as np
from spatialmath import SE3
from spatialmath.base import sym

get_ipython().magic(u'matplotlib notebook')


# The traditional approach to robot kinematics is to use Denavit-Hartenberg notation where each link is deescribed by a very specific sequence of simple (elementary) transformations
# 
# $\mathbf{A}_j = r_z(\theta_j) t_z(d_j) t_x(a_j) r_x(\alpha_j)$
# 
# some of which may be missing.  The parameters $(\theta_j, d_j, a_j, \alpha_j)$ are the Denavit-Hartenberg parameters.  A complete robot is defined as a sequence of $\mathbf{A}_j$, one per joint, 
# 
# $\mathbf{T} = \mathbf{T}_b \prod_j\mathbf{A}_j \mathbf{T}_t$
# 
# sometimes with an addition base transform at the befinning and a tool transform on the end.
# For a revolute joint $\theta_j = q_j$ is the joint variable, whereas for a prismatic (sliding) joint $d_j = q_j$ is the joint variable.
# 
# Because a rigid-link has effectively 6 degrees-of-freedom, using just 4 parameters leads to some constraints.  This all conspires to making the Denavit-Hartenberg notation cumbersome to learn, apply and determine for a new robot.
# 
# If we multiply this out we see it is a sequence of transformations drawn from the set $\{r_x, r_z, t_x, t_z\}$.
# Let's take a step back and consider defining a robot from a sequence of transforms drawn from the complete set of all possible elementary (or canonical) transformations $\{r_x, r_y, r_z, t_x, t_y, t_z\}$ any of which can be a function of a constant or a joint variable.
# 
# The Robotics Toolbox implements this functionality. We will start simply
# 

# In[2]:


from roboticstoolbox import ETS as E

E.rx(45, 'deg')


# In[3]:


E.ty(2)


# In[4]:


E.ry()


# Consider we want to create a simple 2 link planar robot arranged like
# 
# o-------o--<>---x 
# 
# where 'o' is a joint, 'x' is the end effector, the first link is 1m long, and the second link can change its length - it is a prismatic joint.  The robot can be represented by just 4 elementary transforms

# In[5]:


e = E.rz() * E.tx(1) * E.rz() * E.tx()
e


# `e` is an ETS object

# In[6]:


type(e)


# which has a number of methods including 

# In[7]:


len(e)


# indicating it has four transforms

# In[8]:


e.n


# indicating it has three joint variables, and they occur at the indices given by

# In[9]:


e.joints()


# The ETS object acts a lot like a list, and we can slice it

# In[10]:


e[1]


# In[11]:


e[2:]


# There are also a number of predicates

# In[12]:


e[0].isjoint


# In[13]:


e[1].isjoint


# In[14]:


e[1].isrevolute


# In[15]:


e[3].isrevolute


# In[16]:


e[3].isprismatic


# We can substitute in values

# In[17]:


e.eval([0, 0, 1])


# and the result is an SE(3) matrix encapsulated in an `SE3` instance. As expected is a translation of 2m in the x-direction.  
# 
# Let try another configuration

# In[18]:


e.eval([90, -90, 2], 'deg')


# Note that the `'deg'` option only applies to those list elements corresponding to angles.

# ## Joint flip
# 
# Sometimes when we are describing a system of joints and constant transforms, the joint coordinate sense needs to be reversed to meet the user's definition.  In the example above, consider that the first joint increases positively in the clockwise direction. In that case we could write

# In[19]:


e = E.rz(flip=True) * E.tx(1) * E.rz() * E.tx()
e


# and the flip is indicated by a negative sign `Rz(-q0)` when the ETS is displayed.

# In[20]:


e.eval([-90, -90, 2], 'deg')


# Now if we negate the passed value of q0 we see that we have compute the same answer as above.
# 
# Joint flip is handled implicitly, but can be tested

# In[21]:


e[0].isflip


# In[22]:


e[1].isflip


# ## Symbolics
# 
# 

# In[23]:


q = sym.symbol('q_:3')
print(q)


# Which creates a vector of three joint angle variables. The advantage of having the underscore in the name is that Jupyter considers this as LaTeX subscript notation when it pretty prints the symbols

# In[24]:


q[0]


# We can substitute in the symbolic values just as easily as we did the numerical values to find the resulting overall transform

# In[25]:


e.eval(q)


# ## ETS in 2D
# 
# Everything that we've done in 3D we can do in 2D.  In fact the example above was a planar problem, so we can express it easily using 2D ETS

# In[26]:


from roboticstoolbox import ETS2 as E2

e2 = E2.r() * E2.tx(1) * E2.r() * E2.tx()


# The main difference is that in 2D there are three elementary transforms to choose from $\{r, t_x, t_y\}$ - rotation and translation in the plane.
# 
# If we evaluate this

# In[27]:


e2.eval([90, -90, 2], 'deg')


# and the result this time is an SE(2) matrix encapsulated in an `SE2` instance.

# ## A more complex example
# 
# We could consider something more ambitious like a Puma560 robot which six unique lengths
# 
# Reference:
# - [A simple and systematic approach to assigning Denavit-Hartenberg parameters](https://petercorke.com/robotics/a-simple-and-systematic-approach-to-assigning-denavit-hartenberg-parameters), Peter I. Corke, IEEE Transactions on Robotics, 23(3), pp 590-594, June 2007.

# In[28]:


l1 = 0.672;
l2 = 0.2337
l3 = 0.4318
l4 = -0.0837
l5 = 0.4318
l6 = 0.0203


# and we can describe the tip of the robot by a sequence of elementary transforms

# In[29]:


e = E.tz(l1) * E.rz() * E.ty(l2) * E.ry() * E.tz(l3) * E.tx(l6) *  E.ty(l4) * E.ry() * E.tz(l5)     * E.rz() * E.ry() * E.rz() * E.tx(0.2)
e


# In[30]:


e.n


# In[31]:


e.joints()


# In[32]:


e.eval([0]*6)


# We can compute not only forward kinematics, but also the differential kinematics

# In[33]:


J = e.jacobe([0]*6)
J


# which relates the rate of change of the end-effector position to the rate of change of joint coordinates.
# The rate of change of the end-effector position is its velocity, but because it moves in 3D we need to describe its translational velocity (a 3-vector) and its angular velocity (another 3-vector).  We pack these into a 6-vector which is known as _spatial velocity_ and denoted by $\nu$.  These quantities are related by
# 
# $\nu = \mathbf{J}(q) \dot{q}$

# The Hessian is a tensor (3-dimensional matrix) that relates joint velocity to end-effector acceleration $\dot{\nu}$

# In[34]:


e.hessian0([0]*6)


# ## Compilation
# 
# If we were to evaluate this ETS many times we would be peforming lots of unneccessary multiplication of constant terms such as 

# In[35]:


e[4:7]


# To make an ETS more efficient to run we can "compile it"

# In[36]:


ec = e.compile()
ec


# and we can see that consecutive constant transforms have been folded into new "not so elementary" transforms denoted by `Ci`.  The value of the ETS, is of course the same.

# In[37]:


ec.eval([0]*6)


# ## Explicit joint indices
# 
# Consider that we carve out a chunk of this ETS that represents the Puma560, say the wrist bit

# In[38]:


wrist = e[9:]
wrist


# we see that by default the wrist joints, joint 3-5 in the robot are associated with joints 0-2 in the wrist.  If we had a vector of six joint angles, corresponding to the whole robot 

# In[39]:


q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


# then to evaluate the wrist we'd have to write

# In[40]:


wrist.eval(q[3:])


# which is quite doable, but imagine now that the ETS represents the transforms from the right hand to the left hand of a two armed robot.  The joint numbers in that ETS would not be sequential with respect to the overall joint coordinate vector of the whole robot.
# 
# To help deal with such situations we can assign a joint index to each ETS.  We'll do this for the Puma 560

# In[41]:


e = E.tz(l1) * E.rz(j=0) * E.ty(l2) * E.ry(j=1) * E.tz(l3) * E.tx(l6) * E.ty(l4) * E.ry(j=2) * E.tz(l5)     * E.rz(j=3) * E.ry(j=4) * E.rz(j=5) * E.tx(0.2)
e


# and when we display the ETS it looks no different, since the joint indices we assigned match the default sequential assignment.  However if we now take the wrist part

# In[42]:


wrist = e[9:]
wrist


# we see that the explicit joint indices are displayed as arguments of the ETs.
# 
# Now if evaluate the wrist part

# In[43]:


wrist.eval(q)


# we get the same answer as earlier because it is referencing `q[3]`, `q[4]` and `q[5]`.

# ## Handling joint offsets
# 
# When using Denavit-Hartenberg notation, we are often forced to choose a robot joint configuration for zero joint angles that is not what the controller or the user may wish to define as zero joint angles.  The `DHRobot` allows the user to specify joint coordinate offsets, a per-link parameter stored in the robots `DHLink` subclass elements.  This is subtracted from the user's joint angles to obtain the kinematic joint angles on which we compute forward kinematics or Jacobians.  Similarly, these offsets are added to the results of any inverse kinematic solution.
# 
# With the ETS convention things are a lot easier. We just draw (or imagine) the robot in the pose we define as the "zero angle pose" - the configuration where all the joint coordinates are zero.  It might be that this leads to transforms sequences with consecutive elements which are a constant and joint transform about the same axis, for example  `rx(90, 'deg') rx()`.  We could write this in arbitrary order - the result will be the same - but for the toolbox we adopt the convention that a constant transformation on the same axis as a joint is placed before the joint transform.  
# 
# The justification for this is that  because when wen come to assign link frames we place them immediately after a joint transform.  For the Puma560 case
# 
# 
# **{0}** E.tz(l1) * E.rz() **{1}** * E.ty(l2) * E.ry() **{2}** * E.tz(l3) * E.tx(l6) * \
#     E.ty(l4) * E.ry() **{3}** * E.tz(l5) * E.rz() **{4}** * E.ry() **{5}** * E.rz() **{6}** * E.tx(0.2) **{ee}**
# 
# 
# These extra transformation incur minimal computational overhead, since after compilation, they will be folded in with other constant transforms.
# 

# ## Converting an ETS to a robot
# 
# The ETS succinctly describes the forward kinematics of a robot and we can also compute its Jacobian and Hessian.  However Toolbox robot objects have additional capability for graphics, inverse kinematics and dynamics.  We can _promote_ an ETS to a robot by

# In[44]:


robot = rtb.ERobot(e)
print(robot)


# The ETS has been chopped into segments that connect links.  Each link is a rigid-body with an attached coordinate frame.  Relative to that frame is a short ETS that ends with joint transform that describes the coordinate frame of the next link in the chain.
# 
# Note that the last link, with its name in blue, has no joint transform.  It is simply a constant relative pose with respect to the link 5 coordinate frame.

# ## Converting a robot to ETS
# 
# We can also perform the inverse transformation. Consider a model defined using Denavit-Hartenberg notation

# In[45]:


puma = rtb.models.DH.Puma560()
print(puma)


# In[46]:


print(puma.ets())


# Or for a robot with a prismatic joint

# In[47]:


stanford = rtb.models.DH.Stanford()
print(stanford)


# In[48]:


print(stanford.ets())


# In[ ]:




