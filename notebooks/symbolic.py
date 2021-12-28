#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math
import roboticstoolbox as rtb
from spatialmath.base import *
from spatialmath import SE3
import spatialmath.base.symbolic as sym

get_ipython().magic(u'matplotlib notebook')


# We use the Spatial Math Toolbox wrapper of SymPy which is `spatialmath.base.symbolic` which is imported above.
# 
# ## Creating a symbolic variable
# We will first create a symbolic variable

# In[ ]:


theta = sym.symbol('theta')


# In[ ]:


theta


# which displays as a Greek letter, it has the type

# In[ ]:


type(theta)


# The function specifies that the symbolic variables are real valued by default, which will simplify subsequent simplification steps

# In[ ]:


theta.is_real


# We can test if a variable is symbolic

# In[ ]:


sym.issymbol(theta)


# In[ ]:


sym.issymbol(3.7)


# ## Symbolics with the Spatial Math Toolbox
# Many Spatial Toolbox functions handle symbolic variables

# In[ ]:


R = rot2(theta)
R


# and return a NumPy array with symbolic values.  
# 
# The 3D case is similar

# In[ ]:


R = rotx(theta)
R


# In[ ]:


T = trotx(theta)
T


# The elements of this NumPy array are all objects, as indicated by the `dtype`.  However when we index the elements, they will be converted back to numeric types if possible

# In[ ]:


type(T[0,0])


# In[ ]:


type(T[1,1])


# We can perform arithmetic on such matrices, for example

# In[ ]:


T @ T


# ## Pose classes
# The symbolic capability extends to the pose classes

# In[ ]:


T = SE3.Rx(theta)
T


# In[ ]:


T2 = T * T
T2


# but the colored layout is problematic.

# ## Robot forward kinematics

# We will create a symbolic version of the robot model

# In[ ]:


puma = rtb.models.DH.Puma560(symbolic=True)


# In[ ]:


print(puma)


# We see that the $\alpha_j$ values are now given in radians and are colored red.  This means the value is a symbolic expression, for the first link it is $\pi/2$ which is a precise value, compared to the non-symbolic case which is a floating point number 1.5707963267948966 that only  approximates $\pi/2$.

# The next thing we need to do is create a vector of joint coordinates which are all symbols

# In[ ]:


q = sym.symbol('q_:6')
q


# We use the underscore, because the value of the symbol is pretty printed by SymPy as a subscript (just as it is with LaTeX math notation)

# In[ ]:


q[0]


# We are now set to compute the forward kinematics which will be a matrix whose elements will be complicated expressions of symbolic joint variables and kinematic parameters

# In[ ]:


T = puma.fkine(q)
T


# The color coding helps us identify the rotational and translational parts, but the format is not very readable.  We can display the underlying NumPy array

# In[ ]:


T.A


# which is not a lot better.
# 
# As this stage it is far better to convert the result to a SymPy matrix

# In[ ]:


from sympy import Matrix
Matrix(T.A)


# which is decently pretty printed by SymPy.
# 
# Often after a round of symbolic calculations there are simplifications that can be achieved. We can symbolically simplify each element of the `SE3` object by

# In[ ]:


Ts = T.simplify()

M = Matrix(Ts.A)
M


# which is more compact (it takes a few seconds to compute). We can see that a trigometric _sum of angles_  substition has been performed, there are instances of sine and cosine of $q_1 + q_2$, the shoulder and elbow joints. This is to be expected since these joints are adjacent and have with parallel axes.
# 
# We can _slice_ the end-effector translationfrom the SymPy matrix

# In[ ]:


M[:3,3]


# The floating point constants here have been inherited from the kinematic model we imported at [this step](#Robot-forward-kinematics).  It would be possible to replace the non-zero kinematic constants $a_i$ and $d_i$ with symbols created using `sym.symbol` as shown in [this section](#Creating-a-symbolic-variable).

# ## Code generation
# We can now use some of SymPy's superpowers to turn our forward kinematic expression into code

# In[ ]:


from sympy import ccode, pycode, octave_code
print(ccode(M, assign_to="T"))


# which is pure C code that does not require any linear algebra package to compute. We simply need to define the values of `q_0` to `q_5` in order to determine the end-effector pose. The result is computed using symbolically simplified expressions.  The code is not optimized, but we could expect the compiler to perform some additional simplification.
# 
# The equivalent MATLAB code is

# In[ ]:


octave_code(M)


# We can also output Python code

# In[ ]:


print(pycode(M))


# Which constructs an instance of a SymPy `ImmutableDenseMatrix` which we can turn into a function

# In[ ]:


from sympy import lambdify
T_func = lambdify(q, M, modules='numpy')


# If we pass in the zero joint angles we get the familiar forward kinematic solution result

# In[ ]:


T_func(0, 0, 0, 0, 0, 0)


# SymPy also supports printing code in C++ (`cxxcode`), Fortran (`fcode`), JavaScript (`jscode`), Julia (`julia_code`), Rust (`rust_code`), Mathematica (`mathematica_code`) and Octave/MATLAB (`octave_code`).
# 
# The [SymPy autowrap](https://docs.sympy.org/latest/modules/codegen.html#autowrap) capability automatically generates code, writes it to disk, compiles it, and imports it into the current session.  It creates a wrapper using Cython and creates a numerical function.

# ## Robot dynamics
# 
# We can also compute the dynamics symbolically.  To do this we must use the Python version of the inverse dynamics rather than the default efficient C-code implementation.
# 
# We need to setup the problem. Firstly we make the gravitational constant a symbol

# In[ ]:


g = sym.symbol('g')
puma.gravity = [0, 0, g]


# Next we create symbolic vectors for joint velocity and acceleration, just as we did earlier for joint coordinates

# In[ ]:


qd = sym.symbol('qd_:6')
qd


# In[ ]:


qdd = sym.symbol('qdd_:6')
qdd


# Now we compute the inverse dynamics as a function of symbolic joint coordinates, velocities, accelerations and gravity as well as a lot of numerical kinematic and dynamic parameters.
# 
# Note that this next step might take 10 minutes or more to execute (but the result will be impressive and worth the wait!)

# In[ ]:


get_ipython().magic(u'time tau = puma.rne_python(q, qd, qdd)')


# In[ ]:


from sympy import trigsimp, simplify


# In[ ]:


get_ipython().magic(u'time z = simplify(tau[5])')


# In[ ]:


get_ipython().magic(u'time z = simplify(tau[4])')


# In[ ]:


get_ipython().magic(u'time z = simplify(tau[3])')


# In[ ]:


get_ipython().magic(u'time z = simplify(tau[2])')


# In[ ]:


get_ipython().magic(u'time z = simplify(tau[1])')


# In[ ]:


get_ipython().magic(u'time z = simplify(tau[0])')


# In[ ]:





# In[ ]:





# The result `tau` is not a NumPy array as it would be for the numeric case

# In[ ]:


type(tau)


# but it is the symbolic (SymPy) equivalent and it has the expected shape

# In[ ]:


tau.shape


# The torque on the first joint $\tau_0$ is

# In[ ]:


tau[0]


# which is a complicated expression.  To make it easier to work with, we will expand it

# In[ ]:


tau_0 = tau[0].expand()
tau_0


# to form a multinomial, a sum of products of trigonometric functions. The number of product terms is

# In[ ]:


len(tau_0.args)


# With the expression in this form we can find all the terms that contain $\ddot{q}_0$

# In[ ]:


m = tau[0].coeff(qdd[0]).args
m


# so we can write our torque expression as $\tau_0 = m \ddot{q}_0 + \cdots$ which means that $m$ must be element $\textbf{M}_{00}$ of the manipulator inertia matrix.  We see it has contributions due to the centre of mass of the arm which depends on the configuration of joints 1 to 5, and motor inertia. 
# 
# Similarly 

# In[ ]:


m = tau[0].coeff(qdd[1]).args
m


# must be the element $\textbf{M}_{01}$, an off-diagonal element of the inertia matrix.  In this way we can generate expressions for each element of the manipulator inertia matrix.  Remember that this matrix is symmetric so we only need to compute half of the off-diagnonal terms.
# 
# In a similar way we can find elements of the Coriolis and centripetal matrix.  Terms containing $\dot{q}_0$ are

# In[ ]:


c = tau[0].coeff(qd[0]).args
c


# so we can write our torque expression as $\tau_0 = c  \dot{q}_0 + \cdots$
# A subset of these terms still contain $\dot{q}_0$

# In[ ]:


C = tau[0].coeff(qd[0]).coeff(qd[0]).args


# ie. they originally contained $\dot{q}_0^2$.  The squared velocity terms belong on the diagonal of the Coriolis matrix and represent centripetal torques.  This particular expression is the element $\textbf{C}_{00}$.
# 
# The off-diagonal terms are coefficients of $\dot{q}_i \dot{q}_j$ and we can find $\dot{q}_0 \dot{q}_1$ by

# c = tau[0].coeff(qd[0]).coeff(qd[1]).args

# We add $C/2$ to both $C_{01}$ and $C_{10}$ and repeat for all the off-diagonal terms.  $C_{ij} = \Sum_{i=0}^n \Sum_{j=0}^n c
# 
# The elements of this matrix map velocities to force and therefore have the same dimensions as viscous friction.  In fact the motor friction parameter $B_m$ will be mixed up in these expressions, so we should first set $B_m = 0$ for all joints before proceeding.

# SymPy provides tools that allow us to pull apart the expression's parse tree.  The top node is given by `.func` which in this case is

# In[ ]:


tau_0.func


# indicating an addition node, and its arguments are

# In[ ]:


tau_0.args


# expressed as a tuple.  A term, say the first one, can be further decomposed

# In[ ]:


tau_0.args[0].func


# In[ ]:


tau_0.args[0].args


# which is we see here is a product of three terms.  There is a standard ordering of the arguments and it looks like constants are always first.

# We can see that many of the terms have very small coefficients, they literally don't add much to the result, so we can cull them out, or rather, we can select those with significant coefficients

# In[ ]:


signif_terms = [t for t in tau_0.args if abs(t.args[0]) > 1e-6]
signif_terms


# and there are now only

# In[ ]:


len(signif_terms)


# product terms remaining, just 25% of the number we started with.
# 
# We can reconstruct the expression with just the significant terms

# In[ ]:


tau_0s = tau_0.func(*signif_terms)
tau_0s


# and the top-level node function, `tau_0.func` (which is `Add`) acts as a constructor to which we pass the significant terms.
# 
# We can generate executable code in C to compute the torque on joint 0 which could be used in a feed-forward controller for example. 

# In[ ]:


print(ccode(tau_0s, assign_to="tau0"))


# We can do this for any of the expressions we generated above, ie. for elements of the inertia matrix or the velocity matrix.
# 
# The gravity vector is obtained when $\dot{q}_j =0$ and $\ddot{q}_j =0$

# To set these symbolic values to zero we define a substituion dictionary where a key is the symbols to be substituted and the value is its new (substituted) value.

# In[ ]:


subsdict = {}
sqdict = {}
for j in range(puma.n):
    subsdict[qd[j]] = 0
    subsdict[qdd[j]] = 0}
    sqdict[]


# In[ ]:


tau_g = tau.subs(subsdict)
tau_g


# In[ ]:


subsdict = {}
sqdict = {}
for j in range(puma.n):
    S = sym.symbol(f'S{j:d}')
    C = sym.symbol(f'C{j:d}')
    subsdict[sym.sin(q[j])] = S
    subsdict[sym.cos(q[j])] = C
    sqdict[S**2] = 1 - C**2
subsdict


# In[ ]:


sqdict


# In[ ]:


t1 = tau[1].subs(subsdict)


# In[ ]:


t1.expand().subs(sqdict)


# In[ ]:




