#!/usr/bin/env python
# coding: utf-8

# # Calculating probabilities for "Why are amplitudes complex?"

# Doug Sweetser <sweetser@alum.mit.edu>

# Scott Aaronson in his blog titled ["Why are amplitudes complex?"](https://www.scottaaronson.com/blog/?p=4021) set up a simple system to explore with quaternion series quantum mechanics. He came to the conclusion that superluminal travel was possible. This goes against so much experimental data, any further work with such a setup is not worthy of any more time.
# 
# Let me repeat the calculation with my set of tools. Quaternions are a normed division algebra and I use the class QH to do all those operations. Quaternion series is a semi-group with inverses, and the class QHStates are used those.
# 
# Load the needed libraries.

# In[1]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport matplotlib.pyplot as plt\nimport math\n\n# To get equations the look like, well, equations, use the following.\nfrom sympy.interactive import printing\nprinting.init_printing(use_latex=True)\nfrom IPython.display import display\n\n# Tools for manipulating quaternions.\nimport Q_tools as qt\nfrom IPython.core.display import display, HTML, Math, Latex\ndisplay(HTML("<style>.container { width:100% !important; }</style>"))')


# Only a few quaternions will be used, create some convenient abbreviations.

# In[2]:


q_0, q_1, q_i, q_j, q_k = qt.QH().q_0(), qt.QH().q_1(), qt.QH().q_i(), qt.QH().q_j(), qt.QH().q_k()
q_half = qt.QHStates([qt.QH([1/2,0,0,0])])
q_sqrt_half = qt.QHStates([qt.QH([sp.sqrt(1/2),0,0,0])])


# Alice and Bob's state vectors are |1> and |+>. Each can be written in whatever basis one choses. For Alice, let's use the easy one. 

# In[3]:


u = qt.QHStates([q_1, q_0])
d = qt.QHStates([q_0, q_1])

u.print_state("|u>")
d.print_state("|d>")


# Form the ket |+> for Bob and confirm it is orthonormal:

# In[4]:


udp = u.add(d)
udn = u.dif(d)
plus = q_sqrt_half.product(udp).ket()
minus = q_sqrt_half.product(udn).ket()

plus.print_state("|+>", quiet=True)
minus.print_state("|->", quiet=True)
plus.norm_squared().print_state("<+|+>", quiet=True)
minus.norm_squared().print_state("<-|->", quiet=True)
plus.bra().Euclidean_product(minus).print_state("<+|->", quiet=True)


# Define the operators used in the blog.

# In[5]:


U = qt.QHStates([q_1, q_0, q_0, q_j]).op()
U.print_state("U", quiet=True)

V = qt.QHStates([q_1, q_0, q_0, q_i]).op()
V.print_state("V", quiet=True)


# Calculate $<1|U|1>$, $<+|V|+>$, and their product, $<1|U|1><+|V|+>$

# In[6]:


bracket = qt.QHStates().bracket

one_U = bracket(u.bra(), U, u)
plus_V = bracket(plus.bra(), V, plus)
one_U_plus_V = one_U.product(plus_V)

one_U.print_state("<1|U|1>")
plus_V.print_state("<+|V|+>")
one_U_plus_V.print_state("<1|U|1><+|V|+>")


# Good news: the operator U is a Hermitian operator for the ket $|1>$ as indicated by the real value of of $<1|U|1>$. Notice how the imaginary $j$ of U is not used for this representation of $|1>$. This also means any old imaginary 3-vector could be used in the operator U - up to a normalization factor. Sounds like cheating (it is not though).
# 
# Bad news: The assertion that V is a Hermitian operator for the ket |+> is not true since the value calculated for $<+|V|+>$ was not real. This doesn't have anything to do with any value be quaternions or not. The operator V cannot be an observable for the state $|+>$. Bummer.
# 
# Can we ask a better question? If one starts with the state $|0>$, might there be an operator $V$ that would use an imaginary and still generate a real number? Without any more thought, do the calculation...

# In[7]:


d_V = bracket(d.bra(), V, d)
d_V.print_state("<0|V|0>")


# Nice, only off by a factor of -i, simple to adjust.

# In[16]:


Vi = V.product(qt.QHStates([qt.QH().q_i(-1)])).op()
Vi.print_state("Vi")

d_Vi = bracket(d.bra(), Vi, d)
d_Vi.print_state("<0|Vi|0>")


# Now we can ask about odds of making an observation given Alice with a state $|1>$ and Bob with a state $|0>$ using the operators $U$ and $V$:

# In[9]:


one_U.product(d_Vi).print_state("<1|U|1><0|Vi|0>")


# Why is this certain to happen? There is not superposition of states and all the operators can do is play with the phase. 
# 
# What does the suggested rotation operator do to this calculation? I have chosen to rotate the operators $U$ and $Vi$.

# In[10]:


jk = qt.QHStates([q_j, q_k, q_k, q_j], qs_type="op")

jk.print_state("jk operator")


# Again, the normalization posted in the orginal blog does not look correct to me. Not a big deal, but a detail.
# 
# Form the product of this rotation matrix with $U$ and $Vi$, again needing a normalization tweak.

# In[11]:


jkU = jk.product(U).op()
jkVi = jk.product(Vi).op()

jkU.print_state("jk U")
jkVi.print_state("jk Vi")


# Put these rotated operators to work. See if they generate real values as they must to be Hermitian operators.

# In[12]:


bracket(u.bra(), jkU, u).print_state("<1|jkU|1>")
bracket(d.bra(), jkVi, d).print_state("<0|jkVi|0>")


# Both miss the mark by a factor of $-j$.

# In[13]:


jjkU = qt.QHStates([qt.QH().q_j(-1)]).product(jkU).op()
jjkU.print_state("jjkU")
one_jjkU = bracket(u.bra(), jjkU, u)
one_jjkU.print_state("<1|jjkU|1>")


# $ jjkU = \frac{1}{2}\begin{bmatrix}
# 1 & -i \\
# -k & j 
# \end{bmatrix}  $

# The operator $U$ was rotated, but that did not change the odds of seeing this state. There will be no way for Alice to send Bob a signal using a rotation.
# 
# Complete the story with the other operator, $Vi$.

# In[14]:


jjkVi = qt.QHStates([qt.QH().q_j(-1)]).product(jkVi).op()
jjkVi.print_state("jjkVi")
zero_jjkVi = bracket(d.bra(), jjkVi, d)
zero_jjkVi.print_state("<0|jjkVi|0>")


# $ \frac{1}{2}\begin{bmatrix}
# -i & -1 \\
# -i & 1 
# \end{bmatrix}  $

# In[15]:


one_jjkU.product(zero_jjkVi).print_state("<1|jjkU|1><0|jjkVi|0>")


# The odds are not altered by the rotation.

# In[ ]:




