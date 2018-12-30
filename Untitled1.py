
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport matplotlib.pyplot as plt\nimport math\n\n# To get equations the look like, well, equations, use the following.\nfrom sympy.interactive import printing\nprinting.init_printing(use_latex=True)\nfrom IPython.display import display\n\n# Tools for manipulating quaternions.\nimport Q_tools as qt;\n\nfrom IPython.core.display import display, HTML, Math, Latex\ndisplay(HTML("<style>.container { width:100% !important; }</style>"))')


# Try to figure out the identity_op() function for QHStates.

# In[6]:


def identity_op(dim):
    """Identity operator for states."""
    
    q_1 = qt.QHStates([qt.QH().q_1()])
    return qt.QHStates.diagonal(q_1, dim)


# In[7]:


identity_op(3).print_state("op 3")


# In[9]:


At1, Ax1, Ay1, Az1 = sp.symbols("At1 Ax1 Ay1 Az1")
At2, Ax2, Ay2, Az2 = sp.symbols("At2 Ax2 Ay2 Az2")
Aq1 = qt.QH([At1, Ax1, Ay1, Az1], qtype="a₁")
Aq2 = qt.QH([At2, Ax2, Ay2, Az2], qtype="a₂")
A = qt.QHStates([Aq1, Aq2])
A.print_state("A", 1)


# In[11]:


i2 = identity_op(2)
i2A = A.Euclidean_product("ket", operator=i2)
i2A.print_state("i2A")

