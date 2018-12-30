#!/usr/bin/env python
# coding: utf-8

# # Does (AM)B Really equal A(MB)?

# Import the needed libraries.

# In[1]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport matplotlib.pyplot as plt\n\n# To get equations the look like, well, equations, use the following.\nfrom sympy.interactive import printing\nprinting.init_printing(use_latex=True)\nfrom IPython.display import display\n\n# Tools for manipulating quaternions.\nimport Q_tools as qt;\n\nfrom IPython.core.display import display, HTML\ndisplay(HTML("<style>.container { width:100% !important; }</style>"))')


# Make it all abstract. Most of these were in the Lecture 3 notebook.

# In[3]:


At1, Ax1, Ay1, Az1 = sp.symbols("At1 Ax1 Ay1 Az1")
At2, Ax2, Ay2, Az2 = sp.symbols("At2 Ax2 Ay2 Az2")
Aq1 = qt.QH([At1, Ax1, Ay1, Az1], qtype="a₁")
Aq2 = qt.QH([At2, Ax2, Ay2, Az2], qtype="a₂")
A = qt.QHStates([Aq1, Aq2])
A.print_state("A", 1)

Bt1, Bx1, By1, Bz1 = sp.symbols("Bt1 Bx1 Ay1 Bz1")
Bt2, Bx2, By2, Bz2 = sp.symbols("Bt2 Bx2 By2 Bz2")
Bq1 = qt.QH([Bt1, Bx1, By1, Bz1], qtype="b₁")
Bq2 = qt.QH([Bt2, Bx2, By2, Bz2], qtype="b₂")
B = qt.QHStates([Bq1, Bq2])
B.print_state("B", 1)

Mt1, Mx1, My1, Mz1 = sp.symbols("Mt1 Mx1 My1 Mz1")
Mt2, Mx2, My2, Mz2 = sp.symbols("Mt2 Mx2 My2 Mz2")
Mt3, Mx3, My3, Mz3 = sp.symbols("Mt3 Mx3 My3 Mz3")
Mt4, Mx4, My4, Mz4 = sp.symbols("Mt4 Mx4 My4 Mz4")
Mq1 = qt.QH([Mt1, Mx1, My1, Mz1], qtype="m₁")
Mq2 = qt.QH([Mt2, Mx2, My2, Mz2], qtype="m₂")
Mq3 = qt.QH([Mt3, Mx3, My3, Mz3], qtype="m₃")
Mq4 = qt.QH([Mt4, Mx4, My4, Mz4], qtype="m₄")

M = qt.QHStates([Mq1, Mq2, Mq3, Mq4])
M.print_state("M", 1)


# In[10]:


AMthenB = A.Euclidean_product("bra", operator=M).Euclidean_product("bra", ket=B)
MBthenA = B.Euclidean_product("ket", operator=M).Euclidean_product("ket", bra=A)
AMthenB.print_state("AMthenB", 1)
MBthenA.print_state("MBthenA")


# In[12]:


At1, Ax1, Ay1, Az1 = sp.symbols("At1 Ax1 Ay1 Az1")
At2, Ax2, Ay2, Az2 = sp.symbols("At2 Ax2 Ay2 Az2")
Aq1 = qt.QH([1,2, 3, 4], qtype="a₁")
Aq2 = qt.QH([1, 2, 1, 2], qtype="a₂")
A = qt.QHStates([Aq1, Aq2])
A.print_state("A", 1)

Bt1, Bx1, By1, Bz1 = sp.symbols("Bt1 Bx1 Ay1 Bz1")
Bt2, Bx2, By2, Bz2 = sp.symbols("Bt2 Bx2 By2 Bz2")
Bq1 = qt.QH([3, 2, 1, 0], qtype="b₁")
Bq2 = qt.QH([1,2,-1,-2], qtype="b₂")
B = qt.QHStates([Bq1, Bq2])
B.print_state("B", 1)

Mt1, Mx1, My1, Mz1 = sp.symbols("Mt1 Mx1 My1 Mz1")
Mt2, Mx2, My2, Mz2 = sp.symbols("Mt2 Mx2 My2 Mz2")
Mt3, Mx3, My3, Mz3 = sp.symbols("Mt3 Mx3 My3 Mz3")
Mt4, Mx4, My4, Mz4 = sp.symbols("Mt4 Mx4 My4 Mz4")
Mq1 = qt.QH([3,7,3,5], qtype="m₁")
Mq2 = qt.QH([3, 2, 2, -1], qtype="m₂")
Mq3 = qt.QH([3,2,1,1], qtype="m₃")
Mq4 = qt.QH([2, 1, 1, 4], qtype="m₄")

M = qt.QHStates([Mq1, Mq2, Mq3, Mq4])
M.print_state("M", 1)

AMdaggerthenB = A.Euclidean_product("bra", operator=M.dagger()).Euclidean_product("bra", ket=B)
MBthenA = B.Euclidean_product("ket", operator=M).Euclidean_product("ket", bra=A)
AMdaggerthenB.print_state("AMdaggerthenB", 1)
MBthenA.print_state("MBthenA")


# FAIL.

# In[ ]:




