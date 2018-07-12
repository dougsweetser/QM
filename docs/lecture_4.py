
# coding: utf-8

# # Lecture 4: Time and Change

# 
# 
# Load the needed libraries.

# In[1]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport matplotlib.pyplot as plt\nimport math\n\n# To get equations the look like, well, equations, use the following.\nfrom sympy.interactive import printing\nprinting.init_printing(use_latex=True)\nfrom IPython.display import display\n\n# Tools for manipulating quaternions.\nimport Q_tools as qt;\n\nfrom IPython.core.display import display, HTML, Math, Latex\ndisplay(HTML("<style>.container { width:100% !important; }</style>"))')


# whatever

# ![](images/lecture_4/c4_p093_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p094_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p095_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p095_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p096_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p096_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p097_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p097_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p097_q3.jpg)

# whatever

# ![](images/lecture_4/c4_p099_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p099_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p100_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p100_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p100_q3.jpg)

# whatever

# ![](images/lecture_4/c4_p101_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p101_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p102_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p103_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p104_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p105_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p107_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p108_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p109_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p110_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p110_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p111_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p112_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p112_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p113_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p114_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p114_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p114_q3.jpg)

# whatever

# ![](images/lecture_4/c4_p115_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p115_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p115_q3.jpg)

# whatever

# ![](images/lecture_4/c4_p116_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p117_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p117_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p118_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p118_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p119_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p119_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p120_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p120_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p120_q3.jpg)

# whatever

# ![](images/lecture_4/c4_p121_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p121_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p121_q3.jpg)

# whatever

# ![](images/lecture_4/c4_p122_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p123_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p123_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p124_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p125_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p125_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p126_q1.jpg)
