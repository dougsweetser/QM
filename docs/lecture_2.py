
# coding: utf-8

# # Lecture 2

# In[1]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport matplotlib.pyplot as plt\n\n# To get equations the look like, well, equations, use the following.\nfrom sympy.interactive import printing\nprinting.init_printing(use_latex=True)\nfrom IPython.display import display\n\n# Tools for manipulating quaternions.\nimport Q_tools as qt;\n\nfrom IPython.core.display import display, HTML\ndisplay(HTML("<style>.container { width:100% !important; }</style>"))')


# ![](images/lecture_2/lecture_2_p_35_1.50.jpg)

# ![](images/lecture_2/lecture_2_p_36_1.50.jpg)

# ![](images/lecture_2/lecture_2_p_38_1.50.jpg)

# ![](images/lecture_2/lecture_2_p_38_2.50.jpg)

# ![](images/lecture_2/lecture_2_p_47_1.50.jpg)

# ![](images/lecture_2/lecture_2_p_39_1.50.jpg)

# ![](images/lecture_2/lecture_2_p_39_2.50.jpg)

# ![](images/lecture_2/lecture_2_p_39_3.50.jpg)

# ![](images/lecture_2/lecture_2_p_40_1.50.jpg)

# ![](images/lecture_2/lecture_2_p_40_2.50.jpg)

# ![](images/lecture_2/lecture_2_p_41_1.50.jpg)

# ![](images/lecture_2/lecture_2_p_41_2.50.jpg)

# ![](images/lecture_2/lecture_2_p_42_2.50.jpg)

# ![](images/lecture_2/lecture_2_p_42_3.50.jpg)

# ![](images/lecture_2/lecture_2_p_43_1.50.jpg)

# ![](images/lecture_2/lecture_2_p_43_2.50.jpg)

# ![](images/lecture_2/lecture_2_p_44_1.50.jpg)

# ![](images/lecture_2/lecture_2_p_44_2.50.jpg)

# ![](images/lecture_2/lecture_2_p_45_1.50.jpg)
