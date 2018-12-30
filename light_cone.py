
# coding: utf-8

# # Drawing light Cones and Constant Space-times-tim

# Load the needed libraries.

# In[1]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport matplotlib.pyplot as plt\nimport math\n\n# To get equations the look like, well, equations, use the following.\nfrom sympy.interactive import printing\nprinting.init_printing(use_latex=True)\nfrom IPython.display import display\n\n# Tools for manipulating quaternions.\nimport Q_tools as qt;\n\nfrom IPython.core.display import display, HTML, Math, Latex\ndisplay(HTML("<style>.container { width:100% !important; }</style>"))')


# In[2]:


# The numbers.
x4, y4 = np.linspace(-4, 4, 100), np.linspace(-4, 4, 100)
x, y = np.meshgrid(x4, y4)

# The plot.
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x4, y4, ls='--')
ax1.plot(x4, -y4, ls='--')
ax1.contour(x, y, 2 + x**2 - y**2, [0])
ax1.contour(x, y, -2 + x**2 - y**2, [0])
plt.xlabel('space')
plt.ylabel('time')
ax1.axis('square')
plt.show()


# In[3]:


# The numbers.
x4, y4 = np.linspace(-4, 4, 100), np.linspace(-4, 4, 100)
x, y = np.meshgrid(x4, y4)

# The plot.
fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.axvline(0, ls='--')
ax2.axhline(0, ls='--')
ax2.contour(x, y, 2 + x * y, [0])
ax2.contour(x, y, -2 + x * y, [0])
ax2.axis('square')
plt.xlabel('space')
plt.ylabel('time')
plt.show()


# In[7]:


# The numbers.
x4, y4 = np.linspace(-4, 4, 100), np.linspace(-4, 4, 100)
x, y = np.meshgrid(x4, y4)

# The plot.
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(x4, y4, ls='--')
ax1.plot(x4, -y4, ls='--')
ax1.contour(x, y, 2 + x**2 - y**2, [0])
ax1.contour(x, y, -2 + x**2 - y**2, [0])
ax1.set_title("Constant\nintervals")
ax1.axis('square')
ax1.set_xticklabels('')
ax1.set_yticklabels('')
plt.xlabel('dR')
plt.ylabel('dt')

ax2 = fig.add_subplot(122)
ax2.axvline(0, ls='--')
ax2.axhline(0, ls='--')
ax2.contour(x, y, 2 + x * y, [0])
ax2.contour(x, y, -2 + x * y, [0])
ax2.axis('square')
ax2.set_title("Constant\nspace-times-time")
ax2.axis('square')
ax2.set_xlabel('dR')
ax2.set_ylabel('dt')
ax2.set_xticklabels('')
ax2.set_yticklabels('')

plt.subplots_adjust(wspace=.6)
plt.savefig('constant_intervals_space_times_time.png')
plt.show()


# In[32]:


# The numbers.
x4, y4 = np.linspace(0, 4, 100), np.linspace(0, 4, 100)
x, y = np.meshgrid(x4, y4)

x2, y2 = np.linspace(0, 2, 100), np.linspace(0, 2, 100)
x2, y2 = np.meshgrid(x2, y2)

# The plot.
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x4, 2 - y4, ls='--')
ax1.plot(x4, 4 - y4, ls='--')
ax1.set_ylim(0, 4)
ax1.axis('square')
plt.show()


# In[28]:


# The numbers.
x4, y4 = np.linspace(0, 4, 100), np.linspace(0, 4, 100)
x, y = np.meshgrid(x4, y4)

x2, y2 = np.linspace(0, 2, 100), np.linspace(0, 2, 100)
x2, y2 = np.meshgrid(x2, y2)

# The plot.
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x4, 2 - y4, ls='--', label="Measure a Length")
ax1.plot(x4, 4 - y4, ls='--')
ax1.axhline(xmin=0.5, linewidth=8)

ax1.set_xticks([0, 1, 2, 3, 4])
ax1.set_yticks([0, 1, 2, 3, 4])
ax1.spines['top'].set_color(None)
ax1.set_xticklabels('')
ax1.set_yticklabels('')
ax1.set_title("Measure the length of a bar")
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')
ax1.set_xlabel('R')
ax1.set_ylabel('t')
ax1.axis('square')
ax1.set_xlim(0, 4)
ax1.set_ylim(0, 4)
plt.savefig('measure_length.png')
plt.show()

