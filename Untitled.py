
# coding: utf-8

# In[99]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport matplotlib.pyplot as plt\nimport math\n\n# To get equations the look like, well, equations, use the following.\nfrom sympy.interactive import printing\nprinting.init_printing(use_latex=True)\nfrom IPython.display import display\n\n# Tools for manipulating quaternions.\nimport Q_tools as qt;\n\nfrom IPython.core.display import display, HTML, Math, Latex\ndisplay(HTML("<style>.container { width:100% !important; }</style>"))')


# In[118]:


def transpose(self, m=None, n=None):
        """Transposes a series."""
        
        if m is None:
            # test if it is square.
            if math.sqrt(self.dim).is_integer():
                m = sp.sqrt(self.dim)
                n = m
               
        if n is None:
            n = self.dim / m
            
        if m * n != self.dim:
            return None
        
        matrix = [[0 for x in range(m)] for y in range(n)] 
        
        qs = self.qs
        qs_t = []
        
        for mi in range(m):
            for ni in range(n):
                matrix[ni][mi] = qs[n * mi + ni]
                print("mi, ni, n, n * mi + ni: {}, {}, {}, {}".format(mi, ni, n, n * mi + ni))
        
        qs_t = []
        
        for t in matrix:
            for q in t:
                qs_t.append(q)
                
        return(qt.QHStates(qs_t))


# In[119]:


Op = qt.QHStates([qt.QH([0,0,0,0]),qt.QH([0,1,0,0]),qt.QH([0,0,2,0]),qt.QH([0,0,0,3]),qt.QH([4,0,0,0]),qt.
                QH([0,5,0,0])])

transpose(Op, 2,3).print_states("3,2")

Op = qt.QHStates([qt.QH([0,0,0,0]),qt.QH([0,1,0,0]),qt.QH([0,0,2,0]),qt.QH([0,0,0,3]),qt.QH([4,0,0,0]),qt.
                QH([0,5,0,0])])

transpose(Op, 3,2).print_states("3,2")

Op = qt.QHStates([qt.QH([0,0,0,0]),qt.QH([0,1,0,0]),qt.QH([0,0,2,0]),qt.QH([0,0,0,3])])

transpose(Op).print_states("2x2")


# In[107]:


transpose(Op,3,2)


# In[108]:


def transpose(self, m=None, n=None):
        """Transposes a series."""
        
        if m is None:
            # test if it is square.
            if math.sqrt(self.dim).is_integer():
                m = sp.sqrt(self.dim)
                n = m
               
        if n is None:
            n = self.dim / m
            
        if m * n != self.dim:
            return None
        
        matrix = [[0 for x in range(m)] for y in range(n)] 
        
        qs = self.qs
        qs_t = []
        
        for mi in range(m):
            for ni in range(n):
                matrix[ni][mi] = qs[mi  * ni + ni]
        
        qs_t = []
        
        for t in matrix:
            for q in t:
                qs_t.append(q)
                
        return QHStates(qs_t)

