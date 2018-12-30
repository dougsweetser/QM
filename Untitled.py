
# coding: utf-8

# # Inverse Quaternion Series

# In[29]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport math\nimport matplotlib.pyplot as plt\nimport unittest\n\n\n# To get equations the look like, well, equations, use the following.\nfrom sympy.interactive import printing\nprinting.init_printing(use_latex=True)\nfrom IPython.display import display\n\n# Tools for manipulating quaternions.\nimport Q_tools as qt;\n\nfrom IPython.core.display import display, HTML\ndisplay(HTML("<style>.container { width:100% !important; }</style>"))')


# Given eigenvalues and eigenvectors, figure out the matrix operator.

# In[30]:


def eigens_2_matrix(Eigen_values_N, Eigen_vectors_V):
    """
    V xR N V⁻¹ = M, must be fed quaternion series of the same dimensions.
    There is one issue that needs to be addressed with going to quaternions from 
    the real and complex numbers which commute. The reverse product must be used, so
        N|V> = M|V> = V xR N V⁻¹|V> = V xR N = N X V
    """
    
    if Eigen_values_N.dim != Eigen_values_N.dim:
        print("Oops, the dimensions of the diagonal Eigen_value series must be the same as the Eigen_vectors.")
        return
    
    Vinv = Eigen_vectors_V.inverse().normalize(0.5)
    Vinv.print_state("V inv", 1)
    
    Eigen_vectors_V.Euclidean_product("bra", ket=Vinv).print_state("VVinv")
    
    NVinv = Vinv.product("ket", operator=Eigen_values_N)
    
    VNVinv = NVinv.product("ket", operator=Eigen_vectors_V, reverse=True)
    
    return VNVinv


# In[31]:


q_0 = qt.QH([0, 0, 0, 0])
q_1 = qt.QH([1, 0, 0, 0])
q_2 = qt.QH([2, 0, 0, 0])
q_3 = qt.QH([3, 0, 0, 0])
q_n2 = qt.QH([-2, 0, 0, 0])
q_7 = qt.QH([7, 0, 0, 0])

v = qt.QHStates([q_1, q_1, q_2, q_3])
v1 = qt.QHStates([q_1])
v2 = qt.QHStates([q_2])
v3 = qt.QHStates([q_3])
n = qt.QHStates([q_n2, q_0, q_0, q_7])

m = eigens_2_matrix(n, v)
m.print_state("m?", 1)

mv = v.Euclidean_product("ket", operator=m)
nv = v.Euclidean_product("ket", operator=n)

mv1 = v1.Euclidean_product("ket", operator=m)
nv1 = v1.Euclidean_product("ket", operator=n)

mv2 = v2.Euclidean_product("ket", operator=m)
nv2 = v2.Euclidean_product("ket", operator=n)

mv3 = v3.Euclidean_product("ket", operator=m)
nv3 = v3.Euclidean_product("ket", operator=n)




mv.print_state("mv", 1)
nv.print_state("nv",1)

mv1.print_state("mv1", 1)
nv1.print_state("nv1",1)

mv2.print_state("mv2", 1)
nv2.print_state("nv2",1)

mv3.print_state("mv3", 1)
nv3.print_state("nv3")


# In[32]:


def normalize(self, n=1, states=None):
    """Normalize all states."""
        
    new_states = []
        
    zero_norm_count = 0
        
    for bra in self.qs:
        if bra.norm_squared().t == 0:
            zero_norm_count += 1
            new_states.append(QH().q_0())
        else:
            new_states.append(bra.normalize(n))
            print("bra {}, normalized to n {}: {}".format(bra, n, bra.normalize(n)))
        
    new_states_normalized = []
        
    non_zero_states = self.dim - zero_norm_count
        
    for new_state in new_states:
        new_states_normalized.append(new_state.product(qt.QH([1/non_zero_states, 0, 0, 0])))
            
    return qt.QHStates(new_states_normalized)


# In[33]:


def determinant(q):
    """Calculate the determinant of a 'square' quaternion series."""
    
    if q.dim == 1:
        q_det = q.qs[0]
        
    elif q.dim == 4:
        ad = q.qs[0].product(q.qs[3])
        bc = q.qs[1].product(q.qs[2])
        q_det = ad.dif(bc)  
        
    elif q.dim == 9:
        aei = q.qs[0].product(q.qs[4].product(q.qs[8]))
        bfg = q.qs[3].product(q.qs[7].product(q.qs[2]))
        cdh = q.qs[6].product(q.qs[1].product(q.qs[5]))
        ceg = q.qs[6].product(q.qs[4].product(q.qs[2]))
        bdi = q.qs[3].product(q.qs[1].product(q.qs[8]))
        afh = q.qs[0].product(q.qs[7].product(q.qs[5]))
        
        sum_pos = aei.add(bfg.add(cdh))
        sum_neg = ceg.add(bdi.add(afh))
        
        q_det = sum_pos.dif(sum_neg)
        
    else:
        print("Oops, don't know how to calculate the determinant of this one.")
        q_det = qt.QHStates([QH().q_0()])
        
    return q_det


# In[34]:


def inverse(q, operator=False, additive=False):
    """Find the additive or multiplicative inverse of a bra, ket or operator."""
    
    if (operator):
        if additive:
            q_inv = q.flip_signs()
    
        else:    
            if q.dim == 1:
                q_inv = qt.QHStates(q.qs[0].inverse())
        
            elif q.dim == 4:
                print("q is: ", q)
                det = determinant(q)
                detinv = det.inverse()
                print("detinv", detinv)
                q0 = q.qs[3].product(detinv)
                print("q0", q0)
                q1 = q.qs[1].flip_signs().product(detinv)
                print("q1", q1)
                q2 = q.qs[2].flip_signs().product(detinv)
                print("q2", q2)
                q3 = q.qs[0].product(detinv)
                print("q3", q3)
                q_inv = qt.QHStates([q0, q1, q2, q3])
    
            elif q.dim == 9:
                det = determinant(q)
                detinv = det.inverse()
        
                print("detinv", detinv)
                q0 = q.qs[4].product(q.qs[8]).dif(q.qs[5].product(q.qs[7])).product(detinv)
                print("q0", q0)
                q1 = q.qs[7].product(q.qs[2]).dif(q.qs[8].product(q.qs[1])).product(detinv)
                print("q1", q1)
                q2 = q.qs[1].product(q.qs[5]).dif(q.qs[2].product(q.qs[4])).product(detinv)
                print("q2", q2)
                q3 = q.qs[6].product(q.qs[5]).dif(q.qs[8].product(q.qs[3])).product(detinv)
                print("q3", q3)
                q4 = q.qs[0].product(q.qs[8]).dif(q.qs[2].product(q.qs[6])).product(detinv)
                print("q4", q4)
                q5 = q.qs[3].product(q.qs[2]).dif(q.qs[5].product(q.qs[0])).product(detinv)
                print("q5", q5)
                q6 = q.qs[3].product(q.qs[7]).dif(q.qs[4].product(q.qs[6])).product(detinv)
                print("q6", q6)
                q7 = q.qs[6].product(q.qs[1]).dif(q.qs[7].product(q.qs[0])).product(detinv)
                print("q7", q7)
                q8 = q.qs[0].product(q.qs[4]).dif(q.qs[1].product(q.qs[3])).product(detinv)
                print("q8", q8)
        
                q_inv = qt.QHStates([q0, q1, q2, q3, q4, q5, q6, q7, q8])
        
            else:
                print("Oops, don't know how to inverse.")
                q_inv = qt.QHStates([QH().q_0()])

    else:
        if additive:
            q_inv = q.flip_signs()
        
        else:
        
            new_states = []
        
            for bra in q.qs:
                new_states.append(bra.inverse())
                
            q_inv = qt.QHStates(new_states)
            
    return q_inv


# In[35]:


c1 = qt.QH([1, 2, 0, 0])
c2 = qt.QH([4, 2, 0, 0])
c3 = qt.QH([3, 3, 0, 0])
c4 = qt.QH([2, -5, 0, 0])
c5 = qt.QH([-2, .1, 0, 0])

c1_inv = c1.inverse()
c2_inv = c2.inverse()
c3_inv = c3.inverse()
c4_inv = c4.inverse()
c5_inv = c5.inverse()

c1_inv.print_state("c1_inv", 1)
c2_inv.print_state("c2_inv", 1)
c3_inv.print_state("c3_inv", 1)
c4_inv.print_state("c4_inv", 1)
c5_inv.print_state("c5_inv", 1)

cs_1 = qt.QHStates([c1, c2, c3, c4])
cs_2 = qt.QHStates([c5, c4, c2, c3])

cs_1.print_state("cs_1: ", 1)
cs_2.print_state("cs_2: ")


# In[43]:


cs_1_inverse = inverse(cs_1)
cs_2_inverse = inverse(cs_2)
cs_1_inverse.print_state("cs_1_inverse", 1)
cs_2_inverse.print_state("cs_2_inverse")


# In[37]:


cs_1_cs_1_inv = cs_1.product("bra", ket=cs_1_inverse)
cs_2_cs_2_inv = cs_2.product("bra", ket=cs_2_inverse)
cs_1_cs_1_inv.print_state("cs_1_cs_1_inv: ", 1)
cs_2_cs_2_inv.print_state("cs_2_cs_2_inv: ")


# In[44]:


cs_1_cs_1_inv = cs_1.Euclidean_product("bra", ket=cs_1_inverse)
cs_2_cs_2_inv = cs_2.Euclidean_product("bra", ket=cs_2_inverse)
cs_1_cs_1_inv.print_state("cs_1_cs_1_inv: ", 1)
cs_2_cs_2_inv.print_state("cs_2_cs_2_inv: ")


# In[45]:


cs_1_inverse_diagonal = cs_1_inverse.diagonal(4)
cs_2_inverse_diagonal = cs_2_inverse.diagonal(4)
cs_1_inverse_diagonal.print_state("cs_1_inverse_diagonal")
cs_2_inverse_diagonal.print_state("cs_2_inverse_diagonal")


# In[46]:


cs_1_cs_1_inv_op = cs_1.product("ket", operator=cs_1_inverse_diagonal)
cs_2_cs_2_inv_op = cs_2.product("ket", operator=cs_2_inverse_diagonal)
cs_1_cs_1_inv_op.print_state("cs_1_cs_1_inv_op", 1)
cs_2_cs_2_inv_op.print_state("cs_2_cs_2_inv_op")


# In[41]:


print(cs_1_inverse.dim)
print(cs_1.qs)
print(cs_1_inverse.qs)
cs_1_inverse.print_state("cs_1_inverse")
cs_1_inverse_diagonal = cs_1_inverse.diagonal(4)
cs_1_inverse_diagonal.print_state("cs_1 inv diagonal")


# In[39]:


cs_1_inv_series = cs_1.product("ket", operator=cs_1_inverse.diagonal(4))
cs_1_inv_series.print_state("cs_1 inv as series")


# In[40]:


print("cs_1_inverse dim: ", cs_1_inverse.dim)
print("cs_1_inverse len qs: ", len(cs_1_inverse.qs))
cs_1_inverse_diag = cs_1_inverse.diagonal(4)
cs_1_inverse_diag.print_state("cs_1_inverse_diag")


# In[14]:


v.print_state("v", 1)
print(determinant(v))
inverse(v).print_state("v inv")


# In[15]:


v.product("bra", ket=inverse(v)).print_state("v vinverse")


# In[16]:


print(determinant(v))


# In[17]:


v9 = qt.QHStates([q_1, q_1, q_2, q_3, q_1, q_1, q_2, q_3, q_2])
print(v9)
print(determinant(v9))


# In[ ]:


v9i = qt.QHStates([qt.QH([0,1,0,0]), qt.QH([0,2,0,0]), qt.QH([0,3,0,0]), qt.QH([0,4,0,0]), qt.QH([0,5,0,0]), qt.QH([0,6,0,0]), qt.QH([0,7,0,0]), qt.QH([0,8,0,0]), qt.QH([0,9,0,0])])


# In[ ]:


vv9 = v9.add(v9i)
vv9.print_state("vv9")
print(determinant(vv9))


# In[ ]:


v1123 = qt.QHStates([q_1, q_1, q_2, q_3])
dv1123 = determinant(v1123)
print(dv1123)


# In[ ]:


def identity(dim, operator=False):
        """Identity operator for states or operators which are diagonal."""
    
        if operator:
            q_1 = qt.QHStates([qt.QH().q_1()])
            ident = qt.QHStates.diagonal(q_1, dim)    
    
        else:
            i_list = [qt.QH().q_1() for i in range(dim)]
            ident = qt.QHStates(i_list)
            
        return ident


# In[ ]:


class TestDet(unittest.TestCase):
    q_0 = qt.QH().q_0()
    q_1 = qt.QH().q_1()
    q_n1 = qt.QH([-1,0,0,0])
    q_2 = qt.QH([2,0,0,0])
    q_n2 = qt.QH([-2,0,0,0])
    q_3 = qt.QH([3,0,0,0])
    q_n3 = qt.QH([-3,0,0,0])
    q_4 = qt.QH([4,0,0,0])
    q_n5 = qt.QH([-5,0,0,0])
    q_7 = qt.QH([7,0,0,0])
    q_8 = qt.QH([8,0,0,0])
    q_9 = qt.QH([9,0,0,0])
    q_n11 = qt.QH([-11,0,0,0])
    q_21 = qt.QH([21,0,0,0])
    q_n34 = qt.QH([-34,0,0,0])
    v3 = qt.QHStates([q_3])
    v1123 = qt.QHStates([q_1, q_1, q_2, q_3])
    v3n1n21 = qt.QHStates([q_3,q_n1,q_n2,q_1])
    v9 = qt.QHStates([q_1, q_1, q_2, q_3, q_1, q_1, q_2, q_3, q_2])
    v9i = qt.QHStates([qt.QH([0,1,0,0]), qt.QH([0,2,0,0]), qt.QH([0,3,0,0]), qt.QH([0,4,0,0]), qt.QH([0,5,0,0]), qt.QH([0,6,0,0]), qt.QH([0,7,0,0]), qt.QH([0,8,0,0]), qt.QH([0,9,0,0])])
    vv9 = v9.add(v9i)
    qn627 = qt.QH([-6,27,0,0])
    v33 = qt.QHStates([q_7, q_0, q_n3, q_2, q_3, q_4, q_1, q_n1, q_n2])
    v33inv = qt.QHStates([q_n2, q_3, q_9, q_8, q_n11, q_n34, q_n5, q_7, q_21])
    q_i3 = qt.QHStates([q_1, q_1, q_1])
    q_i2d = qt.QHStates([q_1, q_0, q_0, q_1])
            
    def test_identity(self):
        ident = identity(3)
        print("ket 3 identity", ident)
        self.assertTrue(ident.equals(self.q_i3))
        ident = identity(2, operator=True)
        print("operator 2 identity", ident)
        self.assertTrue(ident.equals(self.q_i2d))


    
    def test_determinant(self):
        det_v3 = determinant(self.v3)
        print("det v3:", det_v3)
        self.assertTrue(det_v3.equals(self.q_3))
        det_v1123 = determinant(self.v1123)
        print("det v1123", det_v1123)
        self.assertTrue(det_v1123.equals(self.q_1))
        det_v9 = determinant(self.v9)
        print("det_v9", det_v9)
        self.assertTrue(det_v9.equals(self.q_9))
        det_vv9 = determinant(self.vv9)
        print("det_vv9", det_vv9)
        self.assertTrue(det_vv9.equals(self.qn627))
        
    def test_inverse(self):
        inv_v1123 = inverse(self.v1123, operator=True)
        print("inv_v1123", inv_v1123)
        self.assertTrue(inv_v1123.equals(self.v3n1n21))

        inv_v33 = inverse(self.v33, operator=True)
        print("inv_v33", inv_v33)
        self.assertTrue(inv_v33.equals(self.v33inv))

        
suite = unittest.TestLoader().loadTestsFromModule(TestDet())
unittest.TextTestRunner().run(suite)


# In[ ]:


v2 = qt.QHStates([qt.QH([1,2,3,4]), qt.QH([2,3,2,1])])
v2inv = v2.inverse()
v2v2inv_product = v2.product("bra", ket=v2inv)
v2v2inv_Euclidean_product = v2.Euclidean_product("bra", ket=v2inv)

v2.print_state("v2", 1)
v2inv.print_state("v2inv", 1)
v2v2inv_product.print_state("v2v2inv_product")
v2v2inv_Euclidean_product.print_state("v2v2inv_Euclidean_product")


# In[ ]:


vinverse = v.inverse()
vvinverse_product = v.product("bra", ket=vinverse)
vvinverse_Euclidean_product = v.Euclidean_product("bra", ket=vinverse)

v.print_state("v", 1)
vinverse.print_state("vinverse", 1)
vvinverse_product.print_state("vvinverse_product")
vvinverse_Euclidean_product.print_state("vvinverse_Euclidean_product")


# In[ ]:


u2 = qt.QHStates([qt.QH([1,2,3,4]), qt.QH([3,2,3,1])])
u4 = qt.QHStates([qt.QH([1,2,3,4]), qt.QH([3,2,3,1]), qt.QH([2,2,1,0]), qt.QH([3,-1,-3,2])])
I2 = qt.QHStates([qt.QH().q_1(), qt.QH().q_1()])
I4 = qt.QHStates([qt.QH().q_1(), qt.QH().q_1(), qt.QH().q_1(), qt.QH().q_1()])
u2.print_state("u2", 1)
u2.product("bra", ket=I2).print_state("u2 I2 product",1)
u2.Euclidean_product("ket", bra=I2).print_state("u2* I2 Euclidean product",1)
print("")
u4.print_state("u4", 1)
u4.product("bra", ket=I4).print_state("u4 I4 product",1)
u4.Euclidean_product("ket", bra=I4).print_state("u4* I4 Euclidean product")


# In[ ]:


v.print_state("v")
v.inverse().print_state("v inverse")
v.inverse(operator=True).print_state("v inverse")


# Identity bra, ket or operator(diagonal).

# In[ ]:


identity(3).print_state("i3")
identity(3, operator=True).print_state("i3 op")


# In[ ]:


def inverse(self, operator=False, additive=False):
    """Inverseing bras and kets calls inverse() once for each.
    Inverseing operators is more tricky as one needs a diagonal identity matrix."""
    
    if (operator):
        if additive:
            q_inv = self.flip_signs()
    
        else:    
            if self.dim == 1:
                q_inv = QHStates(self.qs[0].inverse())
        
            elif self.dim == 4:
                det = determinant(q)
                detinv = det.inverse()
                
                q0 = self.qs[3].product(detinv)
                q1 = self.qs[1].flip_signs().product(detinv)
                q2 = self.qs[2].flip_signs().product(detinv)
                q3 = self.qs[0].product(detinv)
                
                q_inv = QHStates([q0, q1, q2, q3])
    
            elif self.dim == 9:
                det = determinant(q)
                detinv = det.inverse()
        
                
                q0 = self.qs[4].product(self.qs[8]).dif(self.qs[5].product(self.qs[7])).product(detinv)
                q1 = self.qs[7].product(self.qs[2]).dif(self.qs[8].product(self.qs[1])).product(detinv)
                q2 = self.qs[1].product(self.qs[5]).dif(self.qs[2].product(self.qs[4])).product(detinv)
                q3 = self.qs[6].product(self.qs[5]).dif(self.qs[8].product(self.qs[3])).product(detinv)
                q4 = self.qs[0].product(self.qs[8]).dif(self.qs[2].product(self.qs[6])).product(detinv)
                q5 = self.qs[3].product(self.qs[2]).dif(self.qs[5].product(self.qs[0])).product(detinv)
                q6 = self.qs[3].product(self.qs[7]).dif(self.qs[4].product(self.qs[6])).product(detinv)
                q7 = self.qs[6].product(self.qs[1]).dif(self.qs[7].product(self.qs[0])).product(detinv)
                q8 = self.qs[0].product(self.qs[4]).dif(self.qs[1].product(self.qs[3])).product(detinv)
                
                q_inv = QHStates([q0, q1, q2, q3, q4, q5, q6, q7, q8])
        
            else:
                print("Oops, don't know yet how to inverse an operator of this size, sorry.")
                q_inv = QHStates([QH().q_0()])

    else:
        if additive:
            q_inv = self.flip_signs()
        
        else:
            new_states = []
        
            for bra in self.qs:
                new_states.append(bra.inverse())
                
            q_inv = QHStates(new_states)
            
    return q_inv


# In[ ]:


def determinant(self):
    """Calculate the determinant of a 'square' quaternion series."""
    
    if self.dim == 1:
        q_det = self.qs[0]
        
    elif self.dim == 4:
        ad = self.qs[0].product(self.qs[3])
        bc = self.qs[1].product(self.qs[2])
        q_det = ad.dif(bc)  
        
    elif self.dim == 9:
        aei = self.qs[0].product(self.qs[4].product(self.qs[8]))
        bfg = self.qs[3].product(self.qs[7].product(self.qs[2]))
        cdh = self.qs[6].product(self.qs[1].product(self.qs[5]))
        ceg = self.qs[6].product(self.qs[4].product(self.qs[2]))
        bdi = self.qs[3].product(self.qs[1].product(self.qs[8]))
        afh = self.qs[0].product(self.qs[7].product(self.qs[5]))
        
        sum_pos = aei.add(bfg.add(cdh))
        sum_neg = ceg.add(bdi.add(afh))
        
        q_det = sum_pos.dif(sum_neg)
        
    else:
        print("Oops, don't know how to calculate the determinant of this one.")
        q_det = QHStates([QH().q_0()])
        
    return q_det

