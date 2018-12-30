
# coding: utf-8

# # Lecture N:

# Load the needed libraries.

# In[2]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport matplotlib.pyplot as plt\nimport math\nimport copy\nimport pprint\n\n# To get equations the look like, well, equations, use the following.\nfrom sympy.interactive import printing\nprinting.init_printing(use_latex=True)\nfrom IPython.display import display\n\n# Tools for manipulating quaternions.\nimport Q_tools as qt;\n\nfrom IPython.core.display import display, HTML, Math, Latex\ndisplay(HTML("<style>.container { width:100% !important; }</style>"))')


# In[3]:


get_ipython().system('pwd')


# In[28]:


def product(self, product_type, bra=None, ket=None, operator=None, kind="", reverse=False):
    """Forms the quaternion product for each state."""
        
    if product_type == 'bra':
        bra = self
    elif product_type == 'ket':
        ket = self
    elif product_type == 'operator':
        if operator is None:
            operator = self
    else:
        print("Oops, need to set product_type to bra, ket, or operator.")
        return None
        
    def _is_square(n):
        return n**0.5 == int(n**0.5)
    
    def _check_dimensions(op_dim=0, state_1_dim=0, state_2_dim=0, equals=False):
        """Make sure the states and operators are the right sizes. The operator dimension is either
               equal to 1 or the product of the bra and ket dimensions."""

        oops = ''
            
        if equals:
            if state_1_dim != state_2_dim:
                oops = "states have different dimensions: {} != {}".format(state_1_dim, state_2_dim)
                    
        elif state_2_dim == 0:
            if (op_dim % state_1_dim != 0) and (op_dim != 1):
                oops = "Operator dimensions don't divide nicely by the state vector: {} % {}".format(
                    op_dim, state_1_dim)
                    
        else:
            if (op_dim != state_1_dim * state_2_dim) and (op_dim == 1 and (state_1_dim != state_2_dim)):
                oops = "Operator dimensions do not equal the product of the states: {} != {} * {}".format(
                   op_dim, state_1_dim, state_2_dim)
                    
        if oops:
            print(oops)
            return False
            
        else:
            return True
        
    new_states = []
    dot_product_flag = False
        
    if bra is None and operator is None:
        return ket
        
    elif ket is None and operator is None:
        return bra
        
    # Op Op
    elif bra is None and ket is None and operator:

        # For now, operator multiplication works only on square matrices of equal size.
        # It should be possible to multiply quaternion series whose dimensions share
        # a common factor, but implimenting that would require more effort than I want to do now.        print("Op op")
        
        if self.dim == operator.dim:
            
            if _is_square(self.dim):
                
                shared_size = int(self.dim**0.5)
                outer_size_1 = shared_size
                outer_size_2 = shared_size
            else:
                shared_size = 1
                outer_size_1 = self.dim
                outer_size_2 = operator.dim
        
        else:
            print("Oops, only can dea with square operators for now.")
            return None
            
        print("shared_size", shared_size)
        print("outer_size_1", outer_size_1)
        print("outer_size_2", outer_size_2)
        # op_chunk_1 = [self.qs[x:x+outer_size_1] for x in range(0, self.dim, outer_size_1)]
        op_chunk_1 = [self.qs[x:x+shared_size] for x in range(0, self.dim)]
        op_chunk_2 = [operator.qs[x:x+shared_size] for x in range(0, operator.dim)]
        
        for op_c_1 in op_chunk_1:
            print("a 1 chunk")
            
            for op_c in op_c_1:
                print("q: ", op_c)
                
        for op_c_2 in op_chunk_2:
            print("a 2 chunk")
            
            for op_c in op_c_2:
                print("q: ", op_c)
                
        # Put in zeros.
        result = [[qt.QH().q_0()] * outer_size_1] * outer_size_2
        
        print("i max: ", len(op_chunk_1))
        print("j max: ", len(op_chunk_2[0]))
        print("k max: ", len(op_chunk_2))

                
        for i in range(len(op_chunk_1)):
   
            # iterate through columns of Y
            for j in range(len(op_chunk_2[0])):
                   
                # iterate through rows of Y
                for k in range(len(op_chunk_2)):
                            
                    result[i][j] = result[i][j].add(op_chunk_1[i][k].product(op_chunk_2[k][j]))
        
        new_states = [item for sublist in result for item in sublist]
                

    # <A|B>                                                     
    elif operator is None:
        if _check_dimensions(state_1_dim=bra.dim, state_2_dim=ket.dim, equals=True):
            dot_product_flag = True
                
            for b, k in zip(bra.qs, ket.qs):
                new_states.append(b.product(k, kind, reverse))
            
    # Op|B>
    elif bra is None:
        if _check_dimensions(op_dim=operator.dim, state_1_dim=ket.dim):
            if operator.dim == 1:
                one_diagonal = operator.diagonal(ket.dim)                    
                opb = one_diagonal.qs

            else:
                opb = operator.qs
                        
            for ops in zip(*[iter(opb)] * ket.dim):
                ok = None
                    
                for op, k in zip(ops, ket.qs): 
                    if ok is None:
                        ok = op.product(k, kind, reverse)
                    else:
                        ok = ok.add(op.product(k, kind, reverse))
                            
                new_states.append(ok)

    # <A|Op
    elif ket is None:
        if _check_dimensions(op_dim=operator.dim, state_1_dim=bra.dim):
            # Operator needs to be transposed.
            opt = operator.transpose(bra.dim)
                
            if operator.dim == 1:
                one_diagonal = operator.diagonal(bra.dim)                    
                aop = one_diagonal.qs

            else:
                aop = opt.qs
                        
            for ops in zip(*[iter(aop)]*bra.dim):
                bop = None
                    
                for b, op in zip(bra.qs, ops):
                    if bop is None:
                        bop = b.product(op, kind, reverse)
                    else:
                        bop = bop.add(b.product(op, kind, reverse))
                            
                new_states.append(bop)

    # <A|Op|B>
    else:
        if _check_dimensions(op_dim=operator.dim, state_1_dim=bra.dim, state_2_dim=ket.dim):
            dot_product_flag = True
            new_ket = []
                
            if operator.dim == 1:
                one_diagonal = operator.diagonal(ket.dim)                    
                opb = one_diagonal.qs

            else:
                opb = operator.qs                                             
                                                             
            for ops in zip(*[iter(opb)]*ket.dim):
                ok = None
                    
                for op, k in zip(ops, ket.qs): 
                    if ok is None:
                        ok = op.product(k, kind, reverse)
                    else:
                        ok = ok.add(op.product(k, kind, reverse))
                    
            new_ket.append(ok)   
                
            new_ket_state = qt.QHStates(new_ket)
                    
            for b, k in zip(bra.qs, new_ket_state.qs):
                new_states.append(b.product(k, kind, reverse))
              
    # Return either the dot product or a new quaternion series.
    if dot_product_flag:
        dot_product = new_states.pop(0)
                
        for new_state in new_states:
            dot_product = dot_product.add(new_state)
                
        return dot_product
        
    else:
        return qt.QHStates(new_states)


# In[29]:


q123 = qt.QHStates([qt.QH([1, 0, 0, 0]),qt.QH([2, 0, 0, 0]),qt.QH([3, 0, 0, 0])])
q456 = qt.QHStates([qt.QH([4, 0, 0, 0]),qt.QH([5, 0, 0, 0]),qt.QH([6, 0, 0, 0])])

q_prod = product(q123, "operator", operator=q456)
    
q_prod.print_state("q123 q456 9", quiet=True)


# In[14]:


q3_orig = []

for i in range(1, 4):
    q3_orig.append(qt.QHStates([qt.QH([i, -i, i+2, i**2]), qt.QH([-i, 2 * i, i-1, i])]))

for q in q3_orig:
    q.print_state("q", 1, 1)
    
q3 = copy.deepcopy(q3_orig)


# In[15]:


q3 = copy.deepcopy(q3_orig)

q_prod = q3.pop()

for q in reversed(q3):
    q_prod = product(q_prod, "ket", operator=q.diagonal(2))
    
q_prod.print_state("Product of two operators and a ket is a quaternion series, q1 q2|q3>", quiet=True)


# In[6]:


q3 = copy.deepcopy(q3_orig)
q0d = q3[0].diagonal(2)
q0d.print_state("q0d")
q1d = q3[1].diagonal(2)
q1d.print_state("q1d")
op01 = product(q0d, "operator", operator=q1d)
op01.print_state("operator 0 * 1")


# In[7]:


print(qt.QH([1,-1,3,1]).product(qt.QH([2,-2,4,4])))


# In[8]:


print(qt.QH([1,-1,3,1]).product(qt.QH([2,-2,4,4])))


# In[9]:


op1 = product(qt.QHStates([qt.QH([1,-1,3,1])]), "operator", operator=qt.QHStates([qt.QH([2,-2,4,4])]))
op1.print_state("operator one state")


# In[10]:


op1 = product(qt.QHStates([qt.QH([1,-1,3,1]), qt.QH([2,0,0,0])]), "operator", operator=qt.QHStates([qt.QH([2,-2,4,4])]))
op1.print_state("operator 2x1 one state")


# In[ ]:


math.gcd(3, 3)


# In[ ]:


def lcm(a, b):
    return (a * b) // math.gcd(a, b)

lcm(20, 8)


# In[ ]:


op1 = product(qt.QHStates([qt.QH([1,-1,3,1]), qt.QH([2,0,2,3])]), "operator", operator=qt.QHStates([qt.QH([1, 2, 3, 4]),qt.QH([2,-2,4,4])]))
op1.print_state("operator 2x2 one state")

