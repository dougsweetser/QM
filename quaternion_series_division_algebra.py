
# coding: utf-8

# # Quaternion Series as a Division Algebra

# I am studying quaternion series, hoping to apply them to quantum mechanics. It would be wrong to think of a quaternion series as a set of Quaternions. A set can be permuted and it is still the same set. A quaternion series is a totally ordered (possibly infinite) array. Each series has associated with it two integers called rows and columns. The product of the rows and columns must equal the length of the array.
# 
# As suggested by the naming of rows and columns, there is much overlap between quaternion series and standard matrix algebra. The one difference will be in how products can be formed between series with the same values for rows and columns.
# 
# I am calling each element in the array a state. Each state is a Quaternion in the full sense of the word as far as the addition and multiplication operators are concerned. Each state can be added, subtracted, multiplied, and divided by a state in the same position in the quaternion series.
# 
# To be a division algebra, both the addition and multiplication operators must be shown to be group operators for quaternion series. Part of the motivaction is to treat the two operations - addition and multiplication - in a similar way. Both are just functions that take in to inputs and create one output. Eliminate any bias in how the two are treated.
# 
# ## Identities
# 
# First things first: start with the identity for a 2 state system which I will make even a little easier by using only complex numbers (the "quaternion"-like properties are not relevant for this analysis).
# 
# Load the needed libraries.

# In[1]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport matplotlib.pyplot as plt\nimport math\nimport copy\n\n# To get equations the look like, well, equations, use the following.\nfrom sympy.interactive import printing\nprinting.init_printing(use_latex=True)\nfrom IPython.display import display\n\n# Tools for manipulating quaternions.\nimport Q_tools as qt;\n\nfrom IPython.core.display import display, HTML, Math, Latex\ndisplay(HTML("<style>.container { width:100% !important; }</style>"))')


# I created a function to return the idenity given a dimeion.

# In[2]:


identity = qt.QHStates().identity(2)
identity.print_state("Identity for 2 state quaternion series.")


# Looks like I expected, a bunch of ones. Do you see an assumption? I did not when I wrote the code. The assumption was that the identity function should only return the multiplicative identity, not the additive one. What darn use is the additive identity, a big fat collection of zeros? That is a utilitarian critique (and a good one, I get it). But if one strives to really treat addition and multiplication similarly, a call to the function identity() should be able to return either the additive or multiplicative identity. Eliminate a natural bias in the code. I added a flag to the function, additive, that is set to False by default, but can be set to True with the following result:

# In[3]:


additive_identity = qt.QHStates().identity(2, additive=True)
additive_identity.print_state("Additive identity for 2 state quaternion series.", 0, 1)


# An identity for an _operator_ will have to be different from that for a bra or ket quaternion series. The values must go along the "diagonal". Fortunately there is a flag for that.

# In[4]:


diagonal_identity = qt.QHStates().identity(2, operator=True)
diagonal_identity.print_state("Diagonal multiplicative operator identity for 2 state quaternion series.")
diagonal_additive_identity = qt.QHStates().identity(2, operator=True, additive=True)
diagonal_additive_identity.print_state("Diagonal addtive operator identity for 2 state quaternion series.")


# This covers the case of square operators. Non-square operators are needed if the bra has a different number of state dimensions from the ket vector. I have made no effort to deal with that case.

# Let's show how these identities work for a simple two state quaternion series.

# In[5]:


c1 = qt.QH([1, 2, 0, 0])
c2 = qt.QH([3, 4, 0, 0])

cs = qt.QHStates([c1, c2])

cs.print_state("cs: ")


# In[6]:


cs.add(additive_identity).print_state("cs + additive identity:")


# In[7]:


diagonal_identity.product(cs).print_state("diagonal operator identity * cs:")


# Here already is an issue. With additive inverse, the two dimensional identity "just worked". One cannot add the diagonal additive identity with its four state dimensions to this two dimensional state.
# 
# With the product, it is the four state dimensional diagonal operator that works as the identity since the result is still a quaternion series, in fact the same series.
# 
# The software has been written so that if and only if two kets of the exact same dimension are used in the product function, then on on the left will be diagonalized. For two bras, the bra on the right will diagonalize. That is operationally how products are defined. Thus one can use the ket identity defined earlier and have it behave like we hope:

# In[8]:


identity.product(cs).print_state("identity * cs:")
identity.product(cs).print_state("cs * identity:")


# The multiplicative identity will commute with all quaternions, an unusual property for a quaternion.

# ## 2. Inverses

# Move on the the additive and multiplicative inverses. At least this time addition will not be so prosaic. Calculate the inverse.

# In[9]:


cs_additive_inv = cs.inverse(additive=True)
cs_additive_inv.print_state("cs_additive_inv", 0, 1)


# Do the obvious to prove this is right:

# In[10]:


cs_plus_additive_inv = cs.add(cs_additive_inv)
cs_plus_additive_inv.print_state("cs_plus_additive_inv, should be zeroes", 0, 1)


# In[11]:


additive_identity = qt.QHStates().identity(2, additive=True)
additive_identity.print_state("additive identity", quiet=True)


# Now repeat the exercise, but for the multiplication operator.

# In[12]:


cs_multi_inv = cs.inverse(additive=False)
cs_multi_inv.print_state("cs_multi_inv")


# In[13]:


cs_times_cs_inv = cs.product(cs_multi_inv)
cs_times_cs_inv.print_state("cs_times_cs_inv")


# What is really going on is that the inverse ket forms a diagonal. Since every non-zero quaternion has an inverse, a ket will always have an inverse - up to those pesky zeroes.
# 
# A different way to accomplish the very same thing is to say the product of two kets is formed using a delta distribution.  The first time of a quaternion series would then be made of the product of first terms of the two input series and no other terms. 

# In[14]:


cs_multi_inv_diag = cs_multi_inv.diagonal(2)
cs_multi_inv_diag.print_state("diagonal form of inverse", 0, 1)


# In[15]:


cs_times_diag_inv = cs_multi_inv_diag.product(cs)
cs_times_diag_inv.print_state("diag_inv * cs")


# That is the multiplicative identity discussed above.

# ## Addition, Multiplication, Vectors and Operators

# This is where working to write "clean code" rather than just code that works. The goals for clean code involve simplicity and directness. Both the identity() and inverse() functions have True/False flags for additive and operator. The additive inverse works if additive=True and operator=False. The multiplicative inverse works if additive=False and operator=True. Nice.

# ## 3. Closure

# Closure for addition is obvious. Let me just create three quaternion kets, add them up, show the result is another quaternion series. It is almost not worth the trouble, but the reason I do it is because the goal is to treat addition similar (but not exactly the same) as multiplication. If something can be done for addition, multiplication should follow - at least that is the proposal.

# In[16]:


q3_orig = []

for i in range(1, 4):
    q3_orig.append(qt.QHStates([qt.QH([i, -i, i+2, i**2]), qt.QH([-i, 2 * i, i-1, i])]))

for i, q in enumerate(q3_orig, 1):
    q.print_state("{}: q".format(i))
    
q3 = copy.deepcopy(q3_orig)

q_sum = q3.pop()

for q in q3:
    q_sum = q_sum.add(q)
    
q_sum.print_state("Sum of three quaternion series is a quaternion series", 0, 1)


# So now how does one show closure for multiplication? The hint comes from the work with identities and operators: multiplication must involve diagonal operators. 

# In[17]:


q3 = copy.deepcopy(q3_orig)

q_prod = q3.pop()

for q in reversed(q3):
    q_prod = q.diagonal(2).product(q_prod)
    
q_prod.print_state("Product of two operators and a ket is a quaternion series, q1 q2|q3>", quiet=True)


# Why did I feel it was necessary to add the reversed() function? What needs to be calculated is $q1 \; q2 \; |q3>$. The pop() function grabs q3 as the ket. Order matters, so the next product has to be with q2, then q1, requiring a reversed array.

# ## 4. Associative

# Just calculate it.

# In[18]:


q3 = copy.deepcopy(q3_orig)
q3[0].add(q3[1]).add(q3[2]).print_state("(q1+q2) + q3")
q3[0].add(q3[1].add(q3[2])).print_state("q1 + (q2+q3)")


# In[19]:


q3 = copy.deepcopy(q3_orig)
q3[0].product(q3[1]).product(q3[2]).print_state("(q1xq2) x q3")
q3[0].product(q3[1].product(q3[2])).print_state("q1 x (q2xq3)")


# This calculation shows how similarly addition and multiplicaiton are being treated by the code now. You will notice that the only change in the two above cells was "add" goes to product and "+" goes to "x". 

# ## Conclusion

# Quaternion series are not numbers like the real numbers, complex numbers and quaternions. Instead they start from quaternions and add more structure: the array, rows, and columns. Addition and multiplicaiton have been shown to be group operations on quaternion series, and thus this quaternion series are a division algebra. It is not obvious to me where there will turn out to be "a good thing". Yet my gut says this is better than just having addition as a group operation.
