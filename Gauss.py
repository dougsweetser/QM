#!/usr/bin/env python
# coding: utf-8

# # Reflections on Gauss' Notebook Using Quaternions

# Gauss did this work around 1819. He recorded it in [one of his notebooks|https://books.google.com/books?id=aecGAAAAYAAJ&pg=PA357#v=onepage&q&f=false], pages 357-361 that got published in 1900. He did not consider it refined to the level of publishable material.
# 
# He had cracked the problem of rotations in 3D space.

# ## Section I.1: Transformations of Space

# ![Mutations of Space/Transformation of space](images/Gauss/100.mutations_of_space.png)

#     Mutations of Space
#     Transformations of Space

# In his notebooks, he did not bother to explain why he started in a particular place. He just "went for it", where it is hard core math.

# ![](images/Gauss/110.first_transformations.png)

#     Set... a, b, c, d
#     
#     so is... 3x3 matrix
#     
#     the general transformation of space coordinates, or if a, b, c, d mean what they
#     want, then 9 magnitudes of the schemes have the folloing proportions... 3x3 matrix

# There is not need for further explanation so long as you have the math skills of Gauss... which would be nobody else. In Goldstein's book, "Classical Mechanics: Second Edition", chapter 4 is devoted to rigid body motion. He takes 25 pages to get to equation 4-67 which is a slight variation on the last 3x3 matrix. There may be 9 terms, but they are not all independent because this is an orthogonal transformation. As such, the product of the off-diagonal terms need to be zero: $$(ad + bc)(bc -ad) = b^2 c^2 - a^2 d^2 = 0$$. No doubt Gauss knew that and just let the math speak for itself.
# 
# What jumped out of the page for me was the definition of a, b, c, and d. In the C_Star_algebra notebook, I wrote:
# 
#     I (probably re-)invented the first and second conjugates.
#     
# $$ A^{*1} \equiv (i q i)^* [= (-t, x, -y, -z)]$$
# $$ A^{*2} \equiv (j q j)^* [= (-t, -x, y, -z)]$$
#     
# Yup! Gauss was using the negatives of the first, second, and third conjugates. Let me demonstrate that.

# In[1]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport matplotlib.pyplot as plt\nimport math\n\n# To get equations the look like, well, equations, use the following.\nfrom sympy.interactive import printing\nprinting.init_printing(use_latex=True)\nfrom IPython.display import display\n\n# Tools for manipulating quaternions.\nimport Q_tools as qt;\n\nfrom IPython.core.display import display, HTML, Math, Latex\ndisplay(HTML("<style>.container { width:100% !important; }</style>"))')


# Square roots cause hassles, so I am going to work with the square of these terms. All the operations will be done from one initial quaternion taking various conjugates.

# In[2]:


p, q, r = sp.symbols("p q r")
Q = qt.QH([1, p, q, r])
half = qt.QH([1/2, 0, 0, 0])
a2 = Q.conj().product(half)
b2 = Q.conj(1).flip_signs().product(half)
c2 = Q.conj(2).flip_signs().product(half)
d2 = Q.conj().conj(1).conj(2).product(half)

a2.print_state("a²", 1)
b2.print_state("b²", 1)
c2.print_state("c²", 1)
d2.print_state("d²")


# I have not used these various conjugates in many equations, so it is a huge amount of fun to see Gauss putting them to use. I doubt few look at the final 3x3 matrix and see what it means exactly. All it takes is a little algebra. First the three off diagonal terms:

# In[3]:


aa = a2.product(a2)
bb = b2.product(b2)
cc = c2.product(c2)
dd = d2.product(d2)

ccddaabb = cc.add(dd).dif(aa).dif(bb).product(half)
bbddaacc = bb.add(dd).dif(aa).dif(cc).product(half)
bbccaadd = bb.add(cc).dif(aa).dif(dd).product(half)

ccddaabb.print_state("(cc + dd - aa - bb)/2", 1)
bbddaacc.print_state("(bb + dd - aa - cc)/2", 1)
bbccaadd.print_state("(bb + cc - aa - dd)/2")


# OK, this was no surprise since it can be seen looking at the previous 3x3 matrix. Look at the off diagonal terms one pair at a time.

# In[4]:


ad = a2.product(d2)
bc = b2.product(c2)

adbc = ad.add(bc)
bcad = bc.dif(ad)

adbc.print_state("ad + bc", 1)
bcad.print_state("bc - ad")


# How neat, two terms are zeros and the zeros do not overlap. Form the product of the two:

# In[5]:


adbc.product(bcad).print_state("(ad + bc)(bc - ad)", quiet=True)


# These two are _orthagonal_ because the first term is zero. The phase terms are complicated, but often in physics, the phase is ignored. Not a good practice, but common.

# In[6]:


ac = a2.product(c2)
bd = b2.product(d2)

acbd  = ac.dif(bd)
nacbd = ac.flip_signs().dif(bd)

acbd.print_state("ac - bd", 1)
nacbd.print_state("-ac - bd", 1)
acbd.product(nacbd).print_state("(ac - bd)(-ac - bd)", quiet=True)


# In[7]:


ab = a2.product(b2)
cd = c2.product(d2)

abcd  = ab.add(cd)
cdab = cd.dif(ab)

abcd.print_state("ab + cd", 1)
cdab.print_state("cd - ab", 1)
abcd.product(cdab).print_state("(ab + cd)(cd - ab)", quiet=True)


# ## Section I.2: Another Transformation

# Gauss was aware of how arbitrary the signs were. In section 2, he looks at a different possibility.

# ![](images/Gauss/120.second_transformations.png)

# In[8]:


A2 = Q.product(half)
B2 = Q.conj().conj(1).flip_signs().product(half)
C2 = Q.conj().conj(2).flip_signs().product(half)
D2 = Q.conj(1).conj(2).product(half)

A2.print_state("A²", 1)
B2.print_state("B²", 1)
C2.print_state("C²", 1)
D2.print_state("D²")


# Generating this variation was simple: I either add or subtracted one conjugate operator from section 1. Look at one diagonal and one off diagonal term:

# In[9]:


AA = A2.product(A2)
BB = B2.product(B2)
CC = C2.product(C2)
DD = D2.product(D2)

AABBCCDD = AA.add(BB).dif(CC).dif(DD).product(half)
AABBCCDD.print_state("AA + BB - CC - DD")


# In[10]:


AD = A2.product(D2)
BC = B2.product(C2)

nADBC = AD.flip_signs().dif(BC)
ADBC = AD.dif(BC)

nADBC.print_state("-ad - bc", 1)
ADBC.print_state("ad - bc")


# No surprises.

# ## Section I.3: The Multiplication Rule for Quaternions

# Since there are many sorts of transformations of space out there, Gauss figured out how to combine them.
# 
# Gauss is not going to use the work "quaternion" since it remains in the King James Bible and not in the lexicon of mathematicians. He used the word "scale", but I will substitute scale -> quaternion in the text below:

# ![](images/Gauss/130.quaternion_multiplication.png)

#     The product of two transformations, 
#     the first of which is a quaternion, 
#     the second of which corresponds to a quaternion, 
#     produces a new quaternion:

# In[11]:


a, b, c, d = sp.symbols("a b c d")
α, β, γ, δ = sp.symbols("α β γ δ")
q1 = qt.QH([a, b, c, d])
q2 = qt.QH([α, β, γ, δ])
q2q1 = q2.product(q1)
q2q1.print_state("q2 x q1")


# The first time through I did not get the answer correct. Gauss looked at this as a way of combining transformations of space. He was not thinking of this as the product of two numbers as we do today. Only in this order is the answer the same as Gauss wrote. Still there is no doubt this is a quaternion product.

# ## Section I.4: Bring in the Half Angles

# Gauss recognized this was a general rule. There is some sort of divide, between a and b, c, d, variables that can be anything at all. Gauss decides to write in trig functions.

# ![](images/Gauss/140.half_angles.png)

#     If the coordinates of the stationary point ξ, η, ζ have an
#     magnitude = nn, rotation = λ, then one can set

# Gauss understoon rotations in a plane using cosines and sines. I am not quite understanding what the "nn" accomplishes. This is a magnitude, how high or low the trig function goes. If b, c, d have their own values of ξ, η, and ζ, then the amplitude of $a$ can be anything. There looks like 5 values one is free to set: the four amplitudes and the angle λ. He may be thinking of 3D space as the central thing going on with his transformations.
# 
# Gauss does not comment in his notebook about the factor of a half on the angle. It takes 6π to return home for a quaternion, a topic that has been discussed extensively. At this juncture, he knows it is needed.

# ## Section I.5 WTF

# I don't understand this section. I will detail a few things I do.

# ![](images/Gauss/150.wtf.png)

#     We write:
#     so is:

# The product of two quaternions is being analyzed to find relationships between the product and the things that go into it. Each term can be isolated using different combinations of conjugates. He also calculates the norm of both quaternions going into the product.

# In[12]:


A = q2q1.add(q2q1.conj()).product(half)
B = q2q1.add(q2q1.conj(1)).product(half)
C = q2q1.add(q2q1.conj(2)).product(half)
D = q2q1.dif(q2q1.conj().conj(1).conj(2)).product(half)
m = q2.norm_squared()
μ = q1.norm_squared()
A.print_state("A", 1)
B.print_state("B", 1)
C.print_state("C", 1)
D.print_state("D", 1)
m.print_state("||m||", 1)
μ.print_state("||μ||")


# So far, so good. Using different combinations of conjugates, I can isolate each sum or difference. Work on the first two lines:

# In[17]:


CD = q2q1.add(q2q1.conj(1).conj()).product(half)
CD.print_state("C + D = q1xq2 + q1xq2 ⃰ ⃰¹", 1)
AB = q2q1.dif(q2q1.conj(1).conj()).product(half)
AB.print_state("A + B= q1xq2 - q1xq2 ⃰ ⃰¹", 1)
cd = q1.add(q1.conj().conj(1)).product(half)
cd.print_state("c + d = q1 + q1 ⃰ ⃰¹", 1)
ab = q1.dif(q1.conj().conj(1)).product(half)
ab.print_state("a + b = q1 - q1 ⃰ ⃰¹")


# Now the greek letter set.

# In[20]:


CmD = q2q1.conj().conj(1).conj(2).add(q2q1.conj(2)).product(half)
αβ = q2.dif(q2.conj().conj(1)).product(half)
γmδ = q2.conj().conj(1).conj(2).add(q2.conj(2)).product(half)

CmD.print_state("C - D = q1xq2 ⃰ ⃰¹ ⃰² + q1xq2 ⃰²", 1, 1)
αβ.print_state("a + b = q2 - q2 ⃰ ⃰¹", 1, 1)
γmδ.print_state("γ - δ = q2 ⃰ ⃰¹ ⃰² + q2 ⃰²", 1, 1)


# **This is not what Gauss wrote**. Just want to make that clear. He had an additional factor of $i$ on the second term. 
