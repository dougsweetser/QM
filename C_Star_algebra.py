
# coding: utf-8

# # Do Quaternion Series form a $C^*$-Algebra?

# D. Sweetser <sweetser@alum.mit.edu>

# A [$C^*$-algebra|https://en.wikipedia.org/wiki/C*-algebra] is part of a study of complex numbers that has been partially motivated by its applications to quantum mechanics. At this time, I only have a superficial appreciation of the subject.
# 
# Quaternions, as a number, contain complex numbers as a subgroup. In a similar way, a quaternion series which is a totally-ordered array of quaternions with integers for rows and columns, will have a totally-ordered array of complex numbers as a subgroup. Perhaps the title question should be modified to ask about an $H^*$-algebra. My expectation is that there will be a few places where things will be different since a quaternion series is in a sense larger than a complex series.

# ## Two Quaternion Products

# Actually, there is only one set of rules for multiplying a totally ordered array of quaternion, don't worry. What can change is what is put into the product. This seamily minor difference in what is going on may be viewed as a big deal by those formally trained in math. Let me try to clarify. First load the needed libraries. 

# In[1]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport matplotlib.pyplot as plt\nimport math\n\n# To get equations the look like, well, equations, use the following.\nfrom sympy.interactive import printing\nprinting.init_printing(use_latex=True)\nfrom IPython.display import display\n\n# Tools for manipulating quaternions.\nimport Q_tools as qt;\n\nfrom IPython.core.display import display, HTML, Math, Latex\ndisplay(HTML("<style>.container { width:100% !important; }</style>"))')


# Work with a simple two state quaternion series.

# In[2]:


At1, Ax1, Ay1, Az1 = sp.symbols("At1 Ax1 Ay1 Az1")
At2, Ax2, Ay2, Az2 = sp.symbols("At2 Ax2 Ay2 Az2")
Aq1 = qt.QH([At1, Ax1, Ay1, Az1], qtype="a₁")
Aq2 = qt.QH([At2, Ax2, Ay2, Az2], qtype="a₂")
A = qt.QHStates([Aq1, Aq2])

A.print_state("Quaternion Series A")


# All that will be done is to take the product of A with itself. A scalar result is desired. That means specifically that the rows and columns are both one. The ket A has to be altered to be a bra using the bra() function. What will be changed is whether or not a conjugate of A is taken.

# In[32]:


A.bra().product(A).print_state("AA")
A.bra().conj().product(A).print_state("A ⃰ A")


# The rest of this notebook, and in fact most of the work done with quantum mechanics, will concern taking the conjuate of the first term times the second one. I have read people far more skilled that I will ever be _argue_ over which of these is the right way to measure things with quaternions. Different input leads to different output, so neither is right or wrong. Instead, in certain situations, one sort of product will be the only one to use in that situation. Change the situation, change the need. For quantum mechanics, the need is for something I call the "Euclidean product" which always takes the conjugate of the first term before multiplying. The first sort of product, $A^2$, is useful for doing special relativity and my new approach to gravity (quaternion gravity).

# The issue may have to due with the focus on _the one number_. Neither $A^2$ nor $A^* A$ results in a solitary number. In both cases, the result is another quaternion, not one number. Yet I have seen professionals treat the value that comes out of $A^* A$ as real number, not a quaternion. This strikes me as wrong. Much of what gets studied in physics is the neighborhood of points. The neighborhood of real number is fundamentally different from the neighborhood of a quaternion. The only point near a real number are other real numbers. For a quaternion, the neighborhood is full of points from time and three dimensions of space, all with specific algebraic relationship to each other.
# 
# There are two different reasons why the norm squared, $A^* A$, has zero for all three imaginaries. The first is that the sine of the angle between $A$ and $A^*$ is zero. There is no cross product for a quaternion and its conjugate. The same can be said for $A^2$. The scalar times the vector add constructively for $A^2$ but destructively for $A^* A$. No cross product and the canceling of terms are the two reasons for the three zeroes of the norm squared. This feels like a deeper way to look at what is going on.

# ## The Three Conjugates

# Anything done on for complex numbers on the real manifold $\mathbb{R}^2$ can also be done on the complex manifold, $\mathbb{C^1}$, because there are two conjugates, $z$ and $z^*$. When I was tutored on this subject, it was apparent this would be a problem for quaternion functions on the real manifold $\mathbb{R}^4$. There would be functions that could not be represented with only $A$ and $A^*$. I recall appreciating this was a total show stopper, 4>2, oops. I (probably re-)invented the first and second conjugates.
# $$ A^{*1} \equiv (i q i)^* $$
# $$ A^{*2} \equiv (j q j)^* $$

# In[33]:


A.conj(1).print_state("A ⃰¹")
A.conj(2).print_state("A ⃰²")


# This is a "cute" result in the sense that the first and second terms are the ones that don't flip signs. It is more fair that way. It will be interesting to see how these other conjugates fit within the normal context of a $C^*$-algebra.

# The wiki article cites four properties for what is needed to be a $C^*$-algebra, and thus by extension, a $H^*$-algebra, if such a beast exists. Let's give it a go...

# ## 1. Involutions

# Here is the definition of an involution: $$A^{**}=(A^{*})^{*}=A$$ 
#     
# See if that holds for quaternion series with the three conjugates.

# In[34]:


A.conj().conj().print_state("(A ⃰) ⃰")
A.conj(1).conj(1).print_state("(A ⃰¹) ⃰¹")
A.conj(2).conj(2).print_state("(A ⃰²) ⃰²")


# OK, that one was easy and I already knew it would work.

# ## 2. Anti-involutive Automorphism

# Here is what I mean by that phrase: $$(A+B)^{*}=A^{*}+B^{*} $$
# 
# $$(AB)^{*}=B^{*}A^{*}$$

# Addition is usually trivial, with multiplicaiton being the challenge.

# In[9]:


Bt1, Bx1, By1, Bz1 = sp.symbols("Bt1 Bx1 By1 Bz1")
Bt2, Bx2, By2, Bz2 = sp.symbols("Bt2 Bx2 By2 Bz2")
Bq1 = qt.QH([Bt1, Bx1, By1, Bz1], qtype="b₁")
Bq2 = qt.QH([Bt2, Bx2, By2, Bz2], qtype="b₂")
B = qt.QHStates([Bq1, Bq2])

B.print_state("Quaternion Series B")


# In[24]:


A.add(B).conj().print_state("(A+B) ⃰")
A.conj().add(B.conj()).print_state("A ⃰ + B ⃰")


# In[22]:


A.add(B).conj(1).print_state("(A+B) ⃰¹")
A.conj(1).add(B.conj(1)).print_state("A ⃰¹ + B ⃰¹")


# In[23]:


A.add(B).conj(2).print_state("(A+B) ⃰²")
A.conj(2).add(B.conj(2)).print_state("A ⃰² + B ⃰²")


# Addition is easy and behaved as expected. On to the product relation. A second quaternion series is needed to see any possible complications from cross products.

# When I see $(A B)^*$, I don't see the norm being taken. I just see the product of $A$ and $B$, then conjugate that. That is my interpretation. As such, one expects the result of multiplying a two state ket times another two state ket to be a two state ket. Note: it normally makes not sense to multiply two kets together, but the entire point of the video and notebook on quaternion series as a division algebras was to show this can make sense by diagonizing one of the kets, and then it all flows.

# In[25]:


ABc = A.product(B).conj()
BcAc = B.conj().product(A.conj())

ABc.print_state("(A B) ⃰")
BcAc.print_state("B ⃰ A ⃰")


# Nice, it all works. Now try with the first conjugate.

# In[26]:


ABc1 = A.product(B).conj(1)
Bc1Ac1 = B.conj(1).product(A.conj(1))

ABc1.print_state("(A B) ⃰¹")
Bc1Ac1.print_state("B ⃰¹ A ⃰¹")


# Total failure... but in an easy to fix way! Every single sign is wrong. The first and second conjuages have this relation:
# 
# $$(A B)^{*1}=-B^{*1}A^{*1}$$
# $$(A B)^{*2}=-B^{*2}A^{*2}$$
# 
# OK, but _why_ did the minus sign show up? Recall the definition of the first and second conjugates.  It was a triple product, with imaginary numbers on the outside. When we go from one conjugate operator to two conjugate operators, that involves going from two imaginaries to four. Two of those are right next to each other and thus introduce a minus sign.
# 
# Prove it works that way for the second conjugate.

# In[27]:


ABc1 = A.product(B).conj(2)
Bc1Ac1 = B.conj(2).product(A.conj(2)).flip_signs()

ABc1.print_state("(A B) ⃰²")
Bc1Ac1.print_state("-B ⃰² A ⃰²")


# Most excellent.

# ## Numbers obey anti-involutive automorphisms

# It is important to know in detail how a number interacts with quaternion series. The easiest thing to do is to make a ket that only has the values of the number in it. After that, the calculation is the same.

# In[13]:


Qt1, Qx1, Qy1, Qz1 = sp.symbols("Qt1 Qx1 Qy1 Qz1")
Qq1 = qt.QH([Qt1, Qx1, Qy1, Qz1], qtype="q₁")
Q2 = qt.QHStates([Qq1, Qq1])

Q2.print_state("Quaternion Ket Q2")


# In[28]:


QAc = Q2.product(A).conj()
AcQc = A.conj().product(Q2.conj())

QAc.print_state("(Q2 A) ⃰")
AcQc.print_state("A ⃰ Q2 ⃰")


# Once the number is written as a ket, the logic for the ket works as before. Here it is for the first conjugate.

# In[29]:


QAc1 = Q2.product(A).conj(1)
Ac1Qc1 = A.conj(1).product(Q2.conj(1)).flip_signs()

QAc1.print_state("(Q A) ⃰¹")
Ac1Qc1.print_state("-A ⃰¹ Q ⃰¹")


# Bingo, bingo.

# ## The $C^*$ identity holds

# Quaternions are a normed division algebra. That helps a lot to show that quaternion series are going to behave like a normed division algebra.
# $$ \|A^{*}A\|=\|A\|\|A^{*}\|.$$

# In[30]:


AcAsq = A.bra().conj().product(A).norm_squared()
Asq = A.norm_squared()
Acsq = A.conj().norm_squared()
Asq_Acsq = Asq.product(Acsq)

print("The parts")
Asq.print_state("||A||")

Acsq.print_state("||A ⃰||")

print("The parts squared.")
AcAsq.print_state("||A ⃰ A ||")

Asq_Acsq.print_state("||A|| ||A ⃰||")



# Bingo, bingo.

# I have yet to really play around with first and second conjugates of identical elements other than to notice in someways it is the opposite of a norm: there is only one zero and three non-zero terms.

# In[31]:


A.conj(1).product(A).print_state("A ⃰¹ A")


# Nature must put this to use somehow...

# ## Conclusions

# It is now technically safe to say that quaternion series have properties identical to those of $C^*$-alebras. Too bad I am not familiar with the literature on the subject. Yet this exercise has pointed out a new sort of relation with the first and second conjugates, namely that there is also a sign flip with these conjugates do an anti-involution. New is good.
