
# coding: utf-8

# # Lecture 3: The Principles of Quantum Mechanics

# In this notebook, the quaternion series are going to have to start to do interesting technical feats. The world of operators, Eigen-values and Eigen-vectors will need to be transported to relationships between quaternion series.
# 
# Load the needed libraries.

# In[1]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport matplotlib.pyplot as plt\nimport math\n\n# To get equations the look like, well, equations, use the following.\nfrom sympy.interactive import printing\nprinting.init_printing(use_latex=True)\nfrom IPython.display import display\n\n# Tools for manipulating quaternions.\nimport Q_tools as qt;\n\nfrom IPython.core.display import display, HTML, Math, Latex\ndisplay(HTML("<style>.container { width:100% !important; }</style>"))')


# ![](images/lecture_3/c3_p52_q1.jpg)

# Vectors are good, just not good enough. A quaternion series has the properties of vectors: they can be added, subtracted, and multiplied by a scalar. Quaternion series can also be multiplied and divided by other perfectly fine quaternion series. This is not the case for vectors. This will work no matter how many state dimensions there are for the series. It is my belief that some exceptionally difficult problems in the foundations of quantum mechanics occur because our tools are indeed powerful, just not powerful enough. In time, I will have to show all expected processes work for quaternion series, no exceptions. Let the games begin.

# ![](images/lecture_3/c3_p52_q3.jpg)

# 
# 
# In the first chapter of these companion notebooks, it was shown that for a quaternion series operator $M$ and bra quaternion series $|A>$ and $|B>$, this was the case. One thing not stated explicitly here is that both $|A>$ and $|B>$ have to have the same number of state dimensions or the expression does not make sense. In the quaternion series approach, the number of states in operator $M$ has to be a multiple of the states in the bras.

# ![](images/lecture_3/c3_p53_q1.jpg)

# If all I can ever do is multiply quaternion series together, this will always be the case because numbers are nice in that way.

# ![](images/lecture_3/c3_p53_q2.jpg)

# This looks like a simple statement, but look at what it presumes. The matrix $M$ has complex values, so $z$ will commute, or $M z = z M$. Now switch to quaternions... does this mean an end to this project because quaternions do not commute? I do think the knee-jerk reaction is to say "yes".

# Careful reflection takes time, so please indulge me. The above in an algebraic expression. Each term has been given a name, $M$ and $z$, and been assigned a specific order at the table. A quaternion product can always be broken into two parts: the even part that commutes, and the odd part or cross product that anti-commutes.
# 
# First define a bunch of quaternions.

# In[2]:


At1, Ax1, Ay1, Az1 = sp.symbols("At1 Ax1 Ay1 Az1")
At2, Ax2, Ay2, Az2 = sp.symbols("At2 Ax2 Ay2 Az2")
Aq1 = qt.QH([At1, Ax1, Ay1, Az1], qtype="a₁")
Aq2 = qt.QH([At2, Ax2, Ay2, Az2], qtype="a₂")
A = qt.QHStates([Aq1, Aq2])
A.print_states("A", 1)

Mt1, Mx1, My1, Mz1 = sp.symbols("Mt1 Mx1 My1 Mz1")
Mt2, Mx2, My2, Mz2 = sp.symbols("Mt2 Mx2 My2 Mz2")
Mt3, Mx3, My3, Mz3 = sp.symbols("Mt3 Mx3 My3 Mz3")
Mt4, Mx4, My4, Mz4 = sp.symbols("Mt4 Mx4 My4 Mz4")
Mq1 = qt.QH([Mt1, Mx1, My1, Mz1], qtype="m₁")
Mq2 = qt.QH([Mt2, Mx2, My2, Mz2], qtype="m₂")
Mq3 = qt.QH([Mt3, Mx3, My3, Mz3], qtype="m₃")
Mq4 = qt.QH([Mt4, Mx4, My4, Mz4], qtype="m₄")

M = qt.QHStates([Mq1, Mq2, Mq3, Mq4])
M.print_states("M", 1)

zt, zx, zy, zz = sp.symbols("zt zx zy zz")
zq = qt.QH([zt, zx, zy, zz], qtype="z")
zqs = qt.QHStates([zq])
z_op = zqs.diagonal(2)
z_op.print_states("z")


# Before trying to understand quaternion series, let's just understand the product of two quaternions. We wish to have an algebraic rule about what exactly it means to reverse two named symbols. When the reversal happens, it means the product of the two reverse quaternions is the difference between the even and odd products. This makes no difference for real and complex numbers since the odd part is always exactly zero. For quaternions, this flips the signs of the thing that flip signs under reversal, so no net change results.

# In[3]:


Mq1z = Mq1.product(zq)
zMq1 = zq.product(Mq1, reverse=True)
print("M, then z, even + odd: ", Mq1z)
print("z, then M, even - odd: ", zMq1)
print("difference: ", Mq1z.dif(zMq1))


# The precise rule about the impact of reversing positions of named terms in algebraic expressions can and will be applied consistently from now on. Am I cheating on the non-commutative nature of quaternions? I think the answer is "no" as demonstrated by the qtype of the difference above, 'm₁xz-zxRm₁'. This says, from left to right, form the product - even plus odd - of m₁ times z, then subtract z "xR" m₁, where xR is the reverse product which is the difference of the even and odd products. The difference of those is exactly zero, always, even though the qtypes are distinct.

# A digression from the book... This idea of a rule for reversing positions in algebraic expressions is not in wide use. I think there is a legitimate reason to never use quaternions in practice: the definition of a quaternion derivative remains a topic of study. There are people who work with left- or right-derivatives because they are different. Consider the standard limit definition:
# 
# $$ \frac{d f(q)}{dq} = \lim_{dq \rightarrow 0} (f(q + dq) - f(q)) \;x \;dq^{-1} $$
# 
# I would have to consult the literature to find out if this was called a left- or right-derivative. Whatever it is called does not matter with this new rule for reversal in place. Yes, the differential can now be moved to the left so long as one uses the reversal product:
# $$\lim_{dq \rightarrow 0} (f(q + dq) - f(q)) \; x \; dq^{-1} = \lim_{dq \rightarrow 0} dq^{-1} \; xR \;(f(q + dq) - f(q)) $$
# Getting one consistent definition for a quaternion derivative may be just as important as anything else in this collection of work, so I think it was worth the digression.

# Back to the question at hand, namely that $Mz|A>=zM|A>$. Notice the $Mz$ and $zM$ can be viewed a product of operators, a subject that has yet to be discussed. This is because $z$ is viewed as "just a number", despite its location in the bracket notation.

# In[4]:


Mq1z = Mq1.product(zq)
Mq2z = Mq2.product(zq)
Mq3z = Mq3.product(zq)
Mq4z = Mq4.product(zq)
Mz = qt.QHStates([Mq1z, Mq2z, Mq3z, Mq4z])
Mz.print_states("Mz", 1)

zMq1 = zq.product(Mq1, reverse=True)
zMq2 = zq.product(Mq2, reverse=True)
zMq3 = zq.product(Mq3, reverse=True)
zMq4 = zq.product(Mq4, reverse=True)
zM = qt.QHStates([zMq1, zMq2, zMq3, zMq4])
zM.print_states("zM")


# If you stare at these long enough, they do look identical. Prove it.

# In[5]:


MzA = A.product("ket", operator=Mz)
zMA = A.product("ket", operator=zM)
print(MzA.dif(zMA))


# ![](images/lecture_3/c3_p54_q1.jpg)

# The next sections use 3x3 matrices, so let's make the ket A have 3 elements. To create an orthornormal basis for j, just use 1, j, and j. Here is the ket A in component form:

# In[6]:


At3, Ax3, Ay3, Az3 = sp.symbols("At3 Ax3 Ay3 Az3")
Aq3 = qt.QH([At3, Ax3, Ay3, Az3], qtype="a")
aj = qt.QHStates([Aq1, Aq2, Aq3]).diagonal(3)
jj = qt.QHStates([qt.QH().q_1(), qt.QH().q_j(), qt.QH().q_j()])
A = jj.product("ket", operator=aj)
A.print_states("A", 1)


# ![](images/lecture_3/c3_p55_q1.jpg)

# Just make M bigger.

# In[7]:


Mt5, Mx5, My5, Mz5 = sp.symbols("Mt5 Mx5 My5 Mz5")
Mt6, Mx6, My6, Mz6 = sp.symbols("Mt6 Mx6 My6 Mz6")
Mt7, Mx7, My7, Mz7 = sp.symbols("Mt7 Mx7 My7 Mz7")
Mt8, Mx8, My8, Mz8 = sp.symbols("Mt8 Mx8 My8 Mz8")
Mt9, Mx9, My9, Mz9 = sp.symbols("Mt9 Mx9 My9 Mz9")

Mq5 = qt.QH([Mt5, Mx5, My5, Mz5], qtype="m₅")
Mq6 = qt.QH([Mt6, Mx6, My6, Mz6], qtype="m₆")
Mq7 = qt.QH([Mt7, Mx7, My7, Mz7], qtype="m₇")
Mq8 = qt.QH([Mt8, Mx8, My8, Mz8], qtype="m₈")
Mq9 = qt.QH([Mt9, Mx9, My9, Mz9], qtype="m₉")

M = qt.QHStates([Mq1, Mq2, Mq3, Mq4, Mq5, Mq6, Mq7, Mq8, Mq9])
M.print_states("M")


# ![](images/lecture_3/c3_p55_q2.jpg)

# Form the product.

# In[8]:


MA = A.Euclidean_product("ket", operator=M)
MA.print_states("MA", quiet=True)


# This simple result is crazy complex. Every term has 4 space-time dimensions. There are three state dimensions, so that makes 12 terms for the ket A. The operator M has nine state dimensions so has 36 spots to fill. All in all the ket $\beta$ is composed of the sum of $3 * 4 * 4 * 3 = 12 * 12 = 144$ terms. Quantum mechanics done over the field of complex numbers would have half this number, or 72, so it is still complicated.
# 
# I made sure the $n=1$ state was simpler. One does see for the first part of the $n=1$ state that 1 pairs with 1. Why all the minus signs? These are quaternions, so a factor of $i$ times a factor of $i$ generates a minus number.

# ![](images/lecture_3/c3_p56_q1.jpg)

# With quaternion series quantum mechanics, the accounting system changes, but the products remain the same. There are no matrices. Take the square matrix and stack them all in a line:
# $$m_{11}\,m_{12}\,m_{13}\,m_{21}\,m_{22}\,m_{23}\,m_{31}\,m_{32}\,m_{33}$$
# $$= (m_{1}\,m_{2}\,m_{3})(m_{4}\,m_{5}\,m_{6})(m_{7}\,m_{8}\,m_{9})$$
# The quaternion series is being treated in the algebraic operations as three sets of three, just like matrix.

# ![](images/lecture_3/c3_p56_q2.jpg)

# With quaternion series, there are two types of representations going on dealing separately with space-time dimensions and with state dimensions. Space-time information can be written in Euclidean coordinates, polar coordinates, spherical coordinates, etc. The state dimensions can be written with an endless variety of basis vectors. There certainly is nothing special about the 1jj basis vector used to express the ket A.

# ![](images/lecture_3/c3_p57_q1.jpg)

# We have a matching pair: an Eigen-value which is a diagonal quaternion series operator and the Eigen-ket. What is impressive about this pair is how simple it is compared to the general case (the one with 144 terms). The Eigen-ket can be whatever. As usual, it has 4 space-time dimensions, but can from one to infinite number of state dimensions. The Eigen-value will not change the "directon" of its Eigen-ket. I put "direction" in quotes because it is important to recall everything lives in space-time. The time part of space-time is not a direction. Much of the sport of quantum mechanics happens by characterizing the order found in these special pairs of Eigen-values and Eigen-kets in a sea of natural chaos. 

# ![](images/lecture_3/c3_p57_q2.jpg)

# Test it:

# In[9]:


q_1 = qt.QH([1, 0, 0, 0])
q_2 = qt.QH([2, 0, 0, 0])
M12 = qt.QHStates([q_1, q_2, q_2, q_1])
k11 = qt.QHStates([q_1, q_1])
M12k11 = k11.product("ket", operator=M12)
M12k11.print_states("M12 k11, an eigen pair?: ")


# The matrix $M$ has an Eigen-value of $3$ and and Eigen-ket of $|1 1>$.

# ![](images/lecture_3/c3_p57_q3.jpg)

# In[10]:


q_n1 = qt.QH([-1, 0, 0, 0])
k1n1 = qt.QHStates([q_1, q_n1])
M12k1n1 = k1n1.product("ket", operator=M12)
M12k11.print_states("M12 k1n1, an eigen pair?: ")


# Nice, same Eigen-value, different Eigen-ket.

# ![](images/lecture_3/c3_p58_q1.jpg)

# In[11]:


q_0 = qt.QH()
k10 = qt.QHStates([q_1, q_0])
M12k10 = k10.product("ket", operator=M12)
M12k10.print_states("M12 k10, an eigen pair?: ")


# Just not the same. So k10 is not an Eigen-ket for the operator $M$. Most kets are not. Being an Eigen-ket for an operator is a rare thing.

# ![](images/lecture_3/c3_p58_q2.jpg)

# In[12]:


q_i = qt.QH([0, 1, 0, 0])
Mn11 = qt.QHStates([q_0, q_n1, q_1, q_0])
k1i = qt.QHStates([q_1, q_i])
Mn11_k1i = k1i.product("ket", operator=Mn11)
Mn11_k1i.print_states("Mn11 k1i, an eigen pair?: ")


# What? This doesn't look like the first Eigen-value, Eigen-ket pair with repeated values. That "easy to spot" quality arose from the fact that both Eigen-values were real number.

# In[13]:


q_ni = qt.QH([0, -1, 0, 0])
op_ni = qt.QHStates([q_ni])
op_ni_k1i = k1i.product("ket", operator=op_ni)
op_ni_k1i.print_states("op_ni_k1i eigen pair?: ")


# This is the same result as above, so for operator $Mn11$, the Eigen-value is $-i$ for Eigen-ket $|1 i>$. This three-way relationship between operator, Eigen-value, and Eigen-ket is tricky, making the study of quantum mechanics every bit as hard as expected.

# ![](images/lecture_3/c3_p60_q1.jpg)

# ![](images/lecture_3/c3_p61_q1.jpg)

# This is straight-forward to write into code.

# In[14]:


M.print_states("M", 1)
M.dagger().print_states("M†")


# Three of the nine states "stay in place", those being n=1, n=5, and n=9. A little game can be made of spotting how the other states shuffle. Every term gets conjugated, no exceptions.

# ![](images/lecture_3/c3_p61_q2.jpg)

# I really am doing an experiment with these notebooks. I have already calculated $MA$, that was the expression with 144 terms cited earlier. Now I need to calculate AM†. Will the conjugated be equal?
# 
# Here's how I _know_ I am doing experiments: the first try did not work. The function Euclidean_product() is quite simple: all it does in the back end is take the conjugate of the bra vector (if there is one) and feeds it into the function product(). The function product() is not simple. It was written to handle all the varieties of bracket operations. It got a detail of how to multiply the operator by the bra vector incorrect. Now let's proceed.

# In[15]:


AMd_conj = A.Euclidean_product("bra", operator=M.dagger()).conj()
MA.print_states("MA", 1, quiet=True)
AMd_conj.print_states("AM†*", 1, quiet=True)
MA.dif(AMd_conj).print_states("M|A> - <A|M†", quiet=True)


# Bingo, bingo. But _why_ is it true, particularly since quaternion series do not commute?
# 
# That change I needed to impose to calculate $<A|M$ correctly was to transpose the operator. The transpose of a transpose is the operator unchanged. There are no conjugates involved in $M|A>$. There are three for $<A|M†*$: one for the bra vector $<A|$, one for the operator $M$, and finally the product of these two. Then there are three parts to analyze in this product: the scalar terms, the even 3-vector terms, and the odd 3-vector terms. One scalar term gets no changes in sign what-so-ever. Three of the scalar terms flip signs twice, so no net change. The even 3-vector term is composed of a scalar and a 3-vector, so it flips signs once as a product, and once more with the final conjugate, so ends up unchanged. The odd 3-vector is the cross of two 3-vector terms, so the conjugates will change signs twice. There is a third sign change brought about by the change in the order of multiplication. The final conjugate levels the odd 3-vector unchanged. Since the scalar, even and odd 3-vectors are unchanged, the two are equal. It is fun to see how all the parts shift under these changes in a way the preserves the final result.

# ![](images/lecture_3/c3_p61_q3.jpg)

# Space-time dimensions are orthogonal to state dimensions. This means not only is there a sense of the two being at "right angles", but there is an intimate connection too. Every state dimension must have 4 space-time dimensions, but the number of state dimensions depends on the system under study. This issue is always framed in terms of real versus imaginary numbers, not something that is physical. With the quaternion series approach to quantum mechanics, the real versus imaginary story remains as true as it ever was. Now however, there is a physical interpretation. To say that the imaginary is zero is to say that an observer is located at the spatial location of (0, 0, 0). A measurement involves time or a time-like thing where the observer is in space or a space-like thing.

# ![](images/lecture_3/c3_p62_q1.jpg)

# Show this is not true in general.

# In[16]:


MD = M.dagger()
MMD = M.dif(MD)
MMD.print_states("Is M† + M?", quiet=True)


# Notice how the diagonal terms **are** the same? That is kind of fascinating. The goal is to find observables where all those other terms drop out.

# ![](images/lecture_3/c3_p63_q1.jpg)

# The conjugate operator flips ths sign of the imaginaries. If $\lamda = \lamba^*$, that can only be the case if the imaginaries are zero. For quaternion series quantum mechanics, that means eigenvalues are time-ish quantities.

# ![](images/lecture_3/c3_p64_q1.jpg)

# ![](images/lecture_3/c3_p64_q2.jpg)

# Most of what goes on in quantum mechanics is fleshing out the details of this fundamental theorem. Recall the three players: a Hermitian series with $N^2$ state dimensions, an Eigen-ket of N state dimensions and N Eigen-values. As discussed earlier, all N Eigen-values are real-valued with all spatial terms equal to zero. 

# ![](images/lecture_3/c3_p64_q3.jpg)

# ![](images/lecture_3/c3_p67_q1.jpg)

# Making a basis quaternion series is trivial - just find the right number to normalize by and one is done with that part. It sounds like a bigger deal to be able to construct a collection of vectors that are all orthogonal. This turns out to be straight forward. It is discussed in the next section which covers the Gram-Schmidt procedure. Recall that a Hermitian operator is quite a special animal. All the Eigen-values are real-valued. One implication of this is that a real-value always necessarily commutes with any other quaternion. What does this mean for the corresponding Eigen-kets? They must be orthogonal to each other or else the Eigen-values could mix.
# 
# The proper way to prove this is to start with one Eigen-value and create an normalized Eigen-ket basis. Then use the Gram-Schmidt procedure to get N-1 other basis quaternions.

# ![](images/lecture_3/c3_p68_q1.jpg)

# It took me a while to see how this works, but once seen clearly, well, of course it is easy. The first quaternion in a quaternion series gets normalized, simple. Before the second one gets normalized, subtract away the Euclidean product with the first quaternion. That assures it will be orthogonal. Rinse and repeat. 

# In[17]:


def orthonormalize(qh):
    """Given a quaternion series, resturn a normalized orthoganl basis."""
    
    last_q = qh.qs.pop(0).normalize(math.sqrt(1/qh.dim))
    orthonormal_qs = [last_q]
    
    for q in qh.qs:
        qp = q.Euclidean_product(last_q)
        orthonormal_q = q.dif(qp).normalize(math.sqrt(1/qh.dim))
        orthonormal_qs.append(orthonormal_q)
        last_q = orthonormal_q
        
    return qt.QHStates(orthonormal_qs)


# In[18]:


qa, qb, qc = qt.QH([0, 1, 2, 3]), qt.QH([1, 1, 3, 2]), qt.QH([2, 1, -2, 3])
qabc = qt.QHStates([qa, qb, qc])
qabc_on = orthonormalize(qabc)
qabc_on.print_states("qabc_orthonormalized", quiet=True)


# In[19]:


qabc_on.norm_squared().print_states("square it up")


# The orthonormalization process can just "always work" because it only involves forming a product, subtraction, and normalization, which will always be legal operations for quaternion series. 

# ![](images/lecture_3/c3_p71_q1.jpg)

# Energy, momentum in the x, y, and z directions, all have familiar examples in the classical world. Yet why are the measurements of quantum mechanics different? Quantum mechanics always, necessarily uses a spatial mirror as part of the accounting process. What this does is assure that for any quaternion series $A$, the value of $<A|A>$ will be zero only if $A$ is zero, otherwise it will be positive definite. When physicists do the work of describing thing with almost nothing, they need a math that handles almost nothing correctly. Nothing is a hard lower bound. Notice how zero is a hard lower bound for $<A|A>$  but __not__ $A^2$.
# 
# A quaternion series that has an orthonormal basis means just that: state dimensions are orthogonal, and the sum of all states adds up to unity. 

# ![](images/lecture_3/c3_p72_q1.jpg)

# In lecture 2, a way was worked out to represent $|u>$, $|d>$, $|L>$, and $|r>$. It is repeated here, but notice two things: how $|L>$ and $|r>$ come from  $|u>$ and $|d>$, and there are an arbitrary number of other ways this could have been done - there is no "correct" way to do this.

# In[20]:


q_0, q_1, q_i, q_j, q_k = qt.QH().q_0(), qt.QH().q_1(), qt.QH().q_i(), qt.QH().q_j(), qt.QH().q_k()

u = qt.QHStates([q_1, q_0])
d = qt.QHStates([q_0, q_1])

sqrt_2op = qt.QHStates([qt.QH([sp.sqrt(1/2), 0, 0, 0])])

u2 = u.Euclidean_product('ket', operator=sqrt_2op)
d2 = d.Euclidean_product('ket', operator=sqrt_2op)

r = u2.add(d2)
L = u2.dif(d2)

u.print_states("u", 1)
d.print_states("d", 1)

r.print_states("r", 1)
L.print_states("L")


# ![](images/lecture_3/c3_p72_q2.jpg)

# This is just a simple math fact. Orthonormal quaternions series are orthonormal. Non-orthonormal are non-orthonormal.

# In[21]:


u.Euclidean_product("bra", ket=d).print_states("<u|d> - orthonormal", 1)
u.Euclidean_product("bra", ket=r).print_states("<u|r> - not orthonormal")


# ![](images/lecture_3/c3_p73_q1.jpg)

# Every so often, it is OK to worry about quaternion series quantum mechanics because as is well-known, in general, quaternions do not commute. Real numbers do commute, and are so much easier to work with. Much of what goes on in quantum mechanics is to construct something that necessarily evaluates to a real value. When one recalls a big goal is plucking out real numbers, then the "flaw" of quaternion series starts to sound manageable.

# ![](images/lecture_3/c3_p73_q2.jpg)

# In fact, the Euclidean product of two quaternion series is just another quaternion series. Getting a real number out of the system is a special thing. If the only in the above expression were to interchange an $A4$ with a $\lambda_i$, then it breaks: 
# $$ <A|\lambda_i><A|\lambda_i> \ne P(\lambda_i)$$ 
# Instead this is the square of a probability amplitude. Note, as the square, the first term will be invariant under a Lorentz transformation, and the other three will be Lorentz variant. Small changes have big consequences.

# ![](images/lecture_3/c3_p74_q1.jpg)

# This is "fingernails on the blackboard" writing to me. Nature works with 3D space-time. A space without a time I believe is technical gibberish. It is impressively common gibberish, for any measure is space happened at a time, and there never was an exception to than, nor will there ever be an exception to the necessity of time for spatial measurements. Three-vectors are poor, numbers are rich beyond measure.

# ![](images/lecture_3/c3_p75_q1.jpg)

# This sounds to my ear like an admission that things are muddled between space-time dimensions and state dimensions. These two quality are orthogonal to each other. Everything we can do is in 3D space-time, no exceptions ever. The number of state dimensions can range from one to infinite. The number depends on what exactly is being studied and the details of that study. In the case of spin, there are only two state dimensions. How we decide to study the two  state dimensions in 3D space-time is up to us, and yes, we have many choices there. Once a choice for how the measurement is to be made in 3D space-time, then we are studying a two state dimension system.

# ![](images/lecture_3/c3_p76_q1.jpg)

# This is a math problem. Principle 3 has already been demonstrated by direct calculation. The only work is to find an operator $\sigma_z$ that has an Eigenvalue of +1 for ket $|u>$ and -1 for ket $|d>$. Since those kets are both real, it is about the easy operator to guess.

# In[22]:


σz = qt.QHStates([q_1, q_0, q_0, q_n1])
σz.print_states("σz operator", 1)
u.product("ket", operator=σz).print_states("σz|u>", 1)
d.product("ket", operator=σz).print_states("σz|d>")


# Bingo, bingo.

# ![](images/lecture_3/c3_p76_q2.jpg)
# ![](images/lecture_3/c3_p77_q1.jpg)

# Oops, this was just done. But why should the result be unique? Quaternions are numbers. If two are multiplied together like the operator sigma and the kets u and d, then the result is unique. This is a defining property of number and of number series.

# ![](images/lecture_3/c3_p78_q1.jpg)

# ![](images/lecture_3/c3_p79_1.jpg)

# Define the operator $\sigma_x$ and run the calculations.

# In[23]:


σx = qt.QHStates([q_0, q_1, q_1, q_0])
σx.print_states("σx operator", 1)
r.product("ket", operator=σx).print_states("σx|r>", 1)
L.product("ket", operator=σx).print_states("σx|L>")


# ![](images/lecture_3/c3_p80_q1.jpg)

# Detail was not provided for this one in the book because the factors of $i$ are just going to make the case look more confusing.

# In[24]:


one_root_two = sp.sqrt(1/2)
q_2 = qt.QHStates( [ qt.QH([sp.sqrt(1/2), 0, 0, 0]) ] )
q_2i = qt.QHStates([qt.QH([0, sp.sqrt(1/2), 0, 0])])

i = u.product("ket", operator=q_2).add(d.product("ket", operator=q_2i))
o = u.product("ket", operator=q_2).dif(d.product("ket", operator=q_2i))

i.print_states("i", 1)
o.print_states("o", 1)


# Show they are orthonormal.

# In[25]:


i.Euclidean_product("bra", ket=i).print_states("<i|i>", 1)
o.Euclidean_product("bra", ket=o).print_states("<o|o>", 1)
i.Euclidean_product("bra", ket=o).print_states("<i|o>")


# Define σy and put it to work.

# In[26]:


σy = qt.QHStates([q_0, q_ni, q_i, q_0])
σy.print_states("σy operator", 1)
i.product("ket", operator=σy).print_states("σy|i>", 1)
o.product("ket", operator=σx).print_states("σy|o>")


# I will need to put some effort into this to make the result look more "obvious". The problem will only become more acute as the terms get more complicated.

# ![](images/lecture_3/c3_p80_p3.jpg)

# I see these three - operators, Eigen-kets, and Eigen-values - as a team. All three are algebraic equals - quaternion series. The smallest as far as its length is the Eigen-ket. The operator and Eigen-value have the Eigen-ket dimension squared. The Eigen-value operator is mostly zeros, with only one term not equal to zero and then always the same, making it quite simple.

# ![](images/lecture_3/c3_p80_q2.jpg)

# The Pauli matrices plus the identity matrix does not make a representation of the quaternions. An essential aspect of quaternions is they form a division algebra. That is not the case for the Pauli matrices with an identity. There is another factor of $i$ that makes much of the math simpler to do, but also breaks the reversibility of multiplication. Why should anyone care if multiplication is always reversible? Please read the first chapter of the first book of this series which explained why physical processes are reversible. Make the physical abstract, and the same should apply to to mathematical processes used to describe the physical ones.

# ![](images/lecture_3/c3_p83_q1.jpg)

# I do not feel this way. Spin is a system that can be completely described by two quantum variables. There are an arbitrary number of different ways one can do that exercise using an n=2 quaternion series. In this chapter, we came up with three, but could have done thirteen. With the choice of three representations, we stabled on the labels "x", "y", and "z". Spatial dimensions have relationships with each other, namely the cross product. One can take advantage of the orthonormal relationship between x, y, and z to do the same for the three sigmas.
# 
# I never use 3-vectors, so it is not my cup of tea.

# There are two state dimensions for spin. There are as always four dimensions for space-time. Since the operator must have a norm of one, that means there are three degrees of freedom in space-time. We can thus come up with three independent representations of spin and embed them in a normalized space-time operator.

# ![](images/lecture_3/c3_p84_q1.jpg)

#  I have tried to construct a general as possible operator.

# In[27]:


def sigma(kind, theta=None, phi=None):
        """Returns a sigma when given a type like, x, y, z, xy, xz, yz, xyz, with optional angles theta and phi."""
        
        q0, q1, qi = qt.QH().q_0(), qt.QH().q_1(), qt.QH().q_i()
        
        # Should work if given angles or not.
        if theta is None:
            sin_theta = 1
            cos_theta = 1
        else:
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            
        if phi is None:
            sin_phi = 1
            cos_phi = 1
        else:
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)
            
        x_factor = q1.product(qt.QH([sin_theta * cos_phi, 0, 0, 0]))
        y_factor = qi.product(qt.QH([sin_theta * sin_phi, 0, 0, 0]))
        z_factor = q1.product(qt.QH([cos_theta, 0, 0, 0]))

        # Create all possibilities.
        sigmas = {}
        sigmas['x'] = qt.QHStates([q0, x_factor, x_factor, q0]).normalize()
        sigmas['y'] = qt.QHStates([q0, y_factor, y_factor.flip_signs(), q0]).normalize() 
        sigmas['z'] = qt.QHStates([z_factor, q0, q0, z_factor.flip_signs()]).normalize()
  
        sigmas['xy'] = sigmas['x'].add(sigmas['y']).normalize()
        sigmas['xz'] = sigmas['x'].add(sigmas['z']).normalize()
        
        sigmas['xyz'] = sigmas['xy'].add(sigmas['z']).normalize()

        if kind not in sigmas:
            print("Oops, I only know about x, y, z, and their combinations.")
            return None
        
        return sigmas[kind]


# See if the function creates the three sigmas discussed so far:

# In[37]:


sigma('x').print_states('σx', 1)
sigma('y').print_states('σy', 1)
sigma('z').print_states('σz', 1)
sigma('z').norm_squared().print_states("σz's norm", 1)


# In[30]:


sigma('xy').norm_squared().print_states('σxy', 1)
sigma('xy').normalize().norm_squared().print_states('σxy', 1)


# It is important not to lose sight about what is going on here. There is a nice picture of a sphere at the end of the lecture:

# ![](images/lecture_3/c3_p89_q1.jpg)

# It is easy of a mind to take the easy road and think spin has something to do with this sphere and spherical coordinates, or x, y, and z for that matter. Spin has two states, that is all it about. Deciding which one to use for $\sigma_x$, $\sigma_y$, and $\sigma_z$ was arbitrary. In fact, the representation for $\sigma_x$ and $\sigma_z$ only involved time-like numbers. Oddly enough it was $\sigma_y$ that involed $x$. One could have easily used $j$ which is "y-like", but that deeply does not matter. Everyone does this calculation since it makes the two states of spin feel like they are part of space-time. With quaternion series quantum mechanics, that need is less needed since space-time is always there.
