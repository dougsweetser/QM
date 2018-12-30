
# coding: utf-8

# # Legal Moves

# This notebook is being created to address an observation of Purple Penguin which makes the math of quaternion series appear inconsistent. No math is ever completely free. I hope to show with reasonable constraints, quaternion series are not internally inconsistent.
# 
# Load the needed libraries.

# In[1]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport matplotlib.pyplot as plt\nimport math\n\n# To get equations the look like, well, equations, use the following.\nfrom sympy.interactive import printing\nprinting.init_printing(use_latex=True)\nfrom IPython.display import display\n\n# Tools for manipulating quaternions.\nimport Q_tools as qt;\n\nfrom IPython.core.display import display, HTML, Math, Latex\ndisplay(HTML("<style>.container { width:100% !important; }</style>"))')


# There are two types of playeres in this notebook: quaternions Q (upper case) and quaternion series qs (lower case). Purple Penguin wrote down the following pair of Quaternions and quaternion series:

# In[2]:


q_0 = qt.QH().q_0()
q_1 = qt.QH().q_1()
q_i = qt.QH().q_i()
q_ni = q_i.flip_signs()

A = q_1
B = q_i
a = qt.QHStates([q_1, q_0])
b = qt.QHStates([q_0, q_i])

A.print_state("Quaternion A: ", 1)
B.print_state("Quaternion B: ", 1)
a.print_state("quaternion series a: ", 1)
b.print_state("quaternion series b: ")


# Notice that the sum of the states in quaternion series a equals the Quaternion A, while the sum of quaternion series b equals Quaternion B.
# 
# Purple Penguin then made the following observation of about the products of the quaternions and quaternion series:

# In[3]:


AB = A.Euclidean_product(B)
ab = a.Euclidean_product("bra", ket=b)

AB.print_state("Quaternion Euclidean product AB: ", 1)
ab.print_state("Quaternion dot product ab: ")


# Since these are clearly different, the math of quaternion series is not logically consistent according to Purple Penguin.

# There are two group operations one can do on quaternions and quaternion series: addition and multiplication. Purple penguin imposed a constraint on addition, namely that the sum of of the states in the quaternion series must be equal to the quaternion. There was no similar constraint on multiplication. It fact, it is easy to impose a reasonable constraint, namely that $AB=<a|b>$.
# 
# How would this be done in practice? What would be the analog in more traditional vector spaces? In vector spaces one has the basis vector and a real or complex number that is the magnitude. One cannot add vectors that have different basis vectors. Only if the vectors have exactly the same basis is one allowed to add them together.
# 
# I can see two ways to fix this issue. The first way is to start with quaternion series b that works for both addition and multiplication.

# In[4]:


b_ok = qt.QHStates([q_i, q_0])
ab_ok = a.product("bra", ket=b_ok)

ab_ok.print_state("<a|b_ok>")
print("Does AB=<a|b>?: ", AB.equals(ab_ok))


# A more interesting fix is to use a spin matrix to rotate the quaternion series b to look exactly like the quaternion series a, analogous to choosing the same basis vector.

# In[5]:


sigma_4_b = qt.QHStates([q_0, q_ni, q_0, q_0])
b_rotated = b.Euclidean_product("ket", operator=sigma_4_b)

sigma_4_b.print_state("sigma_4_b:", 1)
b_rotated.print_state("b_rotated:")


# Now the constraint on addition is violated, so fix that with another operator:

# In[6]:


b_rotated_i = b_rotated.product("ket", operator=qt.QHStates([q_i]))

b_rotated_i.print_state("b_rotated_i", 1)
print("Is b_ok = b_rotated_i?: ", b_ok.equals(b_rotated_i))


# Since the two operators sigma_a and i acting on quaternion series b recreate the quaternion series b_ok which passed the multiplication criteria, this is good.

# Confirm that one could instead decide to work with quaternion series a to pass the multiplication criteria.

# In[7]:


a_ok = qt.QHStates([q_0, q_1])
a_okb = a_ok.Euclidean_product("bra", ket=b)

a_ok.print_state("quaternion series a_ok", 1)
a_okb.print_state("<a_ok|b>")
print("AB=<a_ok|b>?: ", AB.equals(a_okb))


# In[8]:


sigma_4_a = qt.QHStates([q_0, q_1, q_0, q_0])
a_rotated = a.Euclidean_product("bra", operator=sigma_4_a)
a_rotatedb = a_rotated.Euclidean_product("bra", ket=b)

a_rotated.print_state("a_rotated", 1)
a_rotatedb.print_state("<a_rotated|b>", 1)
print("AB=<a_rotated|b>?: ", AB.equals(a_rotatedb))


# To create a large collection of quaternions and a faithful collection of quaternion series would be a tricky affair. The simple part is being consistent with the addition rule. The multiplication constraint would for large numbers might become impossible to figure out. The reason is the constrain requires an examination of every pair of numbers to assure that $AB=<a|b>$.
# 
# If the question was asked in reverse, if you had a big pile of quaternion series, would it be possible to figure out the relevant quaternions? Sure, use the addition rule. Let's test if this is the case with three quaternion series with three state dimensions.

# In[9]:


random_1 = qt.QHStates([qt.QH().q_random(), qt.QH().q_random(), qt.QH().q_random()])
random_2 = qt.QHStates([qt.QH().q_random(), qt.QH().q_random(), qt.QH().q_random()])
random_3 = qt.QHStates([qt.QH().q_random(), qt.QH().q_random(), qt.QH().q_random()])
R_1 = random_1.summation()
R_2 = random_2.summation()
R_3 = random_3.summation()

random_1.print_state("random quaternion series 1: ", 1)
random_2.print_state("random quaternion series 2: ", 1)
random_3.print_state("random quaternion series 3: ", 1)
R_1.print_state("random quaternion series sum 1: ", 1)
R_2.print_state("random quaternion series sum 2: ", 1)
R_3.print_state("random quaternion series sum 3: ", )


# Check if the products are all equal.

# In[10]:


R_1R_2 = R_1.Euclidean_product(R_2)
random_1random_2 = random_1.Euclidean_product("bra", ket=random_2)
R_1R_2.print_state("R_1R_2", 1)
random_1random_2.print_state("random_1random_2")
print("R_1 R_2=<random_1|random_2>?: ", R_1R_2.equals(random_1random_2))


# That hypothesis could not have been more wrong, interesting.

# Here's my take home message about the relationship between Quaternions and quaternion series: it's complicated. Nature can be like that.

# ## Where to Start?

# I want to share something I call a scientific belief. The "belief" part means I take things grounded in the science and I form a specific speculation I cannot support at this time, and it all likelihood, will never be able to support (although it should be possible to imagine such a day). My belief is that with the right mathematical transformations in place, one can create a path between any and all known 4-vectors.

# Let's talk about the first steps on this mathematical unification path. An observer is at now, time zero, and is the center of their own universe, so spatial location zero - no matter what the choice of coordinates. One can do the math on a quaternion manifold $\mathbb{H}^1$ so this is a single number, zero, $0$. This is the hardest path to go down because everything has to be done with automorphisms: addition, subtraction (and their inverses), and the three conjugates ($q^{*}$, $q^{*1} \equiv (iqi)^{*}$, and $q^{*2} \equiv (jqj)^{*} $). The great attraction of such a construct is the unity of every player on the stage. I don't have a quaternion class written to do this. It would be an interesting exercise in abstraction.

# A second approach would be to use what I call space-time numbers. This is a division algebra based on the quaternion group $Q_8$. Space-time numbers should not be considered a new division algebra since there is an equivalence class from space-time numbers to quaternion over $\mathbb{R}^4$. Space-time numbers use the set of positive real numbers plus zero. I do have the class Q8 to work with space-time numbers. I don't do so because it adds complications that only might become of value in relativistic quantum mechanics, a rather distant goal.

# Hamilton's quaternions over the mathematical field of real numbers, $\mathbb{R}^4$, is what I use in practice due to the simplicity of mapping it to one dimension for time and three for space. I choose to start with an event in space-time. I use natural units so that I can work with pure numbers. Natural units take a measurement in time and divide it by the the Planck time. Measurements of distance are divided by the Planck length. This is done consistently, so if one wants to take a time derivative, that must be multiplied by the Planck time so the result remains dimensionless.

# There will always be an observer, which for quaternions will be $o = (0, 0, 0, 0)$. An event will be a quaternion with a non-zero norm, $p = (t, x, y, z)$. The starting place for physics is an observer with the zero quaternion and a non-zero event. I know there are already a huge number of questions one could ask: how does one figure out the value of that non-zero measurement? I do through my hands up in the air at such questions.

# One event is not enough to do any physics. Physics is the science of understanding patterns of events. Once there is a collection of events, then changes in events can be assessed using subtraction, $dp = (dt, dx, dy, dz)$. This has the advantage of subtracting away the origin and thus being all about the object under study.

# The next leap is go from a collection of events in space-time to energy and momentum using Fourier transforms. I really don't get the math here. I recall in quantum mechanics how one can use either a space-time representation or en energy-momentum representation. Fourier transformations are used to convert between the two. It is easy enough to find examples focused on time and energy, or location and momentum. I bet their are examples of space-time and momenergy (a Wheelerism for energy-momentum), but they are harder to find. It is a belief of mine that there is a purely algebraic bridge that can be constructed between events in space-time and momenergy.

# Take as an example a rock in the vacuum of space moving a constant velocity through space-time. That would create an unending pattern of events. Calculate the differences betwee events. In momenergy space, there would be a point for this pattern with a constant energy and momentum. Neither would be right or wrong, just different ways of looking at the same information.

# No matter what sort of transformation happens, the result remains a number subject to additional manipulation. The entire momenergy space could be squared. The result would be $(m^2, ~2 ~E P)$. I don't have a name for it, and suspect it rarely appears in the literature. 

# I can imagine quaternion constructions that always, necessarily have both $U(1)$ and $SU(2)$ symmetry. Will such construct ever have a relationship to electroweak symmetry? All it takes with quaternions is to insist the norm equals unity. As a famous physicist once said of a different idea of mine, "that sounds vague enough to be true." I accept that it will be easily be brushed aside. The reason I cling to this thinnest sheet of ice is that any and all proposals for unifying gravity with the other forces of Nature has to at least address the fundamental gauge symmetries of the standard model. 

# ## Players in Quantum Mechanics

# Stephen Alder has worked on quaternion quantum mechanics (not quaternion series quantum mechanics). He cites a paper in the 30s by Birkhoff and von Neumann that claims quantum mechanics can only be done with a vector field over $\mathbb{R}$, $\mathbb{C}$, $\mathbb{H}$, or $\mathbb{O}$. Alder argues that the real numbers get eliminated because they cannot do quantum interference with the ease and grace available to complex-valued vectors. He has written a book that tried to use quaternions extensively in quantum mechanics. He did not try to make them 100% quaternions through and through.

# ## Managing Non-commuting Quaternions

# There is a literature out there about right-handed and left-handed multiplication. I find such a construct to not be elegant. I thought about this problem yet again in Lecture 3. There they had a matrix $M$ and a complex number $z$, so $Mz|A> = zM|A>$. This is not true for quaternion series quantum mechanics. Here is what I wrote about the issue in Lecture 3: 

# ### Lecture 3

# Careful reflection takes time, so please indulge me. The above is an algebraic expression ($Mz|A>$). Each term has been given a name, $M$ and $z$, and been assigned a specific order at the table. A quaternion product can always be broken into two parts: the even part that commutes, and the odd part or cross product that anti-commutes.
# 
# First define a bunch of quaternions.

# In[11]:


At1, Ax1, Ay1, Az1 = sp.symbols("At1 Ax1 Ay1 Az1")
At2, Ax2, Ay2, Az2 = sp.symbols("At2 Ax2 Ay2 Az2")
Aq1 = qt.QH([At1, Ax1, Ay1, Az1], qtype="a₁")
Aq2 = qt.QH([At2, Ax2, Ay2, Az2], qtype="a₂")
A = qt.QHStates([Aq1, Aq2])
A.print_state("A", 1)

Mt1, Mx1, My1, Mz1 = sp.symbols("Mt1 Mx1 My1 Mz1")
Mt2, Mx2, My2, Mz2 = sp.symbols("Mt2 Mx2 My2 Mz2")
Mt3, Mx3, My3, Mz3 = sp.symbols("Mt3 Mx3 My3 Mz3")
Mt4, Mx4, My4, Mz4 = sp.symbols("Mt4 Mx4 My4 Mz4")
Mq1 = qt.QH([Mt1, Mx1, My1, Mz1], qtype="m₁")
Mq2 = qt.QH([Mt2, Mx2, My2, Mz2], qtype="m₂")
Mq3 = qt.QH([Mt3, Mx3, My3, Mz3], qtype="m₃")
Mq4 = qt.QH([Mt4, Mx4, My4, Mz4], qtype="m₄")

M = qt.QHStates([Mq1, Mq2, Mq3, Mq4])
M.print_state("M", 1)

zt, zx, zy, zz = sp.symbols("zt zx zy zz")
zq = qt.QH([zt, zx, zy, zz], qtype="z")
zqs = qt.QHStates([zq])
z_op = zqs.diagonal(2)
z_op.print_state("z")


# Before trying to understand quaternion series, let's just look at the product of two quaternions. We wish to have an algebraic rule about what **exactly** it means to reverse two named symbols. We can create a rule such that reversing symbols does not change the result. Thus when one writes a reversal of symbols, we mean that the product of the two reverse quaternions is the difference between the even and odd products. This makes no difference for real and complex numbers since the odd part is always exactly zero. For quaternions, this flips the signs of the thing that flip signs under reversal, so no net change results.

# In[12]:


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
# I would have to consult the literature to find out if this was called a left- or right-derivative. Whatever it is called does not matter with this new rule for reversal in place. Yes, the differential can now be moved to the left so long as one uses the reverse product:
# $$\lim_{dq \rightarrow 0} (f(q + dq) - f(q)) \; x \; dq^{-1} = \lim_{dq \rightarrow 0} dq^{-1} \; xR \;(f(q + dq) - f(q)) $$
# Getting one consistent definition for a quaternion derivative may be just as important as anything else in this collection of work, so I think it was worth the digression.

# Back to the question at hand, namely that $Mz|A>=zM|A>$. Notice the $Mz$ and $zM$ can be viewed a product of operators, a subject that has yet to be discussed. This is because $z$ is viewed as "just a number", despite its location in the bracket notation.

# In[13]:


Mq1z = Mq1.product(zq)
Mq2z = Mq2.product(zq)
Mq3z = Mq3.product(zq)
Mq4z = Mq4.product(zq)
Mz = qt.QHStates([Mq1z, Mq2z, Mq3z, Mq4z])
Mz.print_state("Mz", 1)

zMq1 = zq.product(Mq1, reverse=True)
zMq2 = zq.product(Mq2, reverse=True)
zMq3 = zq.product(Mq3, reverse=True)
zMq4 = zq.product(Mq4, reverse=True)
zM = qt.QHStates([zMq1, zMq2, zMq3, zMq4])
zM.print_state("zM")


# If you stare at these long enough, they do look identical. Prove it.

# In[14]:


MzA = A.product("ket", operator=Mz)
zMA = A.product("ket", operator=zM)
print(MzA.dif(zMA))


# Again, the qtypes indicated that different roads were taken to get to the same result.

# ## Bras, Operators, and Kets

# Order matters for quaternion multiplication. Bra comes first, then comes operators, then comes a ket. For quaternion series quantum mechanics, the bra quaternion series gets conjugated. Each of these has the same data structure: an ordered array of quaternions. The rules of the quaternion series libraries are crafted to create the same output as working with a Hilbert vector space over the complex numbers.
