#!/usr/bin/env python
# coding: utf-8

# # Lecture 4: Time and Change

# Since this is an experiment in progress, I am never sure whether I will find a way to do everything using quaternion series quantum mechanics. Some things will be killer easy: unitarity is just another name for having a norm of one. My biggest worry or mystery is how to handle the imaginary factor $i$. Quaternions have the built in $i$, $j$, and $k$. That presents two different issues: one that there are three, and two they don't commute. One of the deep lessons of quantum mechanics is that many of the calculations that are done don't give a darn about what direction in space a state is pointed in because we will be calculating $A^*A$ which removes direction. Not commuting is required in certain situations. Will have to see if the hurdles ahead can be cleared.
# 
# I did come up with a new idea for managing this issue. It is called the complex number/quaternion correspondence principle. Using quaternion tools and quaternions of the form $(a, b, 0, 0)$, one must necessariy get the standard results for quantum mechanics done on a complex-valued Hilber space. There can be no exceptions here since complex numbers are a subgroup of quaternions. Then the question is how going to a full quaternion will tweak an expression.
# 
# Load the needed libraries.

# In[1]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport matplotlib.pyplot as plt\nimport math\n\n# To get equations the look like, well, equations, use the following.\nfrom sympy.interactive import printing\nprinting.init_printing(use_latex=True)\nfrom IPython.display import display\n\n# Tools for manipulating quaternions.\nimport Q_tools as qt;\n\nfrom IPython.core.display import display, HTML, Math, Latex\ndisplay(HTML("<style>.container { width:100% !important; }</style>"))')


# ![](images/lecture_4/c4_p093_q1.jpg)

# The worst of quaternion series quantum mechanics must be over too! Recall every state dimenion has four space-time dimensions. To my eye, a quaternion series is more "real world", less a formal abstract exercise, because of the space-time dimensions. I also don't struggle with there being a few number of state dimensions or infinitely many because that is how series behave. Yet it is easy to construct quaternion series that get crazy complicated quickly.

# ![](images/lecture_4/c4_p094_q1.jpg)

# The one way I know of not changing the amount of information is by multiplying by unity. If I start with seven things and I want to end up with seven things, it is OK to multiply by unity and nothing else. The story is going to get far more subtle and interesting with space-time dimensions imbedded in state dimensions.

# ![](images/lecture_4/c4_p095_q1.jpg)

# I have an issue with how this is commonly discussed. The authors are following an age-old tradition of taking about time $t$. We have known since the development of special relativity that one inertial observer's time is a different inertial observer's space-time. The way I would discuss this issue is to say the state represents all the information about a system at a space-time $Rt$. I use the dash for words to always link space to time. The state $|\psi(Rt)>$ is the entire history of the system no matter where-when it is examined. In classical quantum mechanics this minor transgression is forgiven. The purpose of my work is to make the language and the math consistent as prossible.

# ![](images/lecture_4/c4_p095_q2.jpg)

# Equation 4.1 puts a constraint on the [space-]time-development operator. Image that the space-time was zero, the left-hand like the right. This could only be so if the [space-]time-development operator was equal to the identity at space-time zero. Deciding which point should be space-time zero is arbitrary. This argues that the norm of the [space-]time-development operator should always be equal to one.

# ![](images/lecture_4/c4_p096_q1.jpg)

# Think about the math. If I have a specific expression for the [space-]time-development operator multiplying a specific state, only one state-vector will result. The rules of quaternion series multiplication are _desterministic_ in this way.

# ![](images/lecture_4/c4_p096_q2.jpg)

# The probabilities are assessed in _places-later_, not just later. Many phenomena have interference pattern where probabilities vary according to location as well as time. Every location-time may have a different probability. The authors of course know this, but following tradition, are using the standard lexicon. The advantage to shifting the words used is that the capacity to discuss interference is built in.  

# ![](images/lecture_4/c4_p097_q1.jpg)

# The only tools built so far are linear operators. Quaternion series are not only vectors, but products of states with other states to create new states is allowed, an operation not allowed with vectors. Mostly that new capacity will not be used, but I want to acknowledge it is there.

# ![](images/lecture_4/c4_p097_q2.jpg)

# ![](images/lecture_4/c4_p097_q3.jpg)

# Herein like the "power of zero under multiplication" as I call it. There are two states whose product is zero because they are orthogonal. This means their norm is zero. Quaternion series is :
# 
# ## sidebar 
# I originally wrote: a normed division algebra. This led to an online discussion with Purple Penguin where he convinced me that quaternion series most definitely are **not a division algebra**. Instead, quaternion series are a semi-group with inverses. What this means in practice is that there are many unity elements instead of exactly one. The number of unity elements is easy to calculate: for $n$ states, there are $2^n$ forms of unity. 
# 
# --end sidebare
# 
# Multipying the states by the [space-]time-development operator will not alter the norm of zero. Thus states that are orthogonal will stay that way and conserve distinctions. One should recall that there are many states. The distribution of values will changes via the operation of the [space-]time-development operator, but the sum will be zero always. The fact that you can see shifts in value yet the sum remains zero is pretty cool. Zeroes don't budge, but they do sway in the series of states.

# ![](images/lecture_4/c4_p099_q1.jpg)

# ![](images/lecture_4/c4_p099_q2.jpg)

# There will be times I get lost in the technical details of Hermitian operators acting on Eigen-states and all that jazz. Just remember we are taking a rich look at what unity can do and what unity cannot do. The norm will stay the same, the states will shift. Don't bundge over all, sway in the series of states. I have used the same lingo for the absense of a signal - zero, orthogonal states - as well as for a signal. Information is preserved as promised.

# ![](images/lecture_4/c4_p100_q1.jpg)

# Make sure the norm is one and built to stay that way up to order epsilon (this is a technical comment - if the norm has an epsilon squared it in, that is OK).

# ![](images/lecture_4/c4_p100_q2.jpg)

# ## Life with an Implied Big $I$
# 
# The quaternion series approach has to take a different approach to get to the same result. In the standard method, the factor $i$ communtes with all. With quaternions, there are three possible imaginary terms, $i$, $j$, and $k$. My nomenclature is to use a capital $I$ to indicate a impaginary term that points who-knows-where. An essential property is big $I$ has a norm of one (which is a little unfortunate since it looks **exactly** like the symbol for an identity matrix). For any particular quaternion, there will be a specific big $I$. The factors of big $I$ for two randomly chosen quaternions probably do not commute. In quantum mechanics, to get to observables one would calculate $I^* I$ which has the same value of unity no matter which way it points.
# 
# At this point, I cannot justify an additional imaginary factor big $I$ or small $i$. Every quaternion necessarily has a factor of big $I$ built in. Because that is a fundamental property of quaternions, it may be unnecessary to add an addition imaginary factor in.
# 
# In this experimental notebook, I will see if I can get away with never writing the implicit big $I$. But this is going to create a 'problem' for me over and over again. I put problem in quotes because it is not a problem, it is a difference in convention. The conventions everyone else uses has imaginary factors of $i$ sprinkled where necessary. It is a good, logically consistent system. What I am doing is making a different choice about the phase. The phase does not matter. Yet my choice leads to a different sign convention. Making things simpler. There are no factors of $i$ because all terms whether potentials, source, states, operators, etc., necessarily have a big $I$.
# 
# What is essential is the epsilon $\epsilon$ for continuity.

# $$U(\epsilon) = 1 - \epsilon H$$

# One thing I have figured out is that whatever 3-position I choose in space, say $(x, y, z) = (1, 2, 3)$, that same big $I$ has to be used (something a multiple of $(1, 2, 3)$).

# ![](images/lecture_4/c4_p100_q3.jpg)

# This algebraically simple expression has a correspondingly simple **physical** interpretation. The identity does nothing to a quantum state, go inertia go. The three imaginaries $i$ point in some direction of space - and have zero for time. The epsilon $\epsilon$ is a real number, so is a tiny amount of time. The produt of these two results in a tiny amount of space. Do the calculation.  

# In[2]:


q_0, q_1, q_n1 = qt.QH().q_0(), qt.QH().q_1(), qt.QH().q_1(-1)
I2 = qt.QHStates().identity(2, operator=True)
i123 = qt.QHStates([qt.QH([0, 1, 2, 3])])
ε = sp.symbols("ε")
εs = qt.QHStates([qt.QH([ε, 0, 0, 0])])
σz = qt.QHStates([q_1, q_0, q_0, q_n1], "op")
IaddH = I2.add(i123.product(εs).product(σz.dagger()))
IdifH = I2.dif(i123.product(εs).product(σz))
IaddHIdifH = IaddH.product(IdifH)
IaddHIdifH.print_state("(I+iεH)(I-iεH)=I")


# The authors should have used an approximately equal sign if one wants to be a stickler for detail. It is understood in the field one keeps only first order terms.

# ** Sidebar: Dimensional analysis
# 
# The identity matrix is just that, a bunch of ones, no units. Every term one uses in quaternion series quantum mechanics must have that property: no units. Use [natural units](https://en.wikipedia.org/wiki/Natural_units). Take a measure of space and divide it by the Planck length. Take a measure of time and divide it by the Planck length. No matter what $H$ is, make it dimensionless.

# ![](images/lecture_4/c4_p101_q1.jpg)

# The idea I always come back to is $<\psi|I|\psi>$ made up of n state quaternions evalutes to one real number, magic! But so do does $<\psi|i \epsilon H|\psi>$. Even though there are many complex numbers in the states in standard quantum mechanics, and quaternions in every spot for quaternion series quantum mechanics, the goal is to find one real number hiden in the haystack.

# ![](images/lecture_4/c4_p101_q2.jpg)

# Physicists do not work with the product space-time, $i \epsilon$. I work on a new approach to gravity that is centered on space-time. Instead, separate the time part on the left from the space part on the right.

# ![](images/lecture_4/c4_p102_q1.jpg)

# This is classical quantum mechanics. In this context, what classical means is non-relativistic. Schr&ouml;dinger's first efforts with the wave function were relativistic, and did not work out with Bohr's model for the hydrogen atom. Time and space had to be treated differently.

# ![](images/lecture_4/c4_p103_q1.jpg)

# I prefer to think of this expression rearranged like so:
# 
# $$ \partial |\Psi> = -\frac{i \textbf{H}}{\hbar} |\Psi> \partial t $$
# 
# Math people will hate this approach. It is still about the study of how the state changes. The units for Planck's constant are energy times time. If the Hamiltonian $\textbf{H}$ has units of energy and the differential time has units of time, the entire expression is dimensionless, a necessary thing to have mathematical freedom. Note the spatial imaginary 3-vector $i$ must be dimensionless, a pointer in space with a norm of one. 
# 
# Note that with standard complex quantum mechanics, every state and this factor points in precisely the same direction, namely $(1, 0, 0)$. As I take baby steps with quaternion series quantum mechanics, the same logic applies. If say each and every state dimension points in the direction $(1, 2, 3)/\sqrt{14}$, then that requires that $i = (1, 2, 3)/\sqrt{14}$. At first glance, that might sound insanely specific, not at all general. You demand more freedom! Your desires are profoundly irrelevant. What matters is math that works, then understand it. The math that works is...standard complex quantum mechanics where every state and the imaginary point in precisely the same direction. No new principle is being developed here (like 'point where you like'). When someone says there is an event in space-time, that necessarily fixes a 3-direction. One has to keep that direction fix to understand the mathematical description of the event.
# 
# Numbers Nature uses when made dimensionless are either crazy small or insanely big. There are an insanely large number of atoms in anything humans can pick up. Quantum mechanics is about what individual atoms can do. Therefore Planck's constant has to be super small to fit all those atoms into tiny places.

# ![](images/lecture_4/c4_p104_q1.jpg)

# So true. Being big and heavy and slow means that we don't actually do very much. By not doing much, we can last almost a hundred years. This is all good news.

# ![](images/lecture_4/c4_p105_q1.jpg)

# For each and every state dimension, there is a chance from zero to one of it being in that state. Add up all the odds for all the states and that is the average value.

# ![](images/lecture_4/c4_p107_q1.jpg)

# It is a requirment that $\textbf{L}$ is an observable so that this expression evaluates to a single real number. The single value can be negative if it describes how an observation is anti-correlated. It is easy to forget about the bra/ket sandwich, focusing only on the meat on the inside. Only the complete sandwitch can be observed.

# ![](images/lecture_4/c4_p108_q1.jpg)

# What I see in these statements is a requirement: **Say NO to Torque**. 

# In[3]:


q23 = qt.QHStates([qt.QH([2, 1, 2, 3]).normalize(1/np.sqrt(2)), qt.QH([3, 1, 2, 3]).normalize(1/np.sqrt(2))])
q23_selfie = qt.QHStates().bracket(q23.bra(), I2, q23)
q23.print_state("|q23>")
q23_selfie.print_state("<q23|q23>")


# In[9]:


q1s = qt.QHStates([qt.QH([0,1,2,3])]).diagonal(2)
q1s.print_state("q1s")
q1s_exp123 = q1s.exp()
q1s_exp123.print_state("exp(q123)")


# In[11]:


exp_q23 = q1s_exp123.product(q23)
exp_q23.print_state("exp_q23")


# In[15]:


exp_q23.bra().print_state("ext bra")
I2.print_state("I2") 
exp_q23.print_state("exp")
q23_exp123_selfie = qt.QHStates().bracket(exp_q23.bra(), I2, exp_q23)
q23_selfie.print_state("<q23 exp(0123)*|exp(0123) q23>")


# ![](images/lecture_4/c4_p109_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p110_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p110_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p111_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p112_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p112_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p113_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p114_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p114_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p114_q3.jpg)

# whatever

# ![](images/lecture_4/c4_p115_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p115_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p115_q3.jpg)

# whatever

# ![](images/lecture_4/c4_p116_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p117_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p117_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p118_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p118_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p119_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p119_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p120_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p120_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p120_q3.jpg)

# whatever

# ![](images/lecture_4/c4_p121_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p121_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p121_q3.jpg)

# whatever

# ![](images/lecture_4/c4_p122_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p123_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p123_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p124_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p125_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p125_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p126_q1.jpg)
