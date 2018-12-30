
# coding: utf-8

# # Lecture 4: Time and Change

# Since this is an experiment in progress, I am never sure whether I will find a way to do everything using quaternion series quantum mechanics. Some things will be killer easy: unitarity is just another name for having a norm of one. My biggest worry or mystery is how to handle the imaginary factor $i$. Quaternions have the built in $i$, $j$, and $k$. That presents two different issues: one that there are three, and two they don't commute. One of the deep lessons of quantum mechanics is that many of the calculations that are done don't give a darn about what direction in space a state is pointed in because we will be calculating $A^*A$ which removes direction. Not commuting is required in certain situations. Will have to see if the hurdles ahead can be cleared.
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

# Think about the math. If I have a specific expression for the [space-]time-development operator multiplying a spaceific state, only one state-vector will result. The rules of quaternion series multiplication are _desterministic_ in this way.

# ![](images/lecture_4/c4_p096_q2.jpg)

# The probabilities are assessed in _places-later_, not just later. Many phenomena have interference pattern where probabilities vary according to location as well as time. Every location-time may have a different probability. The authors of course know this, but following tradition, are using the standard lexicon. The advantage to shifting the words used is that the capacity to discuss interference is built in.  

# ![](images/lecture_4/c4_p097_q1.jpg)

# The only tools built so far are linear operators. Quaternion series are not only vectors, but products of states with other states to create new states is allowed, an operation not allowed with vectors. Mostly that new capacity will not be used, but I want to acknowledge it is there.

# ![](images/lecture_4/c4_p097_q2.jpg)

# ![](images/lecture_4/c4_p097_q3.jpg)

# Herein like the "power of zero under multiplication" as I call it. There are two states whose product is zero because they are orthogonal. This means their norm is zero. Quaternion series is a normed division algebra. Multipying the states by the [space-]time-development operator will not alter the norm of zero. Thus states that are orthogonal will stay that way and conserve distinctions. One should recall that there are many states. The distribution of values will changes via the operation of the [space-]time-development operator, but the sum will be zero always. The fact that you can see shifts in value yet the sum remains zero is pretty cool. Zeroes don't budge, but they do sway in the series of states.

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

# gfg

# ![](images/lecture_4/c4_p100_q3.jpg)

# whatever

# ![](images/lecture_4/c4_p101_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p101_q2.jpg)

# whatever

# ![](images/lecture_4/c4_p102_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p103_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p104_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p105_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p107_q1.jpg)

# whatever

# ![](images/lecture_4/c4_p108_q1.jpg)

# whatever

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
