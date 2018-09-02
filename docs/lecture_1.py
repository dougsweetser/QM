
# coding: utf-8

# # Lecture 1: Systems and Experiments Using Quaternion Series

# by Doug Sweetser, email to sweetser@alum.mit.edu

# ## The quaternion series for quantum mechanics hypothesis
# 
# This notebook is being created as a companion to the book "Quantum Mechanics: the Theoretical Minimum" by Susskind and Friedman (QM:TTM for short). Those authors of course never use quaternions as they are a bit player in the crowded field of mathematical tools. Nature has used one accounting system since the beginning of space-time. I will be a jerk in the name of consistency. This leads to a different perspective on what makes an equation quantum mechanical. If a conjugate operator is used, then the expression is about quantum mechanics. It is odd to have such a brief assertion given the complexity of the subject, but that makes the hypothesis fun - and testable by seeing if anything in the book cannot be done with quaternions and their conjugates. Import the tools to work with quaternions in this notebook, the most important one being Q_tools that was tailored for the task at hand.

# In[1]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport matplotlib.pyplot as plt\n\n# To get equations the look like, well, equations, use the following.\nfrom sympy.interactive import printing\nprinting.init_printing(use_latex=True)\nfrom IPython.display import display\n\n# Tools for manipulating quaternions.\nimport Q_tools as qt;\n\nfrom IPython.core.display import display, HTML\ndisplay(HTML("<style>.container { width:100% !important; }</style>"))')


# ![](images/lecture_1/101.02.1.50.jpg)

# I have a vivid memory of learning how to take statements in logic and converting them to simple algebra problems with zero for false and one for true. The converse applies here: by changing the algebra used for all calculations, a consequence is that the logic of quantum mechanics is different from classical physics. 

# ![](images/lecture_1/101.02.2.50.jpg)

# In "Dreams of a Final Theory" by Steven Weinberg, he views quantum physics as a far deeper subject than classical physics. When classical physics is a ball rolling down an inclined plane, it is simple to do. But a piano with all its modes of vibrations of some two hundred strings is not simple. I bring up this note of caution because in my hands anyway, the tools of quantum mechanics will flow through functions that apply to classical physics. The tools sets of classical and quantum physics are distinct and subtly different.

# ![](images/lecture_1/101.03.1.50.jpg)

# This is a sensible caution, but I dream of a day when spin can really be visualized. On that day we can see how they are different and yet related.

# ![](images/lecture_1/101.20.1.50.jpg)

# I remember a lecture in a first year of graduate school quantum mechanics class where the professor derived the uncertainty principle as an outcome of properties of complex numbers. I suspect that every measurable quantity has a conjugate variable, and with that particular variable, the uncertainty principle applies to the pair. It would be odd if there were an exception.

# ![](images/lecture_1/101.21.1.50.jpg)

# Our way of visualizing a ball - it is there going so fast - must not apply to quantum mechanics. There may be a technical way to visualize what happens precisely in quantum mechanics. We do not have the appropriate tools today.

# ![](images/lecture_1/101.22.1.50.jpg)

# Here is my first substantive disagreement, not so much with Susskind as with the traditional explanation as created in the early 1800s by C. Wessel and Argand. This graph can be written in books, published, and studied. It is both useful and good. The common complaint is that there appears to be no difference between the real axis and the imaginary one. Consider the following experiment. Someone draws a point in the complex plane and does either a reflection over the real or imaginary axis. They cover up the label for both axes. In this math thought experiment, there would be no way to distinguish if the reflection had been over the real or the imaginary axis. To an experimental mathematician, this means there is a problem with the definition of real versus imaginary numbers. Most mathematicians project a faith in defined logical precision. If something is define precisely and the logic works, there is no need for discussion and nothing is wrong.
# 
# Space-time physics may provide an animated definition for the difference between real and imaginary numbers. A real number for time can be represented with an animation. The three dimensions of space would be the three imaginary numbers. Make this concrete. Consider a baseball hit from home plate-now that lands in right field-future.

# ![simple.gif](images/lecture_1/baseball_animations/simple.gif)

# The ball starts at here-now, $(0, 0, 0, 0)$. Time is treated as the real number, while the planar graph has the imaginary values of $x i$ and $y j$. I hyphenate space-time and their verbal variations, like the ball landing in right field-future.
# 
# What does a spatial reflection of that baseball look like using a mirror at home plate? There would now be two balls. The distance covered by both balls would be twice the distance but the time would be the same.

# ![simple.gif](images/lecture_1/baseball_animations/space.gif)

# The ball goes from home plate-now to right field-future and left field-future. It takes the same amount of time but twice the distance. The opposite situation happens for time reflection.

# ![time reflection](images/lecture_1/baseball_animations/time.gif)

# The ball goes from right field-past to home plate-now out to right field-future. As with all these looping animations, there is a cyclic boundary condition, a fancy way of saying these animations all repeat.
# 
# There is one last reflection possibility: what if there is a reflection in both space and time?

# ![space-time reflection](images/lecture_1/baseball_animations/space-time.gif)

# The ball starts in left field-past, gets to home plate-now, then continues to right field-future. The two reflections take twice the distance and time as the starting hit to the outfield.

# As animations, the reflection in the real number, time, are distinct from the reflections in imaginary numbers, space, and are visually distinct from the combination. Analytic quaternion animations can be used to describe real world things like the flight of a baseball. **Space is orthogonal to time**. We draw in space but animate in time. As such, what is drawn in space is separate from how we then construct an animation from the drawings.

# ![](images/lecture_1/101.23.1.50.jpg)

# Susskind and Friedman are following the standard play book treating everything as part of a static plane - $r$, $x$, and $y$ are all used for distances. Let's try to shift the thinking to time $t$ for the reals, space $R$ for the imaginary numbers. For complex numbers, there is only one sort of factor of $i$. For quaternions, there is a 3-vector $I$ that can point in an infinite number of ways. For a set of quaternions that all point in the same direction or its opposite, every one of these rules applies as is.

# The polar representation of a quaternion takes as input an amplitude and three "angles" which are ratios of distance to time, making them more like velocities than angles in a plane. These "polar velocities" can take any numerical value, making them different because velocities of material objects are limited to the speed of light. A math construction is more free.
# 
# Here is a demonstration that the product of the polar representation of a quaternion multiplies the Amplitude (the first numbers 2, 3, and 10) and adds the polar velocities (1, 1.5, and 0.5).

# In[2]:


p1 = qt.QH([2, 1, 0, 0], representation="polar")
p2 = qt.QH([3, 1.5, 0, 0], representation="polar")
p3 = qt.QH([10, .5, 0, 0], representation="polar")

print("polar 1: ", p1)
print("polar 2: ", p2)
print("polar 3: ", p3)
print("p1 * p2: ", p1.product(p2))
print("p1*p2*p3 ", p1.product(p2).product(p3))


# The conjugate flips the signs of the polar velocity terms:

# In[3]:


p4 = print(p1.conj())


# If things were done right, the product of a polar with its conjugate is the square of the amplitude.

# In[4]:


print(p1.conj().product(p1))
print(p2.conj().product(p2))
print(p3.conj().product(p3))


# So what happens when there is more than one polar velocity? Think like a mathematical physicist for a moment. One velocity is trivial - everything must point in the same direction. If one now gets two or three velocities in the mix, then one must deal with the confusion of spinning things. The norm does not change going from $2 i$ to $2 i + 3 j + 4k$, only the direction changes. Like all things that spin, the subject gets confusing, and is not germane to this chapter. When I see just a factor of $i$, I see linear physics without rotations.

# ![](images/lecture_1/101.23.2.50.jpg)

# The deeper math idea that is hinted at is that we are all taught about representing complex numbers on the manifold $\mathbb{R}^2$. Much of complex analysis relies on working with complex numbers on the manifold $\mathbb{C}^1$. Everything that can be done on the manifold $\mathbb{R}^2$ can also be done on $\mathbb{C}^1$ so long as one has both $z$ and $z^*$. When I learned this lesson, it immediately sounded like a problem for quaternions. This time there will be functions on $\mathbb{R}^4$ that cannot be constructed using just $q$ and $q^*$. I considered this a show stopper - at least until I found a cute solution. I call them the first and second conjugates. Take a moment to calculate $(i q i)^*$ and $(j q j)^*$. You will see that those flip the signs on all the terms except the $i$ and $j$ respectively. So $q$, $q^*$, $q^{*1}$, and $q^{*2}$ are enough on the quaternion manifold $\mathbb{H}^1$ to cover all possible functions one can construct on $\mathbb{R}^4$. Again, this digression is not germane to the case of just one imaginary factor. I point it out to say more than one can be handled, and it will be darn interesting to see how it plays out technically.

# ![](images/lecture_1/101.24.1.50.jpg)

# The two most important numbers to understand in mathematical physics are zero and one. The duo show up in every context, so their meaning varies by that context. For phase factor terms, just add them up.

# In[5]:


phase_a = qt.QH([1, 0.5, 0, 0], representation="polar")
phase_b = qt.QH([1, 1, 0, 0], representation="polar")
print("phase a:     ", phase_a)
print("phase a+a:   ", phase_a.product(phase_a))
print("phase a+a+b: ", phase_a.product(phase_a).product(phase_b))


# Darn rounding errors. Anyway, you should get the point. When the amplitude is unity, the only thing of interest is the phase.

# ![](images/lecture_1/101.24.2.50.jpg)

# I am not going to say something is *wrong* in QM:TTM. I will contend **it is just not right enough**. Yup, the space of states is not a mathematical set. Agreed, 100%. A vector in a vector space can be added to another vector an multiplied by a scalar value. Agreed, 100%. Yet I want to grant states more power and versatility (addition, addition's inverse subtraction, and multiplication by a scalar). The additional power is to drop the "by a scalar" clause, and put in is place "and multiplication's inverse division." At one level, it does not sound like a big deal. Take the number 6, we can write four expressions using the two operators and their inverses: $2 + 4 = 6$, $6 + (-4) = 2$, $2 * 3 = 6$, $6 * 3^{-1} = 2$. This is third grade math, so the demand is to be consistent with all that math learned long ago.
# 
# On the other hand, this sounds like a huge pain. Quantum mechanics is constructed from complex-valued Hilbert vectors spaces. The entire thing would have to be rebuilt. Everything would have to work, no exceptions. Well, that is why there are these iPython notebooks, to test if the reconstruction work can be done.

# ## Space-time dimensions versus State dimensions

# ![](images/lecture_1/101.25.1.50.jpg)

# As a practice, I never think of space or of time, only space-time. An event in space-time is a number with a real time and three imaginary numbers for space that together form a quaternion. Yet one is free to represent a quaternion as a series of quaternions, using one or an infinite number of quaternions in the series. Each and every element of the series will be in space-time. This is an often unspoken aspect of quantum mechanical state spaces: they all exist in space-time. I think of the space-time dimensions as being orthogonal to the state dimensions. There is a relationship between the two, yet there is also an independence. I will try to always use qualifiers for space-time dimensions or state dimensions. For this book, nearly all the discussion will be about the state dimensions.
# 
# Just to get the toes a little wet, here is one quaternion series ket $|A>$ which has two states $a1$ and $a2$:

# In[6]:


A = qt.QHStates([qt.QH([0,1,2,3], qtype="a1"), qt.QH([2,2,2,2], qtype="a2")], qs_type="ket")
A.print_state("|A>")


#     Notation note:
# 
#     A quaternion series is **not** a quaternion per se. 
#     A quaternion series is totally ordered, with each 
#     quaternion having one and only one position in a 
#     linked list or array. To make a consistent system 
#     of multiplicaiton, each series is assigned a pair of 
#     integers, commonly called rows and columns. If both rows 
#     and columns equals 1, then the series is called a scalar. 
#     If only the columns is 1, then the series is called a ket. 
#     If only the rows is equal to 1, then it is called a ket. 
#     All other cases are called operators.
#     
#     When the QHStates series is created, one can declare
#     "scalar", "bra", "ket", "op"/"operator" with the default
#     being "ket". It is often optional to set the values for
#     rows and columns, although required for operators that
#     are not square.

# Each quaternion in the series will be associated with a value of $n$ that can go from 1 out to infinity. The sum of a series is often a key to the analysis.

# ![](images/lecture_1/101.25.2.50.jpg)

# The game that is afoot is to show that quaternion series which can be finite or infinite behave exactly as one demands by the definition of a Hilbert space. Creativity is required to make this work, but that also makes it fun.

# ![](images/lecture_1/101.25.3.50.jpg)

# This looks so darn obvious, there is nothing much to do. Let's write down a quaternion series $B$ and add them up.

# In[7]:


B = qt.QHStates([qt.QH([-1, -1, -1, -1]), qt.QH([-2,-2,-2,-2]), qt.QH([-3,-3,-3,-3])])
A.add(B)


# When you write software that has to "do all the math", you get confronted with questions like, what if the dimensions are different? I think they should not be added together because a value of zero is different from being undefined. Of course, some one with more technical skills in these arts may already know that I should treat the phantom third element of the quaternion series $A$ as zero and it is all OK. That would require a skilled math nerd. For now, I will treat it as an illegal move and report it. 
# 
# Note: after writing this, a Google search showed that my gut instinct was correct: one cannot add vectors of different lengths.

# Let me redefine $|B>$ to have just two terms, and define a third series $<C|$ .

# In[8]:


B = qt.QHStates([qt.QH([-1, -1, -1, -1], qtype="b1"), qt.QH([-2,-2,-4,-6], qtype="b2")], qs_type="ket")
C = qt.QHStates([qt.QH([0, 0, 0, 4], qtype="c1"), qt.QH([0,0,0,-10], qtype="c2")], qs_type="bra")


# In[9]:


A.add(B).print_state("1. Closure: |A> + |B> = |New>\n")


# So long as neither A nor B is zero, the addition results in a new quaternion series.

# ![](images/lecture_1/101.26.1.50.jpg)

# Whoops, a lot to prove, but nothing too difficult. Instead of a complex number $z$ and $w$, I will use quaternions $q$ and $w$.

# In[10]:


print("2. Addition is commutative, |A> + |B> = |B> + |A>.\n")
A.add(B).print_state("|A> + |B>")
B.add(A).print_state("|B> + |A>")
print("|A> + |B> = |B> + |A>? ", A.add(B).equals(B.add(A)))


# I often focus on the qtype, the strings that report all the operations done for a given calculation. Sometimes the qtype can get quite long, but that is a statement about the calculation in question.

# In[11]:


print("3. Addition is associative, (|A> + |B>) + |C> = |A> + (|B> + |C>).\n")
A.add(B).add(C).print_state("(|A> + |B>) + |C>")
A.add(B.add(C)).print_state("|A> + (|B> + |C>)")
print("(|A> + |B>) + |C> = |A> + (|B> + |C>)? ", A.add(B).add(C).equals(A.add(B.add(C))))


# In[12]:


print("4. An additive identity series zero exists.\n")
Z = qt.QHStates().identity(dim=2, additive=True)
Z.print_state("|Z>")
A.add(Z).print_state("|A> + |Z>")
print("|A> = |A> + |Z>?", A.equals(A.add(Z)))


# In[13]:


print("5. An additive inverse exists, A + (-A) = 0.\n")
A.inverse(additive=True).print_state("-A")
A.add(A.inverse(additive=True)).print_state("A + (-A)")


# In[14]:


print("6. A scalar times a ket produces a new ket, q|A> = |New>")
q = qt.QHStates([qt.QH([4, 3, 2, 1], qtype="q")])
q.print_state("q", 1)
q.product(A).print_state("q|A> = |New>")


# What is going on with the *product()* function? The product function can treat the quaternion series as either a bra or a ket. An operator acts on the ket. If the state dimension of the ket is one, then it creates a diagonal series that is the square of the state dimensions of ket. The ket has 2 state dimensions, so the operator has 4 state dimensions. The zeros of the diagonalized operator appear in the qtype.

# In[15]:


print("7a. Distributive property, q(|A> + |B>) = q|A> + q|B>\n")
qAB = q.product(A.add(B))
qAB.print_state("q(|A> + |B>)")
qAqB = q.product(A).add(q.product(B))
qAqB.print_state("q|A> + q|B>")
print("q(|A> + |B>) = q|A> + q|B>? ", qAB.equals(qAqB))


# In[16]:


w = qt.QHStates([qt.QH([1, -1, 1, 100])])
print("7b. Distributive property, (q + w)|A> = q|A> + w|A>\n")
qwA = q.add(w).product(A)
qwA.print_state("(q + w)|A>)")
qAwA = q.product(A).add(w.product(A))
qAwA.print_state("q|A> + w|A>")
print("(q + w)|A> = q|A> + w|A>? ", qwA.equals(qAwA))


# Thus 7 basic properties of state vectors can be replicated using quaternion series. I could say there was an equivalence relationship between the two. That does not mean they are exactly the same in all detail, but that for the operations done so far, both approaches are equivalent.

# ## BIG Sidebar: Multiplication as a Group Operation Experiment

# For division algebras, both addition and multiplicaiton are group operations. That means there is closure (1 from above), associative (3), an identity exists (4) and and inverse exists (5). The real and complex numbers commute under both addition and multiplication, not so quaternion products.
# 
# For a vector space V, {V, +} is a group due to the work shown above, specifically 1, 3-5. No effort is expended to show the same is true for multiplicaiton because while a ket of exactly the same length can be added to another ket to generate a third ket, the same is not true by the rules of matrix multiplication. The number of rows never equal the number of columns, case closed.
# 
# For quaternion series, the rules for multication are modified in a specific way. If one has a ket of the same dimension, then to form a sensible product, the ket on the left is diagonalized first to form a square matrix. At this point, the rules of matrix multiplication allow a product to be formed.
# 
# Let's make this concrete. Consider a real-valued vector with three state dimensions, $|1 \;2 \;3>$. If we squared each term individual, $|k^2>$, the result is $|1\; 4\; 9>$. If the left ket is made into a diagonal operator with a state dimension of 3 rows by 3 columns, one can form a product with the ket with three rows and a column, so $\rm{diagonal}(k) \times |k> = |1\; 4\; 9>$. Recall how addition of two vectors happens if and only if they are of the same dimensions. This is also the case for multiplicaiton. A similar story is possible for bras, but this time it is the left bra that gets made into a diagonal operator. The result of the multiplaciton is other bra quaternion series of the same dimension.
# 
# Now an experiment can be done. Take the 7 things shown to hold true for the additon operator, and see if they hold for multiplication of quaternion series.

# In[17]:


A.product(B).print_state("1x. Closure |A> x |B> = |New>\n")


# In[18]:


print("2x. Multiplication is NOT commutative, |A> x |B> != |B> x |A>.\n")
A.product(B).print_state("|A> x |B>")
B.product(A).print_state("|B> x |A>")
print("|A> x |B> = |B> x |A>? ", A.product(B).equals(B.product(A)))


# This is a well known property of quaternions. I have argued earlier that one could define a rule that when the order of algebraic symbols is swapped, then the odd part of a quaternion product should be subtracted. One can form a variation on communtivity with this rule in place.

# In[19]:


print("2x'. Order-sensitive multiplication is commutative, |A> x |B> = |B> xR |A>.\n")
A.product(B).print_state("|A> x |B>")
B.product(A, reverse=True).print_state("|B> xR |A>")
print("|A> x |B> = |B> xR |A>? ", A.product(B).equals(B.product(A, reverse=True)))


# One could complain that the xR operator means the order of an expression doesn't matter. With real and complex numbers, that is always the case. A similar story applies with quaternions so long as one uses either the sum of the even and odd products, or the difference. More care is required in the quaternion case, but should be easy enough to manage.

# In[20]:


print("3x. Multiplication is associative, (|A> x |B>) x |C> = |A> x (|B> x |C>).\n")
A.product(B).product(C.ket()).print_state("(|A> x |B>) x |C>")
A.product(B.product(C.ket())).print_state("|A> x (|B> x |C>)")
print("(|A> x |B>) x |C> = |A> x (|B> x |C>)? ", A.product(B).product(C.set_qs_type("ket")).equals(A.product(B.product(C.set_qs_type("ket")))))


# In[21]:


print("4x. An multiplicative identity series identity operator exists.\n")
I = qt.QHStates().identity(dim=2, additive=False, operator=True)
I.print_state("I")
I.product(A).print_state("I x |A>")
print("|A> = I x |A>?", A.equals(I.product(A)))


# In[22]:


print("5x. A multiplicative inverse exists, A A^-1 = 1.\n")
A.inverse(additive=False, operator=True).print_state("A^-1")
A.product(A.inverse(additive=False, operator=True)).print_state("A A^-1")


# Except for a little rounding error, a pair of real-valued ones make up the two state ket. Mutliplying bra and ket quaternion series involves diagonal operators. A ket diagonalized and multiplied by $|1\;1>$ would be the ket for of the diagonal. 

# The results of 6 and 7 involve both addition and multiplication, so don't change.

#     The take home message is that quaternion series form a group with the multiplication operator
#     because multiplication has the same required properties (1x, 3x, 4x, and 5x). All that was
#     needed was a clear rule for multiplying kets with kets (make the first a diagonal operator).

# It should be noted that we have not discovered a new division algebra over the real numbers like the real numbers, complex number or quaternions. Quaternion series have a different structure because of the ordered array of quaternions, and the rows and columns integers used for multiplication only. Part of that structure is that quaternion series behave like a division algebra.

# **End of sidebar on multipying quaternion series.**

# ![](images/lecture_1/101.30.1.50.jpg)

# The world of rows and columns does not port to quaternion series. All series are just that: series. Even operators are quaternion series which have relations based on state dimensions. The conjugate of a series can be calculated, and is a very useful thing.

# In[23]:


A.conj().print_state("A*")


# The signal that one is doing a calculation in quantum mechanics may just be the use of mirrors via the conjugate operator. No wonder the subject is such a struggle to understand!

# ## Conjugates as Magic Mirrors

# ![](images/lecture_1/101.29.2.50.jpg)

# Let's first figure out what exactly is meant by "bra corresponding to [a ket] using complex numbers.

# In[24]:


print("z|Ac> = <Ac|z* ?\n")
Ac = qt.QHStates([qt.QH([1,2,0,0], qtype="b2i"), qt.QH([3,4,0,0], qtype="b4i")], "ket")
zi3 = qt.QHStates([qt.QH([1,3,0,0], qtype="13i")], qs_type="op")
zAc = zi3.product(Ac)
zAc.print_state("z|Ac>")
Acz = Ac.bra().Euclidean_product(zi3.conj())
Acz.print_state("<Ac|z*")
print("z|Ac> = <Ac|z* ?", zAc.equals(Acz))


# Almost, but not quite, off by a conjugate. One more conjugate operation is needed to be equal:

# In[25]:


print("z|Ac> = (<Ac|z*)* ?", zAc.equals(Acz.conj()))


# Now lets do this more precise definition of the relationship for quaternions.

# In[26]:


print("q|A> = (<A|q*)* ?\n")
qA = q.product(A)
qA.print_state("q|A>")
Aq = A.bra().Euclidean_product(q.conj())
Aq.print_state("<A|q*")
print("z|A> = (<A|q*)* ?", qA.equals(Aq.conj()))


# This result surprised me. There are three conjugate operators involved. Any quaternion product can be though of as the sum of an even or commuting product and an odd or anti-commuting product:

# $$\rm{Even}(A, B) = (AB + BA)/2 $$
# $$\rm{Odd}(A, B) = (AB - BA)/2 $$

# When the order of two quaternions is changed, then the product is the *difference* between the even and odd products. In this calculation, not only was the order changed, but one of the terms got conjugated. The conjugate on $A$ times the conjugate on $q$ means the even product flips signs but the odd product does not. Now hit that result with one more conjugate operator. The event product is positive, but the odd is negative, so it will be the difference between the even and odd products. I cannot make the story of three mirrors working together any easier than that.

# ![](images/lecture_1/101.30.3.50.jpg)

# This calculation should feel similar to the one done with ket's. The bra is just a different series that gets conjugated. 

# In[27]:


print("1. linear inner product, <C|(|A> + |B>) = <C|A> + <C|B>\n")
CAB = C.Euclidean_product(A.add(B))
CAB.print_state("<C|(|A> + |B>)", 1)
CABA = C.Euclidean_product(A).add(C.Euclidean_product(B))
CABA.print_state("<C|A> + <C|B>", 1)
print("<C|(|A> + |B>) = <C|A> + <C|B>? ", CAB.equals(CABA))


# ![](images/lecture_1/101.31.1v2.50.jpg)

# Quaternions do not commute, but mirrors can sometimes manage the situation.

# In[28]:


print("2. For inner product, change order AND conjugate, <B|A> = <A|B>*\n")
BdotA = B.bra().Euclidean_product(A)
BdotA.print_state("<B|A>")
AdotBc = A.bra().Euclidean_product(B).conj()
AdotBc.print_state("<A|B>*")
print("<B|A> = <A|B>* ?", BdotA.equals(AdotBc))


# All trained in quantum mechanics know this is true for complex numbers, but it is a minor surprise that the same holds for quaternions without alteration.

# ![](images/lecture_1/101.31.2v2.50.jpg)

# Prove a variation on the linearity of inner products.

# In[29]:


print("Exercise 1.1: A) linear inner products, (<A| + <B|)|C> = <A|C> + <B|C> ?\n")
ABC = A.bra().add(B.bra()).Euclidean_product(C.ket())
ABC.print_state("(<A| + <B|)|C>")
ACBC = A.bra().Euclidean_product(C.ket()).add(B.bra().Euclidean_product(C.ket()))
ACBC.print_state("<A|C> + <B|C>")
print("<C|(|A> + |B>) = <C|A> + <C|B>? ", CAB.equals(CABA))


# Let's show that the inner self-products of A, B, and C evaluate to real numbers.

# In[30]:


print("<q|q> = real scalar?\n")
A.bra().Euclidean_product(A).print_state("<A|A>")
B.norm_squared().print_state("<B|B>")
C.norm_squared().print_state("<C|C>")


# ![](images/lecture_1/101.31.2.50.jpg)

# That is what is going on.

# In[31]:


q_0 = qt.QH().q_0()
q_1 = qt.QH().q_1()
q_i = qt.QH().q_i()
q_j = qt.QH().q_j()
q_k = qt.QH().q_k()

α1, α2, α3, α4, α5  = sp.symbols("α1 α2 α3 α4 α5")
β1, β2, β3, β4, β5  = sp.symbols("β1 β2 β3 β4 β5")


B5 = qt.QHStates([qt.QH([β1,0,0,0]), qt.QH([0,β2,0,0]), qt.QH([0,0,β3,0]), qt.QH([0,0,0,β4]), qt.QH([β5,0,0,0])], qs_type="bra")
A5 = qt.QHStates([qt.QH([α1,0,0,0]), qt.QH([α2,0,0,0]), qt.QH([α3,0,0,0]), qt.QH([α4,0,0,0]), qt.QH([α5,0,0,0])], qs_type="ket")
B5.Euclidean_product(A5).print_state("<B5|A5>")


# ![](images/lecture_1/101.32.2.50.jpg)

# Let's build two quaternion series that are orthogonal. Use only states that are multiplies of i. If we have 4 state dimensions, and half are positive i while the other a minuses, that should cancel out. Here's what I mean:

# In[32]:


Ai = qt.QHStates([q_i, q_i, q_i, q_i]).normalize()
Bi = qt.QHStates([q_i, q_i.conj(), q_i, q_i.conj()]).normalize()
Ai.norm_squared().print_state("<Ai|Ai>")
Bi.norm_squared().print_state("<Bi|Bi>")
Bi.bra().Euclidean_product(Ai).print_state("<Bi|Ai>")


# None of the elements in the series are equal to zero. There should be about an infinite number of ways to make orthogonal states, give or take.

# ![](images/lecture_1/101.34.1.50.jpg)

# The complex numbers $\alpha_i$ have to be made into a diagonal quaternion series so the end result of it acting on the basis series is correct.

# In[33]:


αi = qt.QHStates([qt.QH([1,1,0,0]),qt.QH([2,0,2,0]),qt.QH([3,1,3,0]),qt.QH([4,0,0,4])]).diagonal(4)
αi.print_state("αi")
Asum = αi.product(Ai)
Asum.print_state("|A> = Sum αi|Ai>")


# Good, now just take Euclidean product with <j|.

# In[34]:


Bi.bra().Euclidean_product(αi).product(Ai).print_state("<B|αi|A>")


# I don't think there is much meaning to this calculation. It does show the machinery is in place to do more.

# ![](images/lecture_1/101.34.2.50.jpg)

# The only way this can be is if $|i><i|=1$. We already know $<i|i>=1$. All that has to be done is to change the order, yet they point in the same direction, so changing the order doesn't change a thing. 

# In[35]:


ii_projector = Ai.product(Ai.bra().conj())
ii_projector.print_state("projection operator: |i><i|")


# In[36]:


ii_projector.product(Asum).print_state("|i><i|A")

