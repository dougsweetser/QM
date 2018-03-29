
# coding: utf-8

# # Lecture 1: Systems and Experiments Using Quaternion Series

# by Doug Sweetser, email to sweetser@alum.mit.edu

# **The quaternion series for quantum mechanics hypthesis**
# 
# This notebook is being created as a companion to the book "Quantum Mechanics: the Theoretical Minimum" by Susskind and Friedman (QM:TTM for short). Those authors of course never use quaternions as they are a bit player in the crowded field of mathematical tools. Nature has used one accounting system since the beginning of space-time, so I will be a jerk in the name of consistency. This leads to a different perspective on what makes an equation quantum mechanical. If a conjugate operator is used, then the expression is about quantum mechanics. It is odd to have such a brief assertion given the complexity of the subject, but that make the hypothesis fun - and testable by seeing if anything in the book cannot be done with quaternions and their conjugates. Import the tools to work with quaternions in this notebook.

# In[1]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport matplotlib.pyplot as plt\n\n# To get equations the look like, well, equations, use the following.\nfrom sympy.interactive import printing\nprinting.init_printing(use_latex=True)\nfrom IPython.display import display\n\n# Tools for manipulating quaternions.\nimport Q_tools as qt;')


# ![](images/chapter_1/101.02.1.50.jpg)

# I have a vivid memory of learning how to take statements in logic and converting them to simple algebra problems with zero for false and one for true. The converse applies here: by changing the algebra used for all calculations, a consequence is that the logic of quantum mechanics is different from classical physics. 

# ![](images/chapter_1/101.02.2.50.jpg)

# In "Dreams of a Final Theory" by Steven Weinberg, he views quantum physics as a far deeper subject than classical physics. When classical physics is a ball rolling down an inclined plane, it is simple to do. But a piano with all its modes of vibrations of some two hundred strings is not simple. I bring up this note of caution because in my hands anyway, the tools of quantum mechanics will flow through functions that apply to classical physics. The tools sets are classical and quantum physics are distinct and subtly different.

# ![](images/chapter_1/101.03.1.50.jpg)

# This is a sensible caution, but I dream of a day when spin can really be visualized. On that day we can see how they are different and yet related.

# ![](images/chapter_1/101.20.1.50.jpg)

# I remember a lecture in a first year quantum mechanics class where the professor derived the uncertainty principle as an outcome of properties of complex numbers. I suspect that every measurable quantity has a conjugate variable, and with that particular variable, the uncertainty principle applies to the pair. It would be odd if there were an exception.

# ![](images/chapter_1/101.21.1.50.jpg)

# Our way of visualizing a ball - it is there going so fast - must therefore not apply to quantum mechanics. 

# ![](images/chapter_1/101.22.1.50.jpg)

# Here is my first substantive disagreement, not so much with Susskind as with the traditional explanation as created in the early 1800s by C. Wessel and Argand. This graph can be written in books, published, and studied. It is both useful and good. The common complaint is that there appears to be no difference between the real axis and the imaginary one. Consider the following experiment. Someone draws a point in the complex plane and does either a reflection over the real or imaginary axis. They cover up the label for both axes. In this math thought experiment, there would be no way to distinguish if the reflection had been over the real or the imaginary axis. To an experimental mathematician, this means there is a problem with the definition of real versus imaginary numbers. Most mathematicians project a faith in define logical precision, so if something is define precisely and the logic work, there is no need for discussion and no wrong.
# 
# Space-time physics declares a definition that will work for the nearly extinct bird known as an experimental mathematician. If in the plane one axis is just like the other at 90 degrees, so to is the axis 90 degrees to both of those pointing out of the page. Then imaginary number are for space. What sort of axis would be so orthogonal to these three that they cannot be drawn on the same page? That would be time. Time can be represented with an animation. Consider a baseball hit from home plate-now that lands in right field-future.

# ![simple.gif](images/chapter_1/baseball_animations/simple.gif)

# The ball starts at here-now, $(0, 0, 0, 0)$. Time is treated as the real number, while the planar graph has the imaginary values of $x i$ and $y j$. I hyphenate space-time and their verbal variations, like the ball landing in right field-future.
# 
# What does a spatial reflection of that baseball look like using a mirror at home plate? There would now be two balls. The distance covered by both balls would be twice the distance but the time would be the same.

# ![simple.gif](images/chapter_1/baseball_animations/space.gif)

# The ball goes from home plate-now to right field-future and left field-future. It takes the same amount of time but twice the distance. The opposite situation happens for time reflection.

# ![time reflection](images/chapter_1/baseball_animations/time.gif)

# The ball goes from right field-past to home plate-now out to right field-future. As with all these looping animations, there is a cyclic boundary condition, a fancy way of saying these animations all repeat.
# 
# There is one last reflection possibility: what if there is a reflection in both space and time?

# ![space-time reflection](images/chapter_1/baseball_animations/space-time.gif)

# The ball starts in left field-past, gets to home plate-now, then continues to right field-future. The two reflections take twice the distance and time as the starting hit to the outfield.

# As animations, the reflection in the real number, time, are distinct from the reflections in imaginary numbers, space, and are visually distinct from the combination. Analytic quaternion animations can be used to describe real world things like the flight of a baseball. **Space is orthogonal to time**. We draw in space but animate in time. As such, what is drawn in space is separate from how we then construct an animation from the drawings.

# ![](images/chapter_1/101.23.1.50.jpg)

# Susskind and Friedman are following the standard play book treating everything as part of a static plane - $r$, $x$, and $y$ are all used for distances. Let's try to shift the thinking to time $t$ for the reals, space $R$ for the imaginary numbers. For complex numbers, there is only one sort of factor of $i$. For quaternions, there is a 3-vector $i$ that can point in an infinite number of ways. For a set of quaternions that all point in the same direction or its opposite, every one of these rules applies as is.

# The polar representation of a quaternion takes as input an amplitude and three "angles" which are ratios of distance to time, making them more like velocities than angles in a plane. These "polar velocities" can take any numerical value, making them different because velocities of material objects are limited to the speed of light. A math construction is more free.
# 
# Here is a demonstration that the product of the polar representation of a quaternion multiplies the Amplitude (first numbers 2, 3, and 10) and adds the polar velocities (1, 1.5, and 0.5).

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


# And if we have done things correction, the product of a polar with its conjugate is the square of the amplitude.

# In[4]:


print(p1.conj().product(p1))
print(p2.conj().product(p2))
print(p3.conj().product(p3))


# So what happens when there is more than one polar velocity? Think like a mathematical physicist for a moment. One velocity is trivial - everything must point in the same direction. If one now gets two or three velocities in the mix, then one must deal with the confusion of spinning things. The norm does not change going from $2 i$ to $2 i + 3 j + 4k$, only the direction changes. Like all things that spin, the subject gets confusing, and is not germane to this chapter. When I see just a factor of $i$, I see folks working with a simpler case of physics.

# ![](images/chapter_1/101.23.2.50.jpg)

# The deeper math idea that is hinted at is that we are all taught about representing complex numbers on the manifold $\mathbb{R}^2$. Much of complex analysis relies on working with complex numbers on the manifold $\mathbb{C}^1$. Everything that can be done on the manifold $\mathbb{R}^2$ can also be done on $\mathbb{C}^1$ so long as one has both $z$ and $z^*$. When I learned this lesson, it immediately sounded like a problem for quaternions. This time there will be functions on $\mathbb{R}^4$ that cannot be constructed using just $q$ and $q^*$. I considered this a show stopper - at least until I found a cute solution. I call them the first and second conjugates. Take a moment to calculate $(i q i)^*$ and $(j q j)^*$. You will see that those flip the signs on all the terms except the $i$ and $j$ respectively. So $q$, $q^*$, $q^{*1}$, and $q^{*2}$ are together enough on the quaternion manifold $\mathbb{H}^1$ to cover all possible functions one can construct on $\mathbb{R}^4$. Again, this digression is not germane to the case of just one imaginary factor. I point it out to say more than one can be handled, and it will be darn interesting to see how it plays out technically.

# ![](images/chapter_1/101.24.1.50.jpg)

# The two most important numbers to understand in mathematical physics are zero and one. The duo show up in every context, so their meaning varies by that context. For phase factor terms, just add them up.

# In[5]:


phase_a = qt.QH([1, 0.5, 0, 0], representation="polar")
phase_b = qt.QH([1, 1, 0, 0], representation="polar")
print("phase a:     ", phase_a)
print("phase a+a:   ", phase_a.product(phase_a))
print("phase a+a+b: ", phase_a.product(phase_a).product(phase_b))


# Darn rounding errors. Anyway, you should get the point. When the amplitude is unity, the only thing of interest is the phase.

# ![](images/chapter_1/101.24.2.50.jpg)

# I am not going to say something is *wrong* in QM:TTM. I will contend **it is just not right enough**. Yup, the space of states is not a mathematical set. Agreed, 100%. A vector in a vector space can be added to another vector an multiplied by a scalar value. Agree, 100%. Yet I want to grant states more power and versatility than this two trick pony (addition, addition's inverse subtraction, and multiplication by a scalar). The additional power is to drop the "by a scalar" clause, and put it is place "and multiplication's inverse division." At one level, it does not sound like a big deal. Take the number 6, we can write four expressions using the two operators and their inverses: $2 + 4 = 6$, $6 + (-4) = 2$, $2 * 3 = 6$, $6 * 3^{-1} = 2$. This is third grade math, so the demand is to be consistent with all that math learned long ago.
# 
# On the other hand, this sounds like a huge pain. Quantum mechanics is constructed from complex-valued vectors Hilbert spaces. The entire thing would have to be rebuilt. Everything would have to work, no exceptions. Well, that is why there are these iPython notebooks, to test if the reconstruction work is good.

# ## Spate-time dimensions versus State dimensions

# ![](images/chapter_1/101.25.1.50.jpg)

# As a practice, I never think of space or of time, only space-time. An event in space-time is a number with a real time and three imaginary numbers for space that together form a quaternion. Yet one is free to represent a quaternion as a series of quaternions, using one or an infinite number of quaternions in the series. Each and every element of the series will be in space-time. This is an often unspoken aspect of quantum mechanical states spaces: they all exist in space-time. I think of the space-time dimensions as being orthogonal to the state dimensions. There is a relationship between the two, yet there is also an independence. I will try no never use the word dimension without the qualifier of it being a space-time dimension or a state dimension. For this book, nearly all the discussion will be about the state dimensions.
# 
# Just to get the toes a little wet, here is one quaternion series which has two states:

# In[6]:


A = qt.QHStates([qt.QH([0,1,2,3], qtype="A1"), qt.QH([2,2,2,2], qtype="A2")])
print(A)


# This is just the opening move: each quaternion in the series will be associated with a value of n that can go from 1 out to infinity.

# ![](images/chapter_1/101.25.2.50.jpg)

# The game that is afoot is to show that quaternion series which can be finite or infinite behave exactly as one expect by the definition of a Hilbert space. Creativity is required to make this work! But that also makes it fun.

# ![](images/chapter_1/101.25.3.50.jpg)

# This looks so darn obvious, there is nothing much to do. Let's write down a quaternion series $B$ and add them up.

# In[7]:


B = qt.QHStates([qt.QH([-1, -1, -1, -1]), qt.QH([-2,-2,-2,-2]), qt.QH([-3,-3,-3,-3])])
A.add(B)


# When you write software that has to "do all the math", you get confronted with questions like, what if the dimensions are different? I think they should not be added together because a value of zero is different from being undefined. Of course, some one with more technical skills in these art may already know that I should treat the phantom third element of the quaternion series $A$ as zero and it is all OK. That would require a skilled math nerd. For now, I will treat it as an illegal move and report it. Let me redefine $B$ to have just two terms, and define a third series $C$.

# In[8]:


B = qt.QHStates([qt.QH([-1, -1, -1, -1], qtype="B1"), qt.QH([-2,-2,-2,-2], qtype="B2")])
C = qt.QHStates([qt.QH([0, 0, 0, 4], qtype="C1"), qt.QH([0,0,0,-10], qtype="C2")])
Z = qt.QHStates([qt.QH(), qt.QH()])


# ![](images/chapter_1/101.26.1.50.jpg)

# Whoops, a lot to prove, but nothing too difficult. Instead of a complex number $z$ and $w$, I will use quaternions $q$ and $w$.

# In[9]:


print("2. commutative addition, |A> + |B> = |B> + |A>")
print("|A> + |B>") 
print(A.add(B))
print("\n|B> + |A>")
print(B.add(A))
print("\n|A> + |B> = |B> + |A>? ", A.add(B).equals(B.add(A)))


# In[10]:


print("3. associative addition, (|A> + |B>) + |C> = |A> + (|B> + |C)>")
print("(|A> + |B>) + |C>")
print(A.add(B).add(C))
print("\n|A> + (|B> + |C>)")
print(A.add(B.add(C)))
print("(|A> + |B>) + |C> = |A> + (|B> + |C)>? ", A.add(B).add(C).equals(A.add(B.add(C))))


# In[11]:


print("4. An additive identity series zero exists")
print("A")
print(A)
print("\nA + Z")
print(A.add(Z))
print("\nA = A + Z?", A.equals(A.add(Z)))


# In[12]:


print("5. An additive inverse exists, A + (-A) = 0")
print("-A")
print(A.flip_signs())
print("A + (-A)")
print(A.add(A.flip_signs()))


# In[13]:


print("6. A number times a ket produces a new ket, q|A> = |B>")
q = qt.QHStates([qt.QH([4, 3, 2, 1], qtype="q")])
print("q: ", q)
print("q|A> = |B>")
print(A.product("ket", operator=q))


# What is going on for the *product()* function? The product function can treat the quaternion series as either a bra or a ket. An operator acts on the ket. If the state dimension of the ket is one, then it creates a diagonal series that is a the square of the state dimensions of ket. The ket has 2 state dimensions, so the operator has 4 state dimensions. The zeros of the diagonalized operator appear in the qtype.

# In[14]:


print("7a. Distributive property, q(|A> + |B>) = q|A> + q|B>")
print("q(|A> + |B>)")
qAB = A.add(B).product("ket", operator=q)
print(qAB)
print("q|A> + q|B>")
qAqB = A.product("ket", operator=q).add(B.product("ket", operator=q))
print(qAqB)
print("q(|A> + |B>) = q|A> + q|B>? ", qAB.equals(qAqB))


# In[16]:


w = qt.QHStates([qt.QH([1, -1, 1, 100])])
print("7b. Distributive property, (q + w)|A> = q|A> + w|A>")
print("(q + w)|A>)")
qwA = A.product("ket", operator=q.add(w))
print(qwA)
print("q|A> + w|A>")
qAwA = A.product("ket", operator=q).add(A.product("bra", operator=w))
print(qAwA)
print("(q + w)|A> = q|A> + w|A>? ", qwA.equals(qAwA))


# Thus 7 basic properties of state vectors can be replicated using quaternion series. I could say there was an equivalence relationship between the two. That does not mean they are exactly the same in all detail, but that for the operations done so far, both approaches are equivalent.

# ![](images/chapter_1/101.30.1.50.jpg)

# The world of rows and columns does not port to quaternion series. All series are just that: series. The conjugate of a series can be calculated, and is a very usefule thing.

# In[17]:


print("A*")
print(A.conj())


# The signal that one is doing a calculation in quantum mechanics may just be the use of mirrors via the conjugate operator. No wonder the subject requires such a struggle to understand!

# ## Managing changes in multiplication order

# ![](images/chapter_1/101.29.2.50.jpg)

# Let's first figure out what exactly is meant by "bra corresponding to [a ket] using complex numbers.

# In[23]:


print("z|A> ? <A|z*")
Ac = qt.QHStates([qt.QH([1,2,0,0], qtype="A1i"), qt.QH([3,4,0,0], qtype="A2i")])
i3 = qt.QHStates([qt.QH([1,3,0,0], qtype="13i")])
print("zA")
zA = Ac.product("ket", operator=i3)
print(zA)
print("Az*")
Azc = Ac.conj().product("bra", operator=i3.conj())
print(Azc)


# The "correspondence" is that the two need one more conjugate operation to be equal, like so:

# In[24]:


print("z|A> = (<A|z*)* ?", zA.equals(Azc.conj()))


# Now lets do this more precise definition of the relationship for quaternions.
# 
# This is the point when many folks who have spent years studying math and physics will get off the bus. They know that quaternions do not commute because quaternions do not commute. Let's just wave goodbye to those folks. They are smart, confident, and I wish them well.

# For anyone who stayed, it is time to think a little. Yes indeed, quaternions to not commute with each other. Yet we are not talking about random ass quaternions (random ass quaternions has a technical meaning I chose not to define). Each of these quaternion series has a name: bra, operator, and ket. Because their positions are *named*, rules can be imposed regarding what happens when the order of named terms is changed. Every quaternion product can be split into what I call the "even" or commuting part of the product, and the "odd" or anti-commuting part.

# In[26]:


print("Even product, (AB + BA)/2")
AB_even = A.product("bra", ket=B, kind="even")
print(AB_even)
print("\nOdd product, (AB - BA)/2")
AB_odd = A.product("bra", ket=B, kind="odd")
print(AB_odd)
print("\nAB")
AB = A.product("bra", ket=B)
print(AB)
print("Even + Odd = AB ?", AB_even.add(AB_odd).equals(AB))


# For named positions that change their positions, create a rule that says the product is now the difference between the even and the odd terms.

# In[27]:


print("BA")
BA = B.product("bra", ket=A)
print(BA)
print("Even - Odd = BA ?", AB_even.dif(AB_odd).equals(BA))


# ![](images/chapter_1/101.30.3.50.jpg)

# ![](images/chapter_1/101.31.1.50.jpg)

# ![](images/chapter_1/101.31.2.50.jpg)

# ![](images/chapter_1/101.32.2.50.jpg)

# ![](images/chapter_1/101.34.1.50.jpg)

# ![](images/chapter_1/101.34.2.50.jpg)
