
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

# As animations, the reflection in the real number, time, are distinct from the reflections in imaginary numbers, space, and are visually distinct from the combination. Analytic quaternion animations can be used to describe real world things like the flight of a baseball.

# ![](images/chapter_1/101.23.1.50.jpg)

# Susskind and Friedman are following the standard play book treating everything as part of a static plane - $r$, $x$, and $y$ are all used for distances. Let's try to shift the thinking to time $t$ for the reals, space $R$ for the imaginary numbers. For complex numbers, there is only one sort of factor of $i$. For quaternions, there is a 3-vector $i$ that can point in an infinite number of ways. For a set of quaternions that all point in the same direction or its opposite, every one of these rules applies as is.

# ![](images/chapter_1/101.23.2.50.jpg)

# ![](images/chapter_1/101.24.1.50.jpg)

# ![](images/chapter_1/101.24.2.50.jpg)

# ![](images/chapter_1/101.25.1.50.jpg)

# ![](images/chapter_1/101.25.2.50.jpg)

# ![](images/chapter_1/101.25.3.50.jpg)

# ![](images/chapter_1/101.26.1.50.jpg)

# ![](images/chapter_1/101.29.2.50.jpg)

# ![](images/chapter_1/101.30.1.50.jpg)

# ![](images/chapter_1/101.30.3.50.jpg)

# ![](images/chapter_1/101.31.1.50.jpg)

# ![](images/chapter_1/101.31.2.50.jpg)

# ![](images/chapter_1/101.32.2.50.jpg)

# ![](images/chapter_1/101.34.1.50.jpg)

# ![](images/chapter_1/101.34.2.50.jpg)
