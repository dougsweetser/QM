
# coding: utf-8

# # Lecture 2: Quantum States

# email: dsweetser@alum.mit.edu

# With lecture 1 in the books, this one should be pretty simple and direct. The working hypothesis is that there is an equivalence relation between the way quantum mechanics is represented using a complex-valued vector space and quaternion series. Every calculation done the standard way in lecture 1 was done with quaternion series for the companion iPython notebook. Continue that process with Lecture 2 which fortunately is shorter. It is also simpler, focusing on spin, something that can be described as best we can do with two quantum states. Set up our math tools.

# In[1]:


get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nimport numpy as np\nimport sympy as sp\nimport matplotlib.pyplot as plt\n\n# To get equations the look like, well, equations, use the following.\nfrom sympy.interactive import printing\nprinting.init_printing(use_latex=True)\nfrom IPython.display import display\n\n# Tools for manipulating quaternions.\nimport Q_tools as qt;\n\nfrom IPython.core.display import display, HTML, Math, Latex\ndisplay(HTML("<style>.container { width:100% !important; }</style>"))')


# ![](images/lecture_2/lecture_2_p_35_1.50.jpg)

# Exactly. I was happy to see that phrase be written in italics. Each moment in our lives either is before another moment, after another moment, or exactly the same moment. The set of moments of our lives is a completely ordered set. Now think about a room of people. Each as their ordered set of moments, but how should one organize both the time and place of all these people? The space-time location of all these people cannot form a completely ordered set. Once could design a way to order them, but there is no universal way to do so.
# 
# Hydrogen is the most abundant atom in the Universe. Its quantum behavior is exceptionally precise as indicated by the atomic spectra Balmer series. If a hydrogen atom gets excited, due to the quantization of angular momentum, it can only emit light at precise frequencies. We cannot know when it will emit light, but the frequencies are fixed.

# ![](images/lecture_2/lecture_2_p_36_1.50.jpg)

# I am in the second class. My montra is that quantum mechanics as done today is formally exactly right, but it is not right enough. Most of the progress that needs to be done is in phase spaces that are not even written down. Instead of just the one Schrödinger equation, there are three mixed derivative differential equations that I can write down but do not understand how to use productively. That is for another Lecture. A new proposal I have for how gravity works, quaternion gravity, is based on terms that are not in the canonical lexicon of physics. That example however is not quantum mechanical other that to say if true, the graviton as a material expression of a gravity field does not exist. Instead, special relativity sets some precise rules about what equations can be used to describe Nature. Gravity imposes a different set of rules on all equations. I know these are huge claims not back up by supporting evidence here. Bottom line: things are great, and they can become greater, but it will require a crazy amount of work, hence these notebooks.

# ![](images/lecture_2/lecture_2_p_38_1.50.jpg)

# ![](images/lecture_2/lecture_2_p_47_1.50.jpg)

# These two clips are actually some 11 pages apart. The authors made a decision to use the abtractions of $|u>$ and $|d>$ for from page 38 to 47 so the user would know that no matter what choice was made to represent those two states, it would flow to every statement that followed. That is how abstractions work. The first time I constructed this notebook, I actually used the imaginary $i$ in place of the real number $1$, so I know the abstraction works as is. 
# 
# The working hypothesis a (possibly infinite) series of quaternions has the same algebraic properties of Hilbert spaces when one uses the Euclidean product, $A^* B = \sum_{1}^{n} a_n^* b_n$. For n=2:

# ![AxB.png](images/lecture_2/AxB.png)

# In[2]:


q_0, q_1, q_i, q_j, q_k = qt.QH().q_0(), qt.QH().q_1(), qt.QH().q_i(), qt.QH().q_j(), qt.QH().q_k()

u = qt.QHStates([q_1, q_0])
d = qt.QHStates([q_0, q_1])

u.print_states("u", True)
d.print_states("d")


# ![](images/lecture_2/lecture_2_p_38_2.50.jpg)

# Construct a general A, not setting any values.

# In[3]:


A1t, A1x, A1y, A1z = sp.symbols("A1t A1x A1y A1z")
A2t, A2x, A2y, A2z = sp.symbols("A2t A2x A2y A2z")
A1q = qt.QH([A1t, A1x, A1y, A1z], qtype="a₁")
A2q = qt.QH([A2t, A2x, A2y, A2z], qtype="a₂")
A = qt.QHStates([A1q, A2q])
A.print_states("A")


# Calculation the components.

# In[4]:


uA = u.Euclidean_product("bra", ket=A)
dA = d.Euclidean_product("bra", ket=A)
uA.print_states("αu = <u|A>",1)
dA.print_states("αd = <d|A>")


# That was easy. As different basis vectors are constructed, the 8 values that go into $A$ will shift around.

# ![](images/lecture_2/lecture_2_p_39_1.50.jpg)

# Simpe enough to calculate.

# ![](images/lecture_2/lecture_2_p_39_2.50.jpg)

# In[5]:


uA_norm_sq = uA.Euclidean_product("bra", ket=uA)
dA_norm_sq = dA.Euclidean_product("bra", ket=dA)
uA_norm_sq.print_states("|αu|²", 1)
dA_norm_sq.print_states("|αd|²")


# Everything evaluates to the expected positive real value. The up probability of spin is all from the first quaternion, while the down probability is all from the second quaternion. Quite clean and simple.

# ![](images/lecture_2/lecture_2_p_39_3.50.jpg)

# For this representation, it is obvious so I will do just one.

# In[6]:


ud = u.Euclidean_product("bra", ket=d)
ud.print_states("<u|d>")


# ![](images/lecture_2/lecture_2_p_40_1.50.jpg)

# Nice and clear, and easy to calculate.

# ![](images/lecture_2/lecture_2_p_40_2.50.jpg)

# Normalization is a simple enough trick to do.

# In[7]:


An = A.normalize()
An.print_states("A, normalized", 1)
Anp = An.Euclidean_product("bra", ket=An)
Anp.print_states("<An|An>")
print("simplified t, n=1: ", sp.simplify(Anp.qs[0].t))
print("simplified x, n=1: ", sp.simplify(Anp.qs[0].x))
print("simplified y, n=1: ", sp.simplify(Anp.qs[0].y))
print("simplified z, n=1: ", sp.simplify(Anp.qs[0].z))
print("simplified t, n=2: ", sp.simplify(Anp.qs[1].t))
print("simplified x, n=2: ", sp.simplify(Anp.qs[1].x))
print("simplified y, n=2: ", sp.simplify(Anp.qs[1].y))
print("simplified z, n=2: ", sp.simplify(Anp.qs[1].z))


# These expressions look crazy complex, but it simplifies down to two factors of a half whose sum is unity.

# ![](images/lecture_2/lecture_2_p_41_1.50.jpg)

# Define $|r>$ and $|L>$ using $|u>$ and $|d>$.

# In[8]:


sqrt_2op = qt.QHStates([qt.QH([sp.sqrt(1/2), 0, 0, 0])])

u2 = u.Euclidean_product('ket', operator=sqrt_2op)
d2 = d.Euclidean_product('ket', operator=sqrt_2op)

r = u2.add(d2)
L = u2.dif(d2)

r.print_states("r", True)
L.print_states("L")


# One thing to notices is how complicated the qtype became. For the up and down states, it was either zero or one.  Why is there a sum of four terms? Notices that three of the four terms are zeros. To multiply a quaternion series with two state dimensions by the one over the square root of two takes a diagonal quaternion series with four state dimensions. That requirement is effectively recorded in the qtype.
# 
# Isn't there something wrong with the sum of $|L>$ being equal to zero? What really matters is taking the norm things, $<L|L>=1$, so the zero ket is not going to create any issues.

# ![](images/lecture_2/lecture_2_p_41_2.50.jpg)

# ![](images/lecture_2/lecture_2_p_42_2.50.jpg)

# Just do it.

# In[9]:


r.Euclidean_product('bra', ket=r).print_states("<r|r>", True)
L.Euclidean_product('bra', ket=L).print_states("<L|L>", True)
r.Euclidean_product('bra', ket=L).print_states("<r|L>")


# Just for fun, calculate the probability amplitudes $<A|r>$ and $<A|L>$ to see how this basis mixes around the information in $A$ without destroying it.

# In[10]:


Ar = A.Euclidean_product("bra", ket=r)
Ar.print_states("<A|r>", 1)

AL = A.Euclidean_product("bra", ket=L)
AL.print_states("<A|L>", 1)


# In the up and down representation of quaternion states, one row was all zeroes, nice and simple. Now every seat in the hockey arena is filled. The first element in both series are the same. The second terms all differ by a sign although the magnitudes are the same.

# ![](images/lecture_2/lecture_2_p_42_3.50.jpg)

# ![](images/lecture_2/lecture_2_p_43_1.50.jpg)

# ![](images/lecture_2/lecture_2_p_43_2.50.jpg)

# Oh my, so many conditions!

# ![](images/lecture_2/lecture_2_p_44_1.50.jpg)

# This is not so bad. Let's build this, then see of all the conditions "just work".

# In[11]:


one_root_two = sp.sqrt(1/2)
q_2 = qt.QHStates( [ qt.QH([sp.sqrt(1/2), 0, 0, 0]) ] )
q_2i = qt.QHStates([qt.QH([0, sp.sqrt(1/2), 0, 0])])

i = u.product("ket", operator=q_2).add(d.product("ket", operator=q_2i))
o = u.product("ket", operator=q_2).dif(d.product("ket", operator=q_2i))

i.print_states("i", 1)
o.print_states("o", 1)


# First check the normal and orthogonal properties.

# In[12]:


i.Euclidean_product('bra', ket=i).print_states("<i|i>", 1)
o.Euclidean_product('bra', ket=o).print_states("<o|o>", 1)
i.Euclidean_product('bra', ket=o).print_states("<i|o>")


# Great, these two are orthonormal quaternion series. Now to see how they relate to the other orthonomal series.

# In[13]:


print("Equation 2.8\n")

ou = o.Euclidean_product('bra', ket=u)
uo = u.Euclidean_product('bra', ket=o)
ouuo = ou.product('bra', ket=uo)
ouuo.print_states("<o|u><u|o>", 1)
od = o.Euclidean_product('bra', ket=d)
do = d.Euclidean_product('bra', ket=o)
oddo = od.product('bra', ket=do)
oddo.print_states("<o|d><d|o>", 1)
iu = i.Euclidean_product('bra', ket=u)
ui = u.Euclidean_product('bra', ket=i)
iuui = iu.product('bra', ket=ui)
iuui.print_states("<i|d><d|i>", 1)
id = i.Euclidean_product('bra', ket=d)
di = d.Euclidean_product('bra', ket=i)
iddi = id.product('bra', ket=di)
iddi.print_states("<i|d><d|i>")


# Notice how both a Euclidean product and product are used in the calculation. The amplitudes as quaternion series can be multiplied together to get the correct final result. When I first did this calculation, one of the four was 0.3535, not 0.5. There was a typo in expression. Once correct, four down four to go.

# In[14]:


print("Equation 2.9\n")

Or = o.Euclidean_product('bra', ket=r)
ro = r.Euclidean_product('bra', ket=o)
orro = Or.product('bra', ket=ro)
orro.print_states("<o|r><r|o>", 1)
oL = o.Euclidean_product('bra', ket=L)
Lo = L.Euclidean_product('bra', ket=o)
oLLo = oL.product('bra', ket=Lo)
oLLo.print_states("<o|L><L|o>", 1)
ir = i.Euclidean_product('bra', ket=r)
ri = r.Euclidean_product('bra', ket=i)
irri = ir.product('bra', ket=ri)
irri.print_states("<i|r><r|i>", 1)
iL = i.Euclidean_product('bra', ket=L)
Li = L.Euclidean_product('bra', ket=i)
iLLi = iL.product('bra', ket=Li)
iLLi.print_states("<i|L><L|i>")


# ![](images/lecture_2/lecture_2_p_44_2.50.jpg)

# Some could view this very project as "somewhat tedious". There are so many details that have to be done exactly right to back up a claim that quaternion series can do everything that is right in quantum mechanics, before stepping out onto new ice to say here is something more we can do right. As an example, it took me a few hours to get the normalization done correctly. For a quaternion - not a quaternion series - I had a function to normalize it. For a quaternion series, I just called that function for each quaternion in the series. That produced an incorrect value for the quaternion series. In addition, I needed to normalize for the square root of the number of state dimensions. Once that detail was added, I got the right result for quaternion series.
# 
# Off in my basement theoretical physics isolation chamber, I think Newtonian space-time physics should be done with quaternions. Newton himself could not have done classical physics using complex numbers or quaternions since they had not yet been invented. Search for "quaternion baseball" on YouTube if interested in the subject.

# ![](images/lecture_2/lecture_2_p_45_1.50.jpg)

# Time to get abstract, the wheelhouse of algebra. First define the symbols needed for the four unknown components.

# In[15]:


αt, αx, αy, αz = sp.symbols("αt αx αy αz")
βt, βx, βy, βz = sp.symbols("βt βx βy βz")
γt, γx, γy, γz = sp.symbols("γt γx γy γz")
δt, δx, δy, δz = sp.symbols("δt δx δy δz")

αq = qt.QH([αt, αx, αy, αz])
αs = qt.QHStates([αq])
αs.print_states("α component", 1)

βq = qt.QH([βt, βx, βy, βz])
βs = qt.QHStates([βq])
βs.print_states("β component", 1)

γq = qt.QH([γt, γx, γy, γz])
γs = qt.QHStates([γq])
γs.print_states("γ component", 1)

δq = qt.QH([δt, δx, δy, δz])
δs = qt.QHStates([δq])
δs.print_states("δ component")


# Define the kets $|i>$ and $|o>$.

# In[16]:


iαβ = u.product("bra", operator=αs).add(d.product("ket", operator=βs))
iαβ.print_states("iαβ", 1)

oγδ = u.product("bra", operator=γs).add(d.product("ket", operator=δs))
oγδ.print_states("oγδ")


# Notice we can extract the component alpha from $|i>$ by multiplying it by the bra $<u|$ because that bra is orthonormal to $<d|$. Beta, gamma, and delta components can be extracted the same way.

# In[17]:


alpha = u.Euclidean_product("bra", ket=iαβ)
alpha.print_states("alpha", 1)

beta = d.Euclidean_product("bra", ket=iαβ)
beta.print_states("betaa", 1)

gamma = u.Euclidean_product("bra", ket=oγδ)
gamma.print_states("gamma", 1)

delta = d.Euclidean_product("bra", ket=oγδ)
delta.print_states("delta")


# With the four components precisely defined, we can start forming the products asked for in Exercise 2.3: a):

# In[18]:


print("Exercise 2.3: a)\n")

print("equation 2.8.1: <i|u><u|i> = 1/2 =? α* α")

iu = iαβ.Euclidean_product("bra", ket=u)
ui = u.Euclidean_product("bra", ket=iαβ)
iuui = iu.product("bra", ket=ui)
iuui.print_states("<i|u><u|i>", 1)


print("equation 2.8.2: <i|d><d|i> = 1/2 =? β* β")

id = iαβ.Euclidean_product("bra", ket=d)
di = d.Euclidean_product("bra", ket=iαβ)
iddi = id.product("bra", ket=di)
iddi.print_states("<i|d><d|i>", 1)


print("equation 2.8.3: <o|u><u|o> = 1/2 =? γ* γ")

ou = oγδ.Euclidean_product("bra", ket=u)
uo = u.Euclidean_product("bra", ket=oγδ)
ouuo = ou.product("bra", ket=uo)
ouuo.print_states("<o|u><u|o>", 1)


print("equation 2.8.4: <o|d><d|o> = 1/2 =? δ* δ")

od = oγδ.Euclidean_product("bra", ket=d)
do = d.Euclidean_product("bra", ket=oγδ)
oddo = od.product("bra", ket=do)
oddo.print_states("<o|d><d|o>")


# These products are all real numbers composed of each of the four components.

# In[19]:


print("Exercise 2.3: b)\n")

print("equation 2.9.1: <o|r><r|o> = 1/2 =?")

OR = oγδ.Euclidean_product("bra", ket=r)
ro = r.Euclidean_product("bra", ket=oγδ)
orro = OR.product("bra", ket=ro)
orro.print_states("<o|r><r|o>", 1)


print("equation 2.9.2: <o|L><L|o> = 1/2 =?")

oL = oγδ.Euclidean_product("bra", ket=L)
Lo = L.Euclidean_product("bra", ket=oγδ)
oLLo = oL.product("bra", ket=Lo)
oLLo.print_states("<o|L><L|o>", 1)


print("equation 2.9.3: <i|r><r|i> = 1/2 =?")

ir = iαβ.Euclidean_product("bra", ket=r)
ri = r.Euclidean_product("bra", ket=iαβ)
irri = ir.product("bra", ket=ri)
irri.print_states("<i|r><r|i>", 1)


print("equation 2.9.4: <i|L><L|i> = 1/2 =?")

iL = iαβ.Euclidean_product("bra", ket=L)
Li = L.Euclidean_product("bra", ket=iαβ)
iLLi = iL.product("bra", ket=Li)
iLLi.print_states("<i|L><L|i>")


# The first two have the sum of $\gamma^* \gamma$ and $\delta^* \delta$. The last two are $\alpha^* \alpha$ and $\beta^* \beta$. Those values were figured out in part a). A half of a half plus a half is equal to a half.

# Each of the 4 kets, $|i>$, $|o>$, $|r>$, and $|L>$ are expressed in terms of the orthonormal basis vectors $|u>$ and $|d>$. The products look like this:

# In[20]:


display(Math(r"""\begin{align}
<o|r><r|o> &= \frac{1}{2}(<u|\gamma* + <d|\delta^∗)(|u> + |d>)(<u| + <d|)(\gamma|u> + \delta|d>) \\
&= \frac{1}{2}(\gamma^* + \delta^∗)(\gamma + \delta) \\
&= \frac{1}{2}(\gamma^* \gamma + \delta^∗ \delta + \gamma^* \delta + \delta^* \gamma) \\
&= \frac{1}{2}(1 + \delta^∗ \gamma + \gamma^* \delta ) = \frac{1}{2}\\
\rm{ergo}\quad 0 &= \delta^∗ \gamma + \gamma^* \delta
\end{align}"""))


# In[24]:


display(Math(r"""\begin{align}
<o|L><L|o> &= \frac{1}{2}(<u|\gamma^* + <d|\delta^∗)(|u> - |d>)(<u| - <d|)(\gamma|u> + \delta|d>) \\
&= \frac{1}{2}(\gamma^* - \delta^∗)(\gamma - \delta) \\
&= \frac{1}{2}(\gamma^* \gamma + \delta^∗ \delta - \gamma^* \delta - \delta^* \gamma) \\
&= \frac{1}{2}(1 - \delta^∗ \gamma - \gamma^* \delta ) = \frac{1}{2}\\
\rm{ergo}\quad 0 &= \delta^∗ \gamma + \gamma^* \delta
\end{align}"""))


# In[25]:


display(Math(r"""\begin{align}
<i|r><r|i> &= \frac{1}{2}(<u|\alpha^* + <d|\beta^∗)(|u> + |d>)(<u| + <d|)(\alpha|u> + \beta|d>) \\
&= \frac{1}{2}(\alpha^* + \beta^∗)(\alpha + \beta) \\
&= \frac{1}{2}(\alpha^* \alpha + \beta^∗ \beta + \alpha^* \beta + \beta^* \alpha) \\
&= \frac{1}{2}(1 + \beta^∗ \alpha + \alpha^* \beta ) = \frac{1}{2}\\
\rm{ergo}\quad 0 &= \beta^∗ \alpha + \alpha^* \beta
\end{align}"""))


# In[26]:


display(Math(r"""\begin{align}
<i|L><L|i> &= \frac{1}{2}(<u|\alpha^* + <d|\beta^∗)(|u> - |d>)(<u| - <d|)(\alpha|u> + \beta|d>) \\
&= \frac{1}{2}(\alpha^* - \beta^∗)(\alpha - \beta) \\
&= \frac{1}{2}(\alpha^* \alpha + \beta^∗ \beta - \alpha^* \beta - \beta^* \alpha) \\
&= \frac{1}{2}(1 - \beta^∗ \alpha - \alpha^* \beta ) = \frac{1}{2}\\
\rm{ergo}\quad 0 &= \beta^∗ \alpha + \alpha^* \beta
\end{align}"""))


# Exercise 2.3: c). For any complex number $z$, $z = - z^*$ if and only if $z$ is a purely imaginary number. The conjugate operator flips the sign of the imaginary number, but not the real number. Ergo the real number must be equal to zero. $\alpha^* \beta + \alpha \beta^* = 0$, or $\alpha^* \beta = -\alpha \beta^*$. Based on this observation, we know the imaginary numbers $\alpha^* \beta$ and $\gamma^* \delta$ are pure imaginary numbers. 

# ## Conclusion
# 
# So far so good. I also don't thing there is anything new here. Of course, I had to get ever detail right for the quaternion states or the project would have derailed. 
