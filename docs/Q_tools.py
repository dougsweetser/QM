
# coding: utf-8

# # Developing Quaternion and Space-time Number Tools for iPython3

# In this notebook, tools for working with quaternions for physics issues are developed. The class QH treat quaternions as Hamilton would have done: as a 4-vector over the real numbers. 
# 
# In physics, group theory plays a central role in the fundamental forces of Nature via the standard model. The gauge symmetry U(1) a unit circle in the complex plane leads to electric charge conservation. The unit quaternions $SU(2)$ is the symmetry needed for the weak force which leads to beta decay. The group $SU(3)$ is the symmetry of the strong force that keeps a nucleus together.
# 
# The class Q8 was written in the hope that group theory would be written in first, not added as needed later. I call these "space-time numbers". The problem with such an approach is that one does not use the mathematical field of real numbers. Instead one relies on the set of positive reals. In some ways, this is like reverse engineering some basic computer science. Libraries written in C have a notion of a signed versus unsigned integer. The signed integer behaves like the familiar integers. The unsigned integer is like the positive integers. The difference between the two is whether there is a placeholder for the sign or not. All floats are signed. The modulo operations that work for unsigned integers does not work for floats.
# 
# This set of tools is done 4x:
# 1. QH - Quaternions for Hamilton, can do symbolic manipulations
# 1. QHa - Quaternions for Hamilton numpy arrays
# 1. Q8 - Quaternions that are represented by 8 numbers
# 1. Q8a - Quaternions that are represented by 8 numbers that are numpy arrays
# 
# Test driven development was used. The same tests were used for QH, QHa, Q8, and Q8a.  Either class can be used to study quaternions in physics.

# In[38]:


import IPython
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np
import random
import sympy as sp
import os
import unittest

from IPython.display import display
from os.path import basename
from glob import glob
get_ipython().run_line_magic('matplotlib', 'inline')


# Define the stretch factor $\gamma$ and the $\gamma \beta$ used in special relativity.

# In[39]:


def sr_gamma(beta_x=0, beta_y=0, beta_z=0):
    """The gamma used in special relativity using 3 velocites, some may be zero."""

    return 1 / (1 - beta_x ** 2 - beta_y ** 2 - beta_z ** 2) ** (1/2)

def sr_gamma_betas(beta_x=0, beta_y=0, beta_z=0):
    """gamma and the three gamma * betas used in special relativity."""

    g = sr_gamma(beta_x, beta_y, beta_z)
    
    return [g, g * beta_x, g * beta_y, g * beta_z]


# ## Quaternions for Hamilton

# Define a class QH to manipulate quaternions as Hamilton would have done it so many years ago. The "qtype" is a little bit of text to leave a trail of breadcrumbs about how a particular quaternion was generated.

# In[40]:


class QH(object):
    """Quaternions as Hamilton would have defined them, on the manifold R^4."""

    def __init__(self, values=None, qtype="Q", representation=""):
        if values is None:
            self.t, self.x, self.y, self.z = 0, 0, 0, 0
        elif len(values) == 4:
            self.t, self.x, self.y, self.z = values[0], values[1], values[2], values[3]

        elif len(values) == 8:
            self.t, self.x = values[0] - values[1], values[2] - values[3]
            self.y, self.z = values[4] - values[5], values[6] - values[7]
        self.representation = representation
        
        if representation != "":
            self.t, self.x, self.y, self.z = self.representation_2_txyz(representation)
            
        self.qtype = qtype

    def __str__(self, quiet=False):
        """Customize the output."""
        
        qtype = self.qtype
        
        if quiet:
            qtype = ""
        
        if self.representation == "":
            string = "({t}, {x}, {y}, {z}) {qt}".format(
                t=self.t, x=self.x, y=self.y, z=self.z, qt=qtype)
    
        elif self.representation == "polar":
            rep = self.txyz_2_representation("polar")
            string = "({A} A, {thetaX} 𝜈x, {thetaY} 𝜈y, {thetaZ} 𝜈z) {qt}".format(
                A=rep[0], thetaX=rep[1], thetaY=rep[2], thetaZ=rep[3], qt=qtype)
 
        elif self.representation == "spherical":
            rep = self.txyz_2_representation("spherical")
            string = "({t} t, {R} R, {theta} θ, {phi} φ) {qt}".format(
                t=rep[0], R=rep[1], theta=rep[2], phi=rep[3], qt=qtype)

        return string

    def is_symbolic(self):
        """Figures out if an expression has symbolic terms."""
        
        symbolic = False
        
        if hasattr(self.t, "free_symbols") or hasattr(self.x, "free_symbols") or             hasattr(self.y, "free_symbols") or hasattr(self.z, "free_symbols"): 
            symbolic = True
        
        return symbolic

    def txyz_2_representation(self, representation):
        """Converts Cartesian txyz into an array of 4 values in a different representation."""

        symbolic = self.is_symbolic()
        
        if representation == "":
            rep = [self.t, self.x, self.y, self.z]
        
        elif representation == "polar":
            amplitude = (self.t ** 2 + self.x ** 2 + self.y **2 + self.z **2) ** (1/2)
            
            abs_v = self.abs_of_vector().t
            
            if symbolic:
                theta = sp.atan2(abs_v, self.t)
            else:
                theta = math.atan2(abs_v, self.t)
                
            if abs_v == 0:
                thetaX, thetaY, thetaZ = 0, 0, 0
                
            else:
                thetaX = theta * self.x / abs_v
                thetaY = theta * self.y / abs_v
                thetaZ = theta * self.z / abs_v
                
            rep = [amplitude, thetaX, thetaY, thetaZ]
        
        elif representation == "spherical":
            
            t = self.t
            
            R = (self.x ** 2 + self.y **2 + self.z **2) ** (1/2)
            
            if R == 0:
                theta = 0
            else:
                if symbolic:
                    theta = sp.acos(self.z / R)
                
                else:
                    theta = math.acos(self.z / R)
                
            if symbolic:
                phi = sp.atan2(self.y, self.x)
            else:
                phi = math.atan2(self.y, self.x)
                
            rep = [t, R, theta, phi]
        
        else:
            print("Oops, don't know representation: ", representation)
            
        return rep
    
    def representation_2_txyz(self, representation):
        """Convert from a representation to Cartesian txyz."""
        
        symbolic = False
        
        if hasattr(self.t, "free_symbols") or hasattr(self.x, "free_symbols") or             hasattr(self.y, "free_symbols") or hasattr(self.z, "free_symbols"): 
            symbolic = True

        if representation == "":
            t, x, y, z = self.t, self.x, self.y, self.z
        
        elif representation == "polar":
            amplitude, thetaX, thetaY, thetaZ = self.t, self.x, self.y, self.z
    
            theta = (thetaX ** 2 + thetaY ** 2 + thetaZ ** 2) ** (1/2)
                
            if theta == 0:
                t = self.t
                x, y, z = 0, 0, 0
            
            else:
                if symbolic:
                    t = amplitude * sp.cos(theta)
                    x = self.x / theta * amplitude * sp.sin(theta)
                    y = self.y / theta * amplitude * sp.sin(theta)
                    z = self.z / theta * amplitude * sp.sin(theta)
                else:
                    t = amplitude * math.cos(theta)
                    x = self.x / theta * amplitude * math.sin(theta)
                    y = self.y / theta * amplitude * math.sin(theta)
                    z = self.z / theta * amplitude * math.sin(theta)
    
        elif representation == "spherical":
            t, R, theta, phi = self.t, self.x, self.y, self.z

            if symbolic:
                x = R * sp.sin(theta) * sp.cos(phi)
                y = R * sp.sin(theta) * sp.sin(phi)
                z = R * sp.cos(theta)
            else:
                x = R * math.sin(theta) * math.cos(phi)
                y = R * math.sin(theta) * math.sin(phi)
                z = R * math.cos(theta)
            
        else:
            print("Oops, don't know representation: ", representation)
            
        txyz = [t, x, y, z]
        
        return txyz 
        
    def check_representations(self, q1):
        """If they are the same, report true. If not, kick out an exception. Don't add apples to oranges."""

        if self.representation == q1.representation:
            return True
        
        else:
            raise Exception("Oops, 2 quaternions have different representations: {}, {}".format(self.representation, q1.representation))
            return False
        
    def display_q(self):
        """Display each terms in a pretty way."""

        display(self.t)
        display(self.x)
        display(self.y)
        display(self.z)
        return
    
    def simple_q(self):
        """Simplify each term."""
        
        self.t = sp.simplify(self.t)
        self.x = sp.simplify(self.x)
        self.y = sp.simplify(self.y)
        self.z = sp.simplify(self.z)
        return self
    
    def expand_q(self):
        """Expand each term."""
        
        self.t = sp.expand(self.t)
        self.x = sp.expand(self.x)
        self.y = sp.expand(self.y)
        self.z = sp.expand(self.z)
        return self
    
    def q_0(self, qtype="0"):
        """Return a zero quaternion."""

        q0 = QH([0, 0, 0, 0], qtype=qtype, representation=self.representation)
        return q0

    def q_1(self, qtype="1"):
        """Return a multiplicative identity quaternion."""

        q1 = QH([1, 0, 0, 0], qtype=qtype, representation=self.representation)
        return q1
    
    def q_i(self, qtype="i"):
        """Return i."""

        qi = QH([0, 1, 0, 0], qtype=qtype, representation=self.representation)
        return qi
    
    def q_j(self, qtype="j"):
        """Return j."""

        qj = QH([0, 0, 1, 0], qtype=qtype, representation=self.representation)
        return qj
    
    def q_k(self, qtype="k"):
        """Return k."""

        qk = QH([0, 0, 0, 1], qtype=qtype, representation=self.representation)
        return qk
    
    def q_random(self, qtype="?"):
        """Return a random-valued quaternion."""
        
        qr = QH([random.random(), random.random(), random.random(), random.random()], qtype=qtype)
        return qr
    
    def dupe(self, qtype=""):
        """Return a duplicate copy, good for testing since qtypes persist"""
        
        du = QH([self.t, self.x, self.y, self.z], qtype=self.qtype, representation=self.representation)
        return du
    
    def equals(self, q1):
        """Tests if two quaternions are equal."""
        
        self.check_representations(q1)
        
        self_t, self_x, self_y, self_z = sp.expand(self.t), sp.expand(self.x), sp.expand(self.y), sp.expand(self.z)
        q1_t, q1_x, q1_y, q1_z = sp.expand(q1.t), sp.expand(q1.x), sp.expand(q1.y), sp.expand(q1.z)
        
        if math.isclose(self_t, q1_t) and math.isclose(self_x, q1_x) and math.isclose(self_y, q1_y) and math.isclose(self_z, q1_z):
            return True
        
        else:
            return False
    
    def conj(self, conj_type=0, qtype="*"):
        """Three types of conjugates."""

        t, x, y, z = self.t, self.x, self.y, self.z
        conj_q = QH()

        if conj_type == 0:
            conj_q.t = t
            if x != 0:
                conj_q.x = -1 * x
            if y != 0:
                conj_q.y = -1 * y
            if z != 0:
                conj_q.z = -1 * z

        elif conj_type == 1:
            if t != 0:
                conj_q.t = -1 * t
            conj_q.x = x
            if y != 0:
                conj_q.y = -1 * y
            if z != 0:
                conj_q.z = -1 * z
            qtype += "1"
            
        elif conj_type == 2:
            if t != 0:
                conj_q.t = -1 * t
            if x != 0:
                conj_q.x = -1 * x
            conj_q.y = y
            if z != 0:
                conj_q.z = -1 * z
            qtype += "2"
            
        conj_q.qtype = self.qtype + qtype
        conj_q.representation = self.representation
        
        return conj_q
    
    def flip_signs(self, qtype="-"):
        """Flip the signs of all terms."""
        
        end_qtype = "-{}".format(self.qtype)
        
        t, x, y, z = self.t, self.x, self.y, self.z
        
        flip_q = QH(qtype=end_qtype, representation=self.representation)
        if t != 0:
            flip_q.t = -1 * t
        if x != 0:
            flip_q.x = -1 * x
        if y != 0:
            flip_q.y = -1 * y
        if z != 0:
            flip_q.z = -1 * z
        
        return flip_q

    def vahlen_conj(self, conj_type="-", qtype="vc"):
        """Three types of conjugates -'* done by Vahlen in 1901."""
        
        t, x, y, z = self.t, self.x, self.y, self.z
        conj_q = QH()

        if conj_type == '-':
            conj_q.t = t
            if x != 0:
                conj_q.x = -1 * x
            if y != 0:
                conj_q.y = -1 * y
            if z != 0:
                conj_q.z = -1 * z
            qtype += "*-"

        if conj_type == "'":
            conj_q.t = t
            if x != 0:
                conj_q.x = -1 * x
            if y != 0:
                conj_q.y = -1 * y
            conj_q.z = z
            qtype += "*'"
            
        if conj_type == '*':
            conj_q.t = t
            conj_q.x = x
            conj_q.y = y
            if z != 0:
                conj_q.z = -1 * z
            qtype += "*"
            
        conj_q.qtype = self.qtype + qtype
        conj_q.representation = self.representation
        
        return conj_q
    
    def _commuting_products(self, q1):
        """Returns a dictionary with the commuting products."""

        s_t, s_x, s_y, s_z = self.t, self.x, self.y, self.z
        q1_t, q1_x, q1_y, q1_z = q1.t, q1.x, q1.y, q1.z

        products = {'tt': s_t * q1_t,
                    'xx+yy+zz': s_x * q1_x + s_y * q1_y + s_z * q1_z,
                    'tx+xt': s_t * q1_x + s_x * q1_t,
                    'ty+yt': s_t * q1_y + s_y * q1_t,
                    'tz+zt': s_t * q1_z + s_z * q1_t}

        return products

    def _anti_commuting_products(self, q1):
        """Returns a dictionary with the three anti-commuting products."""

        s_x, s_y, s_z = self.x, self.y, self.z
        q1_x, q1_y, q1_z = q1.x, q1.y, q1.z

        products = {'yz-zy': s_y * q1_z - s_z * q1_y,
                    'zx-xz': s_z * q1_x - s_x * q1_z,
                    'xy-yx': s_x * q1_y - s_y * q1_x,
                    'zy-yz': - s_y * q1_z + s_z * q1_y,
                    'xz-zx': - s_z * q1_x + s_x * q1_z,
                    'yx-xy': - s_x * q1_y + s_y * q1_x
                    }

        return products

    def _all_products(self, q1):
        """Returns a dictionary with all possible products."""

        products = self._commuting_products(q1)
        products.update(self._anti_commuting_products(q1))

        return products

    def square(self, qtype="^2"):
        """Square a quaternion."""

        end_qtype = "{}{}".format(self.qtype, qtype)
        
        qxq = self._commuting_products(self)

        sq_q = QH(qtype=end_qtype, representation=self.representation)
        sq_q.t = qxq['tt'] - qxq['xx+yy+zz']
        sq_q.x = qxq['tx+xt']
        sq_q.y = qxq['ty+yt']
        sq_q.z = qxq['tz+zt']

        return sq_q

    def norm_squared(self, qtype="|| ||^2"):
        """The norm_squared of a quaternion."""

        end_qtype = "||{}||^2".format(self.qtype, qtype)

        qxq = self._commuting_products(self)

        n_q = QH(qtype=end_qtype, representation=self.representation)
        n_q.t = qxq['tt'] + qxq['xx+yy+zz']

        return n_q

    def norm_squared_of_vector(self, qtype="|V( )|^2"):
        """The norm_squared of the vector of a quaternion."""

        end_qtype = "|V({})|^2".format(self.qtype)
        
        qxq = self._commuting_products(self)

        nv_q = QH(qtype=end_qtype, representation=self.representation)
        nv_q.t = qxq['xx+yy+zz']

        return nv_q

    def abs_of_q(self, qtype="||"):
        """The absolute value, the square root of the norm_squared."""

        end_qtype = "|{}|".format(self.qtype)
        
        a = self.norm_squared()
        sqrt_t = a.t ** (1/2)
        a.t = sqrt_t
        a.qtype = end_qtype
        a.representation = self.representation
        
        return a

    def normalize(self, n=1, qtype="U"):
        """Normalize a quaternion"""
        
        end_qtype = "{}{}".format(self.qtype, qtype)
        
        abs_q_inv = self.abs_of_q().invert()
        n_q = self.product(abs_q_inv).product(QH([n, 0, 0, 0]))
        n_q.qtype = end_qtype
        n_q.representation = self.representation
        
        return n_q
    
    def abs_of_vector(self, qtype="|V( )|"):
        """The absolute value of the vector, the square root of the norm_squared of the vector."""

        end_qtype = "|V({})|".format(self.qtype)
        
        av = self.norm_squared_of_vector(qtype=end_qtype)
        sqrt_t = av.t ** (1/2)
        av.t = sqrt_t
        av.representation = self.representation
        
        return av

    def add(self, qh_1, qtype=""):
        """Form a add given 2 quaternions."""

        self.check_representations(qh_1)
        
        end_qtype = "{f}+{s}".format(f=self.qtype, s=qh_1.qtype)
        
        t_1, x_1, y_1, z_1 = self.t, self.x, self.y, self.z
        t_2, x_2, y_2, z_2 = qh_1.t, qh_1.x, qh_1.y, qh_1.z

        add_q = QH(qtype=end_qtype, representation=self.representation)
        add_q.t = t_1 + t_2
        add_q.x = x_1 + x_2
        add_q.y = y_1 + y_2
        add_q.z = z_1 + z_2
        
        return add_q    

    def dif(self, qh_1, qtype=""):
        """Form a add given 2 quaternions."""
        
        self.check_representations(qh_1)

        end_qtype = "{f}-{s}".format(f=self.qtype, s=qh_1.qtype)
        
        t_1, x_1, y_1, z_1 = self.t, self.x, self.y, self.z
        t_2, x_2, y_2, z_2 = qh_1.t, qh_1.x, qh_1.y, qh_1.z

        dif_q = QH(qtype=end_qtype, representation=self.representation)
        dif_q.t = t_1 - t_2
        dif_q.x = x_1 - x_2
        dif_q.y = y_1 - y_2
        dif_q.z = z_1 - z_2
            
        return dif_q

    def product(self, q1, kind="", reverse=False, qtype=""):
        """Form a product given 2 quaternions. Kind can be '' aka standard, even, odd, or even_minus_odd.
        Setting reverse=True is like changing the order."""
        
        self.check_representations(q1)
        
        commuting = self._commuting_products(q1)
        q_even = QH()
        q_even.t = commuting['tt'] - commuting['xx+yy+zz']
        q_even.x = commuting['tx+xt']
        q_even.y = commuting['ty+yt']
        q_even.z = commuting['tz+zt']
        
        anti_commuting = self._anti_commuting_products(q1)
        q_odd = QH()
        
        if reverse:
            q_odd.x = anti_commuting['zy-yz']
            q_odd.y = anti_commuting['xz-zx']
            q_odd.z = anti_commuting['yx-xy']
            
        else:
            q_odd.x = anti_commuting['yz-zy']
            q_odd.y = anti_commuting['zx-xz']
            q_odd.z = anti_commuting['xy-yx']
        
        if kind == "":
            result = q_even.add(q_odd)
            times_symbol = "x"
        elif kind.lower() == "even":
            result = q_even
            times_symbol = "xE"
        elif kind.lower() == "odd":
            result = q_odd
            times_symbol = "xO"
        elif kind.lower() == "even_minus_odd":
            result = q_even.dif(q_odd)
            times_symbol = "xE-O"
        else:
            raise Exception("Four 'kind' values are known: '', 'even', 'odd', and 'even_minus_odd'.")
        
        if reverse:
            times_symbol = times_symbol.replace('x', 'xR')
        
        if qtype:
            result.qtype = qtype
        else:
            result.qtype = "{f}{ts}{s}".format(f=self.qtype, ts=times_symbol, s=q1.qtype)
            
        result.representation = self.representation
            
        return result

    def Euclidean_product(self, q1, kind="", reverse=False, qtype=""):
        """Form a product p* q given 2 quaternions, not associative."""

        self.check_representations(q1)
        
        pq = QH(qtype, representation=self.representation)
        pq = self.conj().product(q1, kind, reverse)
            
        return pq
    
    def invert(self, qtype="^-1"):
        """The inverse of a quaternion."""

        end_qtype = "{}{}".format(self.qtype, qtype)
        
        q_conj = self.conj()
        q_norm_squared = self.norm_squared()

        if (not self.is_symbolic()) and (q_norm_squared.t == 0):
            return self.q_0()

        q_norm_squared_inv = QH([1.0 / q_norm_squared.t, 0, 0, 0])
        q_inv = q_conj.product(q_norm_squared_inv)
        q_inv.qtype = end_qtype
        q_inv.representation = self.representation

        return q_inv

    def divide_by(self, q1, qtype=""):
        """Divide one quaternion by another. The order matters unless one is using a norm_squared (real number)."""
        
        self.check_representations(q1)
        
        end_qtype = "{f}/{s}".format(f=self.qtype, s=q1.qtype)
        
        q1_inv = q1.invert()
        q_div = self.product(q1.invert())
        q_div.qtype = end_qtype
        q_div.representation = self.representation
        
        return q_div

    def triple_product(self, q1, q2):
        """Form a triple product given 3 quaternions."""

        self.check_representations(q1)
        self.check_representations(q2)
        
        triple = self.product(q1).product(q2)
        triple.representation = self.representation
        
        return triple

    # Quaternion rotation involves a triple product:  UQU∗
    # where the U is a unitary quaternion (having a norm_squared of one).
    def rotate(self, a_1=0, a_2=0, a_3=0, qtype="rot"):
        """Do a rotation given up to three angles."""

        end_qtype = "{}{}".format(self.qtype, qtype)
        
        u = QH([0, a_1, a_2, a_3])
        u_abs = u.abs_of_q()
        u_norm_squaredalized = u.divide_by(u_abs)

        q_rot = u_norm_squaredalized.triple_product(self, u_norm_squaredalized.conj())
        q_rot.qtype = end_qtype
        q_rot.representation = self.representation
        
        return q_rot

    # A boost also uses triple products like a rotation, but more of them.
    # This is not a well-known result, but does work.
    # b -> b' = h b h* + 1/2 ((hhb)* -(h*h*b)*)
    # where h is of the form (cosh(a), sinh(a))
    def boost(self, beta_x=0, beta_y=0, beta_z=0, qtype="boost"):
        """A boost along the x, y, and/or z axis."""

        end_qtype = "{}{}".format(self.qtype, qtype)
        
        boost = QH(sr_gamma_betas(beta_x, beta_y, beta_z))      
        b_conj = boost.conj()

        triple_1 = boost.triple_product(self, b_conj)
        triple_2 = boost.triple_product(boost, self).conj()
        triple_3 = b_conj.triple_product(b_conj, self).conj()
      
        triple_23 = triple_2.dif(triple_3)
        half_23 = triple_23.product(QH([0.5, 0, 0, 0]))
        triple_123 = triple_1.add(half_23, qtype=end_qtype)
        triple_123.qtype = end_qtype
        triple_123.representation = self.representation
        
        return triple_123

    # g_shift is a function based on the space-times-time invariance proposal for gravity,
    # which proposes that if one changes the distance from a gravitational source, then
    # squares a measurement, the observers at two different hieghts agree to their
    # space-times-time values, but not the intervals.
    # g_form is the form of the function, either minimal or exponential
    # Minimal is what is needed to pass all weak field tests of gravity
    def g_shift(self, dimensionless_g, g_form="exp", qtype="g_shift"):
        """Shift an observation based on a dimensionless GM/c^2 dR."""

        end_qtype = "{}{}".format(self.qtype, qtype)
        
        if g_form == "exp":
            g_factor = sp.exp(dimensionless_g)
        elif g_form == "minimal":
            g_factor = 1 + 2 * dimensionless_g + 2 * dimensionless_g ** 2
        else:
            print("g_form not defined, should be 'exp' or 'minimal': {}".format(g_form))
            return self

        g_q = QH(qtype=end_qtype)
        g_q.t = self.t / g_factor
        g_q.x = self.x * g_factor
        g_q.y = self.y * g_factor
        g_q.z = self.z * g_factor
        g_q.qtype = end_qtype
        g_q.representation = self.representation
        
        return g_q
    
    def sin(self, qtype="sin"):
        """Take the sine of a quaternion, (sin(t) cosh(|R|), cos(t) sinh(|R|) R/|R|)"""

        end_qtype = "sin({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        
        if abs_v.t == 0:    
            return QH([math.sin(self.t), 0, 0, 0], qtype=end_qtype)
            
        sint = math.sin(self.t)
        cost = math.cos(self.t)
        sinhR = math.sinh(abs_v.t)
        coshR = math.cosh(abs_v.t)
        
        k = cost * sinhR / abs_v.t
            
        q_out = QH()
        q_out.t = sint * coshR
        q_out.x = k * self.x
        q_out.y = k * self.y
        q_out.z = k * self.z
        
        q_out.qtype = end_qtype
        q_out.representation = self.representation
        
        return q_out
     
    def cos(self, qtype="sin"):
        """Take the cosine of a quaternion, (cos(t) cosh(|R|), sin(t) sinh(|R|) R/|R|)"""

        end_qtype = "cos({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        
        if abs_v.t == 0:    
            return QH([math.cos(self.t), 0, 0, 0], qtype=end_qtype)
            
        sint = math.sin(self.t)
        cost = math.cos(self.t)
        sinhR = math.sinh(abs_v.t)
        coshR = math.cosh(abs_v.t)
        
        k = -1 * sint * sinhR / abs_v.t
            
        q_out = QH()
        q_out.t = cost * coshR
        q_out.x = k * self.x
        q_out.y = k * self.y
        q_out.z = k * self.z

        q_out.qtype = end_qtype
        q_out.representation = self.representation
        
        return q_out
    
    def tan(self, qtype="sin"):
        """Take the tan of a quaternion, sin/cos"""

        end_qtype = "tan({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        
        if abs_v.t == 0:    
            return QH([math.tan(self.t), 0, 0, 0], qtype=end_qtype)
            
        sinq = self.sin()
        cosq = self.cos()
        q_out = sinq.divide_by(cosq) 
        
        q_out.qtype = end_qtype
        q_out.representation = self.representation
        
        return q_out
    
    def sinh(self, qtype="sinh"):
        """Take the sinh of a quaternion, (sinh(t) cos(|R|), cosh(t) sin(|R|) R/|R|)"""

        end_qtype = "sinh({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        
        if abs_v.t == 0:    
            return QH([math.sinh(self.t), 0, 0, 0], qtype=end_qtype)
            
        sinht = math.sinh(self.t)
        cosht = math.cosh(self.t)
        sinR = math.sin(abs_v.t)
        cosR = math.cos(abs_v.t)
        
        k = cosht * sinR / abs_v.t
            
        q_out = QH(qtype=end_qtype, representation=self.representation)
        q_out.t = sinht * cosR
        q_out.x = k * self.x
        q_out.y = k * self.y
        q_out.z = k * self.z

        return q_out
     
    def cosh(self, qtype="sin"):
        """Take the cosh of a quaternion, (cosh(t) cos(|R|), sinh(t) sin(|R|) R/|R|)"""

        end_qtype = "cosh({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        
        if abs_v.t == 0:    
            return QH([math.cosh(self.t), 0, 0, 0], qtype=end_qtype)
            
        sinht = math.sinh(self.t)
        cosht = math.cosh(self.t)
        sinR = math.sin(abs_v.t)
        cosR = math.cos(abs_v.t)
        
        k = sinht * sinR / abs_v.t
            
        q_out = QH(qtype=end_qtype, representation=self.representation)
        q_out.t = cosht * cosR
        q_out.x = k * self.x
        q_out.y = k * self.y
        q_out.z = k * self.z

        return q_out
    
    def tanh(self, qtype="tanh"):
        """Take the tanh of a quaternion, sin/cos"""

        end_qtype = "tanh({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        
        if abs_v.t == 0:    
            return QH([math.tanh(self.t), 0, 0, 0], qtype=end_qtype)
            
        sinhq = self.sinh()
        coshq = self.cosh()
            
        q_out = sinhq.divide_by(coshq) 
        
        q_out.qtype = end_qtype
        q_out.representation = self.representation
        
        return q_out
    
    def exp(self, qtype="exp"):
        """Take the exponential of a quaternion."""
        # exp(q) = (exp(t) cos(|R|, exp(t) sin(|R|) R/|R|)
        
        end_qtype = "exp({st})".format(st=self.qtype)
        
        abs_v = self.abs_of_vector()
        et = math.exp(self.t)
        
        if (abs_v.t == 0):
            return QH([et, 0, 0, 0], qtype=end_qtype)
        
        cosR = math.cos(abs_v.t)
        sinR = math.sin(abs_v.t)
        k = et * sinR / abs_v.t
                       
        expq = QH([et * cosR, k * self.x, k * self.y, k * self.z], qtype=end_qtype, representation=self.representation)
                       
        return expq
    
    def ln(self, qtype="ln"):
        """Take the natural log of a quaternion."""
        # ln(q) = (0.5 ln t^2 + R.R, atan2(|R|, t) R/|R|)
        
        end_qtype = "ln({st})".format(st=self.qtype)
        
        abs_v = self.abs_of_vector()
        
        if (abs_v.t == 0):
            if self.t > 0:
                return(QH([math.log(self.t), 0, 0, 0], qtype=end_qtype))
            else:
                # I don't understant this, but mathematica does the same thing.
                return(QH([math.log(-self.t), math.pi, 0, 0], qtype=end_type))   
            
            return QH([lt, 0, 0, 0])
        
        t_value = 0.5 * math.log(self.t * self.t + abs_v.t * abs_v.t)
        k = math.atan2(abs_v.t, self.t) / abs_v.t
                       
        expq = QH([t_value, k * self.x, k * self.y, k * self.z], qtype=end_qtype, representation=self.representation)
                       
        return expq
    
    def q_2_q(self, q1, qtype="P"):
        """Take the natural log of a quaternion."""
        # q^p = exp(ln(q) * p)
        
        self.check_representations(q1)
        end_qtype = "{st}^P".format(st=self.qtype)
        
        q2q = self.ln().product(q1).exp()           
        q2q.qtype = end_qtype
        q2q.representation = self.representation
        
        return q2q
    
    def trunc(self):
        """Truncates values."""
        
        self.t = math.trunc(self.t)
        self.x = math.trunc(self.x)
        self.y = math.trunc(self.y)
        self.z = math.trunc(self.z)
        
        return self


# Write tests the QH class.

# In[41]:


class TestQH(unittest.TestCase):
    """Class to make sure all the functions work as expected."""

    Q = QH([1, -2, -3, -4], qtype="Q")
    P = QH([0, 4, -3, 0], qtype="P")
    R = QH([3, 0, 0, 0], qtype="R")
    C = QH([2, 4, 0, 0], qtype="C")

    def test_qt(self):
        self.assertTrue(self.Q.t == 1)

    def test_q_0(self):
        q_z = self.Q.q_0()
        print("q_0: ", q_z)
        self.assertTrue(q_z.t == 0)
        self.assertTrue(q_z.x == 0)
        self.assertTrue(q_z.y == 0)
        self.assertTrue(q_z.z == 0)

    def test_q_1(self):
        q_z = self.Q.q_1()
        print("q_1: ", q_z)
        self.assertTrue(q_z.t == 1)
        self.assertTrue(q_z.x == 0)
        self.assertTrue(q_z.y == 0)
        self.assertTrue(q_z.z == 0)

    def test_q_i(self):
        q_z = self.Q.q_i()
        print("q_i: ", q_z)
        self.assertTrue(q_z.t == 0)
        self.assertTrue(q_z.x == 1)
        self.assertTrue(q_z.y == 0)
        self.assertTrue(q_z.z == 0)

    def test_q_j(self):
        q_z = self.Q.q_j()
        print("q_j: ", q_z)
        self.assertTrue(q_z.t == 0)
        self.assertTrue(q_z.x == 0)
        self.assertTrue(q_z.y == 1)
        self.assertTrue(q_z.z == 0)

    def test_q_k(self):
        q_z = self.Q.q_k()
        print("q_k: ", q_z)
        self.assertTrue(q_z.t == 0)
        self.assertTrue(q_z.x == 0)
        self.assertTrue(q_z.y == 0)
        self.assertTrue(q_z.z == 1)
        
    def test_q_random(self):
        q_z = QH().q_random()
        print("q_random():", q_z)
        self.assertTrue(q_z.t >= 0 and q_z.t <= 1)
        self.assertTrue(q_z.x >= 0 and q_z.x <= 1)
        self.assertTrue(q_z.y >= 0 and q_z.y <= 1)
        self.assertTrue(q_z.z >= 0 and q_z.z <= 1)
        
    def test_equals(self):
        self.assertTrue(self.Q.equals(self.Q))
        self.assertFalse(self.Q.equals(self.P))

    def test_conj_0(self):
        q_z = self.Q.conj()
        print("q_conj 0: ", q_z)
        self.assertTrue(q_z.t == 1)
        self.assertTrue(q_z.x == 2)
        self.assertTrue(q_z.y == 3)
        self.assertTrue(q_z.z == 4)

    def test_conj_1(self):
        q_z = self.Q.conj(1)
        print("q_conj 1: ", q_z)
        self.assertTrue(q_z.t == -1)
        self.assertTrue(q_z.x == -2)
        self.assertTrue(q_z.y == 3)
        self.assertTrue(q_z.z == 4)

    def test_conj_2(self):
        q_z = self.Q.conj(2)
        print("q_conj 2: ", q_z)
        self.assertTrue(q_z.t == -1)
        self.assertTrue(q_z.x == 2)
        self.assertTrue(q_z.y == -3)
        self.assertTrue(q_z.z == 4)
        
    def sign_flips(self):
        q_z = self.Q.sign_flips()
        print("sign_flips: ", q_z)
        self.assertTrue(q_z.t == -1)
        self.assertTrue(q_z.x == 2)
        self.assertTrue(q_z.y == 3)
        self.assertTrue(q_z.z == 4)
    
    def test_vahlen_conj_minus(self):
        q_z = self.Q.vahlen_conj()
        print("q_vahlen_conj -: ", q_z)
        self.assertTrue(q_z.t == 1)
        self.assertTrue(q_z.x == 2)
        self.assertTrue(q_z.y == 3)
        self.assertTrue(q_z.z == 4)

    def test_vahlen_conj_star(self):
        q_z = self.Q.vahlen_conj('*')
        print("q_vahlen_conj *: ", q_z)
        self.assertTrue(q_z.t == 1)
        self.assertTrue(q_z.x == -2)
        self.assertTrue(q_z.y == -3)
        self.assertTrue(q_z.z == 4)

    def test_vahlen_conj_prime(self):
        q_z = self.Q.vahlen_conj("'")
        print("q_vahlen_conj ': ", q_z)
        self.assertTrue(q_z.t == 1)
        self.assertTrue(q_z.x == 2)
        self.assertTrue(q_z.y == 3)
        self.assertTrue(q_z.z == -4)

    def test_square(self):
        q_z = self.Q.square()
        print("square: ", q_z)
        self.assertTrue(q_z.t == -28)
        self.assertTrue(q_z.x == -4)
        self.assertTrue(q_z.y == -6)
        self.assertTrue(q_z.z == -8)

    def test_norm_squared(self):
        q_z = self.Q.norm_squared()
        print("norm_squared: ", q_z)
        self.assertTrue(q_z.t == 30)
        self.assertTrue(q_z.x == 0)
        self.assertTrue(q_z.y == 0)
        self.assertTrue(q_z.z == 0)

    def test_norm_squared_of_vector(self):
        q_z = self.Q.norm_squared_of_vector()
        print("norm_squared_of_vector: ", q_z)
        self.assertTrue(q_z.t == 29)
        self.assertTrue(q_z.x == 0)
        self.assertTrue(q_z.y == 0)
        self.assertTrue(q_z.z == 0)
        
    def test_abs_of_q(self):
        q_z = self.P.abs_of_q()
        print("abs_of_q: ", q_z)
        self.assertTrue(q_z.t == 5)
        self.assertTrue(q_z.x == 0)
        self.assertTrue(q_z.y == 0)
        self.assertTrue(q_z.z == 0)
        
    def test_normalize(self):
        q_z = self.P.normalize()
        print("q_normalized: ", q_z)
        self.assertTrue(q_z.t == 0)
        self.assertTrue(q_z.x == 0.8)
        self.assertAlmostEqual(q_z.y, -0.6)
        self.assertTrue(q_z.z == 0)
        
    def test_abs_of_vector(self):
        q_z = self.P.abs_of_vector()
        print("abs_of_vector: ", q_z)
        self.assertTrue(q_z.t == 5)
        self.assertTrue(q_z.x == 0)
        self.assertTrue(q_z.y == 0)
        self.assertTrue(q_z.z == 0)
        
    def test_add(self):
        q_z = self.Q.add(self.P)
        print("add: ", q_z)
        self.assertTrue(q_z.t == 1)
        self.assertTrue(q_z.x == 2)
        self.assertTrue(q_z.y == -6)
        self.assertTrue(q_z.z == -4)
        
    def test_dif(self):
        q_z = self.Q.dif(self.P)
        print("dif: ", q_z)
        self.assertTrue(q_z.t == 1)
        self.assertTrue(q_z.x == -6)
        self.assertTrue(q_z.y == 0)
        self.assertTrue(q_z.z == -4) 

    def test_product(self):
        q_z = self.Q.product(self.P)
        print("product: ", q_z)
        self.assertTrue(q_z.t == -1)
        self.assertTrue(q_z.x == -8)
        self.assertTrue(q_z.y == -19)
        self.assertTrue(q_z.z == 18)
        
    def test_product_even(self):
        q_z = self.Q.product(self.P, kind="even")
        print("product, kind even: ", q_z)
        self.assertTrue(q_z.t == -1)
        self.assertTrue(q_z.x == 4)
        self.assertTrue(q_z.y == -3)
        self.assertTrue(q_z.z == 0)
        
    def test_product_odd(self):
        q_z = self.Q.product(self.P, kind="odd")
        print("product, kind odd: ", q_z)
        self.assertTrue(q_z.t == 0)
        self.assertTrue(q_z.x == -12)
        self.assertTrue(q_z.y == -16)
        self.assertTrue(q_z.z == 18)
        
    def test_product_even_minus_odd(self):
        q_z = self.Q.product(self.P, kind="even_minus_odd")
        print("product, kind even_minus_odd: ", q_z)
        self.assertTrue(q_z.t == -1)
        self.assertTrue(q_z.x == 16)
        self.assertTrue(q_z.y == 13)
        self.assertTrue(q_z.z == -18)
        
    def test_product_reverse(self):
        q1q2_rev = self.Q.product(self.P, reverse=True)
        q2q1 = self.P.product(self.Q)
        self.assertTrue(q1q2_rev.equals(q2q1))
        
    def test_Euclidean_product(self):
        q_z = self.Q.Euclidean_product(self.P)
        print("Euclidean product: ", q_z)
        self.assertTrue(q_z.t == 1)
        self.assertTrue(q_z.x == 16)
        self.assertTrue(q_z.y == 13)
        self.assertTrue(q_z.z == -18)
        
    def test_invert(self):
        q_z = self.P.invert()
        print("invert: ", q_z)
        self.assertTrue(q_z.t == 0)
        self.assertTrue(q_z.x == -0.16)
        self.assertTrue(q_z.y == 0.12)
        self.assertTrue(q_z.z == 0)
                
    def test_divide_by(self):
        q_z = self.Q.divide_by(self.Q)
        print("divide_by: ", q_z)
        self.assertTrue(q_z.t == 1)
        self.assertTrue(q_z.x == 0)
        self.assertTrue(q_z.y == 0)
        self.assertTrue(q_z.z == 0) 
        
    def test_triple_product(self):
        q_z = self.Q.triple_product(self.P, self.Q)
        print("triple product: ", q_z)
        self.assertTrue(q_z.t == -2)
        self.assertTrue(q_z.x == 124)
        self.assertTrue(q_z.y == -84)
        self.assertTrue(q_z.z == 8)
        
    def test_rotate(self):
        q_z = self.Q.rotate(1)
        print("rotate: ", q_z)
        self.assertTrue(q_z.t == 1)
        self.assertTrue(q_z.x == -2)
        self.assertTrue(q_z.y == 3)
        self.assertTrue(q_z.z == 4)
        
    def test_boost(self):
        q1_sq = self.Q.square()
        q_z = self.Q.boost(0.003)
        q_z2 = q_z.square()
        print("q1_sq: ", q1_sq)
        print("boosted: ", q_z)
        print("boosted squared: ", q_z2)
        self.assertTrue(round(q_z2.t, 5) == round(q1_sq.t, 5))

    def test_g_shift(self):
        q1_sq = self.Q.square()
        q_z = self.Q.g_shift(0.003)
        q_z2 = q_z.square()
        q_z_minimal = self.Q.g_shift(0.003, g_form="minimal")
        q_z2_minimal = q_z_minimal.square()
        print("q1_sq: ", q1_sq)
        print("g_shift: ", q_z)
        print("g squared: ", q_z2)
        self.assertTrue(q_z2.t != q1_sq.t)
        self.assertTrue(q_z2.x == q1_sq.x)
        self.assertTrue(q_z2.y == q1_sq.y)
        self.assertTrue(q_z2.z == q1_sq.z)
        self.assertTrue(q_z2_minimal.t != q1_sq.t)
        self.assertTrue(q_z2_minimal.x == q1_sq.x)
        self.assertTrue(q_z2_minimal.y == q1_sq.y)
        self.assertTrue(q_z2_minimal.z == q1_sq.z)
        
    def test_sin(self):
        self.assertTrue(QH([0, 0, 0, 0]).sin().equals(QH().q_0()))
        self.assertTrue(self.Q.sin().equals(QH([91.7837157840346691, -21.8864868530291758, -32.8297302795437673, -43.7729737060583517])))
        self.assertTrue(self.P.sin().equals(QH([0,  59.3625684622310033, -44.5219263466732542, 0])))
        self.assertTrue(self.R.sin().equals(QH([0.1411200080598672, 0, 0, 0])))
        self.assertTrue(self.C.sin().equals(QH([24.8313058489463785, -11.3566127112181743, 0, 0])))
     
    def test_cos(self):
        self.assertTrue(QH([0, 0, 0, 0]).cos().equals(QH().q_1()))
        self.assertTrue(self.Q.cos().equals(QH([58.9336461679439481, 34.0861836904655959, 51.1292755356983974, 68.1723673809311919])))
        self.assertTrue(self.P.cos().equals(QH([74.2099485247878476, 0, 0, 0])))
        self.assertTrue(self.R.cos().equals(QH([-0.9899924966004454, 0, 0, 0])))
        self.assertTrue(self.C.cos().equals(QH([-11.3642347064010600, -24.8146514856341867, 0, 0])))
                            
    def test_tan(self):
        self.assertTrue(QH([0, 0, 0, 0]).tan().equals(QH().q_0()))
        self.assertTrue(self.Q.tan().equals(QH([0.0000382163172501, -0.3713971716439372, -0.5570957574659058, -0.7427943432878743])))
        self.assertTrue(self.P.tan().equals(QH([0, 0.7999273634100760, -0.5999455225575570, 0])))
        self.assertTrue(self.R.tan().equals(QH([-0.1425465430742778, 0, 0, 0])))
        self.assertTrue(self.C.tan().equals(QH([-0.0005079806234700, 1.0004385132020521, 0, 0])))
        
    def test_sinh(self):
        self.assertTrue(QH([0, 0, 0, 0]).sinh().equals(QH().q_0()))
        self.assertTrue(self.Q.sinh().equals(QH([0.7323376060463428, 0.4482074499805421, 0.6723111749708131, 0.8964148999610841])))
        self.assertTrue(self.P.sinh().equals(QH([0, -0.7671394197305108, 0.5753545647978831, 0])))
        self.assertTrue(self.R.sinh().equals(QH([10.0178749274099026, 0, 0, 0])))
        self.assertTrue(self.C.sinh().equals(QH([-2.3706741693520015, -2.8472390868488278, 0, 0])))    
        
    def test_cosh(self):
        self.assertTrue(QH([0, 0, 0, 0]).cosh().equals(QH().q_1()))
        self.assertTrue(self.Q.cosh().equals(QH([0.9615851176369565, 0.3413521745610167, 0.5120282618415251, 0.6827043491220334])))
        self.assertTrue(self.P.cosh().equals(QH([0.2836621854632263, 0, 0, 0])))
        self.assertTrue(self.R.cosh().equals(QH([10.0676619957777653, 0, 0, 0])))
        self.assertTrue(self.C.cosh().equals(QH([-2.4591352139173837, -2.7448170067921538, 0, 0])))    
        
    def test_tanh(self):
        self.assertTrue(QH([0, 0, 0, 0]).tanh().equals(QH().q_0()))
        self.assertTrue(self.Q.tanh().equals(QH([1.0248695360556623, 0.1022956817887642, 0.1534435226831462, 0.2045913635775283])))
        self.assertTrue(self.P.tanh().equals(QH([0, -2.7044120049972684, 2.0283090037479505, 0])))
        self.assertTrue(self.R.tanh().equals(QH([0.9950547536867305, 0, 0, 0])))
        self.assertTrue(self.C.tanh().equals(QH([1.0046823121902353, 0.0364233692474038, 0, 0])))    
        
    def test_exp(self):
        self.assertTrue(QH([0, 0, 0, 0]).exp().equals(QH().q_1()))
        self.assertTrue(self.Q.exp().equals(QH([1.6939227236832994, 0.7895596245415588, 1.1843394368123383, 1.5791192490831176])))
        self.assertTrue(self.P.exp().equals(QH([0.2836621854632263, -0.7671394197305108, 0.5753545647978831, 0])))
        self.assertTrue(self.R.exp().equals(QH([20.0855369231876679, 0, 0, 0])))
        self.assertTrue(self.C.exp().equals(QH([-4.8298093832693851, -5.5920560936409816, 0, 0])))    
    
    def test_ln(self):
        self.assertTrue(self.Q.ln().exp().equals(self.Q))
        self.assertTrue(self.Q.ln().equals(QH([1.7005986908310777, -0.5151902926640850, -0.7727854389961275, -1.0303805853281700])))
        self.assertTrue(self.P.ln().equals(QH([1.6094379124341003, 1.2566370614359172, -0.9424777960769379, 0])))
        self.assertTrue(self.R.ln().equals(QH([1.0986122886681098, 0, 0, 0])))
        self.assertTrue(self.C.ln().equals(QH([1.4978661367769954, 1.1071487177940904, 0, 0])))    
        
    def test_q_2_q(self):
        self.assertTrue(self.Q.q_2_q(self.P).equals(QH([-0.0197219653530713, -0.2613955437374326, 0.6496281248064009, -0.3265786562423951])))

suite = unittest.TestLoader().loadTestsFromModule(TestQH())
unittest.TextTestRunner().run(suite);


# In[42]:


class TestQHRep(unittest.TestCase):
    Q12 = QH([1, 2, 0, 0])
    Q1123 = QH([1, 1, 2, 3])
    Q11p = QH([1, 1, 0, 0], representation="polar")
    Q12p = QH([1, 2, 0, 0], representation="polar")
    Q12np = QH([1, -2, 0, 0], representation="polar")
    Q21p = QH([2, 1, 0, 0], representation="polar")
    Q23p = QH([2, 3, 0, 0], representation="polar")
    Q13p = QH([1, 3, 0, 0], representation="polar")
    Q5p = QH([5, 0, 0, 0], representation="polar")
    
    def test_txyz_2_representation(self):
        qr = QH(self.Q12.txyz_2_representation(""))
        self.assertTrue(qr.equals(self.Q12))
        qr = QH(self.Q12.txyz_2_representation("polar"))
        self.assertTrue(qr.equals(QH([2.23606797749979, 1.10714871779409, 0, 0])))
        qr = QH(self.Q1123.txyz_2_representation("spherical"))
        self.assertTrue(qr.equals(QH([1.0, 3.7416573867739413, 0.640522312679424, 1.10714871779409])))
        
    def test_representation_2_txyz(self):
        qr = QH(self.Q12.representation_2_txyz(""))
        self.assertTrue(qr.equals(self.Q12))
        qr = QH(self.Q12.representation_2_txyz("polar"))
        self.assertTrue(qr.equals(QH([-0.4161468365471424, 0.9092974268256817, 0, 0])))
        qr = QH(self.Q1123.representation_2_txyz("spherical"))
        self.assertTrue(qr.equals(QH([1.0, -0.9001976297355174, 0.12832006020245673, -0.4161468365471424])))
    
    def test_polar_products(self):
        qr = self.Q11p.product(self.Q12p)
        print("polar 1 1 0 0 * 1 2 0 0: ", qr)
        self.assertTrue(qr.equals(self.Q13p))
        qr = self.Q12p.product(self.Q21p)
        print("polar 1 2 0 0 * 2 1 0 0: ", qr)
        self.assertTrue(qr.equals(self.Q23p))

    def test_polar_conj(self):
        qr = self.Q12p.conj()
        print("polar conj of 1 2 0 0: ", qr)
        self.assertTrue(qr.equals(self.Q12np))
        
suite = unittest.TestLoader().loadTestsFromModule(TestQHRep())
unittest.TextTestRunner().run(suite);


# ## Numpy Arrays for Hamilton

# A separate class is needed for numpy array due to technical issues I have getting sympy and numpy to play nicely with each other...

# In[43]:


class QHa(object):
    """Quaternions as nparrays."""

    def __init__(self, values=None, qtype="Q", representation=""):
        if values is None:
            self.a = np.array([0.0, 0.0, 0.0, 0.0])
        elif len(values) == 4:
            self.a = np.array([values[0], values[1], values[2], values[3]])

        elif len(values) == 8:
            self.a = np.array([values[0] - values[1], values[2] - values[3], values[4] - values[5], values[6] - values[7]])
        
        self.representation = representation
        
        if representation != "":
            txyz = self.representation_2_txyz(representation)
            self.a = np.array(txyz)
            
        self.qtype = qtype

    def __str__(self, quiet=False):
        """Customize the output."""
        
        qtype = self.qtype
        
        if quiet:
            qtype = ""
            
        if self.representation == "":
            string = "({t}, {x}, {y}, {z}) {qt}".format(t=self.a[0], x=self.a[1], y=self.a[2], z=self.a[3], qt=qtype)
    
        elif self.representation == "polar":
            rep = self.txyz_2_representation("polar")
            
            string = "({A} A, {thetaX} 𝜈x, {thetaY} 𝜈y, {thetaZ} 𝜈z) {qt}".format(
                A=rep[0], thetaX=rep[1], thetaY=rep[2], thetaZ=rep[3], qt=qtype)
 
        elif self.representation == "spherical":
            rep = self.txyz_2_representation("spherical")
            string = "({t} t, {R} R, {theta} θ, {phi} φ) {qt}".format(t=rep[0], R=rep[1], theta=rep[2], phi=rep[3], 
                                                                       qt=qtype)
        
        return string
    
    def is_symbolic(self):
        """Figures out if an expression is symbolic."""
        
        symbolic = False
        
        if hasattr(self.a[0], "free_symbols") or hasattr(self.a[1], "free_symbols") or             hasattr(self.a[2], "free_symbols") or hasattr(self.a[3], "free_symbols"): 
            symbolic = True
        
        return symbolic
    
    def txyz_2_representation(self, representation):
        """Converts Cartesian txyz into an array of 4 values in a different representation."""

        symbolic = self.is_symbolic()
        
        if representation == "":
            rep = [self.a[0], self.a[1], self.a[2], self.a[3]]
        
        elif representation == "polar":
            amplitude = (self.a[0] ** 2 + self.a[1] ** 2 + self.a[2] ** 2 + self.a[3] ** 2) ** (1/2)
            
            abs_v = self.abs_of_vector().a[0]
            
            if symbolic:
                theta = sp.atan2(abs_v, self.a[0])
            else:
                theta = math.atan2(abs_v, self.a[0])
                
            if abs_v == 0:
                thetaX, thetaY, thetaZ = 0, 0, 0
                
            else:
                thetaX = theta * self.a[1] / abs_v
                thetaY = theta * self.a[2] / abs_v
                thetaZ = theta * self.a[3] / abs_v
            
            rep = [amplitude, thetaX, thetaY, thetaZ]
        
        elif representation == "spherical":
            
            t = self.a[0]
            
            R = (self.a[1] ** 2 + self.a[2] **2 + self.a[3] **2) ** (1/2)
            
            if R == 0:
                theta = 0
            else:
                if symbolic:
                    theta = sp.acos(self.a[3] / R)
                
                else:
                    theta = math.acos(self.a[3] / R)
                
            if symbolic:
                phi = sp.atan2(self.a[2], self.a[1])
            else:
                phi = math.atan2(self.a[2], self.a[1])
                
            rep = [t, R, theta, phi]
        
        else:
            print("Oops, don't know representation: ", representation)
            
        return rep
    
    def representation_2_txyz(self, representation):
        """Convert from a representation to Cartesian txyz."""
        
        symbolic = False
        
        if hasattr(self.a[0], "free_symbols") or hasattr(self.a[1], "free_symbols") or             hasattr(self.a[2], "free_symbols") or hasattr(self.a[3], "free_symbols"): 
            symbolic = True

        if representation == "":
            t, x, y, z = self.a[0], self.a[1], self.a[2], self.a[3]
        
        elif representation == "polar":
            amplitude, thetaX, thetaY, thetaZ = self.a[0], self.a[1], self.a[2], self.a[3]
        
            theta = (thetaX ** 2 + thetaY ** 2 + thetaZ ** 2) ** (1/2)
            
            if theta == 0:
                t = amplitude
                x, y, z = 0, 0, 0
                
            else:
                if symbolic:
                    t = amplitude * sp.cos(theta)
                    x = self.a[1] / theta * amplitude * sp.sin(theta)
                    y = self.a[2] / theta * amplitude * sp.sin(theta)
                    z = self.a[3] / theta * amplitude * sp.sin(theta)
                
                else:
                    t = amplitude * math.cos(theta)
                    x = self.a[1] / theta * amplitude * math.sin(theta)
                    y = self.a[2] / theta * amplitude * math.sin(theta)
                    z = self.a[3] / theta * amplitude * math.sin(theta)
                
        elif representation == "spherical":
            t, R, theta, phi = self.a[0], self.a[1], self.a[2], self.a[3]

            if symbolic:
                x = R * sp.sin(theta) * sp.cos(phi)
                y = R * sp.sin(theta) * sp.sin(phi)
                z = R * sp.cos(theta)
            else:
                x = R * math.sin(theta) * math.cos(phi)
                y = R * math.sin(theta) * math.sin(phi)
                z = R * math.cos(theta)
            
        else:
            print("Oops, don't know representation: ", representation)
            
        txyz = [t, x, y, z]
        
        return txyz 
        
    def check_representations(self, q1):
        """If they are the same, report true. If not, kick out an exception. Don't add apples to oranges."""

        if self.representation == q1.representation:
            return True
        
        else:
            raise Exception("Oops, 2 quaternions have different representations: {}, {}".format(self.representation, q1.representation))
            return False
    
    def display_q(self):
        """display each terms in a pretty way."""

        display(self.a[0])
        display(self.a[1])
        display(self.a[2])
        display(self.a[3])
        return

    def simple_q(self):
        """display each terms in a pretty way."""
        
        self.a[0] = sp.simplify(self.a[0])
        self.a[1] = sp.simplify(self.a[1])
        self.a[2] = sp.simplify(self.a[2])
        self.a[3] = sp.simplify(self.a[3])
        return
    
    def q_0(self, qtype="0"):
        """Return a zero quaternion."""

        q0 = QHa(qtype=qtype, representation=self.representation)
        return q0

    def q_1(self, qtype="1"):
        """Return a multiplicative identity quaternion."""
    
        q1 = QHa([1.0, 0.0, 0.0, 0.0], qtype=qtype, representation=self.representation)
        return q1
    
    def q_i(self, qtype="i"):
        """Return i."""

        qi = QHa([0.0, 1.0, 0.0, 0.0], qtype=qtype, representation=self.representation)
        return qi

    def q_j(self, qtype="j"):
        """Return j."""
        
        qj = QHa([0.0, 0.0, 1.0, 0.0], qtype=qtype, representation=self.representation)
        return qj
    
    def q_k(self, qtype="k"):
        """Return k."""

        qk = QHa([0.0, 0.0, 0.0, 1.0], qtype=qtype, representation=self.representation)
        return qk

    def q_random(self, qtype="?"):
        """Return a random-valued quaternion."""

        qr = QHa([random.random(), random.random(), random.random(), random.random()], qtype=qtype, representation=self.representation)
        return qr
    
    def dupe(self, qtype=""):
        """Return a duplicate copy, good for testing since qtypes persist"""
        
        return QHa([self.a[0], self.a[1], self.a[2], self.a[3]], qtype=self.qtype, representation=self.representation)
    
    def equals(self, q1):
        """Tests if two quaternions are equal."""
        
        self.check_representations(q1)
        
        self_t, self_x, self_y, self_z = sp.expand(self.a[0]), sp.expand(self.a[1]), sp.expand(self.a[2]), sp.expand(self.a[3])
        q1_t, q1_x, q1_y, q1_z = sp.expand(q1.a[0]), sp.expand(q1.a[1]), sp.expand(q1.a[2]), sp.expand(q1.a[3])
        
        if math.isclose(self_t, q1_t) and math.isclose(self_x, q1_x) and math.isclose(self_y, q1_y) and math.isclose(self_z, q1_z):
            return True
        
        else:
            return False
        
    def conj(self, conj_type=0, qtype="*"):
        """Three types of conjugates."""

        t, x, y, z = self.a[0], self.a[1], self.a[2], self.a[3]
        conj_q = QHa()

        if conj_type == 0:
            conj_q.a[0] = 1.0 * t
            if x != 0:
                conj_q.a[1] = -1.0 * x
            if y != 0:
                conj_q.a[2] = -1.0 * y
            if z != 0:
                conj_q.a[3] = -1.0 * z

        if conj_type == 1:
            if t != 0:
                conj_q.a[0] = -1.0 * t
            conj_q.a[1] = 1.0 * x
            if y != 0:
                conj_q.a[2] = -1.0 * y
            if z != 0:
                conj_q.a[3] = -1.0 * z
            qtype += "1"
            
        if conj_type == 2:
            if t != 0:
                conj_q.a[0] = -1 * t
            if x != 0:
                conj_q.a[1] = -1 * x
            conj_q.a[2] = 1.0 * y
            if z != 0:
                conj_q.a[3] = -1 * z
            qtype += "2"
            
        conj_q.qtype = self.qtype + qtype
        conj_q.representation = self.representation
    
        return conj_q
    
    def flip_signs(self, conj_type=0, qtype="-"):
        """Flip all the signs, just like multipying by -1."""

        end_qtype = "-{}".format(self.qtype)
        
        t, x, y, z = self.a[0], self.a[1], self.a[2], self.a[3]
        flip_q = QHa(qtype=end_qtype)

        if t != 0:
            flip_q.a[0] = -1.0 * t
        if x != 0:
            flip_q.a[1] = -1.0 * x
        if y != 0:
            flip_q.a[2] = -1.0 * y
        if z != 0:
            flip_q.a[3] = -1.0 * z
        
        flip_q.qtype = end_qtype
        flip_q.representation = self.representation
        
        return flip_q
    
    def vahlen_conj(self, conj_type="-", qtype="vc"):
        """Three types of conjugates -'* done by Vahlen in 1901."""

        t, x, y, z = self.a[0], self.a[1], self.a[2], self.a[3]
        conj_q = QHa()

        if conj_type == '-':
            conj_q.a[0] = 1.0 * t
            if x != 0:
                conj_q.a[1] = -1.0 * x
            if y != 0:
                conj_q.a[2] = -1.0 * y
            if z != 0:
                conj_q.a[3] = -1.0 * z
            qtype += "*-"

        if conj_type == "'":
            conj_q.a[0] = 1.0 * t
            if x != 0:
                conj_q.a[1] = -1.0 * x
            if y != 0:
                conj_q.a[2] = -1.0 * y
            conj_q.a[3] = 1.0 * z
            qtype += "*'"
            
        if conj_type == '*':
            conj_q.a[0] = 1.0 * t
            conj_q.a[1] = 1.0 * x
            conj_q.a[2] = 1.0 * y
            if z != 0:
                conj_q.a[3] = -1.0 * z
            qtype += "*"
            
        conj_q.qtype = self.qtype + qtype
        conj_q.representation = self.representation
        
        return conj_q

    def _commuting_products(self, q1):
        """Returns a dictionary with the commuting products."""

        s_t, s_x, s_y, s_z = self.a[0], self.a[1], self.a[2], self.a[3]
        q1_t, q1_x, q1_y, q1_z = q1.a[0], q1.a[1], q1.a[2], q1.a[3]

        products = {'tt': s_t * q1_t,
                    'xx+yy+zz': s_x * q1_x + s_y * q1_y + s_z * q1_z,
                    'tx+xt': s_t * q1_x + s_x * q1_t,
                    'ty+yt': s_t * q1_y + s_y * q1_t,
                    'tz+zt': s_t * q1_z + s_z * q1_t}

        return products

    def _anti_commuting_products(self, q1):
        """Returns a dictionary with the three anti-commuting products."""

        s_x, s_y, s_z = self.a[1], self.a[2], self.a[3]
        q1_x, q1_y, q1_z = q1.a[1], q1.a[2], q1.a[3]

        products = {'yz-zy': s_y * q1_z - s_z * q1_y,
                    'zx-xz': s_z * q1_x - s_x * q1_z,
                    'xy-yx': s_x * q1_y - s_y * q1_x,
                    'zy-yz': - s_y * q1_z + s_z * q1_y,
                    'xz-zx': - s_z * q1_x + s_x * q1_z,
                    'yx-xy': - s_x * q1_y + s_y * q1_x,
                    }

        return products

    def _all_products(self, q1):
        """Returns a dictionary with all possible products."""

        products = self._commuting_products(q1)
        products.update(self._anti_commuting_products(q1))

        return products

    def square(self, qtype="^2"):
        """Square a quaternion."""

        end_qtype = "{}{}".format(self.qtype, qtype)
        
        qxq = self._commuting_products(self)

        sq_q = QHa(qtype=end_qtype, representation=self.representation)
        sq_q.a[0] = qxq['tt'] - qxq['xx+yy+zz']
        sq_q.a[1] = qxq['tx+xt']
        sq_q.a[2] = qxq['ty+yt']
        sq_q.a[3] = qxq['tz+zt']

        sq_q.qtype = end_qtype
        
        return sq_q

    def norm_squared(self, qtype="|| ||^2"):
        """The norm_squared of a quaternion."""

        end_qtype = "||{}||^2".format(self.qtype, qtype)
        
        qxq = self._commuting_products(self)

        n_q = QHa(qtype=end_qtype)
        n_q.a[0] = qxq['tt'] + qxq['xx+yy+zz']

        n_q.qtype = end_qtype
        n_q.representation = self.representation
        
        return n_q

    def norm_squared_of_vector(self, qtype="norm_squaredV"):
        """The norm_squared of the vector of a quaternion."""

        end_qtype = "||V({})||".format(self.qtype)
        
        qxq = self._commuting_products(self)

        nv_q = QHa(qtype=end_qtype)
        nv_q.a[0] = qxq['xx+yy+zz']

        nv_q.qtype = end_qtype
        nv_q.representation = self.representation
        
        return nv_q

    def abs_of_q(self, qtype="| |"):
        """The absolute value, the square root of the norm_squared."""

        end_qtype = "|{}|".format(self.qtype)
        
        ns = self.norm_squared()
        sqrt_t = ns.a[0] ** (1/2)
        ns.a[0] = sqrt_t

        ns.qtype = end_qtype
        ns.representation = self.representation
        
        return ns

    def abs_of_vector(self, qtype="|V()|"):
        """The absolute value of the vector, the square root of the norm_squared of the vector."""

        end_qtype = "|V({})|".format(self.qtype)
        
        av = self.norm_squared_of_vector()
        sqrt_t = av.a[0] ** (1/2)
        av.a[0] = sqrt_t

        av.qtype = end_qtype
        av.representation = self.representation
        
        return av

    def normalize(self, n=1, qtype="U"):
        """Normalize a quaternion"""
        
        end_qtype = "{}{}".format(self.qtype, qtype)
        
        abs_q_inv = self.abs_of_q().invert()
        n_q = self.product(abs_q_inv).product(QHa([n, 0, 0, 0]))

        n_q.qtype = end_qtype
        n_q.representation = self.representation
        
        return n_q
    
    def add(self, q1, qtype=""):
        """Form a add given 2 quaternions."""

        self.check_representations(q1)
        
        t_1, x_1, y_1, z_1 = self.a[0], self.a[1], self.a[2], self.a[3]
        t_2, x_2, y_2, z_2 = q1.a[0], q1.a[1], q1.a[2], q1.a[3]

        add_q = QHa()
        add_q.a[0] = t_1 + t_2
        add_q.a[1] = x_1 + x_2
        add_q.a[2] = y_1 + y_2
        add_q.a[3] = z_1 + z_2
        
        add_q.qtype = "{f}+{s}".format(f=self.qtype, s=q1.qtype)
        add_q.representation = self.representation
        
        return add_q    

    def dif(self, q1, qtype=""):
        """Form a add given 2 quaternions."""

        self.check_representations(q1)
        
        t_1, x_1, y_1, z_1 = self.a[0], self.a[1], self.a[2], self.a[3]
        t_2, x_2, y_2, z_2 = q1.a[0], q1.a[1], q1.a[2], q1.a[3]

        dif_q = QHa()
        dif_q.a[0] = t_1 - t_2
        dif_q.a[1] = x_1 - x_2
        dif_q.a[2] = y_1 - y_2
        dif_q.a[3] = z_1 - z_2

        dif_q.qtype = "{f}-{s}".format(f=self.qtype, s=q1.qtype)
        dif_q.representation = self.representation
            
        return dif_q

    def product(self, q1, kind="", reverse=False, qtype=""):
        """Form a product given 2 quaternions: standard, even, odd, and even_minus_odd. If reverse=True, reverses the order."""

        self.check_representations(q1)
        
        commuting = self._commuting_products(q1)
        q_even = QHa()
        q_even.a[0] = commuting['tt'] - commuting['xx+yy+zz']
        q_even.a[1] = commuting['tx+xt']
        q_even.a[2] = commuting['ty+yt']
        q_even.a[3] = commuting['tz+zt']
        qxq = self._all_products(q1)
        
        anti_commuting = self._anti_commuting_products(q1)
        q_odd = QHa()
        
        if reverse:
            q_odd.a[1] = anti_commuting['zy-yz']
            q_odd.a[2] = anti_commuting['xz-zx']
            q_odd.a[3] = anti_commuting['yx-xy']
            
        else:
            q_odd.a[1] = anti_commuting['yz-zy']
            q_odd.a[2] = anti_commuting['zx-xz']
            q_odd.a[3] = anti_commuting['xy-yx']
        
        result = QHa()
        
        if kind == "":
            result = q_even.add(q_odd)
            times_symbol = "x"
        elif kind.lower() == "even":
            result = q_even
            times_symbol = "xE"
        elif kind.lower() == "odd":
            result = q_odd
            times_symbol = "xO"
        elif kind.lower() == "even_minus_odd":
            result = q_even.dif(q_odd)
            times_symbol = "xE-O"
        else:
            raise Exception("Four 'kind' values are known: '', 'even', 'odd', and 'even_minus_odd'.")
            
        if reverse:
            times_symbol = times_symbol.replace('x', 'xR')
            
        result.qtype = "{f}{ts}{s}".format(f=self.qtype, ts=times_symbol, s=q1.qtype)
        result.representation = self.representation
            
        return result
    
    def Euclidean_product(self, q1, kind="", reverse=False, qtype=""):
        """Form a product p* q given 2 quaternions, not associative."""

        self.check_representations(q1)
        
        pq = QHa()
        pq = self.conj().product(q1, kind, reverse, qtype)
            
        return pq
    
    def invert(self, qtype="^-1"):
        """The inverse of a quaternion."""

        end_qtype = "{}{}".format(self.qtype, qtype)
        
        q_conj = self.conj()
        q_norm_squared = self.norm_squared()

        if q_norm_squared.a[0] == 0:
            return self.q_0()

        q_norm_squared_inv = QHa([1.0 / q_norm_squared.a[0], 0, 0, 0])
        q_inv = q_conj.product(q_norm_squared_inv, qtype=self.qtype)
        
        q_inv.qtype = end_qtype
        q_inv.representation = self.representation
        
        return q_inv

    def divide_by(self, q1, qtype=""):
        """Divide one quaternion by another. The order matters unless one is using a norm_squared (real number)."""
        
        self.check_representations(q1)
        
        q1_inv = q1.invert()
        q_div = self.product(q1.invert())
        
        q_div.qtype = "{f}/{s}".format(f=self.qtype, s=q1.qtype)
        q_div.representation = self.representation
            
        return q_div

    def triple_product(self, q1, q2):
        """Form a triple product given 3 quaternions."""

        self.check_representations(q1)
        
        triple = self.product(q1).product(q2)
        triple.representation = self.representation
        
        return triple

    # Quaternion rotation involves a triple product:  UQU∗
    # where the U is a unitary quaternion (having a norm_squared of one).
    def rotate(self, a_1=0, a_2=0, a_3=0, qtype="rot"):
        """Do a rotation given up to three angles."""

        end_qtype = "{}{}".format(self.qtype, qtype)
        
        u = QHa([0, a_1, a_2, a_3])
        u_abs = u.abs_of_q()
        u_norm_squaredalized = u.divide_by(u_abs)

        q_rot = u_norm_squaredalized.triple_product(self, u_norm_squaredalized.conj())
  
        q_rot.qtype = end_qtype
        q_rot.representation = self.representation
        
        return q_rot

    # A boost also uses triple products like a rotation, but more of them.
    # This is not a well-known result, but does work.
    # b -> b' = h b h* + 1/2 ((hhb)* -(h*h*b)*)
    # where h is of the form (cosh(a), sinh(a))
    def boost(self, beta_x=0.0, beta_y=0.0, beta_z=0.0, qtype="boost"):
        """A boost along the x, y, and/or z axis."""

        end_qtype = "{}{}".format(self.qtype, qtype)
        
        boost = QHa(sr_gamma_betas(beta_x, beta_y, beta_z))      
        b_conj = boost.conj()

        triple_1 = boost.triple_product(self, b_conj)
        triple_2 = boost.triple_product(boost, self).conj()
        triple_3 = b_conj.triple_product(b_conj, self).conj()
      
        triple_23 = triple_2.dif(triple_3)
        half_23 = triple_23.product(QHa([0.5, 0.0, 0.0, 0.0]))
        triple_123 = triple_1.add(half_23, qtype=self.qtype)
        
        triple_123.qtype = end_qtype
        triple_123.representation = self.representation
        
        return triple_123

    # g_shift is a function based on the space-times-time invariance proposal for gravity,
    # which proposes that if one changes the distance from a gravitational source, then
    # squares a measurement, the observers at two different hieghts agree to their
    # space-times-time values, but not the intervals.
    # g_form is the form of the function, either minimal or exponential
    # Minimal is what is needed to pass all weak field tests of gravity
    def g_shift(self, dimensionless_g, g_form="exp", qtype="g_shift"):
        """Shift an observation based on a dimensionless GM/c^2 dR."""

        end_qtype = "{}{}".format(self.qtype, qtype)
        
        if g_form == "exp":
            g_factor = sp.exp(dimensionless_g)
        elif g_form == "minimal":
            g_factor = 1 + 2 * dimensionless_g + 2 * dimensionless_g ** 2
        else:
            print("g_form not defined, should be 'exp' or 'minimal': {}".format(g_form))
            return self

        g_q = QHa(qtype=end_qtype)
        g_q.a[0] = self.a[0] / g_factor
        g_q.a[1] = self.a[1] * g_factor
        g_q.a[2] = self.a[2] * g_factor
        g_q.a[3] = self.a[3] * g_factor
        g_q.qtype = end_qtype
        g_q.representation = self.representation
        
        return g_q
    
    def sin(self, qtype="sin"):
        """Take the sine of a quaternion, (sin(t) cosh(|R|), cos(t) sinh(|R|) R/|R|)"""

        end_qtype = "sin({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        
        if abs_v.a[0] == 0:    
            return QHa([math.sin(self.a[0]), 0, 0, 0], qtype=end_qtype)
            
        sint = math.sin(self.a[0])
        cost = math.cos(self.a[0])
        sinhR = math.sinh(abs_v.a[0])
        coshR = math.cosh(abs_v.a[0])
        
        k = cost * sinhR / abs_v.a[0]
            
        q_out = QHa(qtype=end_qtype, representation=self.representation)
        q_out.a[0] = sint * coshR
        q_out.a[1] = k * self.a[1]
        q_out.a[2] = k * self.a[2]
        q_out.a[3] = k * self.a[3]

        return q_out
     
    def cos(self, qtype="sin"):
        """Take the cosine of a quaternion, (cos(t) cosh(|R|), sin(t) sinh(|R|) R/|R|)"""

        end_qtype = "cos({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        
        if abs_v.a[0] == 0:    
            return QHa([math.cos(self.a[0]), 0, 0, 0], qtype=end_qtype)
            
        sint = math.sin(self.a[0])
        cost = math.cos(self.a[0])
        sinhR = math.sinh(abs_v.a[0])
        coshR = math.cosh(abs_v.a[0])
        
        k = -1 * sint * sinhR / abs_v.a[0]
            
        q_out = QHa(qtype=end_qtype, representation=self.representation)
        q_out.a[0] = cost * coshR
        q_out.a[1] = k * self.a[1]
        q_out.a[2] = k * self.a[2]
        q_out.a[3] = k * self.a[3]
        
        return q_out
    
    def tan(self, qtype="tan"):
        """Take the tan of a quaternion, sin/cos"""

        end_qtype = "tan({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        
        if abs_v.a[0] == 0:    
            return QHa([math.tan(self.a[0]), 0, 0, 0], qtype=end_qtype)
            
        sinq = self.sin()
        cosq = self.cos()
            
        q_out = sinq.divide_by(cosq) 
        q_out.qtype = end_qtype
        q_out.representation = self.representation
        
        return q_out
    
    def sinh(self, qtype="sinh"):
        """Take the sinh of a quaternion, (sinh(t) cos(|R|), cosh(t) sin(|R|) R/|R|)"""

        end_qtype = "sinh({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        
        if abs_v.a[0] == 0:    
            return QHa([math.sinh(self.a[0]), 0, 0, 0], qtype=end_qtype)
            
        sinht = math.sinh(self.a[0])
        cosht = math.cosh(self.a[0])
        sinR = math.sin(abs_v.a[0])
        cosR = math.cos(abs_v.a[0])
        
        k = cosht * sinR / abs_v.a[0]
        
        q_out = QHa(qtype=end_qtype, representation=self.representation)
        q_out.a[0] = sinht * cosR
        q_out.a[1] = k * self.a[1]
        q_out.a[2] = k * self.a[2]
        q_out.a[3] = k * self.a[3]

        return q_out
     
    def cosh(self, qtype="cosh"):
        """Take the cosh of a quaternion, (cosh(t) cos(|R|), sinh(t) sin(|R|) R/|R|)"""

        end_qtype = "cosh({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        
        if abs_v.a[0] == 0:    
            return QHa([math.cosh(self.a[0]), 0, 0, 0], qtype=end_qtype)
            
        sinht = math.sinh(self.a[0])
        cosht = math.cosh(self.a[0])
        sinR = math.sin(abs_v.a[0])
        cosR = math.cos(abs_v.a[0])
        
        k = sinht * sinR / abs_v.a[0]
            
        q_out = QHa(qtype=end_qtype, representation=self.representation)
        q_out.a[0] = cosht * cosR
        q_out.a[1] = k * self.a[1]
        q_out.a[2] = k * self.a[2]
        q_out.a[3] = k * self.a[3]

        return q_out
    
    def tanh(self, qtype="tanh"):
        """Take the tanh of a quaternion, sin/cos"""

        end_qtype = "tanh({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        
        if abs_v.a[0] == 0:    
            return QHa([math.tanh(self.a[0]), 0, 0, 0], qtype=end_qtype)
            
        sinhq = self.sinh()
        coshq = self.cosh()
            
        q_out = sinhq.divide_by(coshq) 
        q_out.qtype = end_qtype
        q_out.representation = self.representation
        
        return q_out
    
    def exp(self, qtype="exp"):
        """Take the exponential of a quaternion."""
        # exp(q) = (exp(t) cos(|R|, exp(t) sin(|R|) R/|R|)
        
        end_qtype = "exp({st})".format(st=self.qtype)
        
        abs_v = self.abs_of_vector()
        et = math.exp(self.a[0])
        
        if (abs_v.a[0] == 0):
            return QHa([et, 0, 0, 0], qtype=end_qtype, representation=self.representation)
        
        cosR = math.cos(abs_v.a[0])
        sinR = math.sin(abs_v.a[0])
        k = et * sinR / abs_v.a[0]
                       
        expq = QHa([et * cosR, k * self.a[1], k * self.a[2], k * self.a[3]], qtype=end_qtype, representation=self.representation)
        
        return expq
    
    def ln(self, qtype="ln"):
        """Take the natural log of a quaternion."""
        # ln(q) = (0.5 ln t^2 + R.R, atan2(|R|, t) R/|R|)
        
        end_qtype = "ln({st})".format(st=self.qtype)
        
        abs_v = self.abs_of_vector()
        
        if (abs_v.a[0] == 0):
            if self.a[0] > 0:
                return(QHa([math.log(self.a[0]), 0, 0, 0], qtype=end_qtype, representation=self.representation))
            else:
                # I don't understant this, but mathematica does the same thing.
                return(QHa([math.log(-self.a[0]), math.pi, 0, 0], qtype=end_qtype, representation=self.representation))   
            
            return QHa([lt, 0, 0, 0])
        
        t_value = 0.5 * math.log(self.a[0] * self.a[0] + abs_v.a[0] * abs_v.a[0])
        k = math.atan2(abs_v.a[0], self.a[0]) / abs_v.a[0]
                       
        expq = QHa([t_value, k * self.a[1], k * self.a[2], k * self.a[3]], qtype=end_qtype)
        
        return expq
    
    def q_2_q(self, q1, qtype="^P"):
        """Take the natural log of a quaternion."""
        # q^p = exp(ln(q) * p)
        
        self.check_representations(q1)
        
        end_qtype = "{st}^P".format(st=self.qtype)
        
        q2q = self.ln().product(q1).exp()
        q2q.qtype = end_qtype
        q2q.representation = self.representation
        
        return q2q
    
    def trunc(self):
        """Truncates values."""
        
        self.a[0] = math.trunc(self.a[0])
        self.a[1] = math.trunc(self.a[1])
        self.a[2] = math.trunc(self.a[2])
        self.a[3] = math.trunc(self.a[3])
        
        return self


# In[44]:


class TestQHa(unittest.TestCase):
    """Class to make sure all the functions work as expected."""

    Q = QHa([1.0, -2.0, -3.0, -4.0], qtype="Q")
    P = QHa([0.0, 4.0, -3.0, 0.0], qtype="P")
    R = QHa([3.0, 0, 0, 0])
    C = QHa([2.0, 4.0, 0, 0])
    
    def test_qt(self):
        self.assertTrue(self.Q.a[0] == 1)

    def test_q_0(self):
        q_z = self.Q.q_0()
        print("q_0: ", q_z)
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 0)

    def test_q_1(self):
        q_z = self.Q.q_1()
        print("q_1: ", q_z)
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 0)
    
    def test_q_i(self):
        q_z = self.Q.q_i()
        print("q_i: ", q_z)
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == 1)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 0)

    def test_q_j(self):
        q_z = self.Q.q_j()
        print("q_1: ", q_z)
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 1)
        self.assertTrue(q_z.a[3] == 0)

    def test_q_k(self):
        q_z = self.Q.q_k()
        print("q_k: ", q_z)
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 1)
    
    def test_q_random(self):
        q_z = self.Q.q_random()
        print("q_random(): ", q_z)
        self.assertTrue(q_z.a[0] >= 0 and q_z.a[0] <= 1)
        self.assertTrue(q_z.a[1] >= 0 and q_z.a[1] <= 1)
        self.assertTrue(q_z.a[2] >= 0 and q_z.a[2] <= 1)
        self.assertTrue(q_z.a[3] >= 0 and q_z.a[3] <= 1)
            
    def test_equals(self):
        self.assertTrue(self.Q.equals(self.Q))
        self.assertFalse(self.Q.equals(self.P))

    def test_conj_0(self):
        q_z = self.Q.conj()
        print("q_conj 0: ", q_z)
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 2)
        self.assertTrue(q_z.a[2] == 3)
        self.assertTrue(q_z.a[3] == 4)

    def test_conj_1(self):
        q_z = self.Q.conj(1)
        print("q_conj 1: ", q_z)
        self.assertTrue(q_z.a[0] == -1)
        self.assertTrue(q_z.a[1] == -2)
        self.assertTrue(q_z.a[2] == 3)
        self.assertTrue(q_z.a[3] == 4)

    def test_conj_2(self):
        q_z = self.Q.conj(2)
        print("q_conj 2: ", q_z)
        self.assertTrue(q_z.a[0] == -1)
        self.assertTrue(q_z.a[1] == 2)
        self.assertTrue(q_z.a[2] == -3)
        self.assertTrue(q_z.a[3] == 4)

    def test_vahlen_conj_minus(self):
        q_z = self.Q.vahlen_conj()
        print("q_vahlen_conj -: ", q_z)
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 2)
        self.assertTrue(q_z.a[2] == 3)
        self.assertTrue(q_z.a[3] == 4)

    def test_vahlen_conj_star(self):
        q_z = self.Q.vahlen_conj('*')
        print("q_vahlen_conj *: ", q_z)
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == -2)
        self.assertTrue(q_z.a[2] == -3)
        self.assertTrue(q_z.a[3] == 4)

    def test_vahlen_conj_prime(self):
        q_z = self.Q.vahlen_conj("'")
        print("q_vahlen_conj ': ", q_z)
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 2)
        self.assertTrue(q_z.a[2] == 3)
        self.assertTrue(q_z.a[3] == -4)

    def test_square(self):
        q_z = self.Q.square()
        print("square: ", q_z)
        self.assertTrue(q_z.a[0] == -28)
        self.assertTrue(q_z.a[1] == -4)
        self.assertTrue(q_z.a[2] == -6)
        self.assertTrue(q_z.a[3] == -8)

    def test_norm_squared(self):
        q_z = self.Q.norm_squared()
        print("norm_squared: ", q_z)
        self.assertTrue(q_z.a[0] == 30)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 0)

    def test_norm_squared_of_vector(self):
        q_z = self.Q.norm_squared_of_vector()
        print("norm_squared_of_vector: ", q_z)
        self.assertTrue(q_z.a[0] == 29)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 0)
        
    def test_abs_of_q(self):
        q_z = self.P.abs_of_q()
        print("abs_of_q: ", q_z)
        self.assertTrue(q_z.a[0] == 5.0)
        self.assertTrue(q_z.a[1] == 0.0)
        self.assertTrue(q_z.a[2] == 0.0)
        self.assertTrue(q_z.a[3] == 0.0)
        
    def test_abs_of_vector(self):
        q_z = self.P.abs_of_vector()
        print("abs_of_vector: ", q_z)
        self.assertTrue(q_z.a[0] == 5)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 0)
    
    def test_normalize(self):
        q_z = self.P.normalize()
        print("q_normalized: ", q_z)
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == 0.8)
        self.assertAlmostEqual(q_z.a[2], -0.6)
        self.assertTrue(q_z.a[3] == 0)
    
    def test_add(self):
        q_z = self.Q.add(self.P)
        print("add: ", q_z)
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 2)
        self.assertTrue(q_z.a[2] == -6)
        self.assertTrue(q_z.a[3] == -4)
        
    def test_dif(self):
        q_z = self.Q.dif(self.P)
        print("dif: ", q_z)
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == -6)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == -4) 

    def test_product(self):
        q_z = self.Q.product(self.P)
        print("product: ", q_z)
        self.assertTrue(q_z.a[0] == -1)
        self.assertTrue(q_z.a[1] == -8)
        self.assertTrue(q_z.a[2] == -19)
        self.assertTrue(q_z.a[3] == 18)
        
    def test_product_even(self):
        q_z = self.Q.product(self.P, kind="even")
        print("product even: ", q_z)
        self.assertTrue(q_z.a[0] == -1)
        self.assertTrue(q_z.a[1] == 4)
        self.assertTrue(q_z.a[2] == -3)
        self.assertTrue(q_z.a[3] == 0)
        
    def test_product_odd(self):
        q_z = self.Q.product(self.P, kind="odd")
        print("product odd: ", q_z)
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == -12)
        self.assertTrue(q_z.a[2] == -16)
        self.assertTrue(q_z.a[3] == 18)
        
    def test_product_reverse(self):
        q1q2_rev = self.Q.product(self.P, reverse=True)
        q2q1 = self.P.product(self.Q)
        self.assertTrue(q1q2_rev.equals(q2q1))

    def test_product_even_minus_odd(self):
        q_z = self.Q.product(self.P, kind="even_minus_odd")
        print("product even_minus_odd: ", q_z)
        self.assertTrue(q_z.a[0] == -1)
        self.assertTrue(q_z.a[1] == 16)
        self.assertTrue(q_z.a[2] == 13)
        self.assertTrue(q_z.a[3] == -18)
        
    def test_Euclidean_product(self):
        q_z = self.Q.Euclidean_product(self.P)
        print("Euclidean product: ", q_z)
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 16)
        self.assertTrue(q_z.a[2] == 13)
        self.assertTrue(q_z.a[3] == -18)

    def test_invert(self):
        q_z = self.P.invert()
        print("invert: ", q_z)
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == -0.16)
        self.assertTrue(q_z.a[2] == 0.12)
        self.assertTrue(q_z.a[3] == 0)
                
    def test_divide_by(self):
        q_z = self.Q.divide_by(self.Q)
        print("divide_by: ", q_z)
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 0) 
        
    def test_triple_product(self):
        q_z = self.Q.triple_product(self.P, self.Q)
        print("triple product: ", q_z)
        self.assertTrue(q_z.a[0] == -2)
        self.assertTrue(q_z.a[1] == 124)
        self.assertTrue(q_z.a[2] == -84)
        self.assertTrue(q_z.a[3] == 8)
        
    def test_rotate(self):
        q_z = self.Q.rotate(1)
        print("rotate: ", q_z)
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == -2)
        self.assertTrue(q_z.a[2] == 3)
        self.assertTrue(q_z.a[3] == 4)
        
    def test_boost(self):
        q1_sq = self.Q.square()
        q_z = self.Q.boost(0.003)
        q_z2 = q_z.square()
        print("q1_sq: ".format(q1_sq))
        print("boosted: ", q_z)
        print("boosted squared: ".format(q_z2))
        print("{}")
        self.assertTrue(round(q_z2.a[0], 5) == round(q1_sq.a[0], 5))

    def test_g_shift(self):
        q1_sq = self.Q.square()
        q_z = self.Q.g_shift(0.003)
        q_z2 = q_z.square()
        q_z_minimal = self.Q.g_shift(0.003, g_form="minimal")
        q_z2_minimal = q_z_minimal.square()
        print("q1_sq: ".format(q1_sq))
        print("g_shift: ", q_z)
        print("g squared: ".format(q_z2))
        self.assertTrue(q_z2.a[0] != q1_sq.a[0])
        self.assertTrue(q_z2.a[1] == q1_sq.a[1])
        self.assertTrue(q_z2.a[2] == q1_sq.a[2])
        self.assertTrue(q_z2.a[3] == q1_sq.a[3])
        self.assertTrue(q_z2_minimal.a[0] != q1_sq.a[0])
        self.assertTrue(q_z2_minimal.a[1] == q1_sq.a[1])
        self.assertTrue(q_z2_minimal.a[2] == q1_sq.a[2])
        self.assertTrue(q_z2_minimal.a[3] == q1_sq.a[3])
        
    def test_sin(self):
        self.assertTrue(QHa([0, 0, 0, 0]).sin().equals(QHa().q_0()))
        self.assertTrue(self.Q.sin().equals(QHa([91.7837157840346691, -21.8864868530291758, -32.8297302795437673, -43.7729737060583517])))
        self.assertTrue(self.P.sin().equals(QHa([0,  59.3625684622310033, -44.5219263466732542, 0])))
        self.assertTrue(self.R.sin().equals(QHa([0.1411200080598672, 0, 0, 0])))
        self.assertTrue(self.C.sin().equals(QHa([24.8313058489463785, -11.3566127112181743, 0, 0])))
     
    def test_cos(self):
        self.assertTrue(QHa([0, 0, 0, 0]).cos().equals(QHa().q_1()))
        self.assertTrue(self.Q.cos().equals(QHa([58.9336461679439481, 34.0861836904655959, 51.1292755356983974, 68.1723673809311919])))
        self.assertTrue(self.P.cos().equals(QHa([74.2099485247878476, 0, 0, 0])))
        self.assertTrue(self.R.cos().equals(QHa([-0.9899924966004454, 0, 0, 0])))
        self.assertTrue(self.C.cos().equals(QHa([-11.3642347064010600, -24.8146514856341867, 0, 0])))
                            
    def test_tan(self):
        self.assertTrue(QHa([0, 0, 0, 0]).tan().equals(QHa().q_0()))
        self.assertTrue(self.Q.tan().equals(QHa([0.0000382163172501, -0.3713971716439372, -0.5570957574659058, -0.7427943432878743])))
        self.assertTrue(self.P.tan().equals(QHa([0, 0.7999273634100760, -0.5999455225575570, 0])))
        self.assertTrue(self.R.tan().equals(QHa([-0.1425465430742778, 0, 0, 0])))
        self.assertTrue(self.C.tan().equals(QHa([-0.0005079806234700, 1.0004385132020521, 0, 0])))
        
    def test_sinh(self):
        self.assertTrue(QHa([0, 0, 0, 0]).sinh().equals(QHa().q_0()))
        self.assertTrue(self.Q.sinh().equals(QHa([0.7323376060463428, 0.4482074499805421, 0.6723111749708131, 0.8964148999610841])))
        self.assertTrue(self.P.sinh().equals(QHa([0, -0.7671394197305108, 0.5753545647978831, 0])))
        self.assertTrue(self.R.sinh().equals(QHa([10.0178749274099026, 0, 0, 0])))
        self.assertTrue(self.C.sinh().equals(QHa([-2.3706741693520015, -2.8472390868488278, 0, 0])))    
        
    def test_cosh(self):
        self.assertTrue(QHa([0, 0, 0, 0]).cosh().equals(QHa().q_1()))
        self.assertTrue(self.Q.cosh().equals(QHa([0.9615851176369565, 0.3413521745610167, 0.5120282618415251, 0.6827043491220334])))
        self.assertTrue(self.P.cosh().equals(QHa([0.2836621854632263, 0, 0, 0])))
        self.assertTrue(self.R.cosh().equals(QHa([10.0676619957777653, 0, 0, 0])))
        self.assertTrue(self.C.cosh().equals(QHa([-2.4591352139173837, -2.7448170067921538, 0, 0])))    
        
    def test_tanh(self):
        self.assertTrue(QHa([0, 0, 0, 0]).tanh().equals(QHa().q_0()))
        self.assertTrue(self.Q.tanh().equals(QHa([1.0248695360556623, 0.1022956817887642, 0.1534435226831462, 0.2045913635775283])))
        self.assertTrue(self.P.tanh().equals(QHa([0, -2.7044120049972684, 2.0283090037479505, 0])))
        self.assertTrue(self.R.tanh().equals(QHa([0.9950547536867305, 0, 0, 0])))
        self.assertTrue(self.C.tanh().equals(QHa([1.0046823121902353, 0.0364233692474038, 0, 0])))    
        
    def test_exp(self):
        self.assertTrue(QHa([0, 0, 0, 0]).exp().equals(QHa().q_1()))
        self.assertTrue(self.Q.exp().equals(QHa([1.6939227236832994, 0.7895596245415588, 1.1843394368123383, 1.5791192490831176])))
        self.assertTrue(self.P.exp().equals(QHa([0.2836621854632263, -0.7671394197305108, 0.5753545647978831, 0])))
        self.assertTrue(self.R.exp().equals(QHa([20.0855369231876679, 0, 0, 0])))
        self.assertTrue(self.C.exp().equals(QHa([-4.8298093832693851, -5.5920560936409816, 0, 0])))    
    
    def test_ln(self):
        self.assertTrue(self.Q.ln().exp().equals(self.Q))
        self.assertTrue(self.Q.ln().equals(QHa([1.7005986908310777, -0.5151902926640850, -0.7727854389961275, -1.0303805853281700])))
        self.assertTrue(self.P.ln().equals(QHa([1.6094379124341003, 1.2566370614359172, -0.9424777960769379, 0])))
        self.assertTrue(self.R.ln().equals(QHa([1.0986122886681098, 0, 0, 0])))
        self.assertTrue(self.C.ln().equals(QHa([1.4978661367769954, 1.1071487177940904, 0, 0])))    
        
    def test_q_2_q(self):
        self.assertTrue(self.Q.q_2_q(self.P).equals(QHa([-0.0197219653530713, -0.2613955437374326, 0.6496281248064009, -0.3265786562423951])))

suite = unittest.TestLoader().loadTestsFromModule(TestQHa())
unittest.TextTestRunner().run(suite);


# In[45]:


class TestQHaRep(unittest.TestCase):
    Q12 = QHa([1, 2, 0, 0])
    Q1123 = QHa([1, 1, 2, 3])
    Q11p = QHa([1, 1, 0, 0], representation="polar")
    Q12p = QHa([1, 2, 0, 0], representation="polar")
    Q12np = QHa([1, -2, 0, 0], representation="polar")
    Q21p = QHa([2, 1, 0, 0], representation="polar")
    Q23p = QHa([2, 3, 0, 0], representation="polar")
    Q13p = QHa([1, 3, 0, 0], representation="polar")
    Q5p = QHa([5, 0, 0, 0], representation="polar")
    
    def test_txyz_2_representation(self):
        qr = QHa(self.Q12.txyz_2_representation(""))
        self.assertTrue(qr.equals(self.Q12))
        qr = QHa(self.Q12.txyz_2_representation("polar"))
        self.assertTrue(qr.equals(QHa([2.23606797749979, 1.10714871779409, 0, 0])))
        qr = QHa(self.Q1123.txyz_2_representation("spherical"))
        self.assertTrue(qr.equals(QHa([1.0, 3.7416573867739413, 0.640522312679424, 1.10714871779409])))
        
    def test_representation_2_txyz(self):
        qr = QHa(self.Q12.representation_2_txyz(""))
        self.assertTrue(qr.equals(self.Q12))
        qr = QHa(self.Q12.representation_2_txyz("polar"))
        self.assertTrue(qr.equals(QHa([-0.4161468365471424, 0.9092974268256817, 0, 0])))
        qr = QHa(self.Q1123.representation_2_txyz("spherical"))
        self.assertTrue(qr.equals(QHa([1.0, -0.9001976297355174, 0.12832006020245673, -0.4161468365471424])))
    
    def test_polar_products(self):
        qr = self.Q11p.product(self.Q12p)
        print("polar 1 1 0 0 * 1 2 0 0: ", qr)
        self.assertTrue(qr.equals(self.Q13p))
        qr = self.Q12p.product(self.Q21p)
        print("polar 1 2 0 0 * 2 1 0 0: ", qr)
        self.assertTrue(qr.equals(self.Q23p))

    def test_polar_conj(self):
        qr = self.Q12p.conj()
        print("polar conj of {}: {}", self.Q12p, qr)
        print("Q12np: ", self.Q12np)
        self.assertTrue(qr.equals(self.Q12np))
        
suite = unittest.TestLoader().loadTestsFromModule(TestQHaRep())
unittest.TextTestRunner().run(suite);


# ## Using More Numbers via Doublets

# My long term goal is to deal with quaternions on a quaternion manifold. This will have 4 pairs of doublets. Each doublet is paired with its additive inverse. Instead of using real numbers, one uses (3, 0) and (0, 2) to represent +3 and -2 respectively. Numbers such as (5, 6) are allowed. That can be "reduced" to (0, 1).  My sense is that somewhere deep in the depths of relativistic quantum field theory, this will be a "good thing". For now, it is a minor pain to program.

# In[46]:


class Doublet(object):
    """A pair of number that are additive inverses. It can take
    ints, floats, Symbols, or strings."""
    
    def __init__(self, numbers=None):
        
        if numbers is None:
            self.p = 0
            self.n = 0
            
        elif isinstance(numbers, (int, float)):
            if numbers < 0:
                self.n = -1 * numbers
                self.p = 0
            else:
                self.p = numbers
                self.n = 0
        
        elif isinstance(numbers, sp.Symbol):
            self.p = numbers
            self.n = 0
            
        elif isinstance(numbers, list):
            if len(numbers) == 2:
                self.p, self.n = numbers[0], numbers[1]
                      
        elif isinstance(numbers, str):
            n_list = numbers.split()
            
            if (len(n_list) == 1):
                if n_list.isnumeric():
                    n_value = float(numbers)
                      
                    if n_value < 0:
                        self.n = -1 * n_list[0]
                        self.p = 0
                      
                    else:
                        self.p = n_list[0]
                        self.n = 0
                        
                else:
                    self.p = sp.Symbol(n_list[0])
                    self.n = 0
                      
            if (len(n_list) == 2):
                if n_list[0].isnumeric():
                    self.p = float(n_list[0])
                else:
                    self.p = sp.Symbol(n_list[0])
                    
                if n_list[1].isnumeric():
                    self.n = float(n_list[1])
                else:
                    self.n = sp.Symbol(n_list[1])
        else:
            print ("unable to parse this Double.")

    def __str__(self):
        """Customize the output."""
        return "{p}p  {n}n".format(p=self.p, n=self.n)
        
    def d_add(self, d1):
        """Add a doublet to another."""
                        
        pa0, n0 = self.p, self.n
        p1, n1 = d1.p, d1.n
                        
        return Doublet([pa0 + p1, n0 + n1])

    def d_reduce(self):
        """If p and n are not zero, subtract """
        if self.p == 0 or self.n == 0:
            return Doublet([self.p, self.n])
        
        elif self.p > self.n:
            return Doublet([self.p - self.n, 0])
        
        elif self.p < self.n:
            return Doublet([0, self.n - self.p])
        
        else:
            return Doublet()
        
    def d_additive_inverse_up_to_an_automorphism(self, n=0):
        """Creates one additive inverses up to an arbitrary positive n."""
        
        if n == 0:
            return Doublet([self.n + n, self.p + n])
        else:
            red = self.d_reduce()
            return Doublet([red.n + n, red.p + n])
                        
    def d_dif(self, d1, n=0):
        """Take the difference by flipping and adding."""
        d2 = d1.d_additive_inverse_up_to_an_automorphism(n)
                        
        return self.d_add(d2)

    def d_equals(self, d1):
        """Figure out if two doublets are equal up to an equivalence relation."""
        
        self_red = self.d_reduce()
        d1_red = d1.d_reduce()
        
        if math.isclose(self_red.p, d1_red.p) and math.isclose(self_red.n, d1_red.n):
            return True
        
        else:
            return False
        
    def Z2_product(self, d1):
        """Uset the Abelian cyclic group Z2 to form the product of 2 doublets."""
        p1 = self.p * d1.p + self.n * d1.n
        n1 = self.p * d1.n + self.n * d1.p
        
        return Doublet([p1, n1])


# In[47]:


class TestDoublet(unittest.TestCase):
    """Class to make sure all the functions work as expected."""
    
    d1 = Doublet()
    d2 = Doublet(2)
    d3 = Doublet(-3)
    d4 = Doublet([5, 3])
    dstr12 = Doublet("1 2")
    dstr13 = Doublet("3 2")
    
    def test_null(self):
        self.assertTrue(self.d1.p == 0)
        self.assertTrue(self.d1.n == 0)
       
    def test_2(self):
        self.assertTrue(self.d2.p == 2)
        self.assertTrue(self.d2.n == 0)
        
    def test_3(self):
        self.assertTrue(self.d3.p == 0)
        self.assertTrue(self.d3.n == 3)
    
    def test_str12(self):
        self.assertTrue(self.dstr12.p == 1)
        self.assertTrue(self.dstr12.n == 2)
    
    def test_add(self):
        d_add = self.d2.d_add(self.d3)
        self.assertTrue(d_add.p == 2)
        self.assertTrue(d_add.n == 3)
        
    def test_d_additive_inverse_up_to_an_automorphism(self):
        d_f = self.d2.d_additive_inverse_up_to_an_automorphism()
        self.assertTrue(d_f.p == 0)
        self.assertTrue(d_f.n == 2)
        
    def test_dif(self):
        d_d = self.d2.d_dif(self.d3)
        self.assertTrue(d_d.p == 5)
        self.assertTrue(d_d.n == 0)
            
    def test_reduce(self):
        d_add = self.d2.d_add(self.d3)
        d_r = d_add.d_reduce()
        self.assertTrue(d_r.p == 0)
        self.assertTrue(d_r.n == 1)
        
    def test_Z2_product(self):
        Z2p = self.dstr12.Z2_product(self.dstr13)
        self.assertTrue(Z2p.p == 7)
        self.assertTrue(Z2p.n == 8)

    def test_d_equals(self):
        self.assertTrue(self.d2.d_equals(self.d4))
        self.assertFalse(self.d2.d_equals(self.d1))
        
    def test_reduced_product(self):
        """Reduce before or after, should make no difference."""
        Z2p_1 = self.dstr12.Z2_product(self.dstr13)
        Z2p_red = Z2p_1.d_reduce()
        d_r_1 = self.dstr12.d_reduce()
        d_r_2 = self.dstr13.d_reduce()
        Z2p_2 = d_r_1.Z2_product(d_r_2)
        self.assertTrue(Z2p_red.p == Z2p_2.p)
        self.assertTrue(Z2p_red.n == Z2p_2.n)
        
suite = unittest.TestLoader().loadTestsFromModule(TestDoublet())
unittest.TextTestRunner().run(suite);


# Repeat the exercise for arrays.

# In[48]:


class Doubleta(object):
    """A pair of number that are additive inverses. It can take
    ints, floats, Symbols, or strings."""
    
    def __init__(self, numbers=None):
        
        if numbers is None:
            self.d = np.array([0.0, 0.0])
            
        elif isinstance(numbers, (int, float)):
            if numbers < 0:
                self.d = np.array([0, -1 * numbers])
            else:
                self.d = np.array([numbers, 0])
                        
        elif isinstance(numbers, sp.Symbol):
            self.d = np.array([numbers, 0])
            
        elif isinstance(numbers, list):
            
            if len(numbers) == 2:
                self.d = np.array([numbers[0], numbers[1]])
                      
        elif isinstance(numbers, str):
            n_list = numbers.split()
            
            if (len(n_list) == 1):
                if n_list.isnumeric():
                    n_value = float(numbers)
                      
                    if n_value < 0:
                        self.d = np.array([0, -1 * n_list[0]])
                      
                    else:
                        self.d = np.array([n_list[0], 0])
                        
                else:
                    self.d = np.array([sp.Symbol(n_list[0]), 0])
                      
            if (len(n_list) == 2):
                if n_list[0].isnumeric():
                    self.d = np.array([float(n_list[0]), float(n_list[1])])
                else:
                    self.d = np.array([sp.Symbol(n_list[0]), sp.Symbol(n_list[1])]) 
        else:
            print ("unable to parse this Double.")

    def __str__(self):
        """Customize the output."""
        return "{p}p  {n}n".format(p=self.d[0], n=self.d[1])
        
    def d_add(self, d1):
        """Add a doublet to another."""
                        
        pa0, n0 = self.d[0], self.d[1]
        p1, n1 = d1.d[0], d1.d[1]
                        
        return Doubleta([pa0 + p1, n0 + n1])

    def d_reduce(self):
        """If p and n are not zero, subtract """
        if self.d[0] == 0 or self.d[1] == 0:
            return Doubleta([self.d[0], self.d[1]])
        
        elif self.d[0] > self.d[1]:
            return Doubleta([self.d[0] - self.d[1], 0])
        
        elif self.d[0] < self.d[1]:
            return Doubleta([0, self.d[1] - self.d[0]])
        
        else:
            return Doubleta()
        
    def d_additive_inverse_up_to_an_automorphism(self, n=0):
        """Creates one additive inverses up to an arbitrary positive n."""
        
        if n == 0:
            return Doubleta([self.d[1], self.d[0]])
        else:
            red = self.d_reduce()
            return Doubleta([red.d[1] + n, red.d[0] + n])
                        
    def d_dif(self, d1, n=0):
        """Take the difference by flipping and adding."""
        d2 = d1.d_additive_inverse_up_to_an_automorphism(n)
                        
        return self.d_add(d2)

    def d_equals(self, d1):
        """See if two are equals up to an constant value."""
        
        self_red = self.d_reduce()
        d1_red = d1.d_reduce()
        
        if math.isclose(self_red.d[0], d1_red.d[0]) and math.isclose(self_red.d[1], d1_red.d[1]):
            return True
        
        else:
            return False
    
    def Z2_product(self, d1):
        """Uset the Abelian cyclic group Z2 to form the product of 2 doublets."""
        p1 = self.d[0] * d1.d[0] + self.d[1] * d1.d[1]
        n1 = self.d[0] * d1.d[1] + self.d[1] * d1.d[0]
        
        return Doubleta([p1, n1])


# In[49]:


class TestDoubleta(unittest.TestCase):
    """Class to make sure all the functions work as expected."""
    
    d1 = Doubleta()
    d2 = Doubleta(2)
    d3 = Doubleta(-3)
    d4 = Doubleta([5, 3])
    dstr12 = Doubleta("1 2")
    dstr13 = Doubleta("3 2")
    
    def test_null(self):
        self.assertTrue(self.d1.d[0] == 0)
        self.assertTrue(self.d1.d[1] == 0)
       
    def test_2(self):
        self.assertTrue(self.d2.d[0] == 2)
        self.assertTrue(self.d2.d[1] == 0)
        
    def test_3(self):
        self.assertTrue(self.d3.d[0] == 0)
        self.assertTrue(self.d3.d[1] == 3)
    
    def test_str12(self):
        self.assertTrue(self.dstr12.d[0] == 1)
        self.assertTrue(self.dstr12.d[1] == 2)
    
    def test_add(self):
        d_add = self.d2.d_add(self.d3)
        self.assertTrue(d_add.d[0] == 2)
        self.assertTrue(d_add.d[1] == 3)
        
    def test_d_additive_inverse_up_to_an_automorphism(self):
        d_f = self.d2.d_additive_inverse_up_to_an_automorphism()
        self.assertTrue(d_f.d[0] == 0)
        self.assertTrue(d_f.d[1] == 2)
        
    def test_dif(self):
        d_d = self.d2.d_dif(self.d3)
        self.assertTrue(d_d.d[0] == 5)
        self.assertTrue(d_d.d[1] == 0)
            
    def test_reduce(self):
        d_add = self.d2.d_add(self.d3)
        d_r = d_add.d_reduce()
        self.assertTrue(d_r.d[0] == 0)
        self.assertTrue(d_r.d[1] == 1)
        
    def test_Z2_product(self):
        Z2p = self.dstr12.Z2_product(self.dstr13)
        self.assertTrue(Z2p.d[0] == 7)
        self.assertTrue(Z2p.d[1] == 8)

    def test_d_equals(self):
        self.assertTrue(self.d2.d_equals(self.d4))
        self.assertFalse(self.d2.d_equals(self.d1))
        
    def test_reduced_product(self):
        """Reduce before or after, should make no difference."""
        Z2p_1 = self.dstr12.Z2_product(self.dstr13)
        Z2p_red = Z2p_1.d_reduce()
        d_r_1 = self.dstr12.d_reduce()
        d_r_2 = self.dstr13.d_reduce()
        Z2p_2 = d_r_1.Z2_product(d_r_2)
        self.assertTrue(Z2p_red.d[0] == Z2p_2.d[0])
        self.assertTrue(Z2p_red.d[1] == Z2p_2.d[1])


# In[50]:


suite = unittest.TestLoader().loadTestsFromModule(TestDoubleta())
unittest.TextTestRunner().run(suite);


# ## Quaternion Group Q8

# Write a class to handle quaternions given 8 numbers.

# In[51]:


class Q8(object):
    """Quaternions on a quaternion manifold or space-time numbers."""

    def __init__(self, values=None, qtype="Q", representation=""):
        if values is None:
            self.dt, self.dx, self.dy, self.dz = Doublet(), Doublet(),Doublet(), Doublet()
        elif isinstance(values, list):
            if len(values) == 4:
                self.dt = Doublet(values[0])
                self.dx = Doublet(values[1])
                self.dy = Doublet(values[2])
                self.dz = Doublet(values[3])
        
            if len(values) == 8:
                self.dt = Doublet([values[0], values[1]])
                self.dx = Doublet([values[2], values[3]])
                self.dy = Doublet([values[4], values[5]])
                self.dz = Doublet([values[6], values[7]])
                
        self.representation = representation
        
        if representation != "":
            self.dt.p, self.dt.n, self.dx.p, self.dx.n, self.dy.p, self.dy.n, self.dz.p, self.dz.n = self.representation_2_txyz(representation)
                
        self.qtype=qtype
                
    def __str__(self, quiet=False):
        """Customize the output."""
        
        qtype = self.qtype
        
        if quiet:
            qtype = ""
            
        if self.representation == "":
            string = "(({tp}, {tn}), ({xp}, {xn}), ({yp}, {yn}), ({zp}, {zn})) {qt}".format(tp=self.dt.p, tn=self.dt.n, 
                                                             xp=self.dx.p, xn=self.dx.n, 
                                                             yp=self.dy.p, yn=self.dy.n, 
                                                             zp=self.dz.p, zn=self.dz.n,
                                                             qt=qtype)
    
        elif self.representation == "polar":
            rep = self.txyz_2_representation("polar")
            string = "(({Ap}, {An}) A, ({thetaXp}, {thetaXn})  𝜈x, ({thetaYp}, {thetaYn}) 𝜈y, ({thetaZp}, {thetaZn}) 𝜈z) {qt}".format(
                Ap=rep[0], An=rep[1], 
                thetaXp=rep[2], thetaXn=rep[3], 
                thetaYp=rep[4], thetaYn=rep[5], 
                thetaZp=rep[6], thetaZn=rep[7], qt=qtype)
 
        elif self.representation == "spherical":
            rep = self.txyz_2_representation("spherical")
            string = "(({tp}, {tn}) t, ({Rp}, {Rn}) R, ({thetap}, {thetan}) θ , ({phip}, {phin}) φ) {qt}".format(
                tp=rep[0], tn=rep[1], 
                Rp=rep[2], Rn=rep[3], 
                thetap=rep[4], thetan=rep[5], 
                phip=rep[6], phin=rep[7], qt=qtype)
            
        return string 

    def is_symbolic(self):
        """Looks to see if a symbol is inside one of the terms."""
        
        symbolic = False
        
        if hasattr(self.dt.p, "free_symbols") or hasattr(self.dt.n, "free_symbols") or             hasattr(self.dx.p, "free_symbols") or hasattr(self.dx.n, "free_symbols") or             hasattr(self.dy.p, "free_symbols") or hasattr(self.dy.n, "free_symbols") or             hasattr(self.dz.p, "free_symbols") or hasattr(self.dz.n, "free_symbols"): 
            symbolic = True
            
        return symbolic
        
    def txyz_2_representation(self, representation):
        """Converts Cartesian txyz into an array of 4 values in a different representation."""

        symbolic = self.is_symbolic()
                
        if representation == "":
            rep = [self.dt.p, self.dt.n, self.dx.p, self.dx.n, self.dy.p, self.dy.n, self.dz.p, self.dz.n]
            return rep
        
        elif representation == "polar":
            
            dtr = self.dt.p - self.dt.n
            dxr = self.dx.p - self.dx.n
            dyr = self.dy.p - self.dy.n
            dzr = self.dz.p - self.dz.n
            
            amplitude = (dtr ** 2 + dxr ** 2 + dyr **2 + dzr **2) ** (1/2)
            damp = Doublet(amplitude)
            
            abs_v = self.abs_of_vector().dt.p
            
            if symbolic:
                theta = sp.atan2(abs_v, dtr)
            else:
                theta = math.atan2(abs_v, dtr)
                
            if abs_v == 0:
                dthetaX, dthetaY, dthetaZ = Doublet(), Doublet(), Doublet()
                
            else:
                thetaX = dxr / abs_v * theta
                thetaY = dyr / abs_v * theta
                thetaZ = dzr / abs_v * theta
                
                dthetaX = Doublet(thetaX)
                dthetaY = Doublet(thetaY)
                dthetaZ = Doublet(thetaZ)
                
            rep = [damp.p, damp.n, dthetaX.p, dthetaX.n, dthetaY.p, dthetaY.n, dthetaZ.p, dthetaZ.n]
            return rep
        
        elif representation == "spherical":
            
            dtr = self.dt.p - self.dt.n
            dxr = self.dx.p - self.dx.n
            dyr = self.dy.p - self.dy.n
            dzr = self.dz.p - self.dz.n
            
            dt = self.dt
            
            R =(dxr ** 2 + dyr **2 + dzr**2) ** (1/2)
            
            if symbolic:
                theta = sp.acos(dzr / R)
                phi = sp.atan2(dyr, dxr)
            
            else:
                theta = math.acos(dzr / R)
                phi = math.atan2(dyr, dxr)

            dR = Doublet(R)
            dtheta = Doublet(theta)
            dphi = Doublet(phi)
            
            rep = [dt.p, dt.n, dR.p, dR.n, dtheta.p, dtheta.n, dphi.p, dphi.n] 
            return rep
        
        else:
            print("Oops, don't know representation: ", representation)
        
    def representation_2_txyz(self, representation):
        """Convert from a representation to Cartesian txyz."""
        
        symbolic = self.is_symbolic()

        if representation == "":
            dt, dx, dy, dz = self.dt, self.dx, self.dy, self.dz
        
        elif representation == "polar":
                                
            amplitude, thetaX, thetaY, thetaZ = self.dt, self.dx, self.dy, self.dz
    
            amp = amplitude.p - amplitude.n
            thetaXr = thetaX.p - thetaX.n
            thetaYr = thetaY.p - thetaY.n
            thetaZr = thetaZ.p - thetaZ.n
                                
            theta = (thetaXr ** 2 + thetaYr ** 2 + thetaZr ** 2) ** (1/2)
                
            if theta == 0:
                dt = amplitude
                dx, dy, dz = Doublet(), Doublet(), Doublet()
            
            else:
                if symbolic:
                    t = amp * sp.cos(theta)
                    x = thetaXr / theta * amp * sp.sin(theta)
                    y = thetaYr / theta * amp * sp.sin(theta)
                    z = thetaZr / theta * amp * sp.sin(theta)
                else:
                    t = amp * math.cos(theta)
                    x = thetaXr / theta * amp * math.sin(theta)
                    y = thetaYr / theta * amp * math.sin(theta)
                    z = thetaZr / theta * amp * math.sin(theta)
                    
                dt = Doublet(t)
                dx = Doublet(x)
                dy = Doublet(y)
                dz = Doublet(z)
                                
        elif representation == "spherical":
            dt, R, theta, phi = self.dt, self.dx, self.dy, self.dz

            Rr = R.p - R.n
            thetar = theta.p - theta.n
            phir = phi.p - phi.n
            
            if symbolic:
                x = Rr * sp.sin(thetar) * sp.cos(phir)
                y = Rr * sp.sin(thetar) * sp.sin(phir)
                z = Rr * sp.cos(thetar)
                
            else:
                x = Rr * math.sin(thetar) * math.cos(phir)
                y = Rr * math.sin(thetar) * math.sin(phir)
                z = Rr * math.cos(thetar)

                dx = Doublet(x)
                dy = Doublet(y)
                dz = Doublet(z)

        else:
            print("Oops, don't know representation: ", representation)
            
        txyz = [dt.p, dt.n, dx.p, dx.n, dy.p, dy.n, dz.p, dz.n]
        
        return txyz 
        
    def check_representations(self, q1):
        """If they are the same, report true. If not, kick out an exception. Don't add apples to oranges."""

        if self.representation == q1.representation:
            return True
        
        else:
            raise Exception("Oops, 2 quaternions have different representations: {}, {}".format(self.representation, q1.representation))
            return False
        
    def q4(self):
        """Return a 4 element array."""
        return [self.dt.p - self.dt.n, self.dx.p - self.dx.n, self.dy.p - self.dy.n, self.dz.p - self.dz.n]
            
    def q_0(self, qtype="0"):
        """Return a zero quaternion."""
        
        return Q8(qtype=qtype, representation=self.representation)
      
    def q_1(self, qtype="1"):
        """Return a multiplicative identity quaternion."""
        
        return Q8([1, 0, 0, 0], qtype=qtype, representation=self.representation)
    
    def q_i(self, qtype="i"):
        """Return i."""
        
        return Q8([0, 1, 0, 0], qtype=qtype, representation=self.representation)
    
    def q_j(self, qtype="j"):
        """Return j."""
        
        return Q8([0, 0, 1, 0], qtype=qtype, representation=self.representation)
    
    def q_k(self, qtype="k"):
        """Return k."""
        
        return Q8([0, 0, 0, 1], qtype=qtype, representation=self.representation)
    
    def q_random(self, qtype="?"):
        """Return a random-valued quaternion."""

        return Q8([random.random(), random.random(), random.random(), random.random()], qtype=qtype, representation=self.representation)
    
    def equals(self, q1):
        """Tests if two quaternions are equal."""    
        
        if self.dt.d_equals(q1.dt) and self.dx.d_equals(q1.dx) and             self.dy.d_equals(q1.dy) and self.dz.d_equals(q1.dz):
            return True
        
        else:
            return False
    
    def conj(self, conj_type=0, qtype="*"):
        """Three types of conjugates."""
        
        end_qtype = "{st}{qt}".format(st=self.qtype, qt=qtype)
        
        conj_q = Q8()

        if conj_type == 0:
            conj_q.dt = self.dt
            conj_q.dx = self.dx.d_additive_inverse_up_to_an_automorphism()
            conj_q.dy = self.dy.d_additive_inverse_up_to_an_automorphism()
            conj_q.dz = self.dz.d_additive_inverse_up_to_an_automorphism()
        
        if conj_type == 1:
            conj_q.dt = self.dt.d_additive_inverse_up_to_an_automorphism()
            conj_q.dx = self.dx
            conj_q.dy = self.dy.d_additive_inverse_up_to_an_automorphism()
            conj_q.dz = self.dz.d_additive_inverse_up_to_an_automorphism()
            end_qtype += "1"
            
        if conj_type == 2:
            conj_q.dt = self.dt.d_additive_inverse_up_to_an_automorphism()
            conj_q.dx = self.dx.d_additive_inverse_up_to_an_automorphism()
            conj_q.dy = self.dy
            conj_q.dz = self.dz.d_additive_inverse_up_to_an_automorphism()
            end_qtype += "2"
                   
        conj_q.qtype = end_qtype
        conj_q.representation = self.representation
        
        return conj_q
    
    def vahlen_conj(self, conj_type="-", qtype="vc"):
        """Three types of conjugates -'* done by Vahlen in 1901."""
        
        end_qtype = "{st}{qt}".format(st=self.qtype, qt=qtype)
        conj_q = Q8()

        if conj_type == "-":
            conj_q.dt = self.dt
            conj_q.dx = self.dx.d_additive_inverse_up_to_an_automorphism()
            conj_q.dy = self.dy.d_additive_inverse_up_to_an_automorphism()
            conj_q.dz = self.dz.d_additive_inverse_up_to_an_automorphism()
            end_qtype += "-"
            
        if conj_type == "'":
            conj_q.dt = self.dt
            conj_q.dx = self.dx.d_additive_inverse_up_to_an_automorphism()
            conj_q.dy = self.dy.d_additive_inverse_up_to_an_automorphism()
            conj_q.dz = self.dz
            end_qtype += "'"
            
        if conj_type == "*":
            conj_q.dt = self.dt
            conj_q.dx = self.dx
            conj_q.dy = self.dy
            conj_q.dz = self.dz.d_additive_inverse_up_to_an_automorphism()
            end_qtype += "*"

        conj_q.qtype = end_qtype
        conj_q.representation = self.representation
        
        return conj_q

    def flip_signs(self, qtype=""):
        """Flip all the signs, just like multipying by -1."""

        end_qtype = "-{}".format(self.qtype)
        
        dt, dx, dy, dz = self.dt, self.dx, self.dy, self.dz
        flip_q = Q8(qtype=end_qtype)
        
        flip_q.dt.p = dt.n
        flip_q.dt.n = dt.p
        flip_q.dx.p = dx.n
        flip_q.dx.n = dx.p
        flip_q.dy.p = dy.n
        flip_q.dy.n = dy.p
        flip_q.dz.p = dz.n
        flip_q.dz.n = dz.p
        
        flip_q.qtype = end_qtype
        flip_q.representation = self.representation
        
        return flip_q
    
    
    def _commuting_products(self, q1):
        """Returns a dictionary with the commuting products."""

        products = {'tt': self.dt.Z2_product(q1.dt),
                    'xx+yy+zz': self.dx.Z2_product(q1.dx).d_add(self.dy.Z2_product(q1.dy)).d_add(self.dz.Z2_product(q1.dz)),
        
                    'tx+xt': self.dt.Z2_product(q1.dx).d_add(self.dx.Z2_product(q1.dt)),
                    'ty+yt': self.dt.Z2_product(q1.dy).d_add(self.dy.Z2_product(q1.dt)),
                    'tz+zt': self.dt.Z2_product(q1.dz).d_add(self.dz.Z2_product(q1.dt))}
        
        return products
    
    def _anti_commuting_products(self, q1):
        """Returns a dictionary with the three anti-commuting products."""

        products = {'yz-zy': self.dy.Z2_product(q1.dz).d_dif(self.dz.Z2_product(q1.dy)),
                    'zx-xz': self.dz.Z2_product(q1.dx).d_dif(self.dx.Z2_product(q1.dz)),
                    'xy-yx': self.dx.Z2_product(q1.dy).d_dif(self.dy.Z2_product(q1.dx)),
                    'zy-yz': self.dz.Z2_product(q1.dy).d_dif(self.dy.Z2_product(q1.dz)),
                    'xz-zx': self.dx.Z2_product(q1.dz).d_dif(self.dz.Z2_product(q1.dx)),
                    'yx-xy': self.dy.Z2_product(q1.dx).d_dif(self.dx.Z2_product(q1.dy))
                   }
        
        return products
    
    def _all_products(self, q1):
        """Returns a dictionary with all possible products."""

        products = self._commuting_products(q1)
        products.update(self.anti_commuting_products(q1))
        
        return products
    
    def square(self, qtype="^2"):
        """Square a quaternion."""
        
        end_qtype = "{st}{qt}".format(st=self.qtype, qt=qtype)
        
        qxq = self._commuting_products(self)
        
        sq = Q8(qtype=end_qtype, representation=self.representation)
        sq.dt = qxq['tt'].d_dif(qxq['xx+yy+zz'])
        sq.dx = qxq['tx+xt']
        sq.dy = qxq['ty+yt']
        sq.dz = qxq['tz+zt']
        
        return sq
    
    def reduce(self, qtype="reduced"):
        """Put all doublets into the reduced form so one of each pair is zero."""

        end_qtype = "{st}-{qt}".format(st=self.qtype, qt=qtype)
        
        q_red = Q8(qtype=end_qtype)
        q_red.dt = self.dt.d_reduce()
        q_red.dx = self.dx.d_reduce()
        q_red.dy = self.dy.d_reduce()
        q_red.dz = self.dz.d_reduce()
        
        q_red.representation = self.representation
        
        return q_red
    
    def norm_squared(self, qtype="|| ||^2"):
        """The norm_squared of a quaternion."""
        
        end_qtype = "||{st}||^2".format(st=self.qtype)
        
        qxq = self._commuting_products(self)
        n_q = Q8()
        n_q.dt = qxq['tt'].d_add(qxq['xx+yy+zz'])
        result = n_q.reduce()
        result.qtype = end_qtype
        result.representation = self.representation
        
        return result
    
    def norm_squared_of_vector(self, qtype="V(|| ||)^2"):
        """The norm_squared of the vector of a quaternion."""
        
        end_qtype = "||{st}||^2".format(st=self.qtype)
        
        qxq = self._commuting_products(self)
        nv_q = Q8()
        nv_q.dt = qxq['xx+yy+zz']
        result = nv_q.reduce()
        result.qtype = end_qtype
        result.representation = self.representation

        return result
    
    def abs_of_q(self, qtype="| |"):
        """The absolute value, the square root of the norm_squared."""

        end_qtype = "|{st}|".format(st=self.qtype, qt=qtype)
        
        a = self.norm_squared()
        sqrt_t = a.dt.p ** (1/2)
        a.dt.p = sqrt_t
        a.qtype = end_qtype
        a.representation = self.representation
        
        return a

    def abs_of_vector(self, qtype="|V|"):
        """The absolute value of the vector, the square root of the norm_squared of the vector."""

        end_qtype = "|{st}|".format(st=self.qtype, qt=qtype)
        
        av = self.norm_squared_of_vector()
        sqrt_t = av.dt.p ** (1/2)
        av.dt.p = sqrt_t
        
        av.qtype = end_qtype
        av.representation = self.representation
        
        return av
    
    def normalize(self, n=1, qtype="U"):
        """Normalize a quaternion"""
        
        end_qtype = "{st}U".format(st=self.qtype)
        
        abs_q_inv = self.abs_of_q().invert()
        n_q = self.product(abs_q_inv).product(Q8([n, 0, 0, 0]))
        
        n_q.qtype = end_qtype
        n_q.representation = self.representation
        
        return n_q
    
    def add(self, q1, qtype=""):
        """Form a add given 2 quaternions."""

        self.check_representations(q1)
        
        end_qtype = "{f}+{s}".format(f=self.qtype, s=q1.qtype)
        
        add_q = Q8(qtype=end_qtype, representation=self.representation)
        add_q.dt = self.dt.d_add(q1.dt)
        add_q.dx = self.dx.d_add(q1.dx)
        add_q.dy = self.dy.d_add(q1.dy)
        add_q.dz = self.dz.d_add(q1.dz)
        
        return add_q
        
    def dif(self, q1, qtype=""):
        """Form a add given 2 quaternions."""

        self.check_representations(q1)
        
        end_qtype = "{f}-{s}".format(f=self.qtype, s=q1.qtype)
            
        dif_q = Q8(qtype=end_qtype, representation=self.representation)
        dif_q.dt = self.dt.d_dif(q1.dt)
        dif_q.dx = self.dx.d_dif(q1.dx)
        dif_q.dy = self.dy.d_dif(q1.dy)
        dif_q.dz = self.dz.d_dif(q1.dz)
              
        return dif_q
    
    def product(self, q1, kind="", reverse=False, qtype=""):
        """Form a product given 2 quaternions: standard, even, odd, and even_minus_odd."""
    
        self.check_representations(q1)
    
        commuting = self._commuting_products(q1)
        q_even = Q8()
        q_even.dt = commuting['tt'].d_dif(commuting['xx+yy+zz'])
        q_even.dx = commuting['tx+xt']
        q_even.dy = commuting['ty+yt']
        q_even.dz = commuting['tz+zt']
        
        anti_commuting = self._anti_commuting_products(q1)
        q_odd = Q8()
        
        if reverse:
            q_odd.dx = anti_commuting['zy-yz']
            q_odd.dy = anti_commuting['xz-zx']
            q_odd.dz = anti_commuting['yx-xy']
            
        else:
            q_odd.dx = anti_commuting['yz-zy']
            q_odd.dy = anti_commuting['zx-xz']
            q_odd.dz = anti_commuting['xy-yx']
        
        result = Q8(representation=self.representation)
        
        if kind == "":
            result = q_even.add(q_odd)
            times_symbol = "x"
        elif kind.lower() == "even":
            result = q_even
            times_symbol = "xE"
        elif kind.lower() == "odd":
            result = q_odd
            times_symbol = "xO"
        elif kind.lower() == "even_minus_odd":
            result = q_even.dif(q_odd)
            times_symbol = "xE-O"
        else:
            raise Exception("Fouf 'kind' values are known: '', 'even', 'odd', and 'even_minus_odd'")
            
        if reverse:
            times_symbol = times_symbol.replace('x', 'xR')
            
        if qtype:
            result.qtype = qtype
        else:
            result.qtype = "{f}{ts}{s}".format(f=self.qtype, ts=times_symbol, s=q1.qtype)
        
        return result
    
    def Euclidean_product(self, q1, kind="", reverse=False, qtype=""):
        """Form a product p* q given 2 quaternions, not associative."""

        self.check_representations(q1)
        
        pq = Q8(representation=self.representation)
        pq = self.conj().product(q1, kind, reverse, qtype)
            
        return pq
    
    def invert(self, qtype="^-1"):
        """Invert a quaternion."""
        
        end_qtype = "{st}{qt}".format(st=self.qtype, qt=qtype)
        
        q_conj = self.conj()
        q_norm_squared = self.norm_squared().reduce()
        
        if q_norm_squared.dt.p == 0:
            return self.q_0()
        
        q_norm_squared_inv = Q8([1.0 / q_norm_squared.dt.p, 0, 0, 0, 0, 0, 0, 0])

        q_inv = q_conj.product(q_norm_squared_inv, qtype=self.qtype)
        q_inv.qtype = end_qtype
        q_inv.representation = self.representation
        
        return q_inv

    def divide_by(self, q1, qtype=""):
        """Divide one quaternion by another. The order matters unless one is using a norm_squared (real number)."""

        self.check_representations(q1)
        
        end_qtype = "{f}/{s}".format(f=self.qtype, s=q1.qtype)
            
        q_inv = q1.invert()
        q_div = self.product(q_inv) 
        q_div.qtype = end_qtype
        q_div.representation = self.representation
        
        return q_div
    
    def triple_product(self, q1, q2):
        """Form a triple product given 3 quaternions."""
        
        self.check_representations(q1)
        self.check_representations(q2)
        
        triple = self.product(q1).product(q2)
        triple.representation = self.representation
        
        return triple
    
    # Quaternion rotation involves a triple product:  UQU∗
    # where the U is a unitary quaternion (having a norm_squared of one).
    def rotate(self, a_1p=0, a_1n=0, a_2p=0, a_2n=0, a_3p=0, a_3n=0):
        """Do a rotation given up to three angles."""
    
        u = Q8([0, 0, a_1p, a_1n, a_2p, a_2n, a_3p, a_3n])
        u_abs = u.abs_of_q()
        u_norm_squaredalized = u.divide_by(u_abs)
        q_rot = u_norm_squaredalized.triple_product(self, u_norm_squaredalized.conj())
        q_rot.representation = self.representation
        
        return q_rot
    
    # A boost also uses triple products like a rotation, but more of them.
    # This is not a well-known result, but does work.
    def boost(self, beta_x=0, beta_y=0, beta_z=0, qtype="Boost!"):
        """A boost along the x, y, and/or z axis."""
    
        end_qtype = "{st}{qt}".format(st=self.qtype, qt=qtype)
        
        boost = Q8(sr_gamma_betas(beta_x, beta_y, beta_z))
        b_conj = boost.conj()
        
        triple_1 = boost.triple_product(self, b_conj)
        triple_2 = boost.triple_product(boost, self).conj()
        triple_3 = b_conj.triple_product(b_conj, self).conj()
              
        triple_23 = triple_2.dif(triple_3)
        half_23 = triple_23.product(Q8([0.5, 0, 0, 0, 0, 0, 0, 0]))
        triple_123 = triple_1.add(half_23, qtype=self.qtype)
        triple_123.qtype = end_qtype
        triple_123.representation = self.representation
        
        return triple_123
    
    # g_shift is a function based on the space-times-time invariance proposal for gravity,
    # which proposes that if one changes the distance from a gravitational source, then
    # squares a measurement, the observers at two different hieghts agree to their
    # space-times-time values, but not the intervals.
    def g_shift(self, dimensionless_g, g_form="exp", qtype="G_shift"):
        """Shift an observation based on a dimensionless GM/c^2 dR."""
        
        end_qtype = "{st}{qt}".format(st=self.qtype, qt=qtype)
        
        if g_form == "exp":
            g_factor = sp.exp(dimensionless_g)
            if qtype == "g_shift":
                qtype = "g_exp"
        elif g_form == "minimal":
            g_factor = 1 + 2 * dimensionless_g + 2 * dimensionless_g ** 2
            if qtype == "g_shift":
                qtype = "g_minimal"
        else:
            print("g_form not defined, should be 'exp' or 'minimal': {}".format(g_form))
            return self
        exp_g = sp.exp(dimensionless_g)
        
        g_q = Q8(qtype=end_qtype, representation=self.representation)
        g_q.dt = Doublet([self.dt.p / exp_g, self.dt.n / exp_g])
        g_q.dx = Doublet([self.dx.p * exp_g, self.dx.n * exp_g])
        g_q.dy = Doublet([self.dy.p * exp_g, self.dy.n * exp_g])
        g_q.dz = Doublet([self.dz.p * exp_g, self.dz.n * exp_g])
        
        return g_q
    
    def sin(self, qtype="sin"):
        """Take the sine of a quaternion, (sin(t) cosh(|R|), cos(t) sinh(|R|) R/|R|)"""

        end_qtype = "sin({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        red_t = self.dt.d_reduce()

        if red_t.p == 0 and red_t.n != 0:
            if abs_v.dt.p == 0:    
                return Q8([-1 * math.sin(red_t.n), 0, 0, 0], qtype=end_qtype)
            
            sint = math.sin(-1 * red_t.n)
            cost = math.cos(-1 * red_t.n)    
            
        else:
            if abs_v.dt.p == 0:    
                return Q8([math.sin(red_t.p), 0, 0, 0], qtype=end_qtype)

            sint = math.sin(red_t.p)
            cost = math.cos(red_t.p)    
            
        sinhR = math.sinh(abs_v.dt.p)
        coshR = math.cosh(abs_v.dt.p)
        
        k = cost * sinhR / abs_v.dt.p

        q_out = Q8(qtype=end_qtype, representation=self.representation)
        q_out.dt = Doublet(sint * coshR)
        q_out.dx = Doublet(k * (self.dx.p - self.dx.n))
        q_out.dy = Doublet(k * (self.dy.p - self.dy.n))
        q_out.dz = Doublet(k * (self.dz.p - self.dz.n))

        return q_out
     
    def cos(self, qtype="cos"):
        """Take the cosine of a quaternion, (cos(t) cosh(|R|), sin(t) sinh(|R|) R/|R|)"""
        
        end_qtype = "cos({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        red_t = self.dt.d_reduce()
       
        if red_t.p == 0 and red_t.n != 0:
            if abs_v.dt.p == 0:    
                return Q8([math.cos(-1 * red_t.n), 0, 0, 0], qtype=end_qtype)
        
            sint = math.sin(-1 * red_t.n)
            cost = math.cos(-1 * red_t.n) 
            
        else:
            if abs_v.dt.p == 0:    
                return Q8([math.cos(red_t.p), 0, 0, 0], qtype=end_qtype)
        
            sint = math.sin(red_t.p)
            cost = math.cos(red_t.p)
            
        sinhR = math.sinh(abs_v.dt.p)
        coshR = math.cosh(abs_v.dt.p)
        
        k = -1 * sint * sinhR / abs_v.dt.p
        
        q_out = Q8(qtype=end_qtype, representation=self.representation)
        q_out.dt = Doublet(cost * coshR)
        q_out.dx = Doublet(k * (self.dx.p - self.dx.n))
        q_out.dy = Doublet(k * (self.dy.p - self.dy.n))
        q_out.dz = Doublet(k * (self.dz.p - self.dz.n))

        return q_out
    
    def tan(self, qtype="sin"):
        """Take the tan of a quaternion, sin/cos"""

        end_qtype = "tan({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()        
        red_t = self.dt.d_reduce()
        
        if red_t.p == 0 and red_t.n != 0:
            if abs_v.dt == 0:    
                return Q8([math.tan(-1 * red_t.n), 0, 0, 0], qtype=end_qtype)
        else:
            if abs_v.dt.p == 0:    
                return Q8([math.tan(red_t.p), 0, 0, 0], qtype=end_qtype)
            
        sinq = self.sin()
        cosq = self.cos()
            
        q_out = sinq.divide_by(cosq) 
        q_out.qtype = end_qtype
        q_out.representation = self.representation
        
        return q_out
    
    def sinh(self, qtype="sin"):
        """Take the sinh of a quaternion, (sinh(t) cos(|R|), cosh(t) sin(|R|) R/|R|)"""

        end_qtype = "sinh({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        red_t = self.dt.d_reduce()
        
        if red_t.p == 0 and red_t.n != 0: 
            if abs_v.dt.p == 0:    
                return Q8([math.sinh(-1 * red_t.n), 0, 0, 0], qtype=end_qtype)
            
            sinht = math.sinh(-1 * red_t.n)
            cosht = math.cosh(-1 * red_t.n)
            
        else: 
            if abs_v.dt.p == 0:    
                return Q8([math.sinh(red_t.p), 0, 0, 0], qtype=end_qtype)
            
            sinht = math.sinh(red_t.p)
            cosht = math.cosh(red_t.p)
            
        sinR = math.sin(abs_v.dt.p)
        cosR = math.cos(abs_v.dt.p)
        
        k = cosht * sinR / abs_v.dt.p
        
        q_out = Q8(qtype=end_qtype, representation=self.representation)
        q_out.dt = Doublet(sinht * cosR)
        q_out.dx = Doublet(k * (self.dx.p - self.dx.n))
        q_out.dy = Doublet(k * (self.dy.p - self.dy.n))
        q_out.dz = Doublet(k * (self.dz.p - self.dz.n))

        return q_out
    
    def cosh(self, qtype="cosh"):
        """Take the cosh of a quaternion, (cosh(t) cos(|R|), sinh(t) sin(|R|) R/|R|)"""

        end_qtype = "cosh({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()        
        red_t = self.dt.d_reduce()
        
        if red_t.p == 0 and red_t.n != 0:
            if abs_v.dt.p == 0:    
                return Q8([math.cosh(-1 * red_t.n), 0, 0, 0], qtype=end_qtype)
            
            sinht = math.sinh(-1 * red_t.n)
            cosht = math.cosh(-1 * red_t.n)
        
        else:
            if abs_v.dt.p == 0:    
                return Q8([math.cosh(red_t.p), 0, 0, 0], qtype=end_qtype)
            
            sinht = math.sinh(red_t.p)
            cosht = math.cosh(red_t.p)
             
        sinR = math.sin(abs_v.dt.p)
        cosR = math.cos(abs_v.dt.p)
        
        k = sinht * sinR / abs_v.dt.p
            
        q_out = Q8(qtype=end_qtype, representation=self.representation)
        q_out.dt = Doublet(cosht * cosR)
        q_out.dx = Doublet(k * (self.dx.p - self.dx.n))
        q_out.dy = Doublet(k * (self.dy.p - self.dy.n))
        q_out.dz = Doublet(k * (self.dz.p - self.dz.n))
        q_out.qtype = end_qtype
        
        return q_out
    
    def tanh(self, qtype="sin"):
        """Take the tanh of a quaternion, sin/cos"""

        end_qtype = "tanh({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()        
        red_t = self.dt.d_reduce()
        
        if abs_v.dt.p == 0:
            if red_t.p == 0 and red_t.n != 0:
                return Q8([-1 * math.tanh(self.dt.n), 0, 0, 0], qtype=end_qtype)
            
            else:
                return Q8([math.tanh(self.dt.p), 0, 0, 0], qtype=end_qtype)
            
        sinhq = self.sinh()
        coshq = self.cosh()
            
        q_out = sinhq.divide_by(coshq) 
        q_out.qtype = end_qtype
        q_out.representation = self.representation
        
        return q_out
    
    def exp(self, qtype="exp"):
        """Take the exponential of a quaternion."""
        # exp(q) = (exp(t) cos(|R|, exp(t) sin(|R|) R/|R|)
        
        end_qtype = "exp({st})".format(st=self.qtype)
        
        abs_v = self.abs_of_vector()
        red_t = self.dt.d_reduce()
        
        if red_t.p == 0 and red_t.n != 0:
            et = math.exp(-1 * red_t.n)
            
            if (abs_v.dt.p == 0):
                return Q8([et, 0, 0, 0], qtype=end_qtype)
            
            cosR = math.cos(abs_v.dt.p)
            sinR = math.sin(abs_v.dt.p)
    
        else:
            et = math.exp(red_t.p)
            
            if (abs_v.dt.p == 0):
                return Q8([et, 0, 0, 0], qtype=end_qtype)
            
            cosR = math.cos(abs_v.dt.p)
            sinR = math.sin(abs_v.dt.p)
            
        k = et * sinR / abs_v.dt.p
             
        expq = Q8(qtype=end_qtype, representation=self.representation)
        expq.dt = Doublet(et * cosR)
        expq.dx = Doublet(k * (self.dx.p - self.dx.n))
        expq.dy = Doublet(k * (self.dy.p - self.dy.n))
        expq.dz = Doublet(k * (self.dz.p - self.dz.n))
                       
        return expq
    
    def ln(self, qtype="ln"):
        """Take the natural log of a quaternion."""
        # ln(q) = (0.5 ln t^2 + R.R, atan2(|R|, t) R/|R|)
        
        end_qtype = "ln({st})".format(st=self.qtype)
        
        abs_v = self.abs_of_vector()
        red_t = self.dt.d_reduce()
        
        if red_t.p == 0 and red_t.n != 0:
            if (abs_v.dt.p == 0):
                # I don't understant this, but mathematica does the same thing, but it looks wrong to me.
                return(Q8([math.log(-self.dt.n), math.pi, 0, 0], qtype=end_qtype))   
            
            t_value = 0.5 * math.log(red_t.n * red_t.n + abs_v.dt.p * abs_v.dt.p)
            k = math.atan2(abs_v.dt.p, red_t.n) / abs_v.dt.p
        
        else:
            if (abs_v.dt.p == 0):
                return(Q8([math.log(self.dt.p), 0, 0, 0], qtype=end_qtype))
                
            t_value = 0.5 * math.log(red_t.p * red_t.p + abs_v.dt.p * abs_v.dt.p)
            k = math.atan2(abs_v.dt.p, red_t.p) / abs_v.dt.p
            
        lnq = Q8(qtype=end_qtype, representation=self.representation)
        lnq.dt = Doublet(t_value)
        lnq.dx = Doublet(k * (self.dx.p - self.dx.n))
        lnq.dy = Doublet(k * (self.dy.p - self.dy.n))
        lnq.dz = Doublet(k * (self.dz.p - self.dz.n))
                       
        return lnq
    
    def q_2_q(self, q1, qtype="P"):
        """Take the natural log of a quaternion."""
        # q^p = exp(ln(q) * p)
        
        self.check_representations(q1)
        
        end_qtype = "{st}^P".format(st=self.qtype)
        
        q2q = self.ln().product(q1).reduce().exp()           
        q2q.qtype = end_qtype
        q2q.representation = self.representation
        
        return q2q
    
    def trunc(self):
        """Truncates values."""
        
        self.dt = math.trunc(self.dt)
        self.dx = math.trunc(self.dx)
        self.dy = math.trunc(self.dy)
        self.dz = math.trunc(self.dz)
        
        return self


# In[52]:


class TestQ8(unittest.TestCase):
    """Class to make sure all the functions work as expected."""
    
    Q = Q8([1, 0, 0, 2, 0, 3, 0, 4])
    P = Q8([0, 0, 4, 0, 0, 3, 0, 0])
    R = Q8([3, 0, 0, 0, 0, 0, 0, 0])
    C = Q8([2, 0, 4, 0, 0, 0, 0, 0])
    q_big = Q8([1, 2, 3, 4, 5, 6, 7, 8])
    
    def test_qt(self):
        self.assertTrue(self.Q.dt.p == 1)
    
    def test_q_0(self):
        q_z = self.Q.q_0()
        print("q_0: {}", q_z)
        self.assertTrue(q_z.dt.p == 0)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dy.n == 0)
        self.assertTrue(q_z.dz.p == 0)
        
    def test_q_1(self):
        q_z = self.Q.q_1()
        print("q_1: {}", q_z)
        self.assertTrue(q_z.dt.p == 1)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dz.p == 0)
        
    def test_q_i(self):
        q_z = self.Q.q_i()
        print("q_i: {}", q_z)
        self.assertTrue(q_z.dt.p == 0)
        self.assertTrue(q_z.dx.p == 1)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dz.p == 0)
        
    def test_q_j(self):
        q_z = self.Q.q_j()
        print("q_j: {}", q_z)
        self.assertTrue(q_z.dt.p == 0)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dy.p == 1)
        self.assertTrue(q_z.dz.p == 0)
                
    def test_q_k(self):
        q_z = self.Q.q_k()
        print("q_k: {}", q_z)
        self.assertTrue(q_z.dt.p == 0)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dz.p == 1)
        
    def test_q_random(self):
        q_z = self.Q.q_random()
        print("q_random():", q_z)
        self.assertTrue(q_z.dt.p >= 0 and q_z.dt.p <= 1)
        self.assertTrue(q_z.dx.p >= 0 and q_z.dx.p <= 1)
        self.assertTrue(q_z.dy.p >= 0 and q_z.dy.p <= 1)
        self.assertTrue(q_z.dz.p >= 0 and q_z.dz.p <= 1)
        
    def test_equals(self):
        self.assertTrue(self.Q.equals(self.Q))
        self.assertFalse(self.Q.equals(self.P))
        
    def test_conj_0(self):
        q_z = self.Q.conj()
        print("conj 0: {}", q_z)
        self.assertTrue(q_z.dt.p == 1)
        self.assertTrue(q_z.dx.p == 2)
        self.assertTrue(q_z.dy.p == 3)
        self.assertTrue(q_z.dz.p == 4)
                 
    def test_conj_1(self):
        q_z = self.Q.conj(1)
        print("conj 1: {}", q_z)
        self.assertTrue(q_z.dt.n == 1)
        self.assertTrue(q_z.dx.n == 2)
        self.assertTrue(q_z.dy.p == 3)
        self.assertTrue(q_z.dz.p == 4)
                 
    def test_conj_2(self):
        q_z = self.Q.conj(2)
        print("conj 2: {}", q_z)
        self.assertTrue(q_z.dt.n == 1)
        self.assertTrue(q_z.dx.p == 2)
        self.assertTrue(q_z.dy.n == 3)
        self.assertTrue(q_z.dz.p == 4)
        
    def test_vahlen_conj_0(self):
        q_z = self.Q.vahlen_conj()
        print("vahlen conj -: {}", q_z)
        self.assertTrue(q_z.dt.p == 1)
        self.assertTrue(q_z.dx.p == 2)
        self.assertTrue(q_z.dy.p == 3)
        self.assertTrue(q_z.dz.p == 4)
                 
    def test_vahlen_conj_1(self):
        q_z = self.Q.vahlen_conj("'")
        print("vahlen conj ': {}", q_z)
        self.assertTrue(q_z.dt.p == 1)
        self.assertTrue(q_z.dx.p == 2)
        self.assertTrue(q_z.dy.p == 3)
        self.assertTrue(q_z.dz.n == 4)
                 
    def test_vahlen_conj_2(self):
        q_z = self.Q.vahlen_conj('*')
        print("vahlen conj *: {}", q_z)
        self.assertTrue(q_z.dt.p == 1)
        self.assertTrue(q_z.dx.n == 2)
        self.assertTrue(q_z.dy.n == 3)
        self.assertTrue(q_z.dz.p == 4)
        
    def test_square(self):
        q_sq = self.Q.square()
        q_sq_red = q_sq.reduce()
        print("square: {}".format(q_sq))
        print("square reduced: {}".format(q_sq_red))
        self.assertTrue(q_sq.dt.p == 1)
        self.assertTrue(q_sq.dt.n == 29)
        self.assertTrue(q_sq.dx.n == 4)
        self.assertTrue(q_sq.dy.n == 6)
        self.assertTrue(q_sq.dz.n == 8)
        self.assertTrue(q_sq_red.dt.p == 0)
        self.assertTrue(q_sq_red.dt.n == 28)
                
    def test_reduce(self):
        q_red = self.q_big.reduce()
        print("q_big reduced: {}".format(q_red))
        self.assertTrue(q_red.dt.p == 0)
        self.assertTrue(q_red.dt.n == 1)
        self.assertTrue(q_red.dx.p == 0)
        self.assertTrue(q_red.dx.n == 1)
        self.assertTrue(q_red.dy.p == 0)
        self.assertTrue(q_red.dy.n == 1)
        self.assertTrue(q_red.dz.p == 0)
        self.assertTrue(q_red.dz.n == 1)
        
    def test_norm_squared(self):
        q_z = self.Q.norm_squared()
        print("norm_squared: {}", q_z)
        self.assertTrue(q_z.dt.p == 30)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dx.n == 0)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dy.n == 0)
        self.assertTrue(q_z.dz.p == 0)
        self.assertTrue(q_z.dz.n == 0)
        
    def test_norm_squared_of_vector(self):
        q_z = self.Q.norm_squared_of_vector()
        print("norm_squared_of_vector: {}", q_z)
        self.assertTrue(q_z.dt.p == 29)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dx.n == 0)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dy.n == 0)
        self.assertTrue(q_z.dz.p == 0)
        self.assertTrue(q_z.dz.n == 0)
        
    def test_abs_of_q(self):
        q_z = self.P.abs_of_q()
        print("abs_of_q: {}", q_z)
        self.assertTrue(q_z.dt.p == 5)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dz.p == 0)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.n == 0)
        self.assertTrue(q_z.dy.n == 0)
        self.assertTrue(q_z.dz.n == 0)
        
    def test_abs_of_vector(self):
        q_z = self.P.abs_of_vector()
        print("abs_of_vector: {}", q_z)
        self.assertTrue(q_z.dt.p == 5)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dz.p == 0)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.n == 0)
        self.assertTrue(q_z.dy.n == 0)
        self.assertTrue(q_z.dz.n == 0)
        
    def test_normalize(self):
        q_z = self.P.normalize()
        print("q_normalized: {}", q_z)
        self.assertTrue(q_z.dt.p == 0)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.p == 0.8)
        self.assertTrue(q_z.dx.n == 0)
        self.assertTrue(q_z.dy.p == 0)
        self.assertAlmostEqual(q_z.dy.n, 0.6)
        self.assertTrue(q_z.dz.p == 0)    
        self.assertTrue(q_z.dz.n == 0)    
        
    def test_add(self):
        q_z = self.Q.add(self.P)
        print("add: {}", q_z)
        self.assertTrue(q_z.dt.p == 1)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.p == 4)
        self.assertTrue(q_z.dx.n == 2)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dy.n == 6)
        self.assertTrue(q_z.dz.p == 0)
        self.assertTrue(q_z.dz.n == 4)
        
    def test_add_reduce(self):
        q_z_red = self.Q.add(self.P).reduce()
        print("add reduce: {}".format(q_z_red))
        self.assertTrue(q_z_red.dt.p == 1)
        self.assertTrue(q_z_red.dt.n == 0)
        self.assertTrue(q_z_red.dx.p == 2)
        self.assertTrue(q_z_red.dx.n == 0)
        self.assertTrue(q_z_red.dy.p == 0)
        self.assertTrue(q_z_red.dy.n == 6)
        self.assertTrue(q_z_red.dz.p == 0)
        self.assertTrue(q_z_red.dz.n == 4)
        
    def test_dif(self):
        q_z = self.Q.dif(self.P)
        print("dif: {}", q_z)
        self.assertTrue(q_z.dt.p == 1)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dx.n == 6) 
        self.assertTrue(q_z.dy.p == 3)
        self.assertTrue(q_z.dy.n == 3)
        self.assertTrue(q_z.dz.p == 0)
        self.assertTrue(q_z.dz.n == 4) 

    def test_product(self):
        q_z = self.Q.product(self.P).reduce()
        print("product: {}", q_z)
        self.assertTrue(q_z.dt.p == 0)
        self.assertTrue(q_z.dt.n == 1)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dx.n == 8)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dy.n == 19)
        self.assertTrue(q_z.dz.p == 18)
        self.assertTrue(q_z.dz.n == 0)
        
    def test_product_even(self):
        q_z = self.Q.product(self.P, kind="even").reduce()
        print("product, kind even: {}", q_z)
        self.assertTrue(q_z.dt.p == 0)
        self.assertTrue(q_z.dt.n == 1)
        self.assertTrue(q_z.dx.p == 4)
        self.assertTrue(q_z.dx.n == 0)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dy.n == 3)
        self.assertTrue(q_z.dz.p == 0)
        self.assertTrue(q_z.dz.n == 0)
        
    def test_product_odd(self):
        q_z = self.Q.product(self.P, kind="odd").reduce()
        print("product, kind odd: {}", q_z)
        self.assertTrue(q_z.dt.p == 0)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dx.n == 12)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dy.n == 16)
        self.assertTrue(q_z.dz.p == 18)
        self.assertTrue(q_z.dz.n == 0)
    
    def test_product_even_minus_odd(self):
        q_z = self.Q.product(self.P, kind="even_minus_odd").reduce()
        print("product, kind odd: {}", q_z)
        self.assertTrue(q_z.dt.p == 0)
        self.assertTrue(q_z.dt.n == 1)
        self.assertTrue(q_z.dx.p == 16)
        self.assertTrue(q_z.dx.n == 0)
        self.assertTrue(q_z.dy.p == 13)
        self.assertTrue(q_z.dy.n == 0)
        self.assertTrue(q_z.dz.p == 0)
        self.assertTrue(q_z.dz.n == 18)
        
    def test_product_reverse(self):
        QP_rev = self.Q.product(self.P, reverse=True)
        PQ = self.P.product(self.Q)
        self.assertTrue(QP_rev.equals(PQ))
        
    def test_Euclidean_product(self):
        q_z = self.Q.Euclidean_product(self.P).reduce()
        print("Euclidean product: {}", q_z)
        self.assertTrue(q_z.dt.p == 1)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.p == 16)
        self.assertTrue(q_z.dx.n == 0)
        self.assertTrue(q_z.dy.p == 13)
        self.assertTrue(q_z.dy.n == 0)
        self.assertTrue(q_z.dz.p == 0)
        self.assertTrue(q_z.dz.n == 18)
        
    def test_invert(self):
        q_z = self.P.invert().reduce()
        print("inverse: {}", q_z)
        self.assertTrue(q_z.dt.p == 0)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dx.n == 0.16)
        self.assertTrue(q_z.dy.p == 0.12)
        self.assertTrue(q_z.dy.n == 0)
        self.assertTrue(q_z.dz.p == 0)
        self.assertTrue(q_z.dz.n == 0)

    def test_divide_by(self):
        q_z = self.Q.divide_by(self.Q).reduce()
        print("inverse: {}", q_z)
        self.assertTrue(q_z.dt.p == 1)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dx.n == 0)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dy.n == 0)
        self.assertTrue(q_z.dz.p == 0)
        self.assertTrue(q_z.dz.n == 0) 
        
    def test_triple_product(self):
        q_z = self.Q.triple_product(self.P, self.Q).reduce()
        print("triple: {}", q_z)
        self.assertTrue(q_z.dt.p == 0)
        self.assertTrue(q_z.dt.n == 2)
        self.assertTrue(q_z.dx.p == 124)
        self.assertTrue(q_z.dx.n == 0)
        self.assertTrue(q_z.dy.p == 0)
        self.assertTrue(q_z.dy.n == 84)
        self.assertTrue(q_z.dz.p == 8)
        self.assertTrue(q_z.dz.n == 0)
        
    def test_rotate(self):
        q_z = self.Q.rotate(1).reduce()
        print("rotate: {}", q_z)
        self.assertTrue(q_z.dt.p == 1)
        self.assertTrue(q_z.dt.n == 0)
        self.assertTrue(q_z.dx.p == 0)
        self.assertTrue(q_z.dx.n == 2)
        self.assertTrue(q_z.dy.p == 3)
        self.assertTrue(q_z.dy.n == 0)
        self.assertTrue(q_z.dz.p == 4)
        self.assertTrue(q_z.dz.n == 0)
        
    def test_boost(self):
        Q_sq = self.Q.square().reduce()
        q_z = self.Q.boost(0.003)
        q_z2 = q_z.square().reduce()
        print("Q_sq: {}".format(Q_sq))
        print("boosted: {}", q_z)
        print("b squared: {}".format(q_z2))
        self.assertTrue(round(q_z2.dt.n, 12) == round(Q_sq.dt.n, 12))
        
    def test_g_shift(self):
        Q_sq = self.Q.square().reduce()
        q_z = self.Q.g_shift(0.003)
        q_z2 = q_z.square().reduce()
        print("Q_sq: {}".format(Q_sq))
        print("g_shift: {}", q_z)
        print("g squared: {}".format(q_z2))
        self.assertTrue(q_z2.dt.n != Q_sq.dt.n)
        self.assertTrue(q_z2.dx.p == Q_sq.dx.p)
        self.assertTrue(q_z2.dx.n == Q_sq.dx.n)
        self.assertTrue(q_z2.dy.p == Q_sq.dy.p)
        self.assertTrue(q_z2.dy.n == Q_sq.dy.n)
        self.assertTrue(q_z2.dz.p == Q_sq.dz.p)
        self.assertTrue(q_z2.dz.n == Q_sq.dz.n)
        
    def test_sin(self):
        self.assertTrue(Q8([0, 0, 0, 0]).sin().reduce().equals(Q8().q_0()))
        self.assertTrue(self.Q.sin().reduce().equals(Q8([91.7837157840346691, -21.8864868530291758, -32.8297302795437673, -43.7729737060583517])))
        self.assertTrue(self.P.sin().reduce().equals(Q8([0,  59.3625684622310033, -44.5219263466732542, 0])))
        self.assertTrue(self.R.sin().reduce().equals(Q8([0.1411200080598672, 0, 0, 0])))
        self.assertTrue(self.C.sin().reduce().equals(Q8([24.8313058489463785, -11.3566127112181743, 0, 0])))
     
    def test_cos(self):
        self.assertTrue(Q8([0, 0, 0, 0]).cos().equals(Q8().q_1()))
        self.assertTrue(self.Q.cos().equals(Q8([58.9336461679439481, 34.0861836904655959, 51.1292755356983974, 68.1723673809311919])))
        self.assertTrue(self.P.cos().equals(Q8([74.2099485247878476, 0, 0, 0])))
        self.assertTrue(self.R.cos().equals(Q8([-0.9899924966004454, 0, 0, 0])))
        self.assertTrue(self.C.cos().equals(Q8([-11.3642347064010600, -24.8146514856341867, 0, 0])))
                            
    def test_tan(self):
        self.assertTrue(Q8([0, 0, 0, 0]).tan().equals(Q8().q_0()))
        self.assertTrue(self.Q.tan().equals(Q8([0.0000382163172501, -0.3713971716439372, -0.5570957574659058, -0.7427943432878743])))
        self.assertTrue(self.P.tan().equals(Q8([0, 0.7999273634100760, -0.5999455225575570, 0])))
        self.assertTrue(self.R.tan().equals(Q8([-0.1425465430742778, 0, 0, 0])))
        self.assertTrue(self.C.tan().equals(Q8([-0.0005079806234700, 1.0004385132020521, 0, 0])))
        
    def test_sinh(self):
        self.assertTrue(Q8([0, 0, 0, 0]).sinh().equals(Q8().q_0()))
        self.assertTrue(self.Q.sinh().equals(Q8([0.7323376060463428, 0.4482074499805421, 0.6723111749708131, 0.8964148999610841])))
        self.assertTrue(self.P.sinh().equals(Q8([0, -0.7671394197305108, 0.5753545647978831, 0])))
        self.assertTrue(self.R.sinh().equals(Q8([10.0178749274099026, 0, 0, 0])))
        self.assertTrue(self.C.sinh().equals(Q8([-2.3706741693520015, -2.8472390868488278, 0, 0])))    
        
    def test_cosh(self):
        self.assertTrue(Q8([0, 0, 0, 0]).cosh().equals(Q8().q_1()))
        self.assertTrue(self.Q.cosh().equals(Q8([0.9615851176369565, 0.3413521745610167, 0.5120282618415251, 0.6827043491220334])))
        self.assertTrue(self.P.cosh().equals(Q8([0.2836621854632263, 0, 0, 0])))
        self.assertTrue(self.R.cosh().equals(Q8([10.0676619957777653, 0, 0, 0])))
        self.assertTrue(self.C.cosh().equals(Q8([-2.4591352139173837, -2.7448170067921538, 0, 0])))    
        
    def test_tanh(self):
        self.assertTrue(Q8([0, 0, 0, 0]).tanh().equals(Q8().q_0()))
        self.assertTrue(self.Q.tanh().equals(Q8([1.0248695360556623, 0.1022956817887642, 0.1534435226831462, 0.2045913635775283])))
        self.assertTrue(self.P.tanh().equals(Q8([0, -2.7044120049972684, 2.0283090037479505, 0])))
        self.assertTrue(self.R.tanh().equals(Q8([0.9950547536867305, 0, 0, 0])))
        self.assertTrue(self.C.tanh().equals(Q8([1.0046823121902353, 0.0364233692474038, 0, 0])))    
        
    def test_exp(self):
        self.assertTrue(Q8([0, 0, 0, 0]).exp().equals(Q8().q_1()))
        self.assertTrue(self.Q.exp().equals(Q8([1.6939227236832994, 0.7895596245415588, 1.1843394368123383, 1.5791192490831176])))
        self.assertTrue(self.P.exp().equals(Q8([0.2836621854632263, -0.7671394197305108, 0.5753545647978831, 0])))
        self.assertTrue(self.R.exp().equals(Q8([20.0855369231876679, 0, 0, 0])))
        self.assertTrue(self.C.exp().equals(Q8([-4.8298093832693851, -5.5920560936409816, 0, 0])))    
    
    def test_ln(self):
        self.assertTrue(self.Q.ln().exp().equals(self.Q))
        self.assertTrue(self.Q.ln().equals(Q8([1.7005986908310777, -0.5151902926640850, -0.7727854389961275, -1.0303805853281700])))
        self.assertTrue(self.P.ln().equals(Q8([1.6094379124341003, 1.2566370614359172, -0.9424777960769379, 0])))
        self.assertTrue(self.R.ln().equals(Q8([1.0986122886681098, 0, 0, 0])))
        self.assertTrue(self.C.ln().equals(Q8([1.4978661367769954, 1.1071487177940904, 0, 0])))    
        
    def test_q_2_q(self):
        self.assertTrue(self.Q.q_2_q(self.P).equals(Q8([-0.0197219653530713, -0.2613955437374326, 0.6496281248064009, -0.3265786562423951])))

suite = unittest.TestLoader().loadTestsFromModule(TestQ8())
unittest.TextTestRunner().run(suite);


# In[53]:


class TestQ8Rep(unittest.TestCase):
    Q12 = Q8([1.0, 2.0, 0, 0])
    Q1123 = Q8([1.0, 1.0, 2, 3])
    Q11p = Q8([1.0, 1.0, 0, 0], representation="polar")
    Q12p = Q8([1.0, 2.0, 0, 0], representation="polar")
    Q12np = Q8([1.0, -2.0, 0, 0], representation="polar")
    Q21p = Q8([2.0, 1.0, 0, 0], representation="polar")
    Q23p = Q8([2.0, 3.0, 0, 0], representation="polar")
    Q13p = Q8([1.0, 3.0, 0, 0], representation="polar")
    Q5p = Q8([5.0, 0, 0, 0], representation="polar")
    
    # @unittest.skip("problems implementing")
    def test_txyz_2_representation(self):
        qr = Q8(self.Q12.txyz_2_representation("")).reduce()
        self.assertTrue(qr.equals(self.Q12))
        qr = Q8(self.Q12.txyz_2_representation("polar")).reduce()
        self.assertTrue(qr.equals(Q8([2.23606797749979, 1.10714871779409, 0, 0])))
        qr = Q8(self.Q1123.txyz_2_representation("spherical")).reduce()
        self.assertTrue(qr.equals(Q8([1.0, 3.7416573867739413, 0.640522312679424, 1.10714871779409])))

        
    # @unittest.skip("problems implementing")    
    def test_representation_2_txyz(self):
        qr = Q8(self.Q12.representation_2_txyz("")).reduce()
        self.assertTrue(qr.equals(self.Q12))
        qr = Q8(self.Q12.representation_2_txyz("polar")).reduce()
        self.assertTrue(qr.equals(Q8([-0.4161468365471424, 0.9092974268256817, 0, 0])))
        qr = Q8(self.Q1123.representation_2_txyz("spherical")).reduce()
        self.assertTrue(qr.equals(Q8([1.0, -0.9001976297355174, 0.12832006020245673, -0.4161468365471424])))
    
    def test_polar_products(self):
        qr = self.Q11p.product(self.Q12p).reduce()
        print("polar 1 1 0 0 * 1 2 0 0: ", qr)
        self.assertTrue(qr.equals(self.Q13p))
        qr = self.Q12p.product(self.Q21p).reduce()
        print("polar 1 2 0 0 * 2 1 0 0: ", qr)
        self.assertTrue(qr.equals(self.Q23p))

    def test_polar_conj(self):
        qr = self.Q12p.conj().reduce()
        print("polar conj of 1 2 0 0: ", qr)
        self.assertTrue(qr.equals(self.Q12np))
        
suite = unittest.TestLoader().loadTestsFromModule(TestQ8Rep())
unittest.TextTestRunner().run(suite);


# ## Class Q8a as nparrays

# In[54]:


class Q8a(Doubleta):
    """Quaternions on a quaternion manifold or space-time numbers."""

    def __init__(self, values=None, qtype="Q", representation=""):
        if values is None:
            d_zero = Doubleta()
            self.a = np.array([d_zero.d[0], d_zero.d[0], d_zero.d[0], d_zero.d[0], d_zero.d[0], d_zero.d[0], d_zero.d[0], d_zero.d[0]])
     
        elif isinstance(values, list):
            if len(values) == 4:
                self.a = np.array([Doubleta(values[0]).d[0], Doubleta(values[0]).d[1], 
                                   Doubleta(values[1]).d[0], Doubleta(values[1]).d[1], 
                                   Doubleta(values[2]).d[0], Doubleta(values[2]).d[1], 
                                   Doubleta(values[3]).d[0], Doubleta(values[3]).d[1]])
        
            if len(values) == 8:
                self.a = np.array([Doubleta([values[0], values[1]]).d[0], Doubleta([values[0], values[1]]).d[1],
                                   Doubleta([values[2], values[3]]).d[0], Doubleta([values[2], values[3]]).d[1],
                                   Doubleta([values[4], values[5]]).d[0], Doubleta([values[4], values[5]]).d[1],
                                   Doubleta([values[6], values[7]]).d[0], Doubleta([values[6], values[7]]).d[1]])
        
        self.representation = representation
        
        if representation != "":
            rep = self.representation_2_txyz(representation)
            self.a = np.array(rep)
        
        self.qtype=qtype
                
    def __str__(self, quiet=False):
        """Customize the output."""
        
        qtype = self.qtype
        
        if quiet:
            qtype = ""
        
        if self.representation == "":
            string = "(({tp}, {tn}), ({xp}, {xn}), ({yp}, {yn}), ({zp}, {zn})) {qt}".format(tp=self.a[0], tn=self.a[1], 
                                                             xp=self.a[2], xn=self.a[3], 
                                                             yp=self.a[4], yn=self.a[5], 
                                                             zp=self.a[6], zn=self.a[7],
                                                             qt=qtype)
    
        elif self.representation == "polar":
            rep = self.txyz_2_representation("polar")
            string = "(({Ap}, {An}) A, ({thetaXp}, {thetaXn})  𝜈x, ({thetaYp}, {thetaYn}) 𝜈y, ({thetaZp}, {thetaZn}) 𝜈z) {qt}".format(Ap=rep[0], An=rep[1], thetaXp=rep[2], thetaXn=rep[3], thetaYp=rep[4], thetaYn=rep[5], thetaZp=rep[6], thetaZn=rep[7], 
                                                                    qt=qtype)
 
        elif self.representation == "spherical":
            rep = self.txyz_2_representation("spherical")
            string = "(({tp}, {tn}) t, ({Rp}, {Rn}) R, ({thetap}, {thetan}) θ , ({phip}, {phin}) φ) {qt}".format(tp=rep[0], tn=rep[1], Rp=rep[2], Rn=rep[3], thetap=rep[4], thetan=rep[5], phip=rep[6], phin=rep[7], 
                                                                       qt=qtype)
        return string 
        
        
        return "(({tp}, {tn}), ({xp}, {xn}), ({yp}, {yn}), ({zp}, {zn})) {qt}".format(tp=self.a[0], tn=self.a[1], 
                                                             xp=self.a[2], xn=self.a[3], 
                                                             yp=self.a[4], yn=self.a[5], 
                                                             zp=self.a[6], zn=self.a[7],
                                                             qt=qtype)
    
        return string
    
    def is_symbolic(self):
        """Looks to see if a symbol is inside one of the terms."""
        
        symbolic = False
        
        for i in range(8):
            if hasattr(self.a[i], "free_symbols"): 
                symbolic = True
            
        return symbolic
        
    def txyz_2_representation(self, representation):
        """Converts Cartesian txyz into an array of 4 values in a different representation."""

        symbolic = self.is_symbolic()
                
        if representation == "":
            rep = [self.a[0], self.a[1], self.a[2], self.a[3], self.a[4], self.a[5], self.a[6], self.a[7]]
        
        elif representation == "polar":
            
            dtr = self.a[0] - self.a[1]
            dxr = self.a[2] - self.a[3]
            dyr = self.a[4] - self.a[5]
            dzr = self.a[6] - self.a[7]
            
            amplitude = (dtr ** 2 + dxr ** 2 + dyr **2 + dzr **2) ** (1/2)
            
            abs_v = self.abs_of_vector().a[0]
            
            if symbolic:
                theta = sp.atan2(abs_v, dtr)
            else:
                theta = math.atan2(abs_v, dtr)
                
            if abs_v == 0:
                thetaX, thetaY, thetaZ = 0, 0, 0
                
            else:
                thetaX = theta * dxr / abs_v
                thetaY = theta * dyr / abs_v
                thetaZ = theta * dzr / abs_v
                
            damp = Doubleta(amplitude)
            dthetaX = Doubleta(thetaX)
            dthetaY = Doubleta(thetaY)
            dthetaZ = Doubleta(thetaZ)
            
            rep = [damp.d[0], damp.d[1], 
                   dthetaX.d[0], dthetaX.d[1], 
                   dthetaY.d[0], dthetaY.d[1], 
                   dthetaZ.d[0], dthetaZ.d[1]]
        
        elif representation == "spherical":
            
            dtr = self.a[0] - self.a[1]
            dxr = self.a[2] - self.a[3]
            dyr = self.a[4] - self.a[5]
            dzr = self.a[6] - self.a[7]
            
            R = (dxr ** 2 + dyr **2 + dzr**2) ** (1/2)
            
            if symbolic:
                theta = sp.acos(dzr / R)
                phi = sp.atan2(dyr, dxr)
            
            else:
                theta = math.acos(dzr / R)
                phi = math.atan2(dyr, dxr)
                
            dt = Doubleta(dtr)
            dR = Doubleta(R)
            dtheta = Doubleta(theta)
            dphi = Doubleta(phi)
            
            rep = [dt.d[0], dt.d[1], 
                   dR.d[0], dR.d[1], 
                   dtheta.d[0], dtheta.d[1], 
                   dphi.d[0], dphi.d[1]]
        
        else:
            print("Oops, don't know representation: ", representation)
            
        return rep
    
    def representation_2_txyz(self, representation):
        """Convert from a representation to Cartesian txyz."""
        
        symbolic = self.is_symbolic()

        if representation == "":
            rep = [self.a[0], self.a[1], self.a[2], self.a[3], self.a[4], self.a[5], self.a[6], self.a[7]]
        
        elif representation == "polar":
                                
            amplitude1, amplitude2 = self.a[0], self.a[1]
            thetaX1, thetaX2 = self.a[2], self.a[3]
            thetaY1, thetaY2 = self.a[4], self.a[5]
            thetaZ1, thetaZ2 = self.a[6], self.a[7]
    
            amp = amplitude1 - amplitude2
            thetaXr = thetaX1 - thetaX2
            thetaYr = thetaY1 - thetaY2
            thetaZr = thetaZ1 - thetaZ2
                                
            theta = (thetaXr ** 2 + thetaYr ** 2 + thetaZr ** 2) ** (1/2)
                
            if theta == 0:
                t = amp
                x, y, z = 0, 0, 0
            
            else:
                if symbolic:
                    t = amp * sp.cos(theta)
                    x = thetaXr / theta * amp * sp.sin(theta)
                    y = thetaYr / theta * amp * sp.sin(theta)
                    z = thetaZr / theta * amp * sp.sin(theta)
                else:
                    t = amp * math.cos(theta)
                    x = thetaXr / theta * amp * math.sin(theta)
                    y = thetaYr / theta * amp * math.sin(theta)
                    z = thetaZr / theta * amp * math.sin(theta)
                    
            dt = Doubleta(t)
            dx = Doubleta(x)
            dy = Doubleta(y)
            dz = Doubleta(z)
            
            rep = [dt.d[0], dt.d[1],
                   dx.d[0], dx.d[1],
                   dy.d[0], dy.d[1],
                   dz.d[0], dz.d[1]]
            
        elif representation == "spherical":
            t1, t2 = self.a[0], self.a[1]
            R1, R2 = self.a[2], self.a[3]
            theta1, theta2 = self.a[4], self.a[5]
            phi1, phi2 = self.a[6], self.a[7]
    
            t = t1 - t2
            R = R1 - R2
            thetar = theta1 - theta2
            phir = phi1 - phi2
            
            if symbolic:
                x = R * sp.sin(thetar) * sp.cos(phir)
                y = R * sp.sin(thetar) * sp.sin(phir)
                z = R * sp.cos(thetar)
                
            else:
                x = R * math.sin(thetar) * math.cos(phir)
                y = R * math.sin(thetar) * math.sin(phir)
                z = R * math.cos(thetar)
                
            dt = Doubleta(t)
            dx = Doubleta(x)
            dy = Doubleta(y)
            dz = Doubleta(z)

            rep = [dt.d[0], dt.d[1],
                   dx.d[0], dx.d[1],
                   dy.d[0], dy.d[1],
                   dz.d[0], dz.d[1]]

        else:
            print("Oops, don't know representation: ", representation)
        
        return rep
        
    def check_representations(self, q1):
        """If they are the same, report true. If not, kick out an exception. Don't add apples to oranges."""

        if self.representation == q1.representation:
            return True
        
        else:
            raise Exception("Oops, 2 quaternions have different representations: {}, {}".format(self.representation, q1.representation))
            return False
    
    def q4(self):
        """Return a 4 element array."""
        return [self.a[0] - self.a[1], self.a[0] - self.a[1], self.a[4] - self.a[5], self.a[6] - self.a[7]]
            
    def q_0(self, qtype="0"):
        """Return a zero quaternion."""
        
        q0 = Q8a(qtype=qtype, representation=self.representation)
        
        return q0
      
    def q_1(self, qtype="1"):
        """Return a multiplicative identity quaternion."""
        
        q1 = Q8a([1, 0, 0, 0, 0, 0, 0, 0], qtype=qtype, representation=self.representation)
        
        return q1
    
    def q_i(self, qtype="i"):
        """Return i."""
        
        qi = Q8a([0, 0, 1, 0, 0, 0, 0, 0], qtype=qtype, representation=self.representation)
        return qi
    
    def q_j(self, qtype="j"):
        """Return j."""
        
        qj = Q8a([0, 0, 0, 0, 1, 0, 0, 0], qtype=qtype, representation=self.representation)
        
        return qj
    

    def q_k(self, qtype="k"):
        """Return k."""
        
        qk = Q8a([0, 0, 0, 0, 0, 0, 1, 0], qtype=qtype, representation=self.representation)
        
        return qk

    def q_random(self, qtype="?"):
        """Return a random-valued quaternion."""

        qr = Q8a([random.random(), random.random(), random.random(), random.random(), random.random(), random.random(), random.random(), random.random()], qtype=qtype)
        qr.representation = self.representation
        
        return qr
    
    def equals(self, q2):
        """Tests if two quaternions are equal."""
        
        self_red = self.reduce()
        q2_red = q2.reduce()
        result = True
        
        for i in range(8):
            if not math.isclose(self_red.a[i], q2_red.a[i]):
                result = False
        
        return result
    
    def conj(self, conj_type=0, qtype="*"):
        """Three types of conjugates."""
        
        conj_q = Q8a()

        # Flip all but t.                          
        if conj_type == 0:   
            conj_q.a[0] = self.a[0]
            conj_q.a[1] = self.a[1]
            conj_q.a[2] = self.a[3]
            conj_q.a[3] = self.a[2]
            conj_q.a[4] = self.a[5]
            conj_q.a[5] = self.a[4]
            conj_q.a[6] = self.a[7]
            conj_q.a[7] = self.a[6]
        
        # Flip all but x.
        if conj_type == 1:
            conj_q.a[0] = self.a[1]
            conj_q.a[1] = self.a[0]
            conj_q.a[2] = self.a[2]
            conj_q.a[3] = self.a[3]
            conj_q.a[4] = self.a[5]
            conj_q.a[5] = self.a[4]
            conj_q.a[6] = self.a[7]
            conj_q.a[7] = self.a[6]
            qtype += "1"

        # Flip all but y.                                 
        if conj_type == 2:
            conj_q.a[0] = self.a[1]
            conj_q.a[1] = self.a[0]
            conj_q.a[2] = self.a[3]
            conj_q.a[3] = self.a[2]
            conj_q.a[4] = self.a[4]
            conj_q.a[5] = self.a[5]
            conj_q.a[6] = self.a[7]
            conj_q.a[7] = self.a[6]
            qtype += "2"
            
        conj_q.qtype = self.qtype + qtype
        conj_q.representation = self.representation
        
        return conj_q
    
    def vahlen_conj(self, conj_type="-", qtype="vc"):
        """Three types of conjugates -'* done by Vahlen in 1901."""
        
        conj_q = Q8a()

        if conj_type == "-":
            conj_q.a[0] = self.a[0]
            conj_q.a[1] = self.a[1]
            conj_q.a[2] = self.a[3]
            conj_q.a[3] = self.a[2]
            conj_q.a[4] = self.a[5]
            conj_q.a[5] = self.a[4]
            conj_q.a[6] = self.a[7]
            conj_q.a[7] = self.a[6]
            qtype += "-"

        # Flip the sign of x and y.
        if conj_type == "'":
            conj_q.a[0] = self.a[0]
            conj_q.a[1] = self.a[1]
            conj_q.a[2] = self.a[3]
            conj_q.a[3] = self.a[2]
            conj_q.a[4] = self.a[5]
            conj_q.a[5] = self.a[4]
            conj_q.a[6] = self.a[6]
            conj_q.a[7] = self.a[7]
            qtype += "'"
            
        # Flip the sign of only z.
        if conj_type == "*":
            conj_q.a[0] = self.a[0]
            conj_q.a[1] = self.a[1]
            conj_q.a[2] = self.a[2]
            conj_q.a[3] = self.a[3]
            conj_q.a[4] = self.a[4]
            conj_q.a[5] = self.a[5]
            conj_q.a[6] = self.a[7]
            conj_q.a[7] = self.a[6]
            qtype += "*"
            
        conj_q.qtype = self.qtype + qtype
        conj_q.representation = self.representation
        
        return conj_q

    def flip_signs(self, conj_type=0, qtype="-"):
        """Flip all the signs, just like multipying by -1."""

        end_qtype = "-{}".format(self.qtype)
        
        t1, t2 = self.a[0], self.a[1]
        x1, x2 = self.a[2], self.a[3]
        y1, y2 = self.a[4], self.a[5]
        z1, z2 = self.a[6], self.a[7]
        
        flip_q = Q8a(qtype=end_qtype)

        flip_q.a[0] = t2
        flip_q.a[1] = t1
        flip_q.a[2] = x2
        flip_q.a[3] = x1
        flip_q.a[4] = y2
        flip_q.a[5] = y1
        flip_q.a[6] = z2
        flip_q.a[7] = z1

        flip_q.qtype = end_qtype
        flip_q.representation = self.representation
        
        return flip_q
    
    def _commuting_products(self, q1):
        """Returns a dictionary with the commuting products."""

        products = {'tt0': self.a[0] * q1.a[0] + self.a[1] * q1.a[1],
                    'tt1': self.a[0] * q1.a[1] + self.a[1] * q1.a[0],
                    
                    'xx+yy+zz0': self.a[2] * q1.a[2] + self.a[3] * q1.a[3] + self.a[4] * q1.a[4] + self.a[5] * q1.a[5] + self.a[6] * q1.a[6] + self.a[7] * q1.a[7], 
                    'xx+yy+zz1': self.a[2] * q1.a[3] + self.a[3] * q1.a[2] + self.a[4] * q1.a[5] + self.a[5] * q1.a[4] + self.a[6] * q1.a[7] + self.a[7] * q1.a[6], 
                    
                    'tx+xt0': self.a[0] * q1.a[2] + self.a[1] * q1.a[3] + self.a[2] * q1.a[0] + self.a[3] * q1.a[1],
                    'tx+xt1': self.a[0] * q1.a[3] + self.a[1] * q1.a[2] + self.a[3] * q1.a[0] + self.a[2] * q1.a[1],
                    
                    'ty+yt0': self.a[0] * q1.a[4] + self.a[1] * q1.a[5] + self.a[4] * q1.a[0] + self.a[5] * q1.a[1],
                    'ty+yt1': self.a[0] * q1.a[5] + self.a[1] * q1.a[4] + self.a[5] * q1.a[0] + self.a[4] * q1.a[1],
                    
                    'tz+zt0': self.a[0] * q1.a[6] + self.a[1] * q1.a[7] + self.a[6] * q1.a[0] + self.a[7] * q1.a[1],
                    'tz+zt1': self.a[0] * q1.a[7] + self.a[1] * q1.a[6] + self.a[7] * q1.a[0] + self.a[6] * q1.a[1]
                    }
        
        return products
    
    def _anti_commuting_products(self, q1):
        """Returns a dictionary with the three anti-commuting products."""

        yz0 = self.a[4] * q1.a[6] + self.a[5] * q1.a[7]
        yz1 = self.a[4] * q1.a[7] + self.a[5] * q1.a[6]
        zy0 = self.a[6] * q1.a[4] + self.a[7] * q1.a[5]
        zy1 = self.a[6] * q1.a[5] + self.a[7] * q1.a[4]

        zx0 = self.a[6] * q1.a[2] + self.a[7] * q1.a[3]
        zx1 = self.a[6] * q1.a[3] + self.a[7] * q1.a[2]
        xz0 = self.a[2] * q1.a[6] + self.a[3] * q1.a[7]
        xz1 = self.a[2] * q1.a[7] + self.a[3] * q1.a[6]

        xy0 = self.a[2] * q1.a[4] + self.a[3] * q1.a[5]
        xy1 = self.a[2] * q1.a[5] + self.a[3] * q1.a[4]
        yx0 = self.a[4] * q1.a[2] + self.a[5] * q1.a[3]
        yx1 = self.a[4] * q1.a[3] + self.a[5] * q1.a[2]
                                   
        products = {'yz-zy0': yz0 + zy1,
                    'yz-zy1': yz1 + zy0,
                    
                    'zx-xz0': zx0 + xz1,
                    'zx-xz1': zx1 + xz0,
                    
                    'xy-yx0': xy0 + yx1,
                    'xy-yx1': xy1 + yx0,
                   
                    'zy-yz0': yz1 + zy0,
                    'zy-yz1': yz0 + zy1,
                    
                    'xz-zx0': zx1 + xz0,
                    'xz-zx1': zx0 + xz1,
                    
                    'yx-xy0': xy1 + yx0,
                    'yx-xy1': xy0 + yx1
                   }
        
        return products
    
    def _all_products(self, q1):
        """Returns a dictionary with all possible products."""

        products = self._commuting_products(q1)
        products.update(self._anti_commuting_products(q1))
        
        return products
    
    def square(self, qtype="^2"):
        """Square a quaternion."""
        
        end_qtype = "{}{}".format(self.qtype, qtype)
        
        qxq = self._commuting_products(self)
        
        sq_q = Q8a(qtype=self.qtype)        
        sq_q.a[0] = qxq['tt0'] + (qxq['xx+yy+zz1'])
        sq_q.a[1] = qxq['tt1'] + (qxq['xx+yy+zz0'])
        sq_q.a[2] = qxq['tx+xt0']
        sq_q.a[3] = qxq['tx+xt1']
        sq_q.a[4] = qxq['ty+yt0']
        sq_q.a[5] = qxq['ty+yt1']
        sq_q.a[6] = qxq['tz+zt0']
        sq_q.a[7] = qxq['tz+zt1']
        
        sq_q.qtype = end_qtype
        sq_q.representation = self.representation
        
        return sq_q
    
    def reduce(self, qtype="-reduce"):
        """Put all Doubletas into the reduced form so one of each pair is zero."""

        end_qtype = "{}{}".format(self.qtype, qtype)
        
        red_t = Doubleta([self.a[0], self.a[1]]).d_reduce()
        red_x = Doubleta([self.a[2], self.a[3]]).d_reduce()
        red_y = Doubleta([self.a[4], self.a[5]]).d_reduce()
        red_z = Doubleta([self.a[6], self.a[7]]).d_reduce()
            
        q_red = Q8a(qtype=self.qtype)
        q_red.a[0] = red_t.d[0]
        q_red.a[1] = red_t.d[1]
        q_red.a[2] = red_x.d[0]
        q_red.a[3] = red_x.d[1]
        q_red.a[4] = red_y.d[0]
        q_red.a[5] = red_y.d[1]
        q_red.a[6] = red_z.d[0]
        q_red.a[7] = red_z.d[1]

        q_red.qtype = end_qtype
        q_red.representation = self.representation
        
        return q_red
    
    def norm_squared(self, qtype="|| ||^2"):
        """The norm_squared of a quaternion."""
        
        end_qtype = "||{}||^2".format(self.qtype)
        
        qxq = self._commuting_products(self)
        
        n_q = Q8a()        
        n_q.a[0] = qxq['tt0'] + qxq['xx+yy+zz0']
        n_q.a[1] = qxq['tt1'] + qxq['xx+yy+zz1']
        result = n_q.reduce()
        
        result.qtype = end_qtype
        result.representation = self.representation
        
        return result
    
    def norm_squared_of_vector(self, qtype="V(|| ||)^2"):
        """The norm_squared of the vector of a quaternion."""
        
        end_qtype = "V||({})||^2".format(self.qtype)
        
        qxq = self._commuting_products(self)
        
        nv_q = Q8a()
        nv_q.a[0] = qxq['xx+yy+zz0']
        nv_q.a[1] = qxq['xx+yy+zz1']
        result = nv_q.reduce()
        result.qtype = end_qtype
        result.representation = self.representation

        return result
        
    def abs_of_q(self, qtype="| |"):
        """The absolute value, the square root of the norm_squared."""

        end_qtype = "|{}|".format(self.qtype)
        
        abq = self.norm_squared()
        sqrt_t0 = abq.a[0] ** (1/2)
        abq.a[0] = sqrt_t0
        abq.qtype = end_qtype
        abq.representation = self.representation
        
        return abq

    def abs_of_vector(self, qtype="|V()|)"):
        """The absolute value of the vector, the square root of the norm_squared of the vector."""

        end_qtype = "|V({})|".format(self.qtype, qtype)
        
        av = self.norm_squared_of_vector()
        sqrt_t = av.a[0] ** (1/2)
        av.a[0] = sqrt_t
        av.qtype = end_qtype
        av.representation = self.representation
        
        return av
    
    def normalize(self, n=1, qtype="U"):
        """Normalize a quaternion"""
        
        end_qtype = "{}U".format(self.qtype)
        
        abs_q_inv = self.abs_of_q().invert()
        n_q = self.product(abs_q_inv).product(Q8a([n, 0, 0, 0]))
        n_q.qtype = end_qtype
        n_q.representation=self.representation
        
        return n_q
    
    def add(self, q1, qtype="+"):
        """Form a add given 2 quaternions."""

        self.check_representations(q1)
        
        add_q = Q8a()
        for i in range(0, 8):
            add_q.a[i] = self.a[i] + q1.a[i]
                    
        add_q.qtype = "{f}+{s}".format(f=self.qtype, s=q1.qtype)
        add_q.representation = self.representation    
        
        return add_q    

    def dif(self, q1, qtype="-"):
        """Form a add given 2 quaternions."""

        self.check_representations(q1)
        
        dif_q = Q8a()

        dif_q.a[0] = self.a[0] + q1.a[1]
        dif_q.a[1] = self.a[1] + q1.a[0]
        dif_q.a[2] = self.a[2] + q1.a[3]
        dif_q.a[3] = self.a[3] + q1.a[2]
        dif_q.a[4] = self.a[4] + q1.a[5]
        dif_q.a[5] = self.a[5] + q1.a[4]
        dif_q.a[6] = self.a[6] + q1.a[7]
        dif_q.a[7] = self.a[7] + q1.a[6]
     
        dif_q.qtype = "{f}-{s}".format(f=self.qtype, s=q1.qtype)
        dif_q.representation = self.representation
        
        return dif_q
    
    def product(self, q1, kind="", reverse=False, qtype=""):
        """Form a product given 2 quaternions."""

        self.check_representations(q1)
        
        commuting = self._commuting_products(q1)
        q_even = Q8a()
        q_even.a[0] = commuting['tt0'] + commuting['xx+yy+zz1']
        q_even.a[1] = commuting['tt1'] + commuting['xx+yy+zz0']
        q_even.a[2] = commuting['tx+xt0']
        q_even.a[3] = commuting['tx+xt1']
        q_even.a[4] = commuting['ty+yt0']
        q_even.a[5] = commuting['ty+yt1']
        q_even.a[6] = commuting['tz+zt0']
        q_even.a[7] = commuting['tz+zt1']
        
        anti_commuting = self._anti_commuting_products(q1)
        q_odd = Q8a()
        
        if reverse:
            q_odd.a[2] = anti_commuting['zy-yz0']
            q_odd.a[3] = anti_commuting['zy-yz1']
            q_odd.a[4] = anti_commuting['xz-zx0']
            q_odd.a[5] = anti_commuting['xz-zx1']
            q_odd.a[6] = anti_commuting['yx-xy0']
            q_odd.a[7] = anti_commuting['yx-xy1']
            
        else:
            q_odd.a[2] = anti_commuting['yz-zy0']
            q_odd.a[3] = anti_commuting['yz-zy1']
            q_odd.a[4] = anti_commuting['zx-xz0']
            q_odd.a[5] = anti_commuting['zx-xz1']
            q_odd.a[6] = anti_commuting['xy-yx0']
            q_odd.a[7] = anti_commuting['xy-yx1']
        
        if kind == "":
            result = q_even.add(q_odd)
            times_symbol = "x"
        elif kind.lower() == "even":
            result = q_even
            times_symbol = "xE"
        elif kind.lower() == "odd":
            result = q_odd
            times_symbol = "xO"
        else:
            raise Exception("Three 'kind' values are known: '', 'even', and 'odd'")
            
        if reverse:
            times_symbol = times_symbol.replace('x', 'xR')    
            
        result.qtype = "{f}{ts}{s}".format(f=self.qtype, ts=times_symbol, s=q1.qtype)
        result.representation = self.representation
        
        return result

    def Euclidean_product(self, q1, kind="", reverse=False, qtype=""):
        """Form a product p* q given 2 quaternions, not associative."""

        self.check_representations(q1)
        
        pq = Q8a()
        pq = self.conj().product(q1, kind, reverse, qtype)
        pq.representation = self.representation
        
        return pq

    def invert(self, qtype="^-1"):
        """Invert a quaternion."""
        
        end_qtype = "{}{}".format(self.qtype, qtype)
        
        q_conj = self.conj()
        q_norm_squared = self.norm_squared().reduce()
        
        if q_norm_squared.a[0] == 0:
            return self.q_0()
        
        q_norm_squared_inv = Q8a([1.0 / q_norm_squared.a[0], 0, 0, 0, 0, 0, 0, 0])

        q_inv = q_conj.product(q_norm_squared_inv)
        q_inv.qtype = end_qtype
        q_inv.representation = self.representation
        
        return q_inv

    def divide_by(self, q1, qtype=""):
        """Divide one quaternion by another. The order matters unless one is using a norm_squared (real number)."""

        self.check_representations(q1)
        
        q_inv = q1.invert()
        q_div = self.product(q_inv) 
        q_div.qtype = "{f}/{s}".format(f=self.qtype, s=q1.qtype)
        q_div.representation = self.representation    
        
        return q_div
    
    def triple_product(self, q1, q2):
        """Form a triple product given 3 quaternions."""
        
        self.check_representations(q1)
        self.check_representations(q2)
        
        triple = self.product(q1).product(q2)
        
        return triple
    
    # Quaternion rotation involves a triple product:  UQU∗
    # where the U is a unitary quaternion (having a norm_squared of one).
    def rotate(self, a_1p=0, a_1n=0, a_2p=0, a_2n=0, a_3p=0, a_3n=0):
        """Do a rotation given up to three angles."""
    
        u = Q8a([0, 0, a_1p, a_1n, a_2p, a_2n, a_3p, a_3n])
        u_abs = u.abs_of_q()
        u_norm_squaredalized = u.divide_by(u_abs)
        q_rot = u_norm_squaredalized.triple_product(self, u_norm_squaredalized.conj())
        q_rot.representation = self.representation
        
        return q_rot
    
    # A boost also uses triple products like a rotation, but more of them.
    # This is not a well-known result, but does work.
    def boost(self, beta_x=0, beta_y=0, beta_z=0, qtype="boost"):
        """A boost along the x, y, and/or z axis."""
        
        end_qtype = "{}{}".format(self.qtype, qtype)
        
        boost = Q8a(sr_gamma_betas(beta_x, beta_y, beta_z))
        b_conj = boost.conj()
        
        triple_1 = boost.triple_product(self, b_conj)
        triple_2 = boost.triple_product(boost, self).conj()
        triple_3 = b_conj.triple_product(b_conj, self).conj()
              
        triple_23 = triple_2.dif(triple_3)
        half_23 = triple_23.product(Q8a([0.5, 0, 0, 0, 0, 0, 0, 0]))
        triple_123 = triple_1.add(half_23)
        
        triple_123.qtype = end_qtype
        triple_123.representation = self.representation
        
        return triple_123
    
    # g_shift is a function based on the space-times-time invariance proposal for gravity,
    # which proposes that if one changes the distance from a gravitational source, then
    # squares a measurement, the observers at two different hieghts agree to their
    # space-times-time values, but not the intervals.
    def g_shift(self, dimensionless_g, g_form="exp", qtype="g_shift"):
        """Shift an observation based on a dimensionless GM/c^2 dR."""
        
        end_qtype = "{}{}".format(self.qtype, qtype)
        
        if g_form == "exp":
            g_factor = sp.exp(dimensionless_g)
            if qtype == "g_shift":
                qtype = "g_exp"
        elif g_form == "minimal":
            g_factor = 1 + 2 * dimensionless_g + 2 * dimensionless_g ** 2
            if qtype == "g_shift":
                qtype = "g_minimal"
        else:
            print("g_form not defined, should be 'exp' or 'minimal': {}".format(g_form))
            return self
        exp_g = sp.exp(dimensionless_g)
        
        dt = Doubleta([self.a[0] / exp_g, self.a[1] / exp_g])
        dx = Doubleta([self.a[2] * exp_g, self.a[3] * exp_g])
        dy = Doubleta([self.a[4] * exp_g, self.a[5] * exp_g])
        dz = Doubleta([self.a[6] * exp_g, self.a[7] * exp_g])
        
        g_q = Q8a(qtype=self.qtype)
        g_q.a[0] = dt.d[0]
        g_q.a[1] = dt.d[1]
        g_q.a[2] = dx.d[0]
        g_q.a[3] = dx.d[1]
        g_q.a[4] = dy.d[0]
        g_q.a[5] = dy.d[1]
        g_q.a[6] = dz.d[0]
        g_q.a[7] = dz.d[1]
        
        g_q.qtype = end_qtype
        g_q.representation = self.representation
        
        return g_q
    
    def sin(self, qtype="sin"):
        """Take the sine of a quaternion, (sin(t) cosh(|R|), cos(t) sinh(|R|) R/|R|)"""
        
        end_qtype = "sin({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        red_t = Doubleta([self.a[0], self.a[1]]).d_reduce()

        if red_t.d[0] == 0 and red_t.d[1] != 0:
            if abs_v.a[0] == 0:    
                return Q8a([-1 * math.sin(red_t.d[1]), 0, 0, 0], qtype=end_qtype, representation=self.representation)
        
            sint = math.sin(-1 * red_t.d[1])
            cost = math.cos(-1 *red_t.d[1])    
        else:
            if abs_v.a[0] == 0:    
                return Q8a([math.sin(red_t.d[0]), 0, 0, 0], qtype=end_qtype, representation=self.representation)

            sint = math.sin(red_t.d[0])
            cost = math.cos(red_t.d[0])    
            
        sinhR = math.sinh(abs_v.a[0])
        coshR = math.cosh(abs_v.a[0])
        
        k = cost * sinhR / abs_v.a[0]
            
        q_out_dt = Doubleta(sint * coshR)
        q_out_dx = Doubleta(k * (self.a[2] - self.a[3]))
        q_out_dy = Doubleta(k * (self.a[4] - self.a[5]))
        q_out_dz = Doubleta(k * (self.a[6] - self.a[7]))
        q_out = Q8a([q_out_t, q_out_x, q_out_y, q_out_z], qtype=end_qtype, representation=self.representation)
        
        return q_out
     
    def cos(self, qtype="cos"):
        """Take the cosine of a quaternion, (cos(t) cosh(|R|), sin(t) sinh(|R|) R/|R|)"""

        end_qtype = "cos({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        red_t = Doubleta([self.a[0], self.a[1]]).d_reduce()
       
        if red_t.d[0] == 0 and red_t.d[1] != 0:
            if abs_v.a[0] == 0:    
                return Q8a([math.cos(-1 * red_t.d[1]), 0, 0, 0], qtype=end_qtype)
        
            sint = math.sin(-1 * red_t.d[1])
            cost = math.cos(-1 * red_t.d[1]) 
            
        else:
            if abs_v.a[0] == 0:    
                return Q8a([math.cos(red_t.d[0]), 0, 0, 0], qtype=end_qtype)
        
            sint = math.sin(red_t.d[0])
            cost = math.cos(red_t.d[0])
            
        sinhR = math.sinh(abs_v.a[0])
        coshR = math.cosh(abs_v.a[0])
        
        k = -1 * sint * sinhR / abs_v.a[0]
            
        q_out_dt = Doubleta(cost * coshR)
        q_out_dx = Doubleta(k * (self.a[2] - self.a[3]))
        q_out_dy = Doubleta(k * (self.a[4] - self.a[5]))
        q_out_dz = Doubleta(k * (self.a[6] - self.a[7]))
        q_out = Q8a([q_out_t, q_out_x, q_out_y, q_out_z], qtype=end_qtype, representation=self.representation)

        return q_out
    
    def tan(self, qtype="sin"):
        """Take the tan of a quaternion, sin/cos"""

        end_qtype = "tan({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()        
        red_t = Doubleta([self.a[0], self.a[1]]).d_reduce()
        
        if red_t.d[0] == 0 and red_t.d[1] != 0:
            if abs_v.dt == 0:    
                return Q8a([math.tan(-1 * red_t.d[1]), 0, 0, 0], qtype=end_qtype, representation=self.representation)
        else:
            if abs_v.a[0] == 0:    
                return Q8a([math.tan(red_t.d[0]), 0, 0, 0], qtype=end_qtype, representation=self.representation)
            
        sinq = self.sin()
        cosq = self.cos()
            
        q_out = sinq.divide_by(cosq) 
        q_out.qtype = end_qtype
        q_out.representation = self.representation

        return q_out
    
    def sinh(self, qtype="sinh"):
        """Take the sinh of a quaternion, (sinh(t) cos(|R|), cosh(t) sin(|R|) R/|R|)"""
        
        end_qtype = "sinh({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        red_t = Doubleta([self.a[0], self.a[1]]).d_reduce()
        
        if red_t.d[0] == 0 and red_t.d[1] != 0: 
            if abs_v.a[0] == 0:    
                return Q8a([math.sinh(-1 * red_t.d[1]), 0, 0, 0], qtype=end_qtype, representation=self.representation)
            sinht = math.sinh(-1 * red_t.d[1])
            cosht = math.cosh(-1 * red_t.d[1])
        else: 
            if abs_v.a[0] == 0:    
                return Q8a([math.sinh(red_t.d[0]), 0, 0, 0], qtype=end_qtype, representation=self.representation)
            sinht = math.sinh(red_t.d[0])
            cosht = math.cosh(red_t.d[0])
            
        sinR = math.sin(abs_v.a[0])
        cosR = math.cos(abs_v.a[0])
        
        k = cosht * sinR / abs_v.a[0]
            
        q_out_dt = Doubleta(sinht * cosR)
        q_out_dx = Doubleta(k * (self.a[2] - self.a[3]))
        q_out_dy = Doubleta(k * (self.a[4] - self.a[5]))
        q_out_dz = Doubleta(k * (self.a[6] - self.a[7]))
        q_out = Q8a([q_out_t, q_out_x, q_out_y, q_out_z], qtype=end_qtype, representation=self.representation)

        return q_out
     
    def cosh(self, qtype="sin"):
        """Take the cosh of a quaternion, (cosh(t) cos(|R|), sinh(t) sin(|R|) R/|R|)"""

        end_qtype = "cosh({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        red_t = Doubleta([self.a[0], self.a[1]]).d_reduce()
        
        if red_t.d[0] == 0 and red_t.d[1] != 0:
            if abs_v.a[0] == 0:    
                return Q8a([-1 * math.cosh(self.dt.n), 0, 0, 0], qtype=end_qtype, representation=self.representation)
            
            sinht = math.sinh(-1 * red_t.d[1])
            cosht = math.cosh(-1 * red_t.d[1])
        
        else:
            if abs_v.a[0] == 0:    
                return Q8a([math.cosh(self.dt.p), 0, 0, 0], qtype=end_qtype, representation=self.representation)
            
            sinht = math.sinh(red_t.d[0])
            cosht = math.cosh(red_t.d[0])
             
        sinR = math.sin(abs_v.a[0])
        cosR = math.cos(abs_v.a[0])
        
        k = sinht * sinR / abs_v.a[0]
            
        q_out_dt = Doubleta(cosht * cosR)
        q_out_dx = Doubleta(k * (self.a[2] - self.a[3]))
        q_out_dy = Doubleta(k * (self.a[4] - self.a[5]))
        q_out_dz = Doubleta(k * (self.a[6] - self.a[7]))
        q_out = Q8a([q_out_t, q_out_x, q_out_y, q_out_z], qtype=end_qtype, representation=self.representation)

        return q_out
    
    def tanh(self, qtype="sin"):
        """Take the tanh of a quaternion, sin/cos"""
        
        end_qtype = "tanh({sq})".format(sq=self.qtype)
            
        abs_v = self.abs_of_vector()
        red_t = Doubleta([self.a[0], self.a[1]]).d_reduce()
        
        if abs_v.a[0] == 0:
            if red_t.d[0] == 0 and red_t.d[1] != 0:
                return Q8a([-1 * math.tanh(self.dt.n), 0, 0, 0], qtype=end_qtype, representation=self.representation)
            else:
                return Q8a([math.tanh(self.dt.p), 0, 0, 0], qtype=end_qtype, representation=self.representation)
            
        sinhq = self.sinh()
        coshq = self.cosh()
            
        q_out = sinhq.divide_by(coshq) 
        q_out.qtype = end_qtype
        q_out.representation = self.representation
        
        return q_out
    
    def exp(self, qtype="exp"):
        """Take the exponential of a quaternion."""
        # exp(q) = (exp(t) cos(|R|, exp(t) sin(|R|) R/|R|)
        
        end_qtype = "exp({st})".format(st=self.qtype)
        
        abs_v = self.abs_of_vector()
        red_t = Doubleta([self.a[0], self.a[1]]).d_reduce()
        
        if red_t.d[0] == 0 and red_t.d[1] != 0:
            et = math.exp(-1 * red_t.d[1])
            
            if (abs_v.a[0] == 0):
                return Q8a([et, 0, 0, 0], qtype=end_qtype, representation=self.representation)
            
            cosR = math.cos(abs_v.a[0])
            sinR = math.sin(abs_v.a[0])
    
        else:
            et = math.exp(red_t.d[0])
            
            if (abs_v.a[0] == 0):
                return Q8a([et, 0, 0, 0], qtype=end_qtype, representation=self.representation)
            
            cosR = math.cos(abs_v.a[0])
            sinR = math.sin(abs_v.a[0])
            
        k = et * sinR / abs_v.a[0]
                       
        expq_dt = Doubleta(et * cosR)
        expq_dx = Doubleta(k * (self.a[2] - self.a[3]))
        expq_dy = Doubleta(k * (self.a[4] - self.a[5]))
        expq_dz = Doubleta(k * (self.a[6] - self.a[7]))
        expq = Q8a([expq_dt, expq_dt, expq_dt, expq_dt], qtype=end_qtype, representation=self.representation)
                       
        return expq
    
    def ln(self, qtype="ln"):
        """Take the natural log of a quaternion."""
        # ln(q) = (0.5 ln t^2 + R.R, atan2(|R|, t) R/|R|)
        
        end_qtype = "ln({st})".format(st=self.qtype)
        
        abs_v = self.abs_of_vector()
        red_t = Doubleta([self.a[0], self.a[1]]).d_reduce()
        
        if red_t.d[0] == 0 and red_t.d[1] != 0:
            if (abs_v.a[0] == 0):
                # I don't understant this, but mathematica does the same thing, but it looks wrong to me.
                return(Q8a([math.log(-self.dt.n), math.pi, 0, 0], qtype=end_qtype))   
            
            t_value = 0.5 * math.log(red_t.d[1] * red_t.d[1] + abs_v.a[0] * abs_v.a[0])
            k = math.atan2(abs_v.a[0], red_t.d[1]) / abs_v.a[0]
        
        else:
            if (abs_v.a[0] == 0):
                return(Q8a([math.log(self.dt.p), 0, 0, 0], qtype=end_qtype, representation=self.representation))
                
            t_value = 0.5 * math.log(red_t.d[0] * red_t.d[0] + abs_v.a[0] * abs_v.a[0])
            k = math.atan2(abs_v.a[0], red_t.d[0]) / abs_v.a[0]
            
        lnq_dt = Doubleta(t_value)
        lnq_dx = Doubleta(k * (self.a[2] - self.a[3]))
        lnq_dy = Doubleta(k * (self.a[4] - self.a[5]))
        lnq_dz = Doubleta(k * (self.a[6] - self.a[7]))
        lnq = Q8a([lnq_dt, lnq_dx, lnq_dy, lnq_dz], qtype=end_qtype, representation=self.representation)
                       
        return lnq
    
    def q_2_q(self, q1, qtype="P"):
        """Take the natural log of a quaternion, q^p = exp(ln(q) * p)."""
        
        self.check_representations(q1)
        
        end_qtype = "{st}^P".format(st=self.qtype)
        
        q2q = self.ln().product(q1).reduce().exp()
        q2q.qtype = end_qtype
        q2q.representation = self.representation
        
        return q2q
    
    def trunc(self):
        """Truncates values."""
        
        self.dt = math.trunc(self.dt)
        self.dx = math.trunc(self.dx)
        self.dy = math.trunc(self.dy)
        self.dz = math.trunc(self.dz)
        
        return self


# In[55]:


class TestQ8a(unittest.TestCase):

    """Class to make sure all the functions work as expected."""
    
    q1 = Q8a([1, 0, 0, 2, 0, 3, 0, 4])
    q2 = Q8a([0, 0, 4, 0, 0, 3, 0, 0])
    q_big = Q8a([1, 2, 3, 4, 5, 6, 7, 8])
    verbose = True
    
    def test_qt(self):
        self.assertTrue(self.q1.a[0] == 1)
    
    def test_q_zero(self):
        q_z = self.q1.q_0()
        print("q0: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[5] == 0)
        self.assertTrue(q_z.a[6] == 0)
        
    def test_q_1(self):
        q_z = self.q1.q_1()
        print("q_1: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[6] == 0)
        
    def test_q_i(self):
        q_z = self.q1.q_i()
        print("q_i: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[2] == 1)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[6] == 0)
        
    def test_q_j(self):
        q_z = self.q1.q_j()
        print("q_j: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[4] == 1)
        self.assertTrue(q_z.a[6] == 0)
    
    def test_q_k(self):
        q_z = self.q1.q_k()
        print("q_k: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[6] == 1)
                
    def test_q_random(self):
        q_z = self.q1.q_random()
        print("q_random():", q_z)
        self.assertTrue(q_z.a[0] >= 0 and q_z.a[0] <= 1)
        self.assertTrue(q_z.a[2] >= 0 and q_z.a[2] <= 1)
        self.assertTrue(q_z.a[4] >= 0 and q_z.a[4] <= 1)
        self.assertTrue(q_z.a[6] >= 0 and q_z.a[6] <= 1)
            
    def test_conj_0(self):
        q_z = self.q1.conj()
        print("conj 0: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[2] == 2)
        self.assertTrue(q_z.a[4] == 3)
        self.assertTrue(q_z.a[6] == 4)
        
    def test_equals(self):
        self.assertTrue(self.q1.equals(self.q1))
        self.assertFalse(self.q1.equals(self.q2))
                 
    def test_conj_1(self):
        q_z = self.q1.conj(1)
        print("conj 1: {}".format(q_z))
        self.assertTrue(q_z.a[1] == 1)
        self.assertTrue(q_z.a[3] == 2)
        self.assertTrue(q_z.a[4] == 3)
        self.assertTrue(q_z.a[6] == 4)
                 
    def test_conj_2(self):
        q_z = self.q1.conj(2)
        print("conj 2: {}".format(q_z))
        self.assertTrue(q_z.a[1] == 1)
        self.assertTrue(q_z.a[2] == 2)
        self.assertTrue(q_z.a[5] == 3)
        self.assertTrue(q_z.a[6] == 4)
        
    def test_vahlen_conj_0(self):
        q_z = self.q1.vahlen_conj()
        print("vahlen conj -: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[2] == 2)
        self.assertTrue(q_z.a[4] == 3)
        self.assertTrue(q_z.a[6] == 4)
                 
    def test_vahlen_conj_1(self):
        q_z = self.q1.vahlen_conj("'")
        print("vahlen conj ': {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[2] == 2)
        self.assertTrue(q_z.a[4] == 3)
        self.assertTrue(q_z.a[7] == 4)
                 
    def test_vahlen_conj_2(self):
        q_z = self.q1.vahlen_conj('*')
        print("vahlen conj *: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[3] == 2)
        self.assertTrue(q_z.a[5] == 3)
        self.assertTrue(q_z.a[6] == 4)
        
    def test_square(self):
        q_sq = self.q1.square()
        q_sq_red = q_sq.reduce()
        print("square: {}".format(q_sq))
        print("square reduced: {}".format(q_sq_red))
        self.assertTrue(q_sq.a[0] == 1)
        self.assertTrue(q_sq.a[1] == 29)
        self.assertTrue(q_sq.a[3] == 4)
        self.assertTrue(q_sq.a[5] == 6)
        self.assertTrue(q_sq.a[7] == 8)
        self.assertTrue(q_sq_red.a[0] == 0)
        self.assertTrue(q_sq_red.a[1] == 28)
                
    def test_reduce(self):
        q_red = self.q_big.reduce()
        print("q_big reduced: {}".format(q_red))
        self.assertTrue(q_red.a[0] == 0)
        self.assertTrue(q_red.a[1] == 1)
        self.assertTrue(q_red.a[2] == 0)
        self.assertTrue(q_red.a[3] == 1)
        self.assertTrue(q_red.a[4] == 0)
        self.assertTrue(q_red.a[5] == 1)
        self.assertTrue(q_red.a[6] == 0)
        self.assertTrue(q_red.a[7] == 1)
        
    def test_norm_squared(self):
        q_z = self.q1.norm_squared()
        print("norm_squared: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 30)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 0)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[5] == 0)
        self.assertTrue(q_z.a[6] == 0)
        self.assertTrue(q_z.a[7] == 0)
        
    def test_norm_squared_of_vector(self):
        q_z = self.q1.norm_squared_of_vector()
        print("norm_squared_of_vector: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 29)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 0)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[5] == 0)
        self.assertTrue(q_z.a[6] == 0)
        self.assertTrue(q_z.a[7] == 0)
        
    def test_abs_of_q(self):
        q_z = self.q2.abs_of_q()
        print("abs_of_q: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 5)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[6] == 0)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[3] == 0)
        self.assertTrue(q_z.a[5] == 0)
        self.assertTrue(q_z.a[7] == 0)
        
    def test_abs_of_vector(self):
        q_z = self.q2.abs_of_vector()
        print("abs_of_vector: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 5)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[6] == 0)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[3] == 0)
        self.assertTrue(q_z.a[5] == 0)
        self.assertTrue(q_z.a[7] == 0)
        
    def test_normalize(self):
        q_z = self.q2.normalize()
        print("q_normalized: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0.8)
        self.assertTrue(q_z.a[3] == 0)
        self.assertTrue(q_z.a[4] == 0)
        self.assertAlmostEqual(q_z.a[5], 0.6)
        self.assertTrue(q_z.a[6] == 0)
        self.assertTrue(q_z.a[7] == 0)
        
    def test_add(self):
        q_z = self.q1.add(self.q2)
        print("add: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 4)
        self.assertTrue(q_z.a[3] == 2)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[5] == 6)
        self.assertTrue(q_z.a[6] == 0)
        self.assertTrue(q_z.a[7] == 4)
        
    def test_add_reduce(self):
        q_z_red = self.q1.add(self.q2).reduce()
        print("add reduce: {}".format(q_z_red))
        self.assertTrue(q_z_red.a[0] == 1)
        self.assertTrue(q_z_red.a[1] == 0)
        self.assertTrue(q_z_red.a[2] == 2)
        self.assertTrue(q_z_red.a[3] == 0)
        self.assertTrue(q_z_red.a[4] == 0)
        self.assertTrue(q_z_red.a[5] == 6)
        self.assertTrue(q_z_red.a[6] == 0)
        self.assertTrue(q_z_red.a[7] == 4)
        
    def test_dif(self):
        q_z = self.q1.dif(self.q2)
        print("dif: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 6) 
        self.assertTrue(q_z.a[4] == 3)
        self.assertTrue(q_z.a[5] == 3)
        self.assertTrue(q_z.a[6] == 0)
        self.assertTrue(q_z.a[7] == 4) 

    def test_product(self):
        q_z = self.q1.product(self.q2).reduce()
        print("product: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == 1)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 8)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[5] == 19)
        self.assertTrue(q_z.a[6] == 18)
        self.assertTrue(q_z.a[7] == 0)
        
    def test_product_even(self):
        q_z = self.q1.product(self.q2, kind="even").reduce()
        print("product, kind even: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == 1)
        self.assertTrue(q_z.a[2] == 4)
        self.assertTrue(q_z.a[3] == 0)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[5] == 3)
        self.assertTrue(q_z.a[6] == 0)
        self.assertTrue(q_z.a[7] == 0)
        
    def test_product_odd(self):
        q_z = self.q1.product(self.q2, kind="odd").reduce()
        print("product, kind odd: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 12)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[5] == 16)
        self.assertTrue(q_z.a[6] == 18)
        self.assertTrue(q_z.a[7] == 0)
        
    def test_product_reverse(self):
        q1q2_rev = self.q1.product(self.q2, reverse=True)
        q2q1 = self.q2.product(self.q1)
        self.assertTrue(q1q2_rev.equals(q2q1))
        
    def test_Euclidean_product(self):
        q_z = self.q1.Euclidean_product(self.q2).reduce()
        print("Euclidean product: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 16)
        self.assertTrue(q_z.a[3] == 0)
        self.assertTrue(q_z.a[4] == 13)
        self.assertTrue(q_z.a[5] == 0)
        self.assertTrue(q_z.a[6] == 0)
        self.assertTrue(q_z.a[7] == 18)
    
        
    def test_invert(self):
        q_z = self.q2.invert().reduce()
        print("inverse: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 0.16)
        self.assertTrue(q_z.a[4] == 0.12)
        self.assertTrue(q_z.a[5] == 0)
        self.assertTrue(q_z.a[6] == 0)
        self.assertTrue(q_z.a[7] == 0)

    def test_divide_by(self):
        q_z = self.q1.divide_by(self.q1).reduce()
        print("inverse: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 0)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[5] == 0)
        self.assertTrue(q_z.a[6] == 0)
        self.assertTrue(q_z.a[7] == 0) 
        
    def test_triple_product(self):
        q_z = self.q1.triple_product(self.q2, self.q1).reduce()
        print("triple: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 0)
        self.assertTrue(q_z.a[1] == 2)
        self.assertTrue(q_z.a[2] == 124)
        self.assertTrue(q_z.a[3] == 0)
        self.assertTrue(q_z.a[4] == 0)
        self.assertTrue(q_z.a[5] == 84)
        self.assertTrue(q_z.a[6] == 8)
        self.assertTrue(q_z.a[7] == 0)
        
    def test_rotate(self):
        q_z = self.q1.rotate(1).reduce()
        print("rotate: {}".format(q_z))
        self.assertTrue(q_z.a[0] == 1)
        self.assertTrue(q_z.a[1] == 0)
        self.assertTrue(q_z.a[2] == 0)
        self.assertTrue(q_z.a[3] == 2)
        self.assertTrue(q_z.a[4] == 3)
        self.assertTrue(q_z.a[5] == 0)
        self.assertTrue(q_z.a[6] == 4)
        self.assertTrue(q_z.a[7] == 0)
        
    def test_boost(self):
        q1_sq = self.q1.square().reduce()
        q_z = self.q1.boost(0.003)
        q_z2 = q_z.square().reduce()
        print("q1_sq: {}".format(q1_sq))
        print("boosted: {}".format(q_z))
        print("b squared: {}".format(q_z2))
        self.assertTrue(round(q_z2.a[1], 12) == round(q1_sq.a[1], 12))
        
    def test_g_shift(self):
        q1_sq = self.q1.square().reduce()
        q_z = self.q1.g_shift(0.003)
        q_z2 = q_z.square().reduce()
        print("q1_sq: {}".format(q1_sq))
        print("g_shift: {}".format(q_z))
        print("g squared: {}".format(q_z2))
        self.assertTrue(q_z2.a[1] != q1_sq.a[1])
        self.assertTrue(q_z2.a[2] == q1_sq.a[2])
        self.assertTrue(q_z2.a[3] == q1_sq.a[3])
        self.assertTrue(q_z2.a[4] == q1_sq.a[4])
        self.assertTrue(q_z2.a[5] == q1_sq.a[5])
        self.assertTrue(q_z2.a[6] == q1_sq.a[6])
        self.assertTrue(q_z2.a[7] == q1_sq.a[7])
        
suite = unittest.TestLoader().loadTestsFromModule(TestQ8a())
unittest.TextTestRunner().run(suite);


# In[56]:


class TestQ8aRep(unittest.TestCase):
    Q12 = Q8a([1.0, 2.0, 0, 0])
    Q1123 = Q8a([1.0, 1.0, 2.0, 3.0])
    Q11p = Q8a([1.0, 1.0, 0, 0], representation="polar")
    Q12p = Q8a([1.0, 2.0, 0, 0], representation="polar")
    Q12np = Q8a([1.0, -2.0, 0, 0], representation="polar")
    Q21p = Q8a([2.0, 1.0, 0, 0], representation="polar")
    Q23p = Q8a([2.0, 3.0, 0, 0], representation="polar")
    Q13p = Q8a([1.0, 3.0, 0, 0], representation="polar")
    Q5p = Q8a([5.0, 0, 0, 0], representation="polar")
    
    # @unittest.skip("problems implementing")
    def test_txyz_2_representation(self):
        qr = Q8a(self.Q12.txyz_2_representation(""))
        self.assertTrue(qr.equals(self.Q12))
        qr = Q8a(self.Q12.txyz_2_representation("polar"))
        self.assertTrue(qr.equals(Q8a([2.23606797749979, 1.10714871779409, 0, 0])))
        qr = Q8a(self.Q1123.txyz_2_representation("spherical"))
        self.assertTrue(qr.equals(Q8a([1.0, 3.7416573867739413, 0.640522312679424, 1.10714871779409])))

        
    # @unittest.skip("problems implementing")    
    def test_representation_2_txyz(self):
        qr = Q8a(self.Q12.representation_2_txyz(""))
        self.assertTrue(qr.equals(self.Q12))
        qr = Q8a(self.Q12.representation_2_txyz("polar"))
        self.assertTrue(qr.equals(Q8a([-0.4161468365471424, 0.9092974268256817, 0, 0])))
        qr = Q8a(self.Q1123.representation_2_txyz("spherical"))
        self.assertTrue(qr.equals(Q8a([1.0, -0.9001976297355174, 0.12832006020245673, -0.4161468365471424])))
    
    def test_polar_products(self):
        qr = self.Q11p.product(self.Q12p).reduce()
        print("polar 1 1 0 0 * 1 2 0 0: ", qr)
        self.assertTrue(qr.equals(self.Q13p))
        qr = self.Q12p.product(self.Q21p).reduce()
        print("polar 1 2 0 0 * 2 1 0 0: ", qr)
        self.assertTrue(qr.equals(self.Q23p))

    def test_polar_conj(self):
        qr = self.Q12p.conj().reduce()
        print("polar conj of 1 2 0 0: ", qr)
        self.assertTrue(qr.equals(self.Q12np))
        
suite = unittest.TestLoader().loadTestsFromModule(TestQ8aRep())
unittest.TextTestRunner().run(suite);


# ## Equivalence Classes

# Create a class that can figure out if two quaternions are in the same equivalence class. An equivalence class of space-time is a subset of events in space-time. For example, the future equivalence class would have any event that happens in the future. All time-like events have an interval that is positive.
# 
# A series of images were created to show each class.  For the future, here is the equivalence class:
# ![](images/eq_classes/time_future.png)
# There is a smaller class for those that are exactly the same amount in the future. They have a different icon:
# ![](images/eq_classes/time_future_exact.png)
# Such an exact relation is not of much interest to physicists since Einstein showed that holds for only one set of observers. If one is moving relative to the reference observer, the two events would look like they occured at different times in the future, presuming perfectly accurate measuring devices.
# 

# In[57]:


def round_sig_figs(num, sig_figs):
    """Round to specified number of sigfigs.

    # from http://code.activestate.com/recipes/578114-round-number-to-specified-number-of-significant-di/
    """
    if num != 0:
        return round(num, -int(math.floor(math.log10(abs(num))) - (sig_figs - 1)))
    else:
        return 0  # Can't take the log of 0


# In[58]:


class EQ(object):
    """A class that compairs pairs of quaternions."""
    
    # Read images in once for the class.
    eq_images = {}
    qtd_dir =  os.path.dirname(IPython.utils.path.filefind('Q_tools.ipynb'))
    im_dir = "{qd}/images/eq_classes".format(qd=qtd_dir)
    im_files = "{imd}/*png".format(imd=im_dir)

    for eq_image_file in glob(im_files):
        file_name = basename(eq_image_file)
        eq_class_name = (file_name.split(sep='.'))[0]
        eq_images[eq_class_name] = mpimg.imread(eq_image_file)
        
    def __init__(self, q1, q2, sig_figs=10):
        
        # Convert the quaternions into the Q8 reduced form.
        if isinstance(q1, QH):
            self.q1 = Q8([q1.t, q1.x, q1.y, q1.z])
        
        elif(isinstance(q1, Q8)):
            self.q1 = q1.reduce()
            
        if isinstance(q2, QH):
            self.q2 = Q8([q2.t, q2.x, q2.y, q2.z])
            
        elif(isinstance(q2, Q8)):
            self.q2 = q2.reduce()
                
        # The quaternions used by this class are
        # linear, square, and the norm_squared of a quaternion so do the calculations once.
        
        self.q1_square = self.q1.square().reduce()
        self.q2_square = self.q2.square().reduce()
        
        self.q1_norm_squared_minus_1 = self.q1.norm_squared().dif(self.q1.q_1()).reduce()
        self.q2_norm_squared_minus_1 = self.q2.norm_squared().dif(self.q1.q_1()).reduce()

        # Store results here
        self.classes = {}
        self.sig_figs = sig_figs

    def get_class(self, q1, q2, names, position):
        """A general tool to figure out a scalar class. 
           Names is a dictionary that needs values for 'class', 'positive', 'negative', and 'divider'.
           position needs to be dt, dx, dy or dz"""
        
        q1_d = {'dt': q1.dt, 'dy': q1.dy, 'dx': q1.dx, 'dz': q1.dz}
        q2_d = {'dt': q2.dt, 'dy': q2.dy, 'dx': q2.dx, 'dz': q2.dz}

        
        # Since the quaternions in the Q8 form are reduced just look for non-zero values.
        if q1_d[position].p and q2_d[position].p:

            if round_sig_figs(q1_d[position].p, self.sig_figs) == round_sig_figs(q2_d[position].p, self.sig_figs):
                result = "{np}_exact".format(np=names["positive"])
            else:
                result = "{np}".format(np=names["positive"])
                
        elif q1_d[position].n and q2_d[position].n:
            
            if round_sig_figs(q1_d[position].n, self.sig_figs) == round_sig_figs(q2_d[position].n, self.sig_figs):
                result = "{nn}_exact".format(nn=names["negative"])
            else:
                result = "{nn}".format(nn=names["negative"])
                
        elif not q1_d[position].p and not q1_d[position].n and not q2_d[position].p and not q2_d[position].n:
            result = "{nd}_exact".format(nd=names["divider"])
            
        else:
            result = "disjoint"
            
        self.classes[names["class"]] = result
        return result
    
    def time(self):
        """Figure out time equivalence class."""
        
        names = {'class': 'time', 'positive': 'future', 'negative': 'past', 'divider': 'now'}        
        result = self.get_class(self.q1, self.q2, names, 'dt')
        return result
    
    def space(self):
        """Figure out time equivalence class."""
        
        positions = ['dx', 'dy', 'dz']
        
        names = []
        names.append({'class': 'space-1', 'positive': 'right', 'negative': 'left', 'divider': 'here'})
        names.append({'class': 'space-2', 'positive': 'up', 'negative': 'down', 'divider': 'here'})
        names.append({'class': 'space-3', 'positive': 'near', 'negative': 'far', 'divider': 'here'})
        
        results = []
                     
        for name, position in zip(names, positions):
            results.append(self.get_class(self.q1, self.q2, name, position))
        
        return results
    
    def space_time(self):
        """Do both time and space, return an array."""
        
        results = []
        results.append(self.time())
        results.extend(self.space())
        return results
    
    def causality(self):
        """There is only one causality equivalence class."""
        
        names = {'class': 'causality', 'positive': 'time-like', 'negative': 'space-like', 'divider': 'light-like'}
        result = self.get_class(self.q1_square, self.q2_square, names, 'dt')
        return result
    
    def space_times_time(self):
        """Figure out the space-times-time equivalence class used in the quaternion gravity proposal."""
        
        positions = ['dx', 'dy', 'dz']
        
        names = []
        names.append({'class': 'space-times-time-1', 'positive': 'future-right', 
                      'negative': 'future-left', 'divider': 'here-now'})
        names.append({'class': 'space-times-time-2', 'positive': 'future-up', 
                      'negative': 'future-down', 'divider': 'here-now'})
        names.append({'class': 'space-times-time-3', 'positive': 'future-near', 
                      'negative': 'future-far', 'divider': 'here-now'})
        
        results = []
                     
        for name, position in zip(names, positions):
            results.append(self.get_class(self.q1, self.q2, name, position))
        
        return results

    def squared(self):
        """Return both causality and space_times_time as a list."""

        results = []
        results.append(self.causality())
        results.extend(self.space_times_time())
        return results

    def norm_squared_of_unity(self):
        """Find out if the norm_squared of both is greater than, less than, exactly equal or both different from unity."""

        names = {'class': 'norm_squared_of_unity', 'positive': 'greater_than_unity', 'negative': 'less_than_unity', 'divider': 'unity'}
        result = self.get_class(self.q1_norm_squared_minus_1, self.q2_norm_squared_minus_1, names, 'dt')
        return result

    def compare(self, eq_2):
        """Compares one set of equivalence classes to anther."""

        pass

    def get_all_classes(self, eq_2=None):
        """Run them all."""
        
        if eq_2 is None:
            eq_classes = [self]
        else:
            eq_classes = [self, eq_2]
            
        for eq_class in eq_classes:
            if 'time' not in eq_class.classes:
                eq_class.time()
            if 'space' not in eq_class.classes:
                eq_class.space()
            if 'causality' not in eq_class.classes:
                eq_class.causality()
            if 'space-times-time' not in eq_class.classes:
                eq_class.space_times_time()
            if 'norm_squared_of_unity' not in eq_class.classes:
                eq_class.norm_squared_of_unity()

    def visualize(self, eq_2=None):
        """Visualize one or two rows of classes with icons for each of the 5 classes."""
        
        self.get_all_classes(eq_2)
        
        if eq_2 is None:
            fig = plt.figure()
            plt.rcParams["figure.figsize"] = [50, 30]
            
            ax1 = fig.add_subplot(3, 5, 1)
            ax1.imshow(self.eq_images['time_' + self.classes['time']])
            plt.axis('off')
            
            ax21 = fig.add_subplot(3, 5, 2)
            ax21.imshow(self.eq_images['space-1_' + self.classes['space-1']])
            plt.axis('off');

            ax22 = fig.add_subplot(3, 5, 7)
            ax22.imshow(self.eq_images['space-2_' + self.classes['space-2']])
            plt.axis('off');

            ax23 = fig.add_subplot(3, 5, 12)
            ax23.imshow(self.eq_images['space-3_' + self.classes['space-3']])
            plt.axis('off');

            ax3 = fig.add_subplot(3, 5, 3)
            ax3.imshow(self.eq_images['causality_' + self.classes['causality']])
            plt.axis('off');

            ax41 = fig.add_subplot(3, 5, 4)
            ax41.imshow(self.eq_images['space-times-time-1_' + self.classes['space-times-time-1']])
            plt.axis('off');

            ax42 = fig.add_subplot(3, 5, 9)
            ax42.imshow(self.eq_images['space-times-time-2_' + self.classes['space-times-time-2']])
            plt.axis('off');

            ax43 = fig.add_subplot(3, 5, 14)
            ax43.imshow(self.eq_images['space-times-time-3_' + self.classes['space-times-time-3']])
            plt.axis('off');

            ax5 = fig.add_subplot(3, 5, 5)
            ax5.imshow(self.eq_images['norm_squared_of_unity_' + self.classes['norm_squared_of_unity']])
            plt.axis('off');

        else:
            fig = plt.figure()
            plt.rcParams["figure.figsize"] = [50, 60]
            
            ax1 = fig.add_subplot(6, 5, 1)
            ax1.imshow(self.eq_images['time_' + self.classes['time']])
            plt.axis('off')
            
            ax21 = fig.add_subplot(6, 5, 2)
            ax21.imshow(self.eq_images['space-1_' + self.classes['space-1']])
            plt.axis('off');

            ax22 = fig.add_subplot(6, 5, 7)
            ax22.imshow(self.eq_images['space-2_' + self.classes['space-2']])
            plt.axis('off');

            ax23 = fig.add_subplot(6, 5, 12)
            ax23.imshow(self.eq_images['space-3_' + self.classes['space-3']])
            plt.axis('off');

            ax3 = fig.add_subplot(6, 5, 3)
            ax3.imshow(self.eq_images['causality_' + self.classes['causality']])
            plt.axis('off');

            ax41 = fig.add_subplot(6, 5, 4)
            ax41.imshow(self.eq_images['space-times-time-1_' + self.classes['space-times-time-1']])
            plt.axis('off');

            ax42 = fig.add_subplot(6, 5, 9)
            ax42.imshow(self.eq_images['space-times-time-2_' + self.classes['space-times-time-2']])
            plt.axis('off');

            ax43 = fig.add_subplot(6, 5, 14)
            ax43.imshow(self.eq_images['space-times-time-3_' + self.classes['space-times-time-3']])
            plt.axis('off');

            ax5 = fig.add_subplot(6, 5, 5)
            ax5.imshow(self.eq_images['norm_squared_of_unity_' + self.classes['norm_squared_of_unity']])
            plt.axis('off');
            

            ax21 = fig.add_subplot(6, 5, 16)
            ax21.imshow(self.eq_images['time_' + eq_2.classes['time']])
            plt.axis('off')
            
            ax221 = fig.add_subplot(6, 5, 17)
            ax221.imshow(self.eq_images['space-1_' + eq_2.classes['space-1']])
            plt.axis('off');

            ax222 = fig.add_subplot(6, 5, 22)
            ax222.imshow(self.eq_images['space-2_' + eq_2.classes['space-2']])
            plt.axis('off');

            ax223 = fig.add_subplot(6, 5, 27)
            ax223.imshow(self.eq_images['space-3_' + eq_2.classes['space-3']])
            plt.axis('off');

            ax23 = fig.add_subplot(6, 5, 18)
            ax23.imshow(self.eq_images['causality_' + eq_2.classes['causality']])
            plt.axis('off');

            ax241 = fig.add_subplot(6, 5, 19)
            ax241.imshow(self.eq_images['space-times-time-1_' + eq_2.classes['space-times-time-1']])
            plt.axis('off');

            ax242 = fig.add_subplot(6, 5, 24)
            ax242.imshow(self.eq_images['space-times-time-2_' + eq_2.classes['space-times-time-2']])
            plt.axis('off');

            ax243 = fig.add_subplot(6, 5, 29)
            ax243.imshow(self.eq_images['space-times-time-3_' + eq_2.classes['space-times-time-3']])
            plt.axis('off');

            ax25 = fig.add_subplot(6, 5, 20)
            ax25.imshow(self.eq_images['norm_squared_of_unity_' + eq_2.classes['norm_squared_of_unity']])
            plt.axis('off');
            
    def __str__(self):
        """Prints all the equivalence relations."""
        
        self.get_all_classes()
        
        class_names = ["time", "space-1", "space-2", "space-3", "causality", 
                       "space-times-time-1", "space-times-time-2", "space-times-time-3", 
                       "norm_squared_of_unity"]
        
        result = "The equivalence classes for this pair of events are as follows...\n"
        
        result += "q1: {}\n".format(QH(self.q1.q4()))
        result += "q2: {}\n".format(QH(self.q2.q4()))
        result += "q1_squared: {}\n".format(QH(self.q1_square.q4()))
        result += "q2_squared: {}\n".format(QH(self.q2_square.q4()))
        result += "q1_norm_squared -1: {}\n".format(QH(self.q1_norm_squared_minus_1.q4()))
        result += "q2_norm_squared -1: {}\n".format(QH(self.q2_norm_squared_minus_1.q4()))
        
        for class_name in class_names:
            result += "{cn:>20}: {c}\n".format(cn=class_name, c=self.classes[class_name])

        return result
    


# In[59]:


class TestEQ(unittest.TestCase):
    """Class to make sure all the functions work as expected."""
    
    q1 = Q8([1.0, 0, 0, 2.0, 0, 3.0, 0, 4.0])
    q2 = QH([0, 4.0, -3.0, 0])
    eq_11 = EQ(q1, q1)
    eq_12 = EQ(q1, q2)
    
    def test_EQ_assignment(self):
        
        self.assertTrue(self.eq_12.q1.dt.p == 1)
        self.assertTrue(self.eq_12.q1.dt.n == 0)
        self.assertTrue(self.eq_12.q1_square.dt.p == 0)
        self.assertTrue(self.eq_12.q1_square.dt.n == 28)
        self.assertTrue(self.eq_12.q1_norm_squared_minus_1.dt.p == 29)
        self.assertTrue(self.eq_12.q1_norm_squared_minus_1.dt.n == 0)
        self.assertTrue(self.eq_12.q2.dt.p == 0)
        self.assertTrue(self.eq_12.q2.dt.n == 0)
        
    def test_get_class(self):
        """Test all time equivalence classes."""
        names = {'class': 'time', 'positive': 'future', 'negative': 'past', 'divider': 'now'}
        result = self.eq_12.get_class(self.q1, self.q1, names, 'dt')
        self.assertTrue(result == 'future_exact')
    
    def test_time(self):
        """Test all time equivalence classes."""
        q_now = Q8()
        eq_zero = EQ(q_now, q_now)
        self.assertTrue(eq_zero.time() == 'now_exact')
        self.assertTrue(self.eq_12.time() == 'disjoint')
        q1f = QH([4.0, 4.0, 4.0, 4.0])
        q1fe = QH([1.0, 4.0, 4.0, 4.0])
        self.assertTrue(EQ(self.q1, q1f).time() == 'future')
        self.assertTrue(EQ(self.q1, q1fe).time() == 'future_exact')
        q1p = QH([-4.0, 4.0, 4.0, 4.0])
        q1pe = QH([-4.0, 1.0, 2.0, 3.0])
        q1pp = QH([-1.0, 1.0, 2.0, 3.0])
        self.assertTrue(EQ(q1p, q1pp).time() == 'past')
        self.assertTrue(EQ(q1p, q1pe).time() == 'past_exact')
        
    def test_space(self):
        """Test space equivalence class."""
        q_now = Q8()
        eq_zero = EQ(q_now, q_now)
        self.assertTrue(eq_zero.space()[0] == 'here_exact')
        self.assertTrue(eq_zero.space()[1] == 'here_exact')
        self.assertTrue(eq_zero.space()[2] == 'here_exact')
        self.assertTrue(self.eq_11.space()[0] == 'left_exact')
        self.assertTrue(self.eq_11.space()[1] == 'down_exact')
        self.assertTrue(self.eq_11.space()[2] == 'far_exact')
        self.assertTrue(self.eq_12.space()[0] == 'disjoint')
        self.assertTrue(self.eq_12.space()[1] == 'down_exact')
        self.assertTrue(self.eq_12.space()[2] == 'disjoint')
        
        q_sp = Q8([1, 0, 0, 4, 0, 6, 0, 8])
        eq_sp = EQ(self.q1, q_sp)
        self.assertTrue(eq_sp.space()[0] == 'left')
        self.assertTrue(eq_sp.space()[1] == 'down')
        self.assertTrue(eq_sp.space()[2] == 'far')
        
    def test_causality(self):
        """Test all time equivalence classes."""
        q_now = Q8()
        eq_zero = EQ(q_now, q_now)
        self.assertTrue(eq_zero.causality() == 'light-like_exact')
        self.assertTrue(self.eq_12.causality() == 'space-like')
        self.assertTrue(self.eq_11.causality() == 'space-like_exact')
        tl = Q8([4, 0, 0, 0, 0, 0, 0, 0])
        t2 = Q8([5, 0, 0, 3, 0, 0, 0, 0])
        t3 = Q8([5, 0, 3, 0, 1, 0, 0, 0])
        eq_t1_t2 = EQ(tl, t2)
        eq_t1_t3 = EQ(tl, t3)
        self.assertTrue(eq_t1_t2.causality() == 'time-like_exact')
        self.assertTrue(eq_t1_t3.causality() == 'time-like')

    def test_space_times_time(self):
        """Test space equivalence class."""
        q_now = Q8()
        eq_zero = EQ(q_now, q_now)
        self.assertTrue(eq_zero.space_times_time()[0] == 'here-now_exact')
        self.assertTrue(eq_zero.space_times_time()[1] == 'here-now_exact')
        self.assertTrue(eq_zero.space_times_time()[2] == 'here-now_exact')
        self.assertTrue(self.eq_11.space_times_time()[0] == 'future-left_exact')
        self.assertTrue(self.eq_11.space_times_time()[1] == 'future-down_exact')
        self.assertTrue(self.eq_11.space_times_time()[2] == 'future-far_exact')
        self.assertTrue(self.eq_12.space_times_time()[0] == 'disjoint')
        self.assertTrue(self.eq_12.space_times_time()[1] == 'future-down_exact')
        self.assertTrue(self.eq_12.space_times_time()[2] == 'disjoint')

    def test_norm_squared_of_unity(self):
        self.assertTrue(self.eq_11.norm_squared_of_unity() == 'greater_than_unity_exact')
        q_1 = Q8([1, 0, 0, 0, 0, 0, 0, 0])
        q_small = Q8([0.1, 0, 0, 0.2, 0, 0, 0, 0])
        q_tiny = Q8([0.001, 0, 0, 0.002, 0, 0, 0, 0])

        eq_1 = EQ(q_1, q_1)
        eq_q1_small = EQ(q_1, q_small)
        eq_small_small = EQ(q_small, q_small)
        eq_small_tiny = EQ(q_small, q_tiny)
        
        self.assertTrue(eq_1.norm_squared_of_unity() == 'unity_exact')
        self.assertTrue(eq_q1_small.norm_squared_of_unity() == 'disjoint')
        self.assertTrue(eq_small_small.norm_squared_of_unity() == 'less_than_unity_exact')
        self.assertTrue(eq_small_tiny.norm_squared_of_unity() == 'less_than_unity')


# In[60]:


suite = unittest.TestLoader().loadTestsFromModule(TestEQ())
unittest.TextTestRunner().run(suite);


# ## Arrays of Quaternions

# Create a class that can make many, many quaternions.

# In[61]:


class QHArray(QH):
    """A class that can generate many quaternions."""
    
    def __init__(self, q_min=QH([0, 0, 0, 0]), q_max=QH([0, 0, 0, 0]), n_steps=100):
        """Store min, max, and number of step data."""
        self.q_min = q_min
        self.q_max = q_max
        self.n_steps = n_steps
    
    def range(self, q_start, q_delta, n_steps, function=QH.add):
        """Can generate n quaternions"""
        
        functions = {}
        functions["add"] = QH.add
        functions["dif"] = QH.dif
        functions["product"] = QH.product
        
        # To do: figure out the operator used in qtype
        
        q_0 = q_start
        q_0_qtype = q_0.qtype
        self.set_min_max(q_0, first=True)
        yield q_0
        
        for n in range(1, n_steps + 1):
            q_1 = function(q_0, q_delta)
            q_1.qtype = "{q0q}+{n}dQ".format(q0q=q_0_qtype, n=n)
            q_0 = q_1.dupe()
            self.set_min_max(q_1, first=False)
            yield q_1
            
    def set_min_max(self, q1, first=False):
        """Sets the minimum and maximum of a set of quaternions as needed."""
        
        if first:
            self.q_min = q1.dupe()
            self.q_max = q1.dupe()
            
        else:
            if q1.t < self.q_min.t:
                self.q_min.t = q1.t
            elif q1.t > self.q_max.t:
                self.q_max.t = q1.t
                
            if q1.x < self.q_min.x:
                self.q_min.x = q1.x
            elif q1.x > self.q_max.x:
                self.q_max.x = q1.x

            if q1.y < self.q_min.y:
                self.q_min.y = q1.y
            elif q1.y > self.q_max.y:
                self.q_max.y = q1.y
            
            if q1.z < self.q_min.z:
                self.q_min.z = q1.z
            elif q1.z > self.q_max.z:
                self.q_max.z = q1.z
            
    def symbol_sub(self, TXYZ_expression, q1):
        """Given a Symbol expression in terms of T X, Y, and Z, plugs in values for q1.t, q1.x, q1.y, and q1.z"""
        
        new_t = TXYZ_expression.t.subs(T, q1.t).subs(X, q1.x).subs(Y, q1.y).subs(Z, q1.z)
        new_x = TXYZ_expression.x.subs(T, q1.t).subs(X, q1.x).subs(Y, q1.y).subs(Z, q1.z)
        new_y = TXYZ_expression.y.subs(T, q1.t).subs(X, q1.x).subs(Y, q1.y).subs(Z, q1.z)
        new_z = TXYZ_expression.z.subs(T, q1.t).subs(X, q1.x).subs(Y, q1.y).subs(Z, q1.z)
        
        return QH([new_t, new_x, new_y, new_z])


# In[62]:


class TestQHArray(unittest.TestCase):
    """Test array making software."""
    
    t1=QH([1,2,3,4])
    qd=QH([10, .2, .3, 1])
    qha = QHArray()
    
    def test_range(self):
        q_list = list(self.qha.range(self.t1, self.qd, 10))
        self.assertTrue(len(q_list) == 11)
        self.assertTrue(q_list[10].qtype == "Q+10dQ")
        self.assertTrue(q_list[10].z == 14)
    
    def test_min_max(self):
        q_list = list(self.qha.range(self.t1, self.qd, 10))
        self.assertTrue(self.qha.q_min.t < 1.01)
        self.assertTrue(self.qha.q_max.t > 100)
        self.assertTrue(self.qha.q_min.x < 2.01)
        self.assertTrue(self.qha.q_max.x > 2.9)
        self.assertTrue(self.qha.q_min.y < 4.01)
        self.assertTrue(self.qha.q_max.y > 5.8)
        self.assertTrue(self.qha.q_min.z < 6.01)
        self.assertTrue(self.qha.q_max.z > 13.9)


# In[63]:


suite = unittest.TestLoader().loadTestsFromModule(TestQHArray())
unittest.TextTestRunner().run(suite);


# ## Array of nparrays

# In[64]:


class QHaArray(QHa):
    """A class that can generate many quaternions."""
    
    def __init__(self, q_min=QHa([0, 0, 0, 0]), q_max=QHa([0, 0, 0, 0]), n_steps=100):
        """Store min, max, and number of step data."""
        self.q_min = q_min
        self.q_max = q_max
        self.n_steps = n_steps
    
    def range(self, q_start, q_delta, n_steps, function=QHa.add):
        """Can generate n quaternions"""
        
        functions = {}
        functions["add"] = QHa.add
        functions["dif"] = QHa.dif
        functions["product"] = QHa.product
        
        # To do: figure out the operator used in qtype
        
        q_0 = q_start
        q_0_qtype = q_0.qtype
        self.set_min_max(q_0, first=True)
        yield q_0
        
        for n in range(1, n_steps + 1):
            q_1 = function(q_0, q_delta)
            q_1.qtype = "{q0q}+{n}dQ".format(q0q=q_0_qtype, n=n)
            q_0 = q_1.dupe()
            self.set_min_max(q_1, first=False)
            yield q_1
            
    def set_min_max(self, q1, first=False):
        """Sets the minimum and maximum of a set of quaternions as needed."""
        
        if first:
            self.q_min = q1.dupe()
            self.q_max = q1.dupe()
            
        else:
            if q1.a[0] < self.q_min.a[0]:
                self.q_min.a[0] = q1.a[0]
            elif q1.a[0] > self.q_max.a[0]:
                self.q_max.a[0] = q1.a[0]
                
            if q1.a[1] < self.q_min.a[1]:
                self.q_min.a[1] = q1.a[1]
            elif q1.a[1] > self.q_max.a[1]:
                self.q_max.a[1] = q1.a[1]

            if q1.a[2] < self.q_min.a[2]:
                self.q_min.a[2] = q1.a[2]
            elif q1.a[2] > self.q_max.a[2]:
                self.q_max.a[2] = q1.a[2]
            
            if q1.a[3] < self.q_min.a[3]:
                self.q_min.a[3] = q1.a[3]
            elif q1.a[3] > self.q_max.a[3]:
                self.q_max.a[3] = q1.a[3]


# In[65]:


class TestQHaArray(unittest.TestCase):
    """Test array making software."""
    
    t1 = QHa([1,2,3,4])
    qd = QHa([10, .2, .3, 1])
    qha = QHaArray()
    
    def test_range(self):
        q_list = list(self.qha.range(self.t1, self.qd, 10))
        self.assertTrue(len(q_list) == 11)
        self.assertTrue(q_list[10].qtype == "Q+10dQ")
        self.assertTrue(q_list[10].a[3] == 14)
    
    def test_min_max(self):
        q_list = list(self.qha.range(self.t1, self.qd, 10))
        self.assertTrue(self.qha.q_min.a[0] < 1.01)
        self.assertTrue(self.qha.q_max.a[0] > 100)
        self.assertTrue(self.qha.q_min.a[1] < 2.01)
        self.assertTrue(self.qha.q_max.a[1] > 3.9)
        self.assertTrue(self.qha.q_min.a[2] < 3.01)
        self.assertTrue(self.qha.q_max.a[2] > 4.8)
        self.assertTrue(self.qha.q_min.a[3] < 4.01)
        self.assertTrue(self.qha.q_max.a[3] > 13.9)


# In[66]:


suite = unittest.TestLoader().loadTestsFromModule(TestQHaArray())
unittest.TextTestRunner().run(suite);


# ## States - n quaternions that add up to one quaternion

# Any quaternion can be viewed as the sum of n other quaternions. This is common to see in quantum mechanics, whose needs are driving the development of this class and its methods.

# In[67]:


class QHStates(QH):
    """A class made up of many quaternions."""
    
    def __init__(self, qs=None, qtype="", representation=""):
        
        self.qs = qs
        
        if qs is None:
            self.d, self.dim, self.dimensions = 0, 0, 0
        else:
            self.d, self.dim, self.dimensions = len(qs), len(qs), len(qs)
    
        self.qtype = qtype
        self.representation = representation
        
    def __str__(self, quiet=False):
        """Print out all the states."""
        
        states = ''
        
        for n, q in enumerate(self.qs, start=1):
            states = states + "n={}: {}\n".format(n, q.__str__(quiet))
        
        return states.rstrip()
    
    def print_states(self, label, spacer=False, quiet=False):
        """Utility for printing states as a quaternion series."""

        print(label)
        
        for n, q in enumerate(self.qs):
            ##### print("n={}: {}".format(n + 1, q.__str__(quiet)))
            print("n={}: {}".format(n + 1, q.__str__()))
        
        if not quiet:
            print("sum= {ss}".format(ss=self.summation()))
        
        if spacer:
            print("")

    def equals(self, q1):
        """Test if two states are equal."""
   
        if self.dim != q1.dim:
            return False
        
        result = True
    
        for selfq, q1q in zip(self.qs, q1.qs):
            if not selfq.equals(q1q):
                result = False
                
        return result

    def conj(self, conj_type=0):
        """Take the conjgates of states, default is zero, but also can do 1 or 2."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.conj(conj_type))
            
        return QHStates(new_states)
    
    def simple_q(self):
        """Simplify the states."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.simple_q())
            
        return QHStates(new_states)
    
    def flip_signs(self):
        """Flip signs of all states."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.flip_signs())
            
        return QHStates(new_states)
    
    def invert(self):
        """Invert states."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.invert())
            
        return QHStates(new_states)
    
    def norm(self):
        """Norm of states."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.norm())
            
        return QHStates(new_states)
    
    def normalize(self, n=1, states=None):
        """Normalize all states."""
        
        new_states = []
        
        if states is None:
            states = self.dim
        
        zero_norm_count = 0
        
        for bra in self.qs:
            if bra.norm_squared().t == 0:
                zero_norm_count += 1
            
            new_states.append(bra.normalize(n))
        
        new_states_normalized = []
        
        non_zero_states = states - zero_norm_count
        
        for new_state in new_states:
            new_states_normalized.append(new_state.product(QH([math.sqrt(1/non_zero_states), 0, 0, 0])))
            
        return QHStates(new_states_normalized)

    def orthonormalize(self):
        """Given a quaternion series, resturn a normalized orthoganl basis."""
    
        last_q = self.qs.pop(0).normalize(math.sqrt(1/self.dim))
        orthonormal_qs = [last_q]
    
        for q in self.qs:
            qp = q.Euclidean_product(last_q)
            orthonormal_q = q.dif(qp).normalize(math.sqrt(1/self.dim))
            orthonormal_qs.append(orthonormal_q)
            last_q = orthonormal_q
        
        return QHStates(orthonormal_qs)
    
    def add(self, ket):
        """Add two states."""
        
        if self.dim != ket.dim:
            oops = "The dimensions are not the same: {} != {}".format(self.dim, ket.dim)
            print(oops)
            return None
        
        new_states = []
        
        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(bra.add(ket))
            
        return QHStates(new_states)

    def summation(self):
        """Add them all up, return one quaternion."""
        
        result = None
    
        for q in self.qs:
            if result == None:
                result = q
            else:
                result = result.add(q)
            
        return result    
    
    def dif(self, ket):
        """Take the difference of two states."""
        
        new_states = []
        
        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(bra.dif(ket))
            
        return(QHStates(new_states))  
        
    def diagonal(self, dim):
        """Make a state dim*dim with q or qs along the 'diagonal'."""
        
        diagonal = []
        
        if len(self.qs) == 1:
            q_values = [self.qs[0]] * dim
        elif len(self.qs) == dim:
            q_values = self.qs
        else:
            print("Oops, need the length to be equal to the dimensions.")
        
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    diagonal.append(q_values.pop(0))
                else:
                    diagonal.append(QH().q_0())
        
        return QHStates(diagonal)
        
    def product(self, product_type, bra=None, ket=None, operator=None, kind="", reverse=False):
        """Forms the quaternion product for each state."""
        
        if product_type == 'bra':
            bra = self
        elif product_type == 'ket':
            ket = self
        elif product_type == 'operator':
            operator = self
        else:
            print("Oops, need to set product_type to bra, ket, or operator.")
            return None
        
        def _check_dimensions(op_dim=0, state_1_dim=0, state_2_dim=0, equals=False):
            """Make sure the states and operators are the right sizes. The operator dimension is either
               equal to 1 or the product of the bra and ket dimensions."""

            oops = ''
            
            if equals:
                if state_1_dim != state_2_dim:
                    oops = "states have different dimensions: {} != {}".format(state_1_dim, state_2_dim)
                    
            elif state_2_dim == 0:
                if (op_dim % state_1_dim != 0) and (op_dim != 1):
                    oops = "Operator dimensions don't divide nicely by the state vector: {} % {}".format(
                        op_dim, state_1_dim)
                    
            else:
                if (op_dim != state_1_dim * state_2_dim) and (op_dim == 1 and (state_1_dim != state_2_dim)):
                    oops = "Operator dimensions do not equal the product of the states: {} != {} * {}".format(
                        op_dim, state_1_dim, state_2_dim)
                    
            if oops:
                print(oops)
                return False
            
            else:
                return True
        
        new_states = []
        
        if bra is None and ket is None:
            return None
        
        elif bra is None and operator is None:
            return ket
        
        elif ket is None and operator is None:
            return bra
        
        # <A|B>                                                     
        elif operator is None:
            if _check_dimensions(state_1_dim=bra.dim, state_2_dim=ket.dim, equals=True):
                for b, k in zip(bra.qs, ket.qs):
                    new_states.append(b.product(k, kind, reverse))
            
        # Op|B>
        elif bra is None:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=ket.dim):
                if operator.dim == 1:
                    one_diagonal = operator.diagonal(ket.dim)                    
                    opb = one_diagonal.qs

                else:
                    opb = operator.qs
                        
                for ops in zip(*[iter(opb)] * ket.dim):
                    ok = None
                    
                    for op, k in zip(ops, ket.qs): 
                        if ok is None:
                            ok = op.product(k, kind, reverse)
                        else:
                            ok = ok.add(op.product(k, kind, reverse))
                            
                    new_states.append(ok)

        # <A|Op
        elif ket is None:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=bra.dim):
                # Operator needs to be transposed.
                opt = operator.transpose(bra.dim)
                
                if operator.dim == 1:
                    one_diagonal = operator.diagonal(bra.dim)                    
                    aop = one_diagonal.qs

                else:
                    aop = opt.qs
                        
                for ops in zip(*[iter(aop)]*bra.dim):
                    bop = None
                    
                    for b, op in zip(bra.qs, ops):
                        if bop is None:
                            bop = b.product(op, kind, reverse)
                        else:
                            bop = bop.add(b.product(op, kind, reverse))
                            
                    new_states.append(bop)

        # <A|Op|B>
        else:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=bra.dim, state_2_dim=ket.dim):
                new_ket = []
                
                if operator.dim == 1:
                    one_diagonal = operator.diagonal(ket.dim)                    
                    opb = one_diagonal.qs

                else:
                    opb = operator.qs                                             
                                                             
                for ops in zip(*[iter(opb)]*ket.dim):
                    ok = None
                    
                    for op, k in zip(ops, ket.qs): 
                        if ok is None:
                            ok = op.product(k, kind, reverse)
                        else:
                            ok = ok.add(op.product(k, kind, reverse))
                    
                    new_ket.append(ok)   
                
                new_ket_state = QHStates(new_ket)
                    
                for b, k in zip(bra.qs, new_ket_state.qs):
                    new_states.append(b.product(k, kind, reverse))
                
        return QHStates(new_states)

    def Euclidean_product(self, product_type, bra=None, ket=None, operator=None, kind="", reverse=False):
        """Forms the Euclidean product, what is used in QM all the time."""
        
        if bra is not None:
            bra = bra.conj()
            
        if product_type == 'bra':
            self = self.conj()
                    
        return self.product(product_type, bra, ket, operator, kind, reverse)
    
    def op_n(self, n, first=True, kind="", reverse=False):
        """Mulitply an operator times a number, in that order. Set first=false for n * Op"""
    
        new_states = []
    
        for op in self.qs:
        
            if first:
                new_states.append(op.product(n, kind, reverse))
                              
            else:
                new_states.append(n.product(op, kind, reverse))
    
        return QHStates(new_states)
    
    def norm_squared(self):
        """Take the Euclidean product of each state and add it up, returning one quaternion."""
        
        return self.Euclidean_product('bra', ket=self)
    
    def transpose(self, m=None, n=None):
        """Transposes a series."""
        
        if m is None:
            # test if it is square.
            if math.sqrt(self.dim).is_integer():
                m = int(sp.sqrt(self.dim))
                n = m
               
        if n is None:
            n = int(self.dim / m)
            
        if m * n != self.dim:
            return None
        
        matrix = [[0 for x in range(m)] for y in range(n)] 
        qs_t = []
        
        for mi in range(m):
            for ni in range(n):
                matrix[ni][mi] = self.qs[mi * n + ni]
        
        qs_t = []
        
        for t in matrix:
            for q in t:
                qs_t.append(q)
                
        return QHStates(qs_t)
        
    def Hermitian_conj(self, m=None, n=None, conj_type=0):
        """Returns the Hermitian conjugate."""
        
        return self.transpose(m, n).conj(conj_type)
    
    def dagger(self, m=None, n=None, conj_type=0):
        """Just calls Hermitian_conj()"""
        
        return self.Hermitian_conj(m, n, conj_type)
        
    def is_square(self):
        """Tests if a quaternion series is square, meaning the dimenion is n^2."""
                
        return math.sqrt(self.dim).is_integer()

    def is_Hermitian(self):
        """Tests if a series is Hermitian."""
        
        hc = self.Hermitian_conj()
        
        return self.equals(hc)
    
    @staticmethod
    def sigma(kind, theta=None, phi=None):
        """Returns a sigma when given a type like, x, y, z, xy, xz, yz, xyz, with optional angles theta and phi."""
        
        q0, q1, qi = QH().q_0(), QH().q_1(), QH().q_i()
        
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
            
        x_factor = q1.product(QH([sin_theta * cos_phi, 0, 0, 0]))
        y_factor = qi.product(QH([sin_theta * sin_phi, 0, 0, 0]))
        z_factor = q1.product(QH([cos_theta, 0, 0, 0]))

        sigmas = {}
        sigma['x'] = QHStates([q0, x_factor, x_factor, q0])
        sigma['y'] = QHStates([q0, y_factor, y_factor.flip_signs(), q0]) 
        sigma['z'] = QHStates([z_factor, q0, q0, z_factor.flip_signs()])
  
        sigmas['xy'] = sigma['x'].add(sigma['y'])
        sigmas['xz'] = sigma['x'].add(sigma['z'])
        sigmas['yz'] = sigma['y'].add(sigma['z'])
        sigmas['xyz'] = sigma['xy'].add(sigma['z'])

        if kind not in sigma:
            print("Oops, I only know about x, y, z, and their combinations.")
            return None
        
        return signma[kind].normalize()


# In[68]:


class TestQHStates(unittest.TestCase):
    """Test states."""
    
    q0 = QH().q_0()
    q1 = QH().q_1()
    qi = QH().q_i()
    q0_q1 = QHStates([q0, q1])
    q1_q0 = QHStates([q1, q0])
    q1_qi = QHStates([q1, qi])
    A = QHStates([QH([4,0,0,0]),QH([0,1,0,0])])
    B = QHStates([QH([0,0,1,0]),QH([0,0,0,2]),QH([0,3,0,0])])
    Op = QHStates([QH([3,0,0,0]),QH([0,1,0,0]),QH([0,0,2,0]),QH([0,0,0,3]),QH([2,0,0,0]),QH([0,4,0,0])])
    Op4i = QHStates([QH([0,4,0,0])])
    q1234 = QHStates([QH([1, 1, 0, 0]), QH([2, 1, 0, 0]), QH([3, 1, 0, 0]), QH([4, 1, 0, 0])])
    sigma_y = QHStates([QH([1, 0, 0, 0]), QH([0, -1, 0, 0]), QH([0, 1, 0, 0]), QH([-1, 0, 0, 0])])
    qn = QHStates([QH([3,0,0,4])])
    
    def test_init(self):
        self.assertTrue(self.q0_q1.dim == 2)
        
    def test_equals(self):
        self.assertTrue(self.A.equals(self.A))
        self.assertFalse(self.A.equals(self.B))
        
    def test_conj(self):
        qc = self.q1_qi.conj()
        qc1 = self.q1_qi.conj(1)
        print("q1_qi*: ", qc)
        print("q1_qc*1: ", qc1)
        self.assertTrue(qc.qs[1].x == -1)
        self.assertTrue(qc1.qs[1].x == 1)
    
    def test_flip_signs(self):
        qf = self.q1_qi.flip_signs()
        print("-q1_qi: ", qf)
        self.assertTrue(qf.qs[1].x == -1)
        
    def test_normalize(self):
        qn = self.qn.normalize()
        print("Op normalized: ", qn)
        self.assertAlmostEqual(qn.qs[0].t, 0.6)
        self.assertTrue(qn.qs[0].z == 0.8)
        
    def test_summation(self):
        q_01_sum = self.q0_q1.summation()
        print("sum: ", q_01_sum)
        self.assertTrue(type(q_01_sum) is QH)
        self.assertTrue(q_01_sum.t == 1)
        
    def test_add(self):
        q_0110_add = self.q0_q1.add(self.q1_q0)
        print("add 01 10: ", q_0110_add)
        self.assertTrue(q_0110_add.qs[0].t == 1)
        self.assertTrue(q_0110_add.qs[1].t == 1)
        
    def test_dif(self):
        q_0110_dif = self.q0_q1.dif(self.q1_q0)
        print("dif 01 10: ", q_0110_dif)
        self.assertTrue(q_0110_dif.qs[0].t == -1)
        self.assertTrue(q_0110_dif.qs[1].t == 1)
        
    def test_product_AA(self):
        AA = self.A.product('bra', ket=self.A)
        print("AA: ", AA)
        self.assertTrue(AA.qs[0].equals(QH([16, 0, 0, 0])))
        self.assertTrue(AA.qs[1].equals(QH([-1, 0, 0, 0])))
                  
    def test_Euclidean_product_AA(self):
        AA = self.A.Euclidean_product('bra', ket=self.A)
        print("A* A", AA)
        self.assertTrue(AA.qs[0].equals(QH([16, 0, 0, 0])))
        self.assertTrue(AA.qs[1].equals(QH([1, 0, 0, 0])))

    def test_product_AOp(self):
        AOp = self.A.product('bra', operator=self.Op)
        print("A Op: ", AOp)
        self.assertTrue(AOp.qs[0].equals(QH([12, 0, -3, 0])))
        self.assertTrue(AOp.qs[1].equals(QH([0, 6, 0, 0])))
        self.assertTrue(AOp.qs[2].equals(QH([-4, 0, 8, 0])))
                      
    def test_Euclidean_product_AOp(self):
        AOp = self.A.Euclidean_product('bra', operator=self.Op)
        print("A* Op: ", AOp)
        self.assertTrue(AOp.qs[0].equals(QH([12, 0, 3, 0])))
        self.assertTrue(AOp.qs[1].equals(QH([0, 2, 0, 0])))
        self.assertTrue(AOp.qs[2].equals(QH([4, 0, 8, 0])))
        
    def test_product_AOp4i(self):
        AOp4i = self.A.product('bra', operator=self.Op4i)
        print("A Op4i: ", AOp4i)
        self.assertTrue(AOp4i.qs[0].equals(QH([0, 16, 0, 0])))
        self.assertTrue(AOp4i.qs[1].equals(QH([-4, 0, 0, 0])))
                        
    def test_Euclidean_product_AOp4i(self):
        AOp4i = self.A.Euclidean_product('bra', operator=self.Op4i)
        print("A* Op4i: ", AOp4i)
        self.assertTrue(AOp4i.qs[0].equals(QH([0, 16, 0, 0])))
        self.assertTrue(AOp4i.qs[1].equals(QH([4, 0, 0, 0])))

    def test_product_OpB(self):
        OpB = self.B.product('ket', operator=self.Op)
        print("Op B: ", OpB)
        self.assertTrue(OpB.qs[0].equals(QH([0, 0, 1, -6])))
        self.assertTrue(OpB.qs[1].equals(QH([-12, -3, 0, 4])))
                        
    def test_Euclidean_product_OpB(self):
        OpB = self.B.Euclidean_product('ket', operator=self.Op)
        print("Op B: ", OpB)
        self.assertTrue(OpB.qs[0].equals(QH([0, 0, 1, -6])))
        self.assertTrue(OpB.qs[1].equals(QH([-12, -3, 0, 4])))

    def test_product_AOpB(self):
        AOpB = self.A.product('bra', operator=self.Op, ket=self.B)
        print("A Op B: ", AOpB)
        self.assertTrue(AOpB.qs[0].equals(QH([0, 0, 4, -24])))
        self.assertTrue(AOpB.qs[1].equals(QH([3, -12, -4, 0])))
                        
    def test_Euclidean_product_AOpB(self):
        AOpB = self.A.Euclidean_product('bra', operator=self.Op, ket=self.B)
        print("A* Op B: ", AOpB)
        self.assertTrue(AOpB.qs[0].equals(QH([0, 0, 4, -24])))
        self.assertTrue(AOpB.qs[1].equals(QH([-3, 12, 4, 0])))
        
    def test_product_AOp4i(self):
        AOp4i = self.A.product('bra', operator=self.Op4i)
        print("A Op4i: ", AOp4i)
        self.assertTrue(AOp4i.qs[0].equals(QH([0, 16, 0, 0])))
        self.assertTrue(AOp4i.qs[1].equals(QH([-4, 0, 0, 0])))
                        
    def test_Euclidean_product_AOp4i(self):
        AOp4i = self.A.Euclidean_product('bra', operator=self.Op4i)
        print("A* Op4i: ", AOp4i)
        self.assertTrue(AOp4i.qs[0].equals(QH([0, 16, 0, 0])))
        self.assertTrue(AOp4i.qs[1].equals(QH([4, 0, 0, 0])))

    def test_product_Op4iB(self):
        Op4iB = self.B.product('ket', operator=self.Op4i)
        print("Op4i B: ", Op4iB)
        self.assertTrue(Op4iB.qs[0].equals(QH([0, 0, 0, 4])))
        self.assertTrue(Op4iB.qs[1].equals(QH([0, 0, -8, 0])))
        self.assertTrue(Op4iB.qs[2].equals(QH([-12, 0, 0, 0])))
                        
    def test_Euclidean_product_Op4iB(self):
        Op4iB = self.B.Euclidean_product('ket', operator=self.Op4i)
        print("Op4i B: ", Op4iB)
        self.assertTrue(Op4iB.qs[0].equals(QH([0, 0, 0, 4])))
        self.assertTrue(Op4iB.qs[1].equals(QH([0, 0, -8, 0])))
        self.assertTrue(Op4iB.qs[2].equals(QH([-12, 0, 0, 0])))

    def test_product_AOp4iB(self):
        AOp4iB = self.A.product('bra', operator=self.Op4i, ket=self.B)
        print("A* Op4i B: ", AOp4iB)
        self.assertTrue(AOp4iB.dim == 0)
                        
    def test_Euclidean_product_AOp4iB(self):
        AOp4iB = self.A.Euclidean_product('bra', operator=self.Op4i, ket=self.B)
        print("A* Op4i B: ", AOp4iB)
        self.assertTrue(AOp4iB.dim == 0)

    def test_op_n(self):
        opn = self.Op.op_n(n=self.qi)
        print("op_n: ", opn)
        self.assertTrue(opn.qs[0].x == 3)
        
    def test_transpose(self):
        opt = self.q1234.transpose()
        print("op1234 transposed: ", opt)
        self.assertTrue(opt.qs[0].t == 1)
        self.assertTrue(opt.qs[1].t == 3)
        self.assertTrue(opt.qs[2].t == 2)
        self.assertTrue(opt.qs[3].t == 4)
        optt = self.q1234.transpose().transpose()
        self.assertTrue(optt.equals(self.q1234))
        
    def test_Hermitian_conj(self):
        q_hc = self.q1234.Hermitian_conj()
        print("op1234 Hermitian_conj: ", q_hc)
        self.assertTrue(q_hc.qs[0].t == 1)
        self.assertTrue(q_hc.qs[1].t == 3)
        self.assertTrue(q_hc.qs[2].t == 2)
        self.assertTrue(q_hc.qs[3].t == 4)
        self.assertTrue(q_hc.qs[0].x == -1)
        self.assertTrue(q_hc.qs[1].x == -1)
        self.assertTrue(q_hc.qs[2].x == -1)
        self.assertTrue(q_hc.qs[3].x == -1)
        
    def test_is_Hermitian(self):
        self.assertTrue(self.sigma_y.is_Hermitian())
        self.assertFalse(self.q1234.is_Hermitian())
        
    def test_is_square(self):
        self.assertFalse(self.Op.is_square())
        self.assertTrue(self.Op4i.is_square())    
        
suite = unittest.TestLoader().loadTestsFromModule(TestQHStates())
unittest.TextTestRunner().run(suite);


# Repeat this exercise for:
# 
# QHa
# 
# Q8
# Q8a
# 
# by old fashioned cut and paste with minor tweaks (boring).

# In[69]:


class QHaStates(QHa):
    """A class made up of many quaternions."""
    
    def __init__(self, qs=None, qtype="", representation=""):
        
        self.qs = qs
        
        if qs is None:
            self.d, self.dim, self.dimensions = 0, 0, 0
        else:
            self.d, self.dim, self.dimensions = len(qs), len(qs), len(qs)
        
    def __str__(self, quiet=False):
        """Print out all the states."""
        
        states = ''
        
        for n, q in enumerate(self.qs, start=1):
            states = states + "n={}: {}\n".format(n, q.__str__(quiet))
        
        return states.rstrip()
    
    def print_states(self, label, spacer=False, quiet=False):
        """Utility for printing states as a quaternion series."""

        print(label)
        
        for n, q in enumerate(self.qs, start=1):
            print("n={}: {}".format(n, q.__str__(quiet)))
        
        if not quiet:
            print("sum= {ss}".format(ss=self.summation()))
        
        if spacer:
            print("")

    def summation(self):
        """Add them all up, return one quaternion."""
        
        result = None
    
        for q in self.qs:
            if result == None:
                result = q
            else:
                result = result.add(q)
        
        return result
    
    def equals(self, q1):
        """Test if two states are equal."""
   
        if self.dim != q1.dim:
            return False
        
        result = True
    
        for selfq, q1q in zip(self.qs, q1.qs):
            if not selfq.equals(q1q):
                result = False
                
        return result
    
    def conj(self, conj_type=0):
        """Take the conjgates of states, default is zero, but also can do 1 or 2."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.conj(conj_type))
            
        return(QHaStates(new_states))
    
    def flip_signs(self):
        """Flip signs of all states."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.flip_signs())
            
        return QHaStates(new_states)
    
    def norm(self):
        """Norm of states."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.norm())
            
        return QHaStates(new_states)
    
    def normalize(self, n=1, states=None):
        """Normalize all states."""
        
        new_states = []
        
        if states is None:
            states = self.dim
        
        zero_norm_count = 0
        
        for bra in self.qs:
            if bra.norm_squared().a[0] == 0:
                zero_norm_count += 1
            
            new_states.append(bra.normalize(n))
        
        new_states_normalized = []
        
        non_zero_states = states - zero_norm_count
        
        for new_state in new_states:
            new_states_normalized.append(new_state.product(QHa([math.sqrt(1/non_zero_states), 0, 0, 0])))
            
        return QHaStates(new_states_normalized)
    
    def orthonormalize(self):
        """Given a quaternion series, resturn a normalized orthoganl basis."""
    
        last_q = self.qs.pop(0).normalize(math.sqrt(1/self.dim))
        orthonormal_qs = [last_q]
    
        for q in self.qs:
            qp = q.Euclidean_product(last_q)
            orthonormal_q = q.dif(qp).normalize(math.sqrt(1/self.dim))
            orthonormal_qs.append(orthonormal_q)
            last_q = orthonormal_q
        
        return QHaStates(orthonormal_qs)
    
    def add(self, ket):
        """Add two states."""
        
        new_states = []
        
        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(bra.add(ket))
            
        return(QHaStates(new_states))
    
    def dif(self, ket):
        """Take the difference of two states."""
        
        new_states = []
        
        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(bra.dif(ket))
            
        return(QHaStates(new_states))  
    
    def diagonal(self, dim):
        """Make a state dim*dim with q along the 'diagonal'."""
        
        diagonal = []
        
        if len(self.qs) == 1:
            q_values = [self.qs[0]] * dim
        elif len(self.qs) == dim:
            q_values = self.qs
        else:
            print("Oops, need the length to be equal to the dimensions.")
        
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    diagonal.append(q_values.pop(0))

                else:
                    diagonal.append(QHa().q_0())
        
        return QHaStates(diagonal)
        
    def product(self, product_type, bra=None, ket=None, operator=None, kind=""):
        """Forms the quaternion product for each state."""
        
        if product_type == 'bra':
            bra = self
        elif product_type == 'ket':
            ket = self
        elif product_type == 'operator':
            operator = self
        else:
            print("Oops, need to set product_type to bra, ket, or operator.")
            return None
        
        def _check_dimensions(op_dim=0, state_1_dim=0, state_2_dim=0, equals=False):
            """Make sure the states and operators are the right sizes. The operator dimension is either
               equal to 1 or the product of the bra and ket dimensions."""

            oops = ''
            
            if equals:
                if state_1_dim != state_2_dim:
                    oops = "states have different dimensions: {} != {}".format(state_1_dim, state_2_dim)
                    
            elif state_2_dim == 0:
                if (op_dim % state_1_dim != 0) and (op_dim != 1):
                    oops = "Operator dimensions don't divide nicely by the state vector: {} % {}".format(
                        op_dim, state_1_dim)
                    
            else:
                if (op_dim != state_1_dim * state_2_dim) and (op_dim == 1 and (state_1_dim != state_2_dim)):
                    oops = "Operator dimensions do not equal the product of the states: {} != {} * {}".format(
                        op_dim, state_1_dim, state_2_dim)
                    
            if oops:
                print(oops)
                return False
            
            else:
                return True
        
        new_states = []
        
        if bra is None and ket is None:
            return None
        
        elif bra is None and operator is None:
            return ket
        
        elif ket is None and operator is None:
            return bra
        
        # <A|B>                                                     
        elif operator is None:
            if _check_dimensions(state_1_dim=bra.dim, state_2_dim=ket.dim, equals=True):
                for b, k in zip(bra.qs, ket.qs):
                    new_states.append(b.product(k, kind))
            
        # Op|B>
        elif bra is None:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=ket.dim):
                if operator.dim == 1:
                    one_diagonal = operator.diagonal(ket.dim)                    
                    opb = one_diagonal.qs

                else:
                    opb = operator.qs
                        
                for ops in zip(*[iter(opb)] * ket.dim):
                    ok = None
                    
                    for op, k in zip(ops, ket.qs): 
                        if ok is None:
                            ok = op.product(k, kind)
                        else:
                            ok = ok.add(op.product(k, kind))
                            
                    new_states.append(ok)

        # <A|Op
        elif ket is None:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=bra.dim):
                # Operator needs to be transposed.
                opt = operator.transpose(bra.dim)
                
                if operator.dim == 1:
                    one_diagonal = operator.diagonal(bra.dim)                    
                    aop = one_diagonal.qs

                else:
                    aop = opt.qs
                                                             
                for ops in zip(*[iter(aop)]*bra.dim):
                    bop = None
                    
                    for b, op in zip(bra.qs, ops):
                        if bop is None:
                            bop = b.product(op, kind)
                        else:
                            bop = bop.add(b.product(op, kind))
                            
                    new_states.append(bop)

        # <A|Op|B>
        else:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=bra.dim, state_2_dim=ket.dim):
                new_ket = []
                
                if operator.dim == 1:
                    one_diagonal = operator.diagonal(ket.dim)                    
                    opb = one_diagonal.qs

                else:
                    opb = operator.qs                                             
                                                             
                for ops in zip(*[iter(opb)]*ket.dim):
                    ok = None
                    
                    for op, k in zip(ops, ket.qs): 
                        if ok is None:
                            ok = op.product(k, kind)
                        else:
                            ok = ok.add(op.product(k, kind))
                    
                    new_ket.append(ok)
                
                new_ket_state = QHaStates(new_ket)
                    
                for b, k in zip(bra.qs, new_ket_state.qs):
                    new_states.append(b.product(k, kind))
                
        return QHaStates(new_states)

    def Euclidean_product(self, product_type, bra=None, ket=None, operator=None, kind=""):
        """Forms the Euclidean product, what is used in QM all the time."""
        
        if bra is not None:
            bra = bra.conj()
            
        if product_type == 'bra':
            self = self.conj()
                    
        return self.product(product_type, bra, ket, operator, kind)
    
    def op_n(self, n, first=True, kind="", reverse=False):
        """Mulitply an operator times a number, in that order. Set first=false for n * Op"""
    
        new_states = []
    
        for op in self.qs:
        
            if first:
                new_states.append(op.product(n, kind, reverse))
                              
            else:
                new_states.append(n.product(op, kind, reverse))
    
        return QHaStates(new_states)

    def norm_squared(self):
        """Take the Euclidean product of each state and add it up, returning one quaternion."""
        
        norm = self.Euclidean_product(self).summation()
        return norm
    
    def transpose(self, m=None, n=None):
        """Transposes a series."""
        
        if m is None:
            # test if it is square.
            if math.sqrt(self.dim).is_integer():
                m = int(sp.sqrt(self.dim))
                n = m
               
        if n is None:
            n = int(self.dim / m)
            
        if m * n != self.dim:
            return None
        
        matrix = [[0 for x in range(m)] for y in range(n)] 
        
        qs = self.qs
        qs_t = []
        
        for mi in range(m):
            for ni in range(n):
                matrix[ni][mi] = qs[mi * n + ni]
        
        qs_t = []
        
        for t in matrix:
            for q in t:
                qs_t.append(q)
                
        return QHaStates(qs_t)
    
    def Hermitian_conj(self, m=None, n=None, conj_type=0):
        """Returns the Hermitian conjugate."""
        
        return self.transpose(m, n).conj(conj_type)
    
    def dagger(self, m=None, n=None, conj_type=0):
        """Just calls Hermitian_conj()"""
        
        return self.Hermitian_conj(m, n, conj_type)
        
    def is_square(self):
        """Tests if a quaternion series is square, meaning the dimenion is n^2."""
                
        return math.sqrt(self.dim).is_integer()

    def is_Hermitian(self):
        """Tests if a series is Hermitian."""
        
        hc = self.Hermitian_conj()
        
        return self.equals(hc)


# In[70]:


class TestQHaStates(unittest.TestCase):
    """Test states."""
    
    q0 = QHa().q_0()
    q1 = QHa().q_1()
    qi = QHa().q_i()
    q0_q1 = QHaStates([q0, q1])
    q1_q0 = QHaStates([q1, q0])
    q1_qi = QHaStates([q1, qi])
    A = QHaStates([QHa([4,0,0,0]),QHa([0,1,0,0])])
    B = QHaStates([QHa([0,0,1,0]),QHa([0,0,0,2]),QHa([0,3,0,0])])
    Op = QHaStates([QHa([3,0,0,0]),QHa([0,1,0,0]),QHa([0,0,2,0]),QHa([0,0,0,3]),QHa([2,0,0,0]),QHa([0,4,0,0])])
    Op4i = QHaStates([QHa([0,4,0,0])])
    qn = QHaStates([QHa([3,0,0,4])])
    q1234 = QHStates([QHa([1, 1, 0, 0]), QHa([2, 1, 0, 0]), QHa([3, 1, 0, 0]), QHa([4, 1, 0, 0])])
    sigma_y = QHStates([QHa([1, 0, 0, 0]), QHa([0, -1, 0, 0]), QHa([0, 1, 0, 0]), QHa([-1, 0, 0, 0])])
    
    def test_init(self):
        self.assertTrue(self.q0_q1.dim == 2)
        
    def test_equals(self):
        self.assertTrue(self.A.equals(self.A))
        self.assertFalse(self.A.equals(self.B))
        
    def test_conj(self):
        qc = self.q1_qi.conj()
        qc1 = self.q1_qi.conj(1)
        print("q1_qi*: ", qc)
        print("q1_qc*1: ", qc1)
        self.assertTrue(qc.qs[1].a[1] == -1)
        self.assertTrue(qc1.qs[1].a[1] == 1)
    
    def test_flip_signs(self):
        qf = self.q1_qi.flip_signs()
        print("-q1_qi: ", qf)
        self.assertTrue(qf.qs[1].a[1] == -1)  
        
    def test_summation(self):
        q_01_sum = self.q0_q1.summation()
        print("sum: ", q_01_sum)
        self.assertTrue(type(q_01_sum) is QHa)
        self.assertTrue(q_01_sum.a[0] == 1)
        
    def test_normalize(self):
        qn = self.qn.normalize()
        print("Op normalized: ", qn)
        self.assertAlmostEqual(qn.qs[0].a[0], 0.6)
        self.assertTrue(qn.qs[0].a[3] == 0.8)    
        
    def test_add(self):
        q_0110_add = self.q0_q1.add(self.q1_q0)
        print("add 01 10: ", q_0110_add)
        self.assertTrue(q_0110_add.qs[0].a[0] == 1)
        self.assertTrue(q_0110_add.qs[1].a[0] == 1)
        
    def test_dif(self):
        q_0110_dif = self.q0_q1.dif(self.q1_q0)
        print("dif 01 10: ", q_0110_dif)
        self.assertTrue(q_0110_dif.qs[0].a[0] == -1)
        self.assertTrue(q_0110_dif.qs[1].a[0] == 1)
        
    def test_product_AA(self):
        AA = self.A.product('bra', ket=self.A)
        print("AA: ", AA)
        self.assertTrue(AA.qs[0].equals(QHa([16, 0, 0, 0])))
        self.assertTrue(AA.qs[1].equals(QHa([-1, 0, 0, 0])))
                        
    def test_Euclidean_product_AA(self):
        AA = self.A.Euclidean_product('bra', ket=self.A)
        print("A* A", AA)
        self.assertTrue(AA.qs[0].equals(QHa([16, 0, 0, 0])))
        self.assertTrue(AA.qs[1].equals(QHa([1, 0, 0, 0])))

    def test_product_AOp(self):
        AOp = self.A.product('bra', operator=self.Op)
        print("A Op: ", AOp)
        self.assertTrue(AOp.qs[0].equals(QHa([12, 0, -3, 0])))
        self.assertTrue(AOp.qs[1].equals(QHa([0, 6, 0, 0])))
        self.assertTrue(AOp.qs[2].equals(QHa([-4, 0, 8, 0])))
                        
    def test_Euclidean_product_AOp(self):
        AOp = self.A.Euclidean_product('bra', operator=self.Op)
        print("A* Op: ", AOp)
        self.assertTrue(AOp.qs[0].equals(QHa([12, 0, 3, 0])))
        self.assertTrue(AOp.qs[1].equals(QHa([0, 2, 0, 0])))
        self.assertTrue(AOp.qs[2].equals(QHa([4, 0, 8, 0])))

    def test_product_OpB(self):
        OpB = self.B.product('ket', operator=self.Op)
        print("Op B: ", OpB)
        self.assertTrue(OpB.qs[0].equals(QHa([0, 0, 1, -6])))
        self.assertTrue(OpB.qs[1].equals(QHa([-12, -3, 0, 4])))
                        
    def test_Euclidean_product_OpB(self):
        OpB = self.B.Euclidean_product('ket', operator=self.Op)
        print("Op B: ", OpB)
        self.assertTrue(OpB.qs[0].equals(QHa([0, 0, 1, -6])))
        self.assertTrue(OpB.qs[1].equals(QHa([-12, -3, 0, 4])))

    def test_product_AOpB(self):
        AOpB = self.A.product('bra', operator=self.Op, ket=self.B)
        print("A Op B: ", AOpB)
        self.assertTrue(AOpB.qs[0].equals(QHa([0, 0, 4, -24])))
        self.assertTrue(AOpB.qs[1].equals(QHa([3, -12, -4, 0])))
                        
    def test_Euclidean_product_AOpB(self):
        AOpB = self.A.Euclidean_product('bra', operator=self.Op, ket=self.B)
        print("A* Op B: ", AOpB)
        self.assertTrue(AOpB.qs[0].equals(QHa([0, 0, 4, -24])))
        self.assertTrue(AOpB.qs[1].equals(QHa([-3, 12, 4, 0])))
        
    def test_product_AOp4i(self):
        AOp4i = self.A.product('bra', operator=self.Op4i)
        print("A Op4i: ", AOp4i)
        self.assertTrue(AOp4i.qs[0].equals(QHa([0, 16, 0, 0])))
        self.assertTrue(AOp4i.qs[1].equals(QHa([-4, 0, 0, 0])))
                        
    def test_Euclidean_product_AOp4i(self):
        AOp4i = self.A.Euclidean_product('bra', operator=self.Op4i)
        print("A* Op4i: ", AOp4i)
        self.assertTrue(AOp4i.qs[0].equals(QHa([0, 16, 0, 0])))
        self.assertTrue(AOp4i.qs[1].equals(QHa([4, 0, 0, 0])))

    def test_product_Op4iB(self):
        Op4iB = self.B.product('ket', operator=self.Op4i)
        print("Op4i B: ", Op4iB)
        self.assertTrue(Op4iB.qs[0].equals(QHa([0, 0, 0, 4])))
        self.assertTrue(Op4iB.qs[1].equals(QHa([0, 0, -8, 0])))
        self.assertTrue(Op4iB.qs[2].equals(QHa([-12, 0, 0, 0])))
                        
    def test_Euclidean_product_Op4iB(self):
        Op4iB = self.B.Euclidean_product('ket', operator=self.Op4i)
        print("Op4i B: ", Op4iB)
        self.assertTrue(Op4iB.qs[0].equals(QHa([0, 0, 0, 4])))
        self.assertTrue(Op4iB.qs[1].equals(QHa([0, 0, -8, 0])))
        self.assertTrue(Op4iB.qs[2].equals(QHa([-12, 0, 0, 0])))

    def test_product_AOp4iB(self):
        AOp4iB = self.A.product('bra', operator=self.Op4i, ket=self.B)
        print("A* Op4i B: ", AOp4iB)
        self.assertTrue(AOp4iB.dim == 0)
                        
    def test_Euclidean_product_AOp4iB(self):
        AOp4iB = self.A.Euclidean_product('bra', operator=self.Op4i, ket=self.B)
        print("A* Op4i B: ", AOp4iB)
        self.assertTrue(AOp4iB.dim == 0)
        
    def test_op_n(self):
        opn = self.Op.op_n(n=self.qi)
        print("op_n: ", opn)
        self.assertTrue(opn.qs[0].a[1] == 3)
        
    def test_transpose(self):
        opt = self.q1234.transpose()
        print("op1234 transposed: ", opt)
        self.assertTrue(opt.qs[0].a[0] == 1)
        self.assertTrue(opt.qs[1].a[0] == 3)
        self.assertTrue(opt.qs[2].a[0] == 2)
        self.assertTrue(opt.qs[3].a[0] == 4)
        optt = self.q1234.transpose().transpose()
        self.assertTrue(optt.equals(self.q1234))
        
    def test_Hermitian_conj(self):
        q_hc = self.q1234.Hermitian_conj()
        print("op1234 Hermitian_conj: ", q_hc)
        self.assertTrue(q_hc.qs[0].a[0] == 1)
        self.assertTrue(q_hc.qs[1].a[0] == 3)
        self.assertTrue(q_hc.qs[2].a[0] == 2)
        self.assertTrue(q_hc.qs[3].a[0] == 4)
        self.assertTrue(q_hc.qs[0].a[1] == -1)
        self.assertTrue(q_hc.qs[1].a[1] == -1)
        self.assertTrue(q_hc.qs[2].a[1] == -1)
        self.assertTrue(q_hc.qs[3].a[1] == -1)
        
    def test_is_Hermitian(self):
        self.assertTrue(self.sigma_y.is_Hermitian())
        self.assertFalse(self.q1234.is_Hermitian())
        
    def test_is_square(self):
        self.assertFalse(self.Op.is_square())
        self.assertTrue(self.Op4i.is_square()) 
        
suite = unittest.TestLoader().loadTestsFromModule(TestQHaStates())
unittest.TextTestRunner().run(suite);


# In[71]:


class Q8States(Q8):
    """A class made up of many quaternions."""
    
    def __init__(self, qs=None, qtype="", representation=""):
        
        self.qs = qs
        
        if qs is None:
            self.d, self.dim, self.dimensions = 0, 0, 0
        else:
            self.d, self.dim, self.dimensions = len(qs), len(qs), len(qs)
        
    def __str__(self, quiet=False):
        """Print out all the states."""
        
        states = ''
        
        for n, q in enumerate(self.qs, start=1):
            states = states + "n={}: {}\n".format(n, q.__str__(quiet))
        
        return states.rstrip()
    
    def print_states(self, label, spacer=False, quiet=False):
        """Utility for printing states as a quaternion series."""

        print(label)
        
        for n, q in enumerate(self.qs, start=1):
            print("n={}: {}".format(n, q.__str__(quiet)))
        
        if not quiet:
            print("sum= {ss}".format(ss=self.summation()))
        
        if spacer:
            print("")

    def equals(self, q1):
        """Test if two states are equal."""
   
        if self.dim != q1.dim:
            return False
        
        result = True
    
        for selfq, q1q in zip(self.qs, q1.qs):
            if not selfq.equals(q1q):
                result = False
                
        return result        
            
    def conj(self, conj_type=0):
        """Take the conjgates of states, default is zero, but also can do 1 or 2."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.conj(conj_type))
            
        return(Q8States(new_states))
    
    def flip_signs(self):
        """Flip signs of all states."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.flip_signs())
            
        return Q8States(new_states)
    
    def norm(self):
        """Norm of states."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.norm())
            
        return Q8States(new_states)
    
    def normalize(self, n=1, states=None):
        """Normalize all states."""
        
        new_states = []
        
        if states is None:
            states = self.dim
        
        zero_norm_count = 0
        
        for bra in self.qs:
            if bra.norm_squared().reduce().dt.p == 0:
                zero_norm_count += 1
            
            new_states.append(bra.normalize(n))
        
        new_states_normalized = []
        
        non_zero_states = states - zero_norm_count
        
        for new_state in new_states:
            new_states_normalized.append(new_state.product(Q8([math.sqrt(1/non_zero_states), 0, 0, 0])))
            
        return Q8States(new_states_normalized)

    def orthonormalize(self):
        """Given a quaternion series, resturn a normalized orthoganl basis."""
    
        last_q = self.qs.pop(0).normalize(math.sqrt(1/self.dim))
        orthonormal_qs = [last_q]
    
        for q in self.qs:
            qp = q.Euclidean_product(last_q)
            orthonormal_q = q.dif(qp).normalize(math.sqrt(1/self.dim))
            orthonormal_qs.append(orthonormal_q)
            last_q = orthonormal_q
        
        return Q8States(orthonormal_qs)
    
    def summation(self):
        """Add them all up, return one quaternion."""
        
        result = None
    
        for q in self.qs:
            if result == None:
                result = q
            else:
                result = result.add(q)
            
        return result
    
    def add(self, ket):
        """Add two states."""
        
        new_states = []
        
        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(bra.add(ket))
            
        return(Q8States(new_states))
    
    def dif(self, ket):
        """Take the difference of two states."""
        
        new_states = []
        
        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(bra.dif(ket))
            
        return(Q8States(new_states))  
    
    def diagonal(self, dim):
        """Make a state dim*dim with q along the 'diagonal'."""
        
        diagonal = []
        
        if len(self.qs) == 1:
            q_values = [self.qs[0]] * dim
        elif len(self.qs) == dim:
            q_values = self.qs
        else:
            print("Oops, need the length to be equal to the dimensions.")
        
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    diagonal.append(q_values.pop(0))

                else:
                    diagonal.append(Q8().q_0())
        
        return Q8States(diagonal)
        
    def product(self, product_type, bra=None, ket=None, operator=None, kind=""):
        """Forms the quaternion product for each state."""
        
        if product_type == 'bra':
            bra = self
        elif product_type == 'ket':
            ket = self
        elif product_type == 'operator':
            operator = self
        else:
            print("Oops, need to set product_type to bra, ket, or operator.")
            return None
        
        def _check_dimensions(op_dim=0, state_1_dim=0, state_2_dim=0, equals=False):
            """Make sure the states and operators are the right sizes. The operator dimension is either
               equal to 1 or the product of the bra and ket dimensions."""

            oops = ''
            
            if equals:
                if state_1_dim != state_2_dim:
                    oops = "states have different dimensions: {} != {}".format(state_1_dim, state_2_dim)
                    
            elif state_2_dim == 0:
                if (op_dim % state_1_dim != 0) and (op_dim != 1):
                    oops = "Operator dimensions don't divide nicely by the state vector: {} % {}".format(
                        op_dim, state_1_dim)
                    
            else:
                if (op_dim != state_1_dim * state_2_dim) and (op_dim == 1 and (state_1_dim != state_2_dim)):
                    oops = "Operator dimensions do not equal the product of the states: {} != {} * {}".format(
                        op_dim, state_1_dim, state_2_dim)
                    
            if oops:
                print(oops)
                return False
            
            else:
                return True
        
        new_states = []
        
        if bra is None and ket is None:
            return None
        
        elif bra is None and operator is None:
            return ket
        
        elif ket is None and operator is None:
            return bra
        
        # <A|B>                                                     
        elif operator is None:
            if _check_dimensions(state_1_dim=bra.dim, state_2_dim=ket.dim, equals=True):
                for b, k in zip(bra.qs, ket.qs):
                    new_states.append(b.product(k, kind))
            
        # Op|B>
        elif bra is None:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=ket.dim):
                if operator.dim == 1:
                    one_diagonal = operator.diagonal(ket.dim)                    
                    opb = one_diagonal.qs

                else:
                    opb = operator.qs
                        
                for ops in zip(*[iter(opb)] * ket.dim):
                    ok = None
                    
                    for op, k in zip(ops, ket.qs): 
                        if ok is None:
                            ok = op.product(k, kind)
                        else:
                            ok = ok.add(op.product(k, kind))
                            
                    new_states.append(ok)

        # <A|Op
        elif ket is None:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=bra.dim):
                # Operator needs to be transposed.
                opt = operator.transpose(bra.dim)
                
                if operator.dim == 1:
                    one_diagonal = operator.diagonal(bra.dim)                    
                    aop = one_diagonal.qs

                else:
                    aop = operator.qs
                                                             
                for ops in zip(*[iter(aop)]*bra.dim):
                    bop = None
                    
                    for b, op in zip(bra.qs, ops):
                        if bop is None:
                            bop = b.product(op, kind)
                        else:
                            bop = bop.add(b.product(op, kind))
                            
                    new_states.append(bop)

        # <A|Op|B>
        else:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=bra.dim, state_2_dim=ket.dim):
                new_ket = []
                
                if operator.dim == 1:
                    one_diagonal = operator.diagonal(ket.dim)                    
                    opb = one_diagonal.qs

                else:
                    opb = operator.qs                                             
                                                             
                for ops in zip(*[iter(opb)]*ket.dim):
                    ok = None
                    
                    for op, k in zip(ops, ket.qs): 
                        if ok is None:
                            ok = op.product(k, kind)
                        else:
                            ok = ok.add(op.product(k, kind))
                    
                    new_ket.append(ok)
                
                new_ket_state = Q8States(new_ket)
                    
                for b, k in zip(bra.qs, new_ket_state.qs):
                    new_states.append(b.product(k, kind))
                
        return Q8States(new_states)

    def Euclidean_product(self, product_type, bra=None, ket=None, operator=None, kind=""):
        """Forms the Euclidean product, what is used in QM all the time."""
        
        if bra is not None:
            bra = bra.conj()
            
        if product_type == 'bra':
            self = self.conj()
                    
        return self.product(product_type, bra, ket, operator, kind)
    
    def op_n(self, n, first=True, kind="", reverse=False):
        """Mulitply an operator times a number, in that order. Set first=false for n * Op"""
    
        new_states = []
    
        for op in self.qs:
        
            if first:
                new_states.append(op.product(n, kind, reverse))
                              
            else:
                new_states.append(n.product(op, kind, reverse))
    
        return Q8States(new_states)

    def norm_squared(self):
        """Take the Euclidean product of each state and add it up, returning one quaternion."""
        
        norm = self.Euclidean_product(self).summation()
        return norm
    
    def transpose(self, m=None, n=None):
        """Transposes a series."""
        
        if m is None:
            # test if it is square.
            if math.sqrt(self.dim).is_integer():
                m = int(sp.sqrt(self.dim))
                n = m
               
        if n is None:
            n = int(self.dim / m)
            
        if m * n != self.dim:
            return None
        
        matrix = [[0 for x in range(m)] for y in range(n)] 
        
        qs = self.qs
        qs_t = []
        
        for mi in range(m):
            for ni in range(n):
                matrix[ni][mi] = qs[mi * n + ni]
        
        qs_t = []
        
        for t in matrix:
            for q in t:
                qs_t.append(q)
                
        return Q8States(qs_t)


# In[72]:


class TestQ8States(unittest.TestCase):
    """Test states."""
    
    q0 = Q8().q_0()
    q1 = Q8().q_1()
    qi = Q8().q_i()
    q0_q1 = Q8States([q0, q1])
    q1_q0 = Q8States([q1, q0])
    q1_qi = Q8States([q1, qi])
    A = Q8States([Q8([4,0,0,0]),Q8([0,1,0,0])])
    B = Q8States([Q8([0,0,1,0]),Q8([0,0,0,2]),Q8([0,3,0,0])])
    Op = Q8States([Q8([3,0,0,0]),Q8([0,1,0,0]),Q8([0,0,2,0]),Q8([0,0,0,3]),Q8([2,0,0,0]),Q8([0,4,0,0])])
    Op4i = Q8States([Q8([0,4,0,0])])
    qn = Q8States([Q8([3,0,0,4])])
    q1234 = Q8States([Q8([1, 1, 0, 0]), Q8([2, 1, 0, 0]), Q8([3, 1, 0, 0]), Q8([4, 1, 0, 0])])
    sigma_y = Q8States([Q8([1, 0, 0, 0]), Q8([0, -1, 0, 0]), Q8([0, 1, 0, 0]), Q8([-1, 0, 0, 0])])
        
    def test_init(self):
        self.assertTrue(self.q0_q1.dim == 2)
        
    def test_equals(self):
        self.assertTrue(self.A.equals(self.A))
        self.assertFalse(self.A.equals(self.B))
         
    def test_conj(self):
        qc = self.q1_qi.conj()
        qc1 = self.q1_qi.conj(1)
        print("q1_qi*: ", qc)
        print("q1_qc*1: ", qc1)
        print("qc.qs[1]: ", qc.qs[1])
        self.assertTrue(qc.qs[1].dx.n == 1)
        self.assertTrue(qc1.qs[1].dx.p == 1)   

    def test_flip_signs(self):
        qf = self.q1_qi.flip_signs()
        print("-q1_qi: ", qf)
        self.assertTrue(qf.qs[1].dx.n == 1)
    
    def test_normalize(self):
        qn = self.qn.normalize()
        print("Op normalized: ", qn)
        self.assertAlmostEqual(qn.qs[0].dt.p, 0.6)
        self.assertTrue(qn.qs[0].dz.p == 0.8)
        
    def test_summation(self):
        q_01_sum = self.q0_q1.summation()
        print("sum: ", q_01_sum)
        self.assertTrue(type(q_01_sum) is Q8)
        self.assertTrue(q_01_sum.dt.p== 1)
        
    def test_add(self):
        q_0110_add = self.q0_q1.add(self.q1_q0)
        print("add 01 10: ", q_0110_add)
        self.assertTrue(q_0110_add.qs[0].dt.p== 1)
        self.assertTrue(q_0110_add.qs[1].dt.p== 1)
        
    def test_dif(self):
        q_0110_dif = self.q0_q1.dif(self.q1_q0)
        print("dif 01 10: ", q_0110_dif)
        self.assertTrue(q_0110_dif.qs[0].dt.n== 1)
        self.assertTrue(q_0110_dif.qs[1].dt.p== 1)
        
    def test_product_AA(self):
        AA = self.A.product('bra', ket=self.A)
        print("AA: ", AA)
        self.assertTrue(AA.qs[0].equals(Q8([16, 0, 0, 0])))
        self.assertTrue(AA.qs[1].equals(Q8([-1, 0, 0, 0])))
                        
    def test_Euclidean_product_AA(self):
        AA = self.A.Euclidean_product('bra', ket=self.A)
        print("A* A", AA)
        self.assertTrue(AA.qs[0].equals(Q8([16, 0, 0, 0])))
        self.assertTrue(AA.qs[1].equals(Q8([1, 0, 0, 0])))

    def test_product_AOp(self):
        AOp = self.A.product('bra', operator=self.Op)
        print("A Op: ", AOp)
        self.assertTrue(AOp.qs[0].equals(Q8([11, 0, 0, 0])))
        self.assertTrue(AOp.qs[1].equals(Q8([0, 0, 5, 0])))
        self.assertTrue(AOp.qs[2].equals(Q8([4, 0, 0, 0])))
                        
    def test_Euclidean_product_AOp(self):
        AOp = self.A.Euclidean_product('bra', operator=self.Op)
        print("A* Op: ", AOp)
        self.assertTrue(AOp.qs[0].equals(Q8([13, 0, 0, 0])))
        self.assertTrue(AOp.qs[1].equals(Q8([0, 0, 11, 0])))
        self.assertTrue(AOp.qs[2].equals(Q8([12, 0, 0, 0])))

    def test_product_OpB(self):
        OpB = self.B.product('ket', operator=self.Op)
        print("Op B: ", OpB)
        self.assertTrue(OpB.qs[0].equals(Q8([0, 0, 1, -6])))
        self.assertTrue(OpB.qs[1].equals(Q8([-12, -3, 0, 4])))
                        
    def test_Euclidean_product_OpB(self):
        OpB = self.B.Euclidean_product('ket', operator=self.Op)
        print("Op B: ", OpB)
        self.assertTrue(OpB.qs[0].equals(Q8([0, 0, 1, -6])))
        self.assertTrue(OpB.qs[1].equals(Q8([-12, -3, 0, 4])))

    def test_product_AOpB(self):
        AOpB = self.A.product('bra', operator=self.Op, ket=self.B)
        print("A Op B: ", AOpB)
        self.assertTrue(AOpB.qs[0].equals(Q8([0, 0, 4, -24])))
        self.assertTrue(AOpB.qs[1].equals(Q8([3, -12, -4, 0])))
                        
    def test_Euclidean_product_AOpB(self):
        AOpB = self.A.Euclidean_product('bra', operator=self.Op, ket=self.B)
        print("A* Op B: ", AOpB)
        self.assertTrue(AOpB.qs[0].equals(Q8([0, 0, 4, -24])))
        self.assertTrue(AOpB.qs[1].equals(Q8([-3, 12, 4, 0])))
        
    def test_product_AOp4i(self):
        AOp4i = self.A.product('bra', operator=self.Op4i)
        print("A Op4i: ", AOp4i)
        self.assertTrue(AOp4i.qs[0].equals(Q8([0, 16, 0, 0])))
        self.assertTrue(AOp4i.qs[1].equals(Q8([-4, 0, 0, 0])))
                        
    def test_Euclidean_product_AOp4i(self):
        AOp4i = self.A.Euclidean_product('bra', operator=self.Op4i)
        print("A* Op4i: ", AOp4i)
        self.assertTrue(AOp4i.qs[0].equals(Q8([0, 16, 0, 0])))
        self.assertTrue(AOp4i.qs[1].equals(Q8([4, 0, 0, 0])))

    def test_product_Op4iB(self):
        Op4iB = self.B.product('ket', operator=self.Op4i)
        print("Op4i B: ", Op4iB)
        self.assertTrue(Op4iB.qs[0].equals(Q8([0, 0, 0, 4])))
        self.assertTrue(Op4iB.qs[1].equals(Q8([0, 0, -8, 0])))
        self.assertTrue(Op4iB.qs[2].equals(Q8([-12, 0, 0, 0])))
                        
    def test_Euclidean_product_Op4iB(self):
        Op4iB = self.B.Euclidean_product('ket', operator=self.Op4i)
        print("Op4i B: ", Op4iB)
        self.assertTrue(Op4iB.qs[0].equals(Q8([0, 0, 0, 4])))
        self.assertTrue(Op4iB.qs[1].equals(Q8([0, 0, -8, 0])))
        self.assertTrue(Op4iB.qs[2].equals(Q8([-12, 0, 0, 0])))

    def test_product_AOp4iB(self):
        AOp4iB = self.A.product('bra', operator=self.Op4i, ket=self.B)
        print("A* Op4i B: ", AOp4iB)
        self.assertTrue(AOp4iB.dim == 0)
                        
    def test_Euclidean_product_AOp4iB(self):
        AOp4iB = self.A.Euclidean_product('bra', operator=self.Op4i, ket=self.B)
        print("A* Op4i B: ", AOp4iB)
        self.assertTrue(AOp4iB.dim == 0)
        
    def test_op_n(self):
        opn = self.Op.op_n(n=self.qi)
        print("op_n: ", opn)
        self.assertTrue(opn.qs[0].dx.p == 3)
        
suite = unittest.TestLoader().loadTestsFromModule(TestQ8States())
unittest.TextTestRunner().run(suite);


# In[73]:


class Q8aStates(Q8a):
    """A class made up of many quaternions."""
    
    def __init__(self, qs=None, qtype="Q", representation=""):
        
        self.qs = qs
        
        if qs is None:
            self.d, self.dim, self.dimensions = 0, 0, 0
        else:
            self.d, self.dim, self.dimensions = len(qs), len(qs), len(qs)
        
        self.qtype = qtype
        self.representation = representation
        
    def __str__(self, quiet=False):
        """Print out all the states."""
        
        states = ''
        
        for n, q in enumerate(self.qs, start=1):
            states = states + "n={}: {}\n".format(n, q.__str__(quiet))
        
        return states.rstrip()
    
    def print_states(self, label, spacer=False, quiet=False):
        """Utility for printing states as a quaternion series."""

        print(label)
        
        for n, q in enumerate(self.qs, start=1):
            #print("n={}: {}".format(n, q.__str__(quiet)))
            print("n={}: {}".format(n, q.__str__()))
        
        if not quiet:
            print("sum= {ss}".format(ss=self.summation()))
        
        if spacer:
            print("")

    def equals(self, q1):
        """Test if two states are equal."""
   
        if self.dim != q1.dim:
            return False
        
        result = True
    
        for selfq, q1q in zip(self.qs, q1.qs):
            if not selfq.equals(q1q):
                result = False
                
        return result
            
    def conj(self, conj_type=0):
        """Take the conjgates of states, default is zero, but also can do 1 or 2."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.conj(conj_type))
            
        return(Q8aStates(new_states))
    
    def flip_signs(self):
        """Flip signs of all states."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.flip_signs())
            
        return Q8aStates(new_states)
    
    def norm(self):
        """Norm of states."""
        
        new_states = []
        
        for bra in self.qs:
            new_states.append(bra.norm())
            
        return Q8aStates(new_states)
    
    def normalize(self, n=1, states=None):
        """Normalize all states."""
        
        new_states = []
        
        if states is None:
            states = self.dim
        
        zero_norm_count = 0
        
        for bra in self.qs:
            if bra.norm_squared().reduce().a[0] == 0:
                zero_norm_count += 1
            
            new_states.append(bra.normalize(n))
        
        new_states_normalized = []
        
        non_zero_states = states - zero_norm_count
        
        for new_state in new_states:
            new_states_normalized.append(new_state.product(Q8a([math.sqrt(1/non_zero_states), 0, 0, 0])))
            
        return Q8aStates(new_states_normalized)
    
    def orthonormalize(self):
        """Given a quaternion series, resturn a normalized orthoganl basis."""
    
        last_q = self.qs.pop(0).normalize(math.sqrt(1/self.dim))
        orthonormal_qs = [last_q]
    
        for q in self.qs:
            qp = q.Euclidean_product(last_q)
            orthonormal_q = q.dif(qp).normalize(math.sqrt(1/self.dim))
            orthonormal_qs.append(orthonormal_q)
            last_q = orthonormal_q
        
        return Q8aStates(orthonormal_qs)
    
    def summation(self):
        """Add them all up, return one quaternion."""
        
        result = None
    
        for q in self.qs:
            if result == None:
                result = q
            else:
                result = result.add(q)
    
        for q in self.qs:
            result = result.add(q)
            
        return result
    
    def add(self, ket):
        """Add two states."""
        
        new_states = []
        
        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(bra.add(ket))
            
        return(Q8aStates(new_states))
    
    def dif(self, ket):
        """Take the difference of two states."""
        
        new_states = []
        
        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(bra.dif(ket))
            
        return(Q8aStates(new_states)) 
    
    def diagonal(self, dim):
        """Make a state dim*dim with q along the 'diagonal'."""
        
        diagonal = []
        
        if len(self.qs) == 1:
            q_values = [self.qs[0]] * dim
        elif len(self.qs) == dim:
            q_values = self.qs
        else:
            print("Oops, need the length to be equal to the dimensions.")
        
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    diagonal.append(q_values.pop(0))

                else:
                    diagonal.append(Q8a().q_0())
        
        return Q8aStates(diagonal)
        
    def product(self, product_type, bra=None, ket=None, operator=None, kind=""):
        """Forms the quaternion product for each state."""
        
        if product_type == 'bra':
            bra = self
        elif product_type == 'ket':
            ket = self
        elif product_type == 'operator':
            operator = self
        else:
            print("Oops, need to set product_type to bra, ket, or operator.")
            return None
        
        def _check_dimensions(op_dim=0, state_1_dim=0, state_2_dim=0, equals=False):
            """Make sure the states and operators are the right sizes. The operator dimension is either
               equal to 1 or the product of the bra and ket dimensions."""

            oops = ''
            
            if equals:
                if state_1_dim != state_2_dim:
                    oops = "states have different dimensions: {} != {}".format(state_1_dim, state_2_dim)
                    
            elif state_2_dim == 0:
                if (op_dim % state_1_dim != 0) and (op_dim != 1):
                    oops = "Operator dimensions don't divide nicely by the state vector: {} % {}".format(
                        op_dim, state_1_dim)
                    
            else:
                if (op_dim != state_1_dim * state_2_dim) and (op_dim == 1 and (state_1_dim != state_2_dim)):
                    oops = "Operator dimensions do not equal the product of the states: {} != {} * {}".format(
                        op_dim, state_1_dim, state_2_dim)
                    
            if oops:
                print(oops)
                return False
            
            else:
                return True
        
        new_states = []
        
        if bra is None and ket is None:
            return None
        
        elif bra is None and operator is None:
            return ket
        
        elif ket is None and operator is None:
            return bra
        
        # <A|B>                                                     
        elif operator is None:
            if _check_dimensions(state_1_dim=bra.dim, state_2_dim=ket.dim, equals=True):
                for b, k in zip(bra.qs, ket.qs):
                    new_states.append(b.product(k, kind))
            
        # Op|B>
        elif bra is None:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=ket.dim):
                if operator.dim == 1:
                    one_diagonal = operator.diagonal(ket.dim)                    
                    opb = one_diagonal.qs

                else:
                    opb = operator.qs
                        
                for ops in zip(*[iter(opb)] * ket.dim):
                    ok = None
                    
                    for op, k in zip(ops, ket.qs): 
                        if ok is None:
                            ok = op.product(k, kind)
                        else:
                            ok = ok.add(op.product(k, kind))
                            
                    new_states.append(ok)

        # <A|Op
        elif ket is None:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=bra.dim):
                # Operator needs to be transposed.
                opt = operator.transpose(bra.dim)
                
                if operator.dim == 1:
                    one_diagonal = operator.diagonal(bra.dim)                    
                    aop = one_diagonal.qs

                else:
                    aop = opt.qs
                                                             
                for ops in zip(*[iter(aop)]*bra.dim):
                    bop = None
                    
                    for b, op in zip(bra.qs, ops):
                        if bop is None:
                            bop = b.product(op, kind)
                        else:
                            bop = bop.add(b.product(op, kind))
                            
                    new_states.append(bop)

        # <A|Op|B>
        else:
            if _check_dimensions(op_dim=operator.dim, state_1_dim=bra.dim, state_2_dim=ket.dim):
                new_ket = []
                
                if operator.dim == 1:
                    one_diagonal = operator.diagonal(ket.dim)                    
                    opb = one_diagonal.qs

                else:
                    opb = operator.qs                                             
                                                             
                for ops in zip(*[iter(opb)]*ket.dim):
                    ok = None
                    
                    for op, k in zip(ops, ket.qs): 
                        if ok is None:
                            ok = op.product(k, kind)
                        else:
                            ok = ok.add(op.product(k, kind))
                    
                    new_ket.append(ok)
                
                new_ket_state = Q8aStates(new_ket)
                    
                for b, k in zip(bra.qs, new_ket_state.qs):
                    new_states.append(b.product(k, kind))
                
        return Q8aStates(new_states)

    def Euclidean_product(self, product_type, bra=None, ket=None, operator=None, kind=""):
        """Forms the Euclidean product, what is used in QM all the time."""
        
        if bra is not None:
            bra = bra.conj()
            
        if product_type == 'bra':
            self = self.conj()
                    
        return self.product(product_type, bra, ket, operator, kind)
    
    def op_n(self, n, first=True, kind="", reverse=False):
        """Mulitply an operator times a number, in that order. Set first=false for n * Op"""
    
        new_states = []
    
        for op in self.qs:
        
            if first:
                new_states.append(op.product(n, kind, reverse))
                              
            else:
                new_states.append(n.product(op, kind, reverse))
    
        return Q8aStates(new_states)

    def norm_squared(self):
        """Take the Euclidean product of each state and add it up, returning one quaternion."""
        
        norm = self.Euclidean_product(self).summation()
        return norm
    
    def transpose(self, m=None, n=None):
        """Transposes a series."""
        
        if m is None:
            # test if it is square.
            if math.sqrt(self.dim).is_integer():
                m = int(sp.sqrt(self.dim))
                n = m
               
        if n is None:
            n = int(self.dim / m)
            
        if m * n != self.dim:
            return None
        
        matrix = [[0 for x in range(m)] for y in range(n)] 
        
        qs_t = []
        
        for mi in range(m):
            for ni in range(n):
                matrix[ni][mi] = self.qs[mi * n + ni]
        
        qs_t = []
        
        for t in matrix:
            for q in t:
                qs_t.append(q)
                
        return Q8aStates(qs_t)


# In[74]:


class TestQ8aStates(unittest.TestCase):
    """Test states."""
    
    q0 = Q8a().q_0()
    q1 = Q8a().q_1()
    qi = Q8a().q_i()
    q0_q1 = Q8aStates([q0, q1])
    q1_q0 = Q8aStates([q1, q0])
    q1_qi = Q8aStates([q1, qi])
    A = Q8aStates([Q8a([4,0,0,0]),Q8a([0,1,0,0])])
    B = Q8aStates([Q8a([0,0,1,0]),Q8a([0,0,0,2]),Q8a([0,3,0,0])])
    Op = Q8aStates([Q8a([3,0,0,0]),Q8a([0,1,0,0]),Q8a([0,0,2,0]),Q8a([0,0,0,3]),Q8a([2,0,0,0]),Q8a([0,4,0,0])])
    Op4i = Q8aStates([Q8a([0,4,0,0])])
    qn = Q8aStates([Q8a([3,0,0,4])])
    q1234 = Q8aStates([Q8a([1, 1, 0, 0]), Q8a([2, 1, 0, 0]), Q8a([3, 1, 0, 0]), Q8a([4, 1, 0, 0])])
    sigma_y = Q8aStates([Q8a([1, 0, 0, 0]), Q8a([0, -1, 0, 0]), Q8a([0, 1, 0, 0]), Q8a([-1, 0, 0, 0])])
    
    def test_init(self):
        self.assertTrue(self.q0_q1.dim == 2)

    def test_equals(self):
        self.assertTrue(self.A.equals(self.A))
        self.assertFalse(self.A.equals(self.B))
        
    def test_conj(self):
        qc = self.q1_qi.conj()
        qc1 = self.q1_qi.conj(1)
        print("q1_qi*: ", qc)
        print("q1_qc*1: ", qc1)
        self.assertTrue(qc.qs[1].a[3] == 1)
        self.assertTrue(qc1.qs[1].a[2] == 1)
    
    def test_flip_signs(self):
        qf = self.q1_qi.flip_signs()
        print("-q1_qi: ", qf)
        self.assertTrue(qf.qs[1].a[3] == 1)    
    
    def test_normalize(self):
        qn = self.qn.normalize()
        print("Op normalized: ", qn)
        self.assertAlmostEqual(qn.qs[0].a[0], 0.6)
        self.assertTrue(qn.qs[0].a[6] == 0.8)
    
    def test_summation(self):
        q_01_sum = self.q0_q1.summation()
        print("sum: ", q_01_sum)
        self.assertTrue(type(q_01_sum) is Q8a)
        self.assertTrue(q_01_sum.a[0]== 2)
        
    def test_add(self):
        q_0110_add = self.q0_q1.add(self.q1_q0)
        print("add 01 10: ", q_0110_add)
        self.assertTrue(q_0110_add.qs[0].a[0]== 1)
        self.assertTrue(q_0110_add.qs[1].a[0]== 1)
        
    def test_dif(self):
        q_0110_dif = self.q0_q1.dif(self.q1_q0)
        print("dif 01 10: ", q_0110_dif)
        self.assertTrue(q_0110_dif.qs[0].a[1]== 1)
        self.assertTrue(q_0110_dif.qs[1].a[0]== 1)
        
    def test_product_AA(self):
        AA = self.A.product('bra', ket=self.A)
        print("AA: ", AA)
        self.assertTrue(AA.qs[0].equals(Q8a([16, 0, 0, 0])))
        self.assertTrue(AA.qs[1].equals(Q8a([-1, 0, 0, 0])))
                        
    def test_Euclidean_product_AA(self):
        AA = self.A.Euclidean_product('bra', ket=self.A)
        print("A* A", AA)
        self.assertTrue(AA.qs[0].equals(Q8a([16, 0, 0, 0])))
        self.assertTrue(AA.qs[1].equals(Q8a([1, 0, 0, 0])))

    def test_product_AOp(self):
        AOp = self.A.product('bra', operator=self.Op)
        print("(A|Op: ", AOp)
        self.assertTrue(AOp.qs[0].equals(Q8a([12, 0, -3, 0])))
        self.assertTrue(AOp.qs[1].equals(Q8a([0, 6, 0, 0])))
        self.assertTrue(AOp.qs[2].equals(Q8a([-4, 0,8, 0])))
                        
    def test_Euclidean_product_AOp(self):
        AOp = self.A.Euclidean_product('bra', operator=self.Op)
        print("<A*|Op: ", AOp)
        self.assertTrue(AOp.qs[0].equals(Q8a([12, 0, 3, 0])))
        self.assertTrue(AOp.qs[1].equals(Q8a([0, 2, 0, 0])))
        self.assertTrue(AOp.qs[2].equals(Q8a([4, 0, 8, 0])))

    def test_product_OpB(self):
        OpB = self.B.product('ket', operator=self.Op)
        print("Op B: ", OpB)
        self.assertTrue(OpB.qs[0].equals(Q8a([0, 0, 1, -6])))
        self.assertTrue(OpB.qs[1].equals(Q8a([-12, -3, 0, 4])))
                        
    def test_Euclidean_product_OpB(self):
        OpB = self.B.Euclidean_product('ket', operator=self.Op)
        print("Op B: ", OpB)
        self.assertTrue(OpB.qs[0].equals(Q8a([0, 0, 1, -6])))
        self.assertTrue(OpB.qs[1].equals(Q8a([-12, -3, 0, 4])))

    def test_product_AOpB(self):
        AOpB = self.A.product('bra', operator=self.Op, ket=self.B)
        print("A Op B: ", AOpB)
        self.assertTrue(AOpB.qs[0].equals(Q8a([0, 0, 4, -24])))
        self.assertTrue(AOpB.qs[1].equals(Q8a([3, -12, -4, 0])))
                        
    def test_Euclidean_product_AOpB(self):
        AOpB = self.A.Euclidean_product('bra', operator=self.Op, ket=self.B)
        print("A* Op B: ", AOpB)
        self.assertTrue(AOpB.qs[0].equals(Q8a([0, 0, 4, -24])))
        self.assertTrue(AOpB.qs[1].equals(Q8a([-3, 12, 4, 0])))
        
    def test_product_AOp4i(self):
        AOp4i = self.A.product('bra', operator=self.Op4i)
        print("A Op4i: ", AOp4i)
        self.assertTrue(AOp4i.qs[0].equals(Q8a([0, 16, 0, 0])))
        self.assertTrue(AOp4i.qs[1].equals(Q8a([-4, 0, 0, 0])))
                        
    def test_Euclidean_product_AOp4i(self):
        AOp4i = self.A.Euclidean_product('bra', operator=self.Op4i)
        print("A* Op4i: ", AOp4i)
        self.assertTrue(AOp4i.qs[0].equals(Q8a([0, 16, 0, 0])))
        self.assertTrue(AOp4i.qs[1].equals(Q8a([4, 0, 0, 0])))

    def test_product_Op4iB(self):
        Op4iB = self.B.product('ket', operator=self.Op4i)
        print("Op4i B: ", Op4iB)
        self.assertTrue(Op4iB.qs[0].equals(Q8a([0, 0, 0, 4])))
        self.assertTrue(Op4iB.qs[1].equals(Q8a([0, 0, -8, 0])))
        self.assertTrue(Op4iB.qs[2].equals(Q8a([-12, 0, 0, 0])))
                        
    def test_Euclidean_product_Op4iB(self):
        Op4iB = self.B.Euclidean_product('ket', operator=self.Op4i)
        print("Op4i B: ", Op4iB)
        self.assertTrue(Op4iB.qs[0].equals(Q8a([0, 0, 0, 4])))
        self.assertTrue(Op4iB.qs[1].equals(Q8a([0, 0, -8, 0])))
        self.assertTrue(Op4iB.qs[2].equals(Q8a([-12, 0, 0, 0])))

    def test_product_AOp4iB(self):
        AOp4iB = self.A.product('bra', operator=self.Op4i, ket=self.B)
        print("A* Op4i B: ", AOp4iB)
        self.assertTrue(AOp4iB.dim == 0)
                        
    def test_Euclidean_product_AOp4iB(self):
        AOp4iB = self.A.Euclidean_product('bra', operator=self.Op4i, ket=self.B)
        print("A* Op4i B: ", AOp4iB)
        self.assertTrue(AOp4iB.dim == 0)
        
    def test_op_n(self):
        opn = self.Op.op_n(n=self.qi)
        print("op_n: ", opn)
        self.assertTrue(opn.qs[0].a[2] == 3)
        
suite = unittest.TestLoader().loadTestsFromModule(TestQ8aStates())
unittest.TextTestRunner().run(suite);

