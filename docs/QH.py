#!/usr/bin/env python
# coding: utf-8

# # Developing Quaternions for iPython






import math
import numpy as np
import pdb
import random
import sympy as sp
import unittest
from copy import deepcopy
from IPython.display import display


# Define the stretch factor $\gamma$ and the $\gamma \beta$ used in special relativity.




def sr_gamma(beta_x=0, beta_y=0, beta_z=0):
    """The gamma used in special relativity using 3 velocites, some may be zero."""

    return 1 / (1 - beta_x ** 2 - beta_y ** 2 - beta_z ** 2) ** (1 / 2)


def sr_gamma_betas(beta_x=0, beta_y=0, beta_z=0):
    """gamma and the three gamma * betas used in special relativity."""

    g = sr_gamma(beta_x, beta_y, beta_z)

    return [g, g * beta_x, g * beta_y, g * beta_z]


# ## Quaternions for Hamilton

# Define a class QH to manipulate quaternions as Hamilton would have done it so many years ago. The "qtype" is a little bit of text to leave a trail of breadcrumbs about how a particular quaternion was generated.




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
                t=self.t, x=self.x, y=self.y, z=self.z, qt=qtype
            )

        elif self.representation == "polar":
            rep = self.txyz_2_representation("polar")
            string = "({A} A, {thetaX} 𝜈x, {thetaY} 𝜈y, {thetaZ} 𝜈z) {qt}".format(
                A=rep[0], thetaX=rep[1], thetaY=rep[2], thetaZ=rep[3], qt=qtype
            )

        elif self.representation == "spherical":
            rep = self.txyz_2_representation("spherical")
            string = "({t} t, {R} R, {theta} θ, {phi} φ) {qt}".format(
                t=rep[0], R=rep[1], theta=rep[2], phi=rep[3], qt=qtype
            )

        return string

    def print_state(self, label, spacer=False, quiet=True):
        """Utility for printing a quaternion."""

        print(label)

        print(self.__str__(quiet))

        if spacer:
            print("")

    def is_symbolic(self):
        """Figures out if an expression has symbolic terms."""

        symbolic = False

        if (
            hasattr(self.t, "free_symbols")
            or hasattr(self.x, "free_symbols")
            or hasattr(self.y, "free_symbols")
            or hasattr(self.z, "free_symbols")
        ):
            symbolic = True

        return symbolic

    def txyz_2_representation(self, representation):
        """Converts Cartesian txyz into an array of 4 values in a different representation."""

        symbolic = self.is_symbolic()

        if representation == "":
            rep = [self.t, self.x, self.y, self.z]

        elif representation == "polar":
            amplitude = (self.t ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2) ** (
                1 / 2
            )

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

            R = (self.x ** 2 + self.y ** 2 + self.z ** 2) ** (1 / 2)

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

        if (
            hasattr(self.t, "free_symbols")
            or hasattr(self.x, "free_symbols")
            or hasattr(self.y, "free_symbols")
            or hasattr(self.z, "free_symbols")
        ):
            symbolic = True

        if representation == "":
            t, x, y, z = self.t, self.x, self.y, self.z

        elif representation == "polar":
            amplitude, thetaX, thetaY, thetaZ = self.t, self.x, self.y, self.z

            theta = (thetaX ** 2 + thetaY ** 2 + thetaZ ** 2) ** (1 / 2)

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
            raise Exception(
                "Oops, 2 quaternions have different representations: {}, {}".format(
                    self.representation, q1.representation
                )
            )
            return False

    def display_q(self, label=""):
        """Display each terms in a pretty way."""

        if label:
            print(label)
        display(self.t)
        display(self.x)
        display(self.y)
        display(self.z)
        return

    def simple_q(self, label=""):
        """Simplify each term."""

        if label:
            print(label)
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

    def subs(self, symbol_value_dict):
        """Evaluates a quaternion using sympy values and a dictionary {t:1, x:2, etc}."""

        t1 = self.t.subs(symbol_value_dict)
        x1 = self.x.subs(symbol_value_dict)
        y1 = self.y.subs(symbol_value_dict)
        z1 = self.z.subs(symbol_value_dict)

        q_txyz = QH(
            [t1, x1, y1, z1], qtype=self.qtype, representation=self.representation
        )

        return q_txyz

    def scalar(self, qtype="scalar"):
        """Returns the scalar part of a quaternion."""

        end_qtype = "scalar({})".format(self.qtype)

        s = QH([self.t, 0, 0, 0], qtype=end_qtype, representation=self.representation)
        return s

    def vector(self, qtype="v"):
        """Returns the vector part of a quaternion."""

        end_qtype = "vector({})".format(self.qtype)

        v = QH(
            [0, self.x, self.y, self.z],
            qtype=end_qtype,
            representation=self.representation,
        )
        return v

    def xyz(self):
        """Returns the vector as an np.array."""

        return np.array([self.x, self.y, self.z])

    def q_0(self, qtype="0"):
        """Return a zero quaternion."""

        q0 = QH([0, 0, 0, 0], qtype=qtype, representation=self.representation)
        return q0

    def q_1(self, n=1, qtype="1"):
        """Return a multiplicative identity quaternion."""

        q1 = QH([n, 0, 0, 0], qtype=qtype, representation=self.representation)
        return q1

    def q_i(self, n=1, qtype="i"):
        """Return i."""

        qi = QH([0, n, 0, 0], qtype=qtype, representation=self.representation)
        return qi

    def q_j(self, n=1, qtype="j"):
        """Return j."""

        qj = QH([0, 0, n, 0], qtype=qtype, representation=self.representation)
        return qj

    def q_k(self, n=1, qtype="k"):
        """Return k."""

        qk = QH([0, 0, 0, n], qtype=qtype, representation=self.representation)
        return qk

    def q_random(self, qtype="?"):
        """Return a random-valued quaternion."""

        qr = QH(
            [random.random(), random.random(), random.random(), random.random()],
            qtype=qtype,
        )
        return qr

    def dupe(self, qtype=""):
        """Return a duplicate copy, good for testing since qtypes persist"""

        du = QH(
            [self.t, self.x, self.y, self.z],
            qtype=self.qtype,
            representation=self.representation,
        )
        return du

    def equals(self, q1):
        """Tests if two quaternions are equal."""

        self.check_representations(q1)

        self_t, self_x, self_y, self_z = (
            sp.expand(self.t),
            sp.expand(self.x),
            sp.expand(self.y),
            sp.expand(self.z),
        )
        q1_t, q1_x, q1_y, q1_z = (
            sp.expand(q1.t),
            sp.expand(q1.x),
            sp.expand(q1.y),
            sp.expand(q1.z),
        )

        if (
            math.isclose(self_t, q1_t)
            and math.isclose(self_x, q1_x)
            and math.isclose(self_y, q1_y)
            and math.isclose(self_z, q1_z)
        ):
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

    def conj_q(self, q1):
        """Given a quaternion with 0's or 1's, will do the standard conjugate, first conjugate
           second conjugate, sign flip, or all combinations of the above."""

        _conj = deepcopy(self)

        if q1.t:
            _conj = _conj.conj(conj_type=0)

        if q1.x:
            _conj = _conj.conj(conj_type=1)

        if q1.y:
            _conj = _conj.conj(conj_type=2)

        if q1.z:
            _conj = _conj.flip_signs()

        return _conj

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

        if conj_type == "-":
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

        if conj_type == "*":
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

        products = {
            "tt": s_t * q1_t,
            "xx+yy+zz": s_x * q1_x + s_y * q1_y + s_z * q1_z,
            "tx+xt": s_t * q1_x + s_x * q1_t,
            "ty+yt": s_t * q1_y + s_y * q1_t,
            "tz+zt": s_t * q1_z + s_z * q1_t,
        }

        return products

    def _anti_commuting_products(self, q1):
        """Returns a dictionary with the three anti-commuting products."""

        s_x, s_y, s_z = self.x, self.y, self.z
        q1_x, q1_y, q1_z = q1.x, q1.y, q1.z

        products = {
            "yz-zy": s_y * q1_z - s_z * q1_y,
            "zx-xz": s_z * q1_x - s_x * q1_z,
            "xy-yx": s_x * q1_y - s_y * q1_x,
            "zy-yz": -s_y * q1_z + s_z * q1_y,
            "xz-zx": -s_z * q1_x + s_x * q1_z,
            "yx-xy": -s_x * q1_y + s_y * q1_x,
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
        sq_q.t = qxq["tt"] - qxq["xx+yy+zz"]
        sq_q.x = qxq["tx+xt"]
        sq_q.y = qxq["ty+yt"]
        sq_q.z = qxq["tz+zt"]

        return sq_q

    def norm_squared(self, qtype="|| ||^2"):
        """The norm_squared of a quaternion."""

        end_qtype = "||{}||^2".format(self.qtype, qtype)

        qxq = self._commuting_products(self)

        n_q = QH(qtype=end_qtype, representation=self.representation)
        n_q.t = qxq["tt"] + qxq["xx+yy+zz"]

        return n_q

    def norm_squared_of_vector(self, qtype="|V( )|^2"):
        """The norm_squared of the vector of a quaternion."""

        end_qtype = "|V({})|^2".format(self.qtype)

        qxq = self._commuting_products(self)

        nv_q = QH(qtype=end_qtype, representation=self.representation)
        nv_q.t = qxq["xx+yy+zz"]

        return nv_q

    def abs_of_q(self, qtype="||"):
        """The absolute value, the square root of the norm_squared."""

        end_qtype = "|{}|".format(self.qtype)

        a = self.norm_squared()
        sqrt_t = a.t ** (1 / 2)
        a.t = sqrt_t
        a.qtype = end_qtype
        a.representation = self.representation

        return a

    def normalize(self, n=1, qtype="U"):
        """Normalize a quaternion"""

        end_qtype = "{}{}".format(self.qtype, qtype)

        abs_q_inv = self.abs_of_q().inverse()
        n_q = self.product(abs_q_inv).product(QH([n, 0, 0, 0]))
        n_q.qtype = end_qtype
        n_q.representation = self.representation

        return n_q

    def abs_of_vector(self, qtype="|V( )|"):
        """The absolute value of the vector, the square root of the norm_squared of the vector."""

        end_qtype = "|V({})|".format(self.qtype)

        av = self.norm_squared_of_vector(qtype=end_qtype)
        sqrt_t = av.t ** (1 / 2)
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
        q_even.t = commuting["tt"] - commuting["xx+yy+zz"]
        q_even.x = commuting["tx+xt"]
        q_even.y = commuting["ty+yt"]
        q_even.z = commuting["tz+zt"]

        anti_commuting = self._anti_commuting_products(q1)
        q_odd = QH()

        if reverse:
            q_odd.x = anti_commuting["zy-yz"]
            q_odd.y = anti_commuting["xz-zx"]
            q_odd.z = anti_commuting["yx-xy"]

        else:
            q_odd.x = anti_commuting["yz-zy"]
            q_odd.y = anti_commuting["zx-xz"]
            q_odd.z = anti_commuting["xy-yx"]

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
            raise Exception(
                "Four 'kind' values are known: '', 'even', 'odd', and 'even_minus_odd'."
            )

        if reverse:
            times_symbol = times_symbol.replace("x", "xR")

        if qtype:
            result.qtype = qtype
        else:
            result.qtype = "{f}{ts}{s}".format(
                f=self.qtype, ts=times_symbol, s=q1.qtype
            )

        result.representation = self.representation

        return result

    def Euclidean_product(self, q1, kind="", reverse=False, qtype=""):
        """Form a product p* q given 2 quaternions, not associative."""

        self.check_representations(q1)

        pq = QH(qtype, representation=self.representation)
        pq = self.conj().product(q1, kind, reverse)

        return pq

    def inverse(self, qtype="^-1", additive=False):
        """The additive or multiplicative inverse of a quaternion."""

        if additive:
            end_qtype = "-{}".format(self.qtype, qtype)
            q_inv = self.flip_signs()
            q_inv.qtype = end_qtype

        else:
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

        q1_inv = q1.inverse()
        q_div = self.product(q1.inverse())
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

    # Quaternion rotation involves a triple product:  u R 1/u
    def rotate(self, u, qtype="rot"):
        """Do a rotation using a triple product: u R 1/u."""

        end_qtype = "{}{}".format(self.qtype, qtype)

        u_abs = u.abs_of_q()
        u_norm_squaredalized = u.divide_by(u_abs)

        q_rot = u_norm_squaredalized.triple_product(self, u_norm_squaredalized.conj())
        q_rot.qtype = end_qtype
        q_rot.representation = self.representation

        return q_rot

    # A boost also uses triple products like a rotation, but more of them.
    # This is not a well-known result, but does work.
    # b -> b' = h b h* + 1/2 ((hhb)* -(h*h*b)*)
    # where h is of the form (cosh(a), sinh(a)) OR (0, a, b, c)
    def boost(self, h, qtype="boost"):
        """A boost or rotation or both."""

        end_qtype = "{}{}".format(self.qtype, qtype)

        boost = h
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

        if abs_v.t == 0:
            return QH([et, 0, 0, 0], qtype=end_qtype)

        cosR = math.cos(abs_v.t)
        sinR = math.sin(abs_v.t)
        k = et * sinR / abs_v.t

        expq = QH(
            [et * cosR, k * self.x, k * self.y, k * self.z],
            qtype=end_qtype,
            representation=self.representation,
        )

        return expq

    def ln(self, qtype="ln"):
        """Take the natural log of a quaternion."""
        # ln(q) = (0.5 ln t^2 + R.R, atan2(|R|, t) R/|R|)

        end_qtype = "ln({st})".format(st=self.qtype)

        abs_v = self.abs_of_vector()

        if abs_v.t == 0:
            if self.t > 0:
                return QH([math.log(self.t), 0, 0, 0], qtype=end_qtype)
            else:
                # I don't understant this, but mathematica does the same thing.
                return QH([math.log(-self.t), math.pi, 0, 0], qtype=end_type)

            return QH([lt, 0, 0, 0])

        t_value = 0.5 * math.log(self.t * self.t + abs_v.t * abs_v.t)
        k = math.atan2(abs_v.t, self.t) / abs_v.t

        expq = QH(
            [t_value, k * self.x, k * self.y, k * self.z],
            qtype=end_qtype,
            representation=self.representation,
        )

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




if __name__ == "__main__":

    class TestQH(unittest.TestCase):
        """Class to make sure all the functions work as expected."""

        Q = QH([1, -2, -3, -4], qtype="Q")
        P = QH([0, 4, -3, 0], qtype="P")
        R = QH([3, 0, 0, 0], qtype="R")
        C = QH([2, 4, 0, 0], qtype="C")
        t, x, y, z = sp.symbols("t x y z")
        q_sym = QH([t, x, y, x * y * z])

        def test_qt(self):
            self.assertTrue(self.Q.t == 1)

        def test_subs(self):
            q_z = self.q_sym.subs({self.t: 1, self.x: 2, self.y: 3, self.z: 4})
            print("t x y xyz sub 1 2 3 4: ", q_z)
            self.assertTrue(q_z.equals(QH([1, 2, 3, 24])))

        def test_scalar(self):
            q_z = self.Q.scalar()
            print("scalar(q): ", q_z)
            self.assertTrue(q_z.t == 1)
            self.assertTrue(q_z.x == 0)
            self.assertTrue(q_z.y == 0)
            self.assertTrue(q_z.z == 0)

        def test_vector(self):
            q_z = self.Q.vector()
            print("vector(q): ", q_z)
            self.assertTrue(q_z.t == 0)
            self.assertTrue(q_z.x == -2)
            self.assertTrue(q_z.y == -3)
            self.assertTrue(q_z.z == -4)

        def test_xyz(self):
            q_z = self.Q.xyz()
            print("q.xyz()): ", q_z)
            self.assertTrue(q_z[0] == -2)
            self.assertTrue(q_z[1] == -3)
            self.assertTrue(q_z[2] == -4)

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

        def test_conj_q(self):
            q_z = self.Q.conj_q(self.Q)
            print("conj_q(conj_q): ", q_z)
            self.assertTrue(q_z.t == -1)
            self.assertTrue(q_z.x == 2)
            self.assertTrue(q_z.y == 3)
            self.assertTrue(q_z.z == -4)

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
            q_z = self.Q.vahlen_conj("*")
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

        def test_inverse(self):
            q_z = self.P.inverse()
            print("inverse: ", q_z)
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
            q_z = self.Q.rotate(QH([0, 1, 0, 0]))
            print("rotate: ", q_z)
            self.assertTrue(q_z.t == 1)
            self.assertTrue(q_z.x == -2)
            self.assertTrue(q_z.y == 3)
            self.assertTrue(q_z.z == 4)

        def test_boost(self):
            q1_sq = self.Q.square()
            h = QH(sr_gamma_betas(0.003))
            q_z = self.Q.boost(h)
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
            self.assertTrue(
                self.Q.sin().equals(
                    QH(
                        [
                            91.7837157840346691,
                            -21.8864868530291758,
                            -32.8297302795437673,
                            -43.7729737060583517,
                        ]
                    )
                )
            )
            self.assertTrue(
                self.P.sin().equals(
                    QH([0, 59.3625684622310033, -44.5219263466732542, 0])
                )
            )
            self.assertTrue(self.R.sin().equals(QH([0.1411200080598672, 0, 0, 0])))
            self.assertTrue(
                self.C.sin().equals(
                    QH([24.8313058489463785, -11.3566127112181743, 0, 0])
                )
            )

        def test_cos(self):
            self.assertTrue(QH([0, 0, 0, 0]).cos().equals(QH().q_1()))
            self.assertTrue(
                self.Q.cos().equals(
                    QH(
                        [
                            58.9336461679439481,
                            34.0861836904655959,
                            51.1292755356983974,
                            68.1723673809311919,
                        ]
                    )
                )
            )
            self.assertTrue(self.P.cos().equals(QH([74.2099485247878476, 0, 0, 0])))
            self.assertTrue(self.R.cos().equals(QH([-0.9899924966004454, 0, 0, 0])))
            self.assertTrue(
                self.C.cos().equals(
                    QH([-11.3642347064010600, -24.8146514856341867, 0, 0])
                )
            )

        def test_tan(self):
            self.assertTrue(QH([0, 0, 0, 0]).tan().equals(QH().q_0()))
            self.assertTrue(
                self.Q.tan().equals(
                    QH(
                        [
                            0.0000382163172501,
                            -0.3713971716439372,
                            -0.5570957574659058,
                            -0.7427943432878743,
                        ]
                    )
                )
            )
            self.assertTrue(
                self.P.tan().equals(QH([0, 0.7999273634100760, -0.5999455225575570, 0]))
            )
            self.assertTrue(self.R.tan().equals(QH([-0.1425465430742778, 0, 0, 0])))
            self.assertTrue(
                self.C.tan().equals(QH([-0.0005079806234700, 1.0004385132020521, 0, 0]))
            )

        def test_sinh(self):
            self.assertTrue(QH([0, 0, 0, 0]).sinh().equals(QH().q_0()))
            self.assertTrue(
                self.Q.sinh().equals(
                    QH(
                        [
                            0.7323376060463428,
                            0.4482074499805421,
                            0.6723111749708131,
                            0.8964148999610841,
                        ]
                    )
                )
            )
            self.assertTrue(
                self.P.sinh().equals(
                    QH([0, -0.7671394197305108, 0.5753545647978831, 0])
                )
            )
            self.assertTrue(self.R.sinh().equals(QH([10.0178749274099026, 0, 0, 0])))
            self.assertTrue(
                self.C.sinh().equals(
                    QH([-2.3706741693520015, -2.8472390868488278, 0, 0])
                )
            )

        def test_cosh(self):
            self.assertTrue(QH([0, 0, 0, 0]).cosh().equals(QH().q_1()))
            self.assertTrue(
                self.Q.cosh().equals(
                    QH(
                        [
                            0.9615851176369565,
                            0.3413521745610167,
                            0.5120282618415251,
                            0.6827043491220334,
                        ]
                    )
                )
            )
            self.assertTrue(self.P.cosh().equals(QH([0.2836621854632263, 0, 0, 0])))
            self.assertTrue(self.R.cosh().equals(QH([10.0676619957777653, 0, 0, 0])))
            self.assertTrue(
                self.C.cosh().equals(
                    QH([-2.4591352139173837, -2.7448170067921538, 0, 0])
                )
            )

        def test_tanh(self):
            self.assertTrue(QH([0, 0, 0, 0]).tanh().equals(QH().q_0()))
            self.assertTrue(
                self.Q.tanh().equals(
                    QH(
                        [
                            1.0248695360556623,
                            0.1022956817887642,
                            0.1534435226831462,
                            0.2045913635775283,
                        ]
                    )
                )
            )
            self.assertTrue(
                self.P.tanh().equals(
                    QH([0, -2.7044120049972684, 2.0283090037479505, 0])
                )
            )
            self.assertTrue(self.R.tanh().equals(QH([0.9950547536867305, 0, 0, 0])))
            self.assertTrue(
                self.C.tanh().equals(QH([1.0046823121902353, 0.0364233692474038, 0, 0]))
            )

        def test_exp(self):
            self.assertTrue(QH([0, 0, 0, 0]).exp().equals(QH().q_1()))
            self.assertTrue(
                self.Q.exp().equals(
                    QH(
                        [
                            1.6939227236832994,
                            0.7895596245415588,
                            1.1843394368123383,
                            1.5791192490831176,
                        ]
                    )
                )
            )
            self.assertTrue(
                self.P.exp().equals(
                    QH([0.2836621854632263, -0.7671394197305108, 0.5753545647978831, 0])
                )
            )
            self.assertTrue(self.R.exp().equals(QH([20.0855369231876679, 0, 0, 0])))
            self.assertTrue(
                self.C.exp().equals(
                    QH([-4.8298093832693851, -5.5920560936409816, 0, 0])
                )
            )

        def test_ln(self):
            self.assertTrue(self.Q.ln().exp().equals(self.Q))
            self.assertTrue(
                self.Q.ln().equals(
                    QH(
                        [
                            1.7005986908310777,
                            -0.5151902926640850,
                            -0.7727854389961275,
                            -1.0303805853281700,
                        ]
                    )
                )
            )
            self.assertTrue(
                self.P.ln().equals(
                    QH([1.6094379124341003, 1.2566370614359172, -0.9424777960769379, 0])
                )
            )
            self.assertTrue(self.R.ln().equals(QH([1.0986122886681098, 0, 0, 0])))
            self.assertTrue(
                self.C.ln().equals(QH([1.4978661367769954, 1.1071487177940904, 0, 0]))
            )

        def test_q_2_q(self):
            self.assertTrue(
                self.Q.q_2_q(self.P).equals(
                    QH(
                        [
                            -0.0197219653530713,
                            -0.2613955437374326,
                            0.6496281248064009,
                            -0.3265786562423951,
                        ]
                    )
                )
            )

    suite = unittest.TestLoader().loadTestsFromModule(TestQH())
    _results = unittest.TextTestRunner().run(suite)





if __name__ == "__main__":

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
            self.assertTrue(
                qr.equals(
                    QH([1.0, 3.7416573867739413, 0.640522312679424, 1.10714871779409])
                )
            )

        def test_representation_2_txyz(self):
            qr = QH(self.Q12.representation_2_txyz(""))
            self.assertTrue(qr.equals(self.Q12))
            qr = QH(self.Q12.representation_2_txyz("polar"))
            self.assertTrue(
                qr.equals(QH([-0.4161468365471424, 0.9092974268256817, 0, 0]))
            )
            qr = QH(self.Q1123.representation_2_txyz("spherical"))
            self.assertTrue(
                qr.equals(
                    QH(
                        [
                            1.0,
                            -0.9001976297355174,
                            0.12832006020245673,
                            -0.4161468365471424,
                        ]
                    )
                )
            )

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
    _results = unittest.TextTestRunner().run(suite)


# ## QHStates - n quaternions that are a semi-group with inverses

# Any quaternion can be viewed as the sum of n other quaternions. This is common to see in quantum mechanics, whose needs are driving the development of this class and its methods.




class QHStates(QH):
    """A class made up of many quaternions."""

    QS_TYPES = ["scalar", "bra", "ket", "op", "operator"]

    def __init__(self, qs=None, qs_type="ket", rows=0, columns=0):

        self.qs = qs
        self.qs_type = qs_type
        self.rows = rows
        self.columns = columns
        self.qtype = ""

        if qs_type not in self.QS_TYPES:
            print(
                "Oops, only know of these quaternion series types: {}".format(
                    self.QS_TYPES
                )
            )
            return None

        if qs is None:
            self.d, self.dim, self.dimensions = 0, 0, 0
        else:
            self.d, self.dim, self.dimensions = int(len(qs)), int(len(qs)), int(len(qs))

        self.set_qs_type(qs_type, rows, columns, copy=False)

    def set_qs_type(self, qs_type="", rows=0, columns=0, copy=True):
        """Set the qs_type to something sensible."""

        # Checks.
        if (rows) and (columns) and rows * columns != self.dim:
            print(
                "Oops, check those values again for rows:{} columns:{} dim:{}".format(
                    rows, columns, self.dim
                )
            )
            self.qs, self.rows, self.columns = None, 0, 0
            return None

        new_q = self

        if copy:
            new_q = deepcopy(self)

        # Assign values if need be.
        if new_q.qs_type != qs_type:
            new_q.rows = 0

        if qs_type == "ket" and not new_q.rows:
            new_q.rows = new_q.dim
            new_q.columns = 1

        elif qs_type == "bra" and not new_q.rows:
            new_q.rows = 1
            new_q.columns = new_q.dim

        elif qs_type in ["op", "operator"] and not new_q.rows:
            # Square series
            root_dim = math.sqrt(new_q.dim)

            if root_dim.is_integer():
                new_q.rows = int(root_dim)
                new_q.columns = int(root_dim)
                qs_type = "op"

        elif rows * columns == new_q.dim and not new_q.qs_type:
            if new_q.dim == 1:
                qs_type = "scalar"
            elif new_q.rows == 1:
                qs_type = "bra"
            elif new_q.columns == 1:
                qs_type = "ket"
            else:
                qs_type = "op"

        if not qs_type:
            print(
                "Oops, please set rows and columns for this quaternion series operator. Thanks."
            )
            return None

        if new_q.dim == 1:
            qs_type = "scalar"

        new_q.qs_type = qs_type

        return new_q

    def bra(self):
        """Quickly set the qs_type to bra by calling set_qs_type()."""

        if self.qs_type == "bra":
            return self

        bra = deepcopy(self).conj()
        bra.rows = 1
        bra.columns = self.dim

        if self.dim > 1:
            bra.qs_type = "bra"

        return bra

    def ket(self):
        """Quickly set the qs_type to ket by calling set_qs_type()."""

        if self.qs_type == "ket":
            return self

        ket = deepcopy(self).conj()
        ket.rows = self.dim
        ket.columns = 1

        if self.dim > 1:
            ket.qs_type = "ket"

        return ket

    def op(self, rows, columns):
        """Quickly set the qs_type to op by calling set_qs_type()."""

        if rows * columns != self.dim:
            print(
                "Oops, rows * columns != dim: {} * {}, {}".formaat(
                    rows, columns, self.dim
                )
            )
            return None

        op_q = deepcopy(self)

        op_q.rows = rows
        op_q.columns = columns

        if self.dim > 1:
            op_q.qs_type = "op"

        return op_q

    def __str__(self, quiet=False):
        """Print out all the states."""

        states = ""

        for n, q in enumerate(self.qs, start=1):
            states = states + "n={}: {}\n".format(n, q.__str__(quiet))

        return states.rstrip()

    def print_state(self, label, spacer=True, quiet=True, sum=False):
        """Utility for printing states as a quaternion series."""

        print(label)

        # Warn if empty.
        if self.qs is None or len(self.qs) == 0:
            print("Oops, no quaternions in the series.")
            return

        for n, q in enumerate(self.qs):
            print("n={}: {}".format(n + 1, q.__str__(quiet)))

        if sum:
            print("sum= {ss}".format(ss=self.summation()))

        print("{t}: {r}/{c}".format(t=self.qs_type, r=self.rows, c=self.columns))

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

        for ket in self.qs:
            new_states.append(ket.conj(conj_type))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def conj_q(self, q1):
        """Does multicate conjugate operators."""

        new_states = []

        for ket in self.qs:
            new_states.append(ket.conj_q(q1))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def display_q(self, label):
        """Try to display algebra in a pretty way."""

        if label:
            print(label)

        for i, ket in enumerate(self.qs, start=1):
            print(f"n={i}")
            ket.display_q()
            print("")

    def simple_q(self):
        """Simplify the states."""

        new_states = []

        for ket in self.qs:
            new_states.append(ket.simple_q())

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def subs(self, symbol_value_dict, qtype="scalar"):
        """Substitutes values into ."""

        new_states = []

        for ket in self.qs:
            new_states.append(ket.subs(symbol_value_dict))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def scalar(self, qtype="scalar"):
        """Returns the scalar part of a quaternion."""

        new_states = []

        for ket in self.qs:
            new_states.append(ket.scalar())

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def vector(self, qtype="v"):
        """Returns the vector part of a quaternion."""

        new_states = []

        for ket in self.qs:
            new_states.append(ket.vector())

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def xyz(self):
        """Returns the vector as an np.array."""

        new_states = []

        for ket in self.qs:
            new_states.append(ket.xyz())

        return new_states

    def flip_signs(self):
        """Flip signs of all states."""

        new_states = []

        for ket in self.qs:
            new_states.append(ket.flip_signs())

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def inverse(self, additive=False):
        """Inverseing bras and kets calls inverse() once for each.
        Inverseing operators is more tricky as one needs a diagonal identity matrix."""

        if self.qs_type in ["op", "operator"]:

            if additive:

                q_flip = self.inverse(additive=True)
                q_inv = q_flip.diagonal(self.dim)

            else:
                if self.dim == 1:
                    q_inv = QHStates(self.qs[0].inverse())

                elif self.qs_type in ["bra", "ket"]:

                    new_qs = []

                    for q in self.qs:
                        new_qs.append(q.inverse())

                    q_inv = QHStates(
                        new_qs,
                        qs_type=self.qs_type,
                        rows=self.rows,
                        columns=self.columns,
                    )

                elif self.dim == 4:
                    det = self.determinant()
                    detinv = det.inverse()

                    q0 = self.qs[3].product(detinv)
                    q1 = self.qs[1].flip_signs().product(detinv)
                    q2 = self.qs[2].flip_signs().product(detinv)
                    q3 = self.qs[0].product(detinv)

                    q_inv = QHStates(
                        [q0, q1, q2, q3],
                        qs_type=self.qs_type,
                        rows=self.rows,
                        columns=self.columns,
                    )

                elif self.dim == 9:
                    det = self.determinant()
                    detinv = det.inverse()

                    q0 = (
                        self.qs[4]
                        .product(self.qs[8])
                        .dif(self.qs[5].product(self.qs[7]))
                        .product(detinv)
                    )
                    q1 = (
                        self.qs[7]
                        .product(self.qs[2])
                        .dif(self.qs[8].product(self.qs[1]))
                        .product(detinv)
                    )
                    q2 = (
                        self.qs[1]
                        .product(self.qs[5])
                        .dif(self.qs[2].product(self.qs[4]))
                        .product(detinv)
                    )
                    q3 = (
                        self.qs[6]
                        .product(self.qs[5])
                        .dif(self.qs[8].product(self.qs[3]))
                        .product(detinv)
                    )
                    q4 = (
                        self.qs[0]
                        .product(self.qs[8])
                        .dif(self.qs[2].product(self.qs[6]))
                        .product(detinv)
                    )
                    q5 = (
                        self.qs[3]
                        .product(self.qs[2])
                        .dif(self.qs[5].product(self.qs[0]))
                        .product(detinv)
                    )
                    q6 = (
                        self.qs[3]
                        .product(self.qs[7])
                        .dif(self.qs[4].product(self.qs[6]))
                        .product(detinv)
                    )
                    q7 = (
                        self.qs[6]
                        .product(self.qs[1])
                        .dif(self.qs[7].product(self.qs[0]))
                        .product(detinv)
                    )
                    q8 = (
                        self.qs[0]
                        .product(self.qs[4])
                        .dif(self.qs[1].product(self.qs[3]))
                        .product(detinv)
                    )

                    q_inv = QHStates(
                        [q0, q1, q2, q3, q4, q5, q6, q7, q8],
                        qs_type=self.qs_type,
                        rows=self.rows,
                        columns=self.columns,
                    )

                else:
                    print("Oops, don't know how to inverse.")
                    q_inv = QHStates([QH().q_0()])

        else:
            new_states = []

            for bra in self.qs:
                new_states.append(bra.inverse(additive=additive))

            q_inv = QHStates(
                new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
            )

        return q_inv

    def norm(self):
        """Norm of states."""

        new_states = []

        for bra in self.qs:
            new_states.append(bra.norm())

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def normalize(self, n=1, states=None):
        """Normalize all states."""

        new_states = []

        zero_norm_count = 0

        for bra in self.qs:
            if bra.norm_squared().t == 0:
                zero_norm_count += 1
                new_states.append(QH().q_0())
            else:
                new_states.append(bra.normalize(n))

        new_states_normalized = []

        non_zero_states = self.dim - zero_norm_count

        for new_state in new_states:
            new_states_normalized.append(
                new_state.product(QH([math.sqrt(1 / non_zero_states), 0, 0, 0]))
            )

        return QHStates(
            new_states_normalized,
            qs_type=self.qs_type,
            rows=self.rows,
            columns=self.columns,
        )

    def orthonormalize(self):
        """Given a quaternion series, resturn a normalized orthoganl basis."""

        last_q = self.qs.pop(0).normalize(math.sqrt(1 / self.dim))
        orthonormal_qs = [last_q]

        for q in self.qs:
            qp = q.Euclidean_product(last_q)
            orthonormal_q = q.dif(qp).normalize(math.sqrt(1 / self.dim))
            orthonormal_qs.append(orthonormal_q)
            last_q = orthonormal_q

        return QHStates(
            orthonormal_qs, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def determinant(self):
        """Calculate the determinant of a 'square' quaternion series."""

        if self.dim == 1:
            q_det = self.qs[0]

        elif self.dim == 4:
            ad = self.qs[0].product(self.qs[3])
            bc = self.qs[1].product(self.qs[2])
            q_det = ad.dif(bc)

        elif self.dim == 9:
            aei = self.qs[0].product(self.qs[4].product(self.qs[8]))
            bfg = self.qs[3].product(self.qs[7].product(self.qs[2]))
            cdh = self.qs[6].product(self.qs[1].product(self.qs[5]))
            ceg = self.qs[6].product(self.qs[4].product(self.qs[2]))
            bdi = self.qs[3].product(self.qs[1].product(self.qs[8]))
            afh = self.qs[0].product(self.qs[7].product(self.qs[5]))

            sum_pos = aei.add(bfg.add(cdh))
            sum_neg = ceg.add(bdi.add(afh))

            q_det = sum_pos.dif(sum_neg)

        else:
            print("Oops, don't know how to calculate the determinant of this one.")
            return None

        return q_det

    def add(self, ket):
        """Add two states."""

        if (self.rows != ket.rows) or (self.columns != ket.columns):
            print("Oops, can only add if rows and columns are the same.")
            print(
                "rows are: {}/{}, columns are: {}/{}".format(
                    self.rows, ket.rows, self.columns, ket.columns
                )
            )
            return None

        new_states = []

        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(bra.add(ket))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

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

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def diagonal(self, dim):
        """Make a state dim*dim with q or qs along the 'diagonal'. Always returns an operator."""

        diagonal = []

        if len(self.qs) == 1:
            q_values = [self.qs[0]] * dim
        elif len(self.qs) == dim:
            q_values = self.qs
        elif self.qs is None:
            print("Oops, the qs here is None.")
            return None
        else:
            print("Oops, need the length to be equal to the dimensions.")
            return None

        for i in range(dim):
            for j in range(dim):
                if i == j:
                    diagonal.append(q_values.pop(0))
                else:
                    diagonal.append(QH().q_0())

        return QHStates(diagonal, qs_type="op", rows=dim, columns=dim)

    @staticmethod
    def identity(dim, operator=False, additive=False, non_zeroes=None, qs_type="ket"):
        """Identity operator for states or operators which are diagonal."""

        if additive:
            id_q = [QH().q_0() for i in range(dim)]

        elif non_zeroes is not None:
            id_q = []

            if len(non_zeroes) != dim:
                print(
                    "Oops, len(non_zeroes)={nz}, should be: {d}".format(
                        nz=len(non_zeroes), d=dim
                    )
                )
                return QHStates([QH().q_0()])

            else:
                for non_zero in non_zeroes:
                    if non_zero:
                        id_q.append(QH().q_1())
                    else:
                        id_q.append(QH().q_0())

        else:
            id_q = [QH().q_1() for i in range(dim)]

        if operator:
            q_1 = QHStates(id_q)
            ident = QHStates.diagonal(q_1, dim)

        else:
            ident = QHStates(id_q, qs_type=qs_type)

        return ident

    def product(self, q1, kind="", reverse=False):
        """Forms the quaternion product for each state."""

        self_copy = deepcopy(self)
        q1_copy = deepcopy(q1)

        # Diagonalize if need be.
        if ((self.rows == q1.rows) and (self.columns == q1.columns)) or (
            "scalar" in [self.qs_type, q1.qs_type]
        ):

            if self.columns == 1:
                qs_right = q1_copy
                qs_left = self_copy.diagonal(qs_right.rows)

            elif q1.rows == 1:
                qs_left = self_copy
                qs_right = q1_copy.diagonal(qs_left.columns)

            else:
                qs_left = self_copy
                qs_right = q1_copy

        # Typical matrix multiplication criteria.
        elif self.columns == q1.rows:
            qs_left = self_copy
            qs_right = q1_copy

        else:
            print(
                "Oops, cannot multiply series with row/column dimensions of {}/{} to {}/{}".format(
                    self.rows, self.columns, q1.rows, q1.columns
                )
            )
            return None

        # Operator products need to be transposed.
        operator_flag = False
        if qs_left in ["op", "operator"] and qs_right in ["op", "operator"]:
            operator_flag = True

        outer_row_max = qs_left.rows
        outer_column_max = qs_right.columns
        shared_inner_max = qs_left.columns
        projector_flag = (
            (shared_inner_max == 1) and (outer_row_max > 1) and (outer_column_max > 1)
        )

        result = [
            [QH().q_0(qtype="") for i in range(outer_column_max)]
            for j in range(outer_row_max)
        ]

        for outer_row in range(outer_row_max):
            for outer_column in range(outer_column_max):
                for shared_inner in range(shared_inner_max):

                    # For projection operators.
                    left_index = outer_row
                    right_index = outer_column

                    if outer_row_max >= 1 and shared_inner_max > 1:
                        left_index = outer_row + shared_inner * outer_row_max

                    if outer_column_max >= 1 and shared_inner_max > 1:
                        right_index = shared_inner + outer_column * shared_inner_max

                    result[outer_row][outer_column] = result[outer_row][
                        outer_column
                    ].add(
                        qs_left.qs[left_index].product(
                            qs_right.qs[right_index], kind=kind, reverse=reverse
                        )
                    )

        # Flatten the list.
        new_qs = [item for sublist in result for item in sublist]
        new_states = QHStates(new_qs, rows=outer_row_max, columns=outer_column_max)

        if projector_flag or operator_flag:
            return new_states.transpose()

        else:
            return new_states

    def Euclidean_product(self, q1, kind="", reverse=False):
        """Forms the Euclidean product, what is used in QM all the time."""

        return self.conj().product(q1, kind, reverse)

    @staticmethod
    def bracket(bra, op, ket):
        """Forms <bra|op|ket>. Note: if fed 2 kets, will take a conjugate."""

        flip = 0

        if bra.qs_type == "ket":
            bra = bra.bra()
            flip += 1

        if ket.qs_type == "bra":
            ket = ket.ket()
            flip += 1

        if flip == 1:
            print("fed 2 bras or kets, took a conjugate. Double check.")

        else:
            print("Assumes your <bra| already has been conjugated. Double check.")

        b = bra.product(op).product(ket)

        return b

    @staticmethod
    def braket(bra, ket):
        """Forms <bra|ket>, no operator. Note: if fed 2 kets, will take a conjugate."""

        flip = 0

        if bra.qs_type == "ket":
            bra = bra.bra()
            flip += 1

        if ket.qs_type == "bra":
            ket = ket.ket()
            flip += 1

        if flip == 1:
            print("fed 2 bras or kets, took a conjugate. Double check.")

        else:
            print("Assumes your <bra| already has been conjugated. Double check.")

        b = bra.product(ket)

        return b

    def op_n(self, n, first=True, kind="", reverse=False):
        """Mulitply an operator times a number, in that order. Set first=false for n * Op"""

        new_states = []

        for op in self.qs:

            if first:
                new_states.append(op.product(n, kind, reverse))

            else:
                new_states.append(n.product(op, kind, reverse))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def norm_squared(self):
        """Take the Euclidean product of each state and add it up, returning a scalar series."""

        return self.set_qs_type("bra").Euclidean_product(self.set_qs_type("ket"))

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

        # Switch rows and columns.
        return QHStates(qs_t, rows=self.columns, columns=self.rows)

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

        sigma = {}
        sigma["x"] = QHStates([q0, x_factor, x_factor, q0], "op")
        sigma["y"] = QHStates([q0, y_factor, y_factor.flip_signs(), q0], "op")
        sigma["z"] = QHStates([z_factor, q0, q0, z_factor.flip_signs()], "op")

        sigma["xy"] = sigma["x"].add(sigma["y"])
        sigma["xz"] = sigma["x"].add(sigma["z"])
        sigma["yz"] = sigma["y"].add(sigma["z"])
        sigma["xyz"] = sigma["x"].add(sigma["y"]).add(sigma["z"])

        if kind not in sigma:
            print("Oops, I only know about x, y, z, and their combinations.")
            return None

        return sigma[kind].normalize()

    def sin(self):
        """sine of states."""

        new_states = []

        for ket in self.qs:
            new_states.append(ket.sin(qtype=""))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def cos(self):
        """cosine of states."""

        new_states = []

        for ket in self.qs:
            new_states.append(ket.cos(qtype=""))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def tan(self):
        """tan of states."""

        new_states = []

        for ket in self.qs:
            new_states.append(ket.tan(qtype=""))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def sinh(self):
        """sinh of states."""

        new_states = []

        for ket in self.qs:
            new_states.append(ket.sinh(qtype=""))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def cosh(self):
        """cosh of states."""

        new_states = []

        for ket in self.qs:
            new_states.append(ket.cosh(qtype=""))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def tanh(self):
        """tanh of states."""

        new_states = []

        for ket in self.qs:
            new_states.append(ket.tanh(qtype=""))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def exp(self):
        """exponential of states."""

        new_states = []

        for ket in self.qs:
            new_states.append(ket.exp(qtype=""))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )





if __name__ == "__main__":

    class TestQHStates(unittest.TestCase):
        """Test states."""

        q_0 = QH().q_0()
        q_1 = QH().q_1()
        q_i = QH().q_i()
        q_n1 = QH([-1, 0, 0, 0])
        q_2 = QH([2, 0, 0, 0])
        q_n2 = QH([-2, 0, 0, 0])
        q_3 = QH([3, 0, 0, 0])
        q_n3 = QH([-3, 0, 0, 0])
        q_4 = QH([4, 0, 0, 0])
        q_5 = QH([5, 0, 0, 0])
        q_6 = QH([6, 0, 0, 0])
        q_10 = QH([10, 0, 0, 0])
        q_n5 = QH([-5, 0, 0, 0])
        q_7 = QH([7, 0, 0, 0])
        q_8 = QH([8, 0, 0, 0])
        q_9 = QH([9, 0, 0, 0])
        q_n11 = QH([-11, 0, 0, 0])
        q_21 = QH([21, 0, 0, 0])
        q_n34 = QH([-34, 0, 0, 0])
        v3 = QHStates([q_3])
        v1123 = QHStates([q_1, q_1, q_2, q_3])
        v3n1n21 = QHStates([q_3, q_n1, q_n2, q_1])
        v9 = QHStates([q_1, q_1, q_2, q_3, q_1, q_1, q_2, q_3, q_2])
        v9i = QHStates(
            [
                QH([0, 1, 0, 0]),
                QH([0, 2, 0, 0]),
                QH([0, 3, 0, 0]),
                QH([0, 4, 0, 0]),
                QH([0, 5, 0, 0]),
                QH([0, 6, 0, 0]),
                QH([0, 7, 0, 0]),
                QH([0, 8, 0, 0]),
                QH([0, 9, 0, 0]),
            ]
        )
        vv9 = v9.add(v9i)
        q_1d0 = QH([1.0, 0, 0, 0])
        q12 = QHStates([q_1d0, q_1d0])
        q14 = QHStates([q_1d0, q_1d0, q_1d0, q_1d0])
        q19 = QHStates([q_1d0, q_0, q_1d0, q_1d0, q_1d0, q_1d0, q_1d0, q_1d0, q_1d0])
        qn627 = QH([-6, 27, 0, 0])
        v33 = QHStates([q_7, q_0, q_n3, q_2, q_3, q_4, q_1, q_n1, q_n2])
        v33inv = QHStates([q_n2, q_3, q_9, q_8, q_n11, q_n34, q_n5, q_7, q_21])
        q_i3 = QHStates([q_1, q_1, q_1])
        q_i2d = QHStates([q_1, q_0, q_0, q_1])
        q_i3_bra = QHStates([q_1, q_1, q_1], "bra")
        q_6_op = QHStates([q_1, q_0, q_0, q_1, q_i, q_i], "op")
        q_6_op_32 = QHStates([q_1, q_0, q_0, q_1, q_i, q_i], "op", rows=3, columns=2)
        q_i2d_op = QHStates([q_1, q_0, q_0, q_1], "op")
        q_i4 = QH([0, 4, 0, 0])
        q_0_q_1 = QHStates([q_0, q_1])
        q_1_q_0 = QHStates([q_1, q_0])
        q_1_q_i = QHStates([q_1, q_i])
        q_1_q_0 = QHStates([q_1, q_0])
        q_0_q_i = QHStates([q_0, q_i])
        A = QHStates([QH([4, 0, 0, 0]), QH([0, 1, 0, 0])], "bra")
        B = QHStates([QH([0, 0, 1, 0]), QH([0, 0, 0, 2]), QH([0, 3, 0, 0])])
        Op = QHStates(
            [
                QH([3, 0, 0, 0]),
                QH([0, 1, 0, 0]),
                QH([0, 0, 2, 0]),
                QH([0, 0, 0, 3]),
                QH([2, 0, 0, 0]),
                QH([0, 4, 0, 0]),
            ],
            "op",
            rows=2,
            columns=3,
        )
        Op4i = QHStates([q_i4, q_0, q_0, q_i4, q_2, q_3], "op", rows=2, columns=3)
        Op_scalar = QHStates([q_i4], "scalar")
        q_1234 = QHStates(
            [QH([1, 1, 0, 0]), QH([2, 1, 0, 0]), QH([3, 1, 0, 0]), QH([4, 1, 0, 0])]
        )
        sigma_y = QHStates(
            [QH([1, 0, 0, 0]), QH([0, -1, 0, 0]), QH([0, 1, 0, 0]), QH([-1, 0, 0, 0])]
        )
        qn = QHStates([QH([3, 0, 0, 4])])
        q_bad = QHStates([q_1], rows=2, columns=3)

        b = QHStates([q_1, q_2, q_3], qs_type="bra")
        k = QHStates([q_4, q_5, q_6], qs_type="ket")
        o = QHStates([q_10], qs_type="op")

        def test_1000_init(self):
            self.assertTrue(self.q_0_q_1.dim == 2)

        def test_1010_set_qs_type(self):
            bk = self.b.set_qs_type("ket")
            self.assertTrue(bk.rows == 3)
            self.assertTrue(bk.columns == 1)
            self.assertTrue(bk.qs_type == "ket")
            self.assertTrue(self.q_bad.qs is None)

        def test_1020_set_rows_and_columns(self):
            self.assertTrue(self.q_i3.rows == 3)
            self.assertTrue(self.q_i3.columns == 1)
            self.assertTrue(self.q_i3_bra.rows == 1)
            self.assertTrue(self.q_i3_bra.columns == 3)
            self.assertTrue(self.q_i2d_op.rows == 2)
            self.assertTrue(self.q_i2d_op.columns == 2)
            self.assertTrue(self.q_6_op_32.rows == 3)
            self.assertTrue(self.q_6_op_32.columns == 2)

        def test_1030_equals(self):
            self.assertTrue(self.A.equals(self.A))
            self.assertFalse(self.A.equals(self.B))

        def test_1031_subs(self):

            t, x, y, z = sp.symbols("t x y z")
            q_sym = QHStates([QH([t, x, y, x * y * z])])

            q_z = q_sym.subs({t: 1, x: 2, y: 3, z: 4})
            print("t x y xyz sub 1 2 3 4: ", q_z)
            self.assertTrue(q_z.equals(QHStates([QH([1, 2, 3, 24])])))

        def test_1032_scalar(self):
            qs = self.q_1_q_i.scalar()
            print("scalar(q_1_q_i)", qs)
            self.assertTrue(qs.equals(self.q_1_q_0))

        def test_1033_vector(self):
            qv = self.q_1_q_i.vector()
            print("vector(q_1_q_i)", qv)
            self.assertTrue(qv.equals(self.q_0_q_i))

        def test_1034_xyz(self):
            qxyz = self.q_1_q_i.xyz()
            print("q_1_q_i.xyz()", qxyz)
            self.assertTrue(qxyz[0][0] == 0)
            self.assertTrue(qxyz[1][0] == 1)

        def test_1040_conj(self):
            qc = self.q_1_q_i.conj()
            qc1 = self.q_1_q_i.conj(1)
            print("q_1_q_i*: ", qc)
            print("q_1_qc*1: ", qc1)
            self.assertTrue(qc.qs[1].x == -1)
            self.assertTrue(qc1.qs[1].x == 1)

        def test_1042_conj_q(self):
            qc = self.q_1_q_i.conj_q(self.q_1)
            qc1 = self.q_1_q_i.conj_q(self.q_1)
            print("q_1_q_i conj_q: ", qc)
            print("q_1_qc*1 conj_q: ", qc1)
            self.assertTrue(qc.qs[1].x == -1)
            self.assertTrue(qc1.qs[1].x == -1)

        def test_1050_flip_signs(self):
            qf = self.q_1_q_i.flip_signs()
            print("-q_1_q_i: ", qf)
            self.assertTrue(qf.qs[1].x == -1)

        def test_1060_inverse(self):
            inv_v1123 = self.v1123.inverse()
            print("inv_v1123 operator", inv_v1123)
            vvinv = inv_v1123.product(self.v1123)
            vvinv.print_state("vinvD x v")
            self.assertTrue(vvinv.equals(self.q14))

            inv_v33 = self.v33.inverse()
            print("inv_v33 operator", inv_v33)
            vv33 = inv_v33.product(self.v33)
            vv33.print_state("inv_v33D x v33")
            self.assertTrue(vv33.equals(self.q19))

            Ainv = self.A.inverse()
            print("A ket inverse, ", Ainv)
            AAinv = self.A.product(Ainv)
            AAinv.print_state("A x AinvD")
            self.assertTrue(AAinv.equals(self.q12))

        def test_1070_normalize(self):
            qn = self.qn.normalize()
            print("Op normalized: ", qn)
            self.assertAlmostEqual(qn.qs[0].t, 0.6)
            self.assertTrue(qn.qs[0].z == 0.8)

        def test_1080_determinant(self):
            det_v3 = self.v3.determinant()
            print("det v3:", det_v3)
            self.assertTrue(det_v3.equals(self.q_3))
            det_v1123 = self.v1123.determinant()
            print("det v1123", det_v1123)
            self.assertTrue(det_v1123.equals(self.q_1))
            det_v9 = self.v9.determinant()
            print("det_v9", det_v9)
            self.assertTrue(det_v9.equals(self.q_9))
            det_vv9 = self.vv9.determinant()
            print("det_vv9", det_vv9)
            self.assertTrue(det_vv9.equals(self.qn627))

        def test_1090_summation(self):
            q_01_sum = self.q_0_q_1.summation()
            print("sum: ", q_01_sum)
            self.assertTrue(type(q_01_sum) is QH)
            self.assertTrue(q_01_sum.t == 1)

        def test_1100_add(self):
            q_0110_add = self.q_0_q_1.add(self.q_1_q_0)
            print("add 01 10: ", q_0110_add)
            self.assertTrue(q_0110_add.qs[0].t == 1)
            self.assertTrue(q_0110_add.qs[1].t == 1)

        def test_1110_dif(self):
            q_0110_dif = self.q_0_q_1.dif(self.q_1_q_0)
            print("dif 01 10: ", q_0110_dif)
            self.assertTrue(q_0110_dif.qs[0].t == -1)
            self.assertTrue(q_0110_dif.qs[1].t == 1)

        def test_1120_diagonal(self):
            Op4iDiag2 = self.Op_scalar.diagonal(2)
            print("Op4i on a diagonal 2x2", Op4iDiag2)
            self.assertTrue(Op4iDiag2.qs[0].equals(self.q_i4))
            self.assertTrue(Op4iDiag2.qs[1].equals(QH().q_0()))

        def test_1130_identity(self):
            I2 = QHStates().identity(2, operator=True)
            print("Operator Idenity, diagonal 2x2", I2)
            self.assertTrue(I2.qs[0].equals(QH().q_1()))
            self.assertTrue(I2.qs[1].equals(QH().q_0()))
            I2 = QHStates().identity(2)
            print("Idenity on 2 state ket", I2)
            self.assertTrue(I2.qs[0].equals(QH().q_1()))
            self.assertTrue(I2.qs[1].equals(QH().q_1()))

        def test_1140_product(self):
            self.assertTrue(
                self.b.product(self.o).equals(
                    QHStates([QH([10, 0, 0, 0]), QH([20, 0, 0, 0]), QH([30, 0, 0, 0])])
                )
            )
            self.assertTrue(
                self.b.product(self.k).equals(QHStates([QH([32, 0, 0, 0])]))
            )
            self.assertTrue(
                self.b.product(self.o)
                .product(self.k)
                .equals(QHStates([QH([320, 0, 0, 0])]))
            )
            self.assertTrue(
                self.b.product(self.b).equals(
                    QHStates([QH([1, 0, 0, 0]), QH([4, 0, 0, 0]), QH([9, 0, 0, 0])])
                )
            )
            self.assertTrue(
                self.o.product(self.k).equals(
                    QHStates([QH([40, 0, 0, 0]), QH([50, 0, 0, 0]), QH([60, 0, 0, 0])])
                )
            )
            self.assertTrue(
                self.o.product(self.o).equals(QHStates([QH([100, 0, 0, 0])]))
            )
            self.assertTrue(
                self.k.product(self.k).equals(
                    QHStates([QH([16, 0, 0, 0]), QH([25, 0, 0, 0]), QH([36, 0, 0, 0])])
                )
            )
            self.assertTrue(
                self.k.product(self.b).equals(
                    QHStates(
                        [
                            QH([4, 0, 0, 0]),
                            QH([5, 0, 0, 0]),
                            QH([6, 0, 0, 0]),
                            QH([8, 0, 0, 0]),
                            QH([10, 0, 0, 0]),
                            QH([12, 0, 0, 0]),
                            QH([12, 0, 0, 0]),
                            QH([15, 0, 0, 0]),
                            QH([18, 0, 0, 0]),
                        ]
                    )
                )
            )

        def test_1150_product_AA(self):
            AA = self.A.product(self.A.set_qs_type("ket"))
            print("AA: ", AA)
            self.assertTrue(AA.equals(QHStates([QH([15, 0, 0, 0])])))

        def test_1160_Euclidean_product_AA(self):
            AA = self.A.Euclidean_product(self.A.set_qs_type("ket"))
            print("A* A", AA)
            self.assertTrue(AA.equals(QHStates([QH([17, 0, 0, 0])])))

        def test_1170_product_AOp(self):
            AOp = self.A.product(self.Op)
            print("A Op: ", AOp)
            self.assertTrue(AOp.qs[0].equals(QH([11, 0, 0, 0])))
            self.assertTrue(AOp.qs[1].equals(QH([0, 0, 5, 0])))
            self.assertTrue(AOp.qs[2].equals(QH([4, 0, 0, 0])))

        def test_1180_Euclidean_product_AOp(self):
            AOp = self.A.Euclidean_product(self.Op)
            print("A* Op: ", AOp)
            self.assertTrue(AOp.qs[0].equals(QH([13, 0, 0, 0])))
            self.assertTrue(AOp.qs[1].equals(QH([0, 0, 11, 0])))
            self.assertTrue(AOp.qs[2].equals(QH([12, 0, 0, 0])))

        def test_1190_product_AOp4i(self):
            AOp4i = self.A.product(self.Op4i)
            print("A Op4i: ", AOp4i)
            self.assertTrue(AOp4i.qs[0].equals(QH([0, 16, 0, 0])))
            self.assertTrue(AOp4i.qs[1].equals(QH([-4, 0, 0, 0])))

        def test_1200_Euclidean_product_AOp4i(self):
            AOp4i = self.A.Euclidean_product(self.Op4i)
            print("A* Op4i: ", AOp4i)
            self.assertTrue(AOp4i.qs[0].equals(QH([0, 16, 0, 0])))
            self.assertTrue(AOp4i.qs[1].equals(QH([4, 0, 0, 0])))

        def test_1210_product_OpB(self):
            OpB = self.Op.product(self.B)
            print("Op B: ", OpB)
            self.assertTrue(OpB.qs[0].equals(QH([0, 10, 3, 0])))
            self.assertTrue(OpB.qs[1].equals(QH([-18, 0, 0, 1])))

        def test_1220_Euclidean_product_OpB(self):
            OpB = self.Op.Euclidean_product(self.B)
            print("Op B: ", OpB)
            self.assertTrue(OpB.qs[0].equals(QH([0, 2, 3, 0])))
            self.assertTrue(OpB.qs[1].equals(QH([18, 0, 0, -1])))

        def test_1230_product_AOpB(self):
            AOpB = self.A.product(self.Op).product(self.B)
            print("A Op B: ", AOpB)
            self.assertTrue(AOpB.equals(QHStates([QH([0, 22, 11, 0])])))

        def test_1240_Euclidean_product_AOpB(self):
            AOpB = self.A.Euclidean_product(self.Op).product(self.B)
            print("A* Op B: ", AOpB)
            self.assertTrue(AOpB.equals(QHStates([QH([0, 58, 13, 0])])))

        def test_1250_product_AOp4i(self):
            AOp4i = self.A.product(self.Op4i)
            print("A Op4i: ", AOp4i)
            self.assertTrue(AOp4i.qs[0].equals(QH([0, 16, 0, 0])))
            self.assertTrue(AOp4i.qs[1].equals(QH([-4, 0, 0, 0])))

        def test_1260_Euclidean_product_AOp4i(self):
            AOp4i = self.A.Euclidean_product(self.Op4i)
            print("A* Op4i: ", AOp4i)
            self.assertTrue(AOp4i.qs[0].equals(QH([0, 16, 0, 0])))
            self.assertTrue(AOp4i.qs[1].equals(QH([4, 0, 0, 0])))

        def test_1270_product_Op4iB(self):
            Op4iB = self.Op4i.product(self.B)
            print("Op4i B: ", Op4iB)
            self.assertTrue(Op4iB.qs[0].equals(QH([0, 6, 0, 4])))
            self.assertTrue(Op4iB.qs[1].equals(QH([0, 9, -8, 0])))

        def test_1280_Euclidean_product_Op4iB(self):
            Op4iB = self.Op4i.Euclidean_product(self.B)
            print("Op4i B: ", Op4iB)
            self.assertTrue(Op4iB.qs[0].equals(QH([0, 6, 0, -4])))
            self.assertTrue(Op4iB.qs[1].equals(QH([0, 9, 8, 0])))

        def test_1290_product_AOp4iB(self):
            AOp4iB = self.A.product(self.Op4i).product(self.B)
            print("A* Op4i B: ", AOp4iB)
            self.assertTrue(AOp4iB.equals(QHStates([QH([-9, 24, 0, 8])])))

        def test_1300_Euclidean_product_AOp4iB(self):
            AOp4iB = self.A.Euclidean_product(self.Op4i).product(self.B)
            print("A* Op4i B: ", AOp4iB)
            self.assertTrue(AOp4iB.equals(QHStates([QH([9, 24, 0, 24])])))

        def test_1305_bracket(self):
            bracket1234 = QHStates().bracket(
                self.q_1234, QHStates().identity(4, operator=True), self.q_1234
            )
            print("bracket <1234|I|1234>: ", bracket1234)
            self.assertTrue(bracket1234.equals(QHStates([QH([34, 0, 0, 0])])))

        def test_1310_op_n(self):
            opn = self.Op.op_n(n=self.q_i)
            print("op_n: ", opn)
            self.assertTrue(opn.qs[0].x == 3)

        def test_1315_norm_squared(self):
            ns = self.q_1_q_i.norm_squared()
            ns.print_state("q_1_q_i norm squared")
            self.assertTrue(ns.equals(QHStates([QH([2, 0, 0, 0])])))

        def test_1320_transpose(self):
            opt = self.q_1234.transpose()
            print("op1234 transposed: ", opt)
            self.assertTrue(opt.qs[0].t == 1)
            self.assertTrue(opt.qs[1].t == 3)
            self.assertTrue(opt.qs[2].t == 2)
            self.assertTrue(opt.qs[3].t == 4)
            optt = self.q_1234.transpose().transpose()
            self.assertTrue(optt.equals(self.q_1234))

        def test_1330_Hermitian_conj(self):
            q_hc = self.q_1234.Hermitian_conj()
            print("op1234 Hermtian_conj: ", q_hc)
            self.assertTrue(q_hc.qs[0].t == 1)
            self.assertTrue(q_hc.qs[1].t == 3)
            self.assertTrue(q_hc.qs[2].t == 2)
            self.assertTrue(q_hc.qs[3].t == 4)
            self.assertTrue(q_hc.qs[0].x == -1)
            self.assertTrue(q_hc.qs[1].x == -1)
            self.assertTrue(q_hc.qs[2].x == -1)
            self.assertTrue(q_hc.qs[3].x == -1)

        def test_1340_is_Hermitian(self):
            self.assertTrue(self.sigma_y.is_Hermitian())
            self.assertFalse(self.q_1234.is_Hermitian())

        def test_1350_is_square(self):
            self.assertFalse(self.Op.is_square())
            self.assertTrue(self.Op_scalar.is_square())

    suite = unittest.TestLoader().loadTestsFromModule(TestQHStates())
    _results = unittest.TextTestRunner().run(suite)





if __name__ == "__main__":

    get_ipython().system("jupyter nbconvert --to script QH.ipynb")
    get_ipython().system("black QH.py")
    get_ipython().system("In_remover.sh QH.py")



