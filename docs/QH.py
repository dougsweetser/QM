#!/usr/bin/env python
# coding: utf-8
"""
Developing Quaternions for iPython

Define a class QH to manipulate quaternions as Hamilton would have done it so many years ago.
The "q_type" is a little bit of text to leave a trail of breadcrumbs about how a particular quaternion was generated.

The class QHStates is a semi-group with inverses, that has a row * column = dimensions as seen in
quantum mechanics.

The function calls for QH and QHStates are meant to be very similar.
"""

from __future__ import annotations
import math
from copy import deepcopy

import numpy as np
import sympy as sp
from typing import Dict, List
from IPython.display import display
from bunch import Bunch


# noinspection PyTypeChecker
class QH(object):
    """
    Quaternions as Hamilton would have defined them, on the manifold R^4.
    Add the usual operations should be here: add, dif, product, trig functions.
    """

    def __init__(self, values: List = None, q_type: str = "Q", representation: str = "") -> QH:
        if values is None:
            self.t, self.x, self.y, self.z = 0, 0, 0, 0
        elif len(values) == 4:
            self.t, self.x, self.y, self.z = values[0], values[1], values[2], values[3]

        elif len(values) == 8:
            self.t, self.x = values[0] - values[1], values[2] - values[3]
            self.y, self.z = values[4] - values[5], values[6] - values[7]

        else:
            raise ValueError(f"The program accepts lists/arrays of 4 or 8 dimensions, not {len(values)}")

        self.representation = representation

        # "Under the hood", all quaternions are manipulated in a Cartesian coordinate system.
        if representation != "":
            self.t, self.x, self.y, self.z = self.representation_2_txyz(representation)

        self.q_type = q_type

    def __str__(self, quiet: bool = False) -> str:
        """
        Customizes the output of a quaternion
        as a tuple given a particular representation.
        Since all quaternions 'look the same',
        the q_type after the tuple tries to summarize
        how this quaternions came into being.
        Quiet turns off printing the q_type.

        Args:
            quiet: bool

        Return: str
        """

        q_type = self.q_type

        if quiet:
            q_type = ""

        string = ""

        if self.representation == "":
            string = f"({self.t}, {self.x}, {self.y}, {self.z}) {q_type}"

        elif self.representation == "polar":
            rep = self.txyz_2_representation("polar")
            string = f"({rep[0]} A, {rep[1]} 𝜈x, {rep[2]} 𝜈y, {rep[3]} 𝜈z) {q_type}"

        elif self.representation == "spherical":
            rep = self.txyz_2_representation("spherical")
            string = f"({rep[0]} t, {rep[1]} R, {rep[2]} θ, {rep[3]} φ) {q_type}"

        return string

    def print_state(self: QH, label: str = "", spacer: bool = True, quiet: bool = True) -> None:
        """
        Utility to print a quaternion with a label.

        Args:
            label: str     User chosen
            spacer: bool   Adds a line return
            quiet: bool    Does not print q_type

        Return: None

        """

        print(label)

        print(self.__str__(quiet))

        if spacer:
            print("")

    def is_symbolic(self: QH) -> bool:
        """
        Figures out if a quaternion has any symbolic terms.

        Return: bool

        """

        symbolic = False

        if (
                hasattr(self.t, "free_symbols")
                or hasattr(self.x, "free_symbols")
                or hasattr(self.y, "free_symbols")
                or hasattr(self.z, "free_symbols")
        ):
            symbolic = True

        return symbolic

    def txyz_2_representation(self: QH, representation: str = "") -> List:
        """
        Given a quaternion in Cartesian coordinates
        returns one in another representation.
        Only 'polar' and 'spherical' are done so far.

        Args:
            representation: bool

        Return: QH

        """

        symbolic = self.is_symbolic()
        rep = ""

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
                theta_x, theta_y, theta_z = 0, 0, 0

            else:
                theta_x = theta * self.x / abs_v
                theta_y = theta * self.y / abs_v
                theta_z = theta * self.z / abs_v

            rep = [amplitude, theta_x, theta_y, theta_z]

        elif representation == "spherical":

            spherical_t = self.t

            spherical_r = (self.x ** 2 + self.y ** 2 + self.z ** 2) ** (1 / 2)

            if spherical_r == 0:
                theta = 0
            else:
                if symbolic:
                    theta = sp.acos(self.z / spherical_r)

                else:
                    theta = math.acos(self.z / spherical_r)

            if symbolic:
                phi = sp.atan2(self.y, self.x)
            else:
                phi = math.atan2(self.y, self.x)

            rep = [spherical_t, spherical_r, theta, phi]

        else:
            raise ValueError(f"Oops, don't know representation: representation")

        return rep

    def representation_2_txyz(self: QH, representation: str = "") -> List:
        """
        Converts something in a representation such as
        polar, spherical
        and returns a Cartesian representation.

        Args:
            representation: str   can be polar or spherical

        Return: QH

        """

        symbolic = False

        if (
                hasattr(self.t, "free_symbols")
                or hasattr(self.x, "free_symbols")
                or hasattr(self.y, "free_symbols")
                or hasattr(self.z, "free_symbols")
        ):
            symbolic = True

        if representation == "":
            box_t, box_x, box_y, box_z = self.t, self.x, self.y, self.z

        elif representation == "polar":
            amplitude, theta_x, theta_y, theta_z = self.t, self.x, self.y, self.z

            theta = (theta_x ** 2 + theta_y ** 2 + theta_z ** 2) ** (1 / 2)

            if theta == 0:
                box_t = self.t
                box_x, box_y, box_z = 0, 0, 0

            else:
                if symbolic:
                    box_t = amplitude * sp.cos(theta)
                    box_x = self.x / theta * amplitude * sp.sin(theta)
                    box_y = self.y / theta * amplitude * sp.sin(theta)
                    box_z = self.z / theta * amplitude * sp.sin(theta)
                else:
                    box_t = amplitude * math.cos(theta)
                    box_x = self.x / theta * amplitude * math.sin(theta)
                    box_y = self.y / theta * amplitude * math.sin(theta)
                    box_z = self.z / theta * amplitude * math.sin(theta)

        elif representation == "spherical":
            box_t, R, theta, phi = self.t, self.x, self.y, self.z

            if symbolic:
                box_x = R * sp.sin(theta) * sp.cos(phi)
                box_y = R * sp.sin(theta) * sp.sin(phi)
                box_z = R * sp.cos(theta)
            else:
                box_x = R * math.sin(theta) * math.cos(phi)
                box_y = R * math.sin(theta) * math.sin(phi)
                box_z = R * math.cos(theta)

        elif representation == "hyperbolic":
            u, v, theta, phi = self.t, self.x, self.y, self.z

            if symbolic:
                box_t = v * sp.exp(u)
                box_x = v * sp.exp(-u)
                box_y = v * sp.sin(theta) * sp.sin(phi)
                box_z = v * sp.cos(theta)

            else:
                box_t = v * math.exp(u)
                box_x = v * math.exp(-u)
                box_y = v * math.sin(theta) * sp.sin(phi)
                box_z = v * math.cos(theta)

        else:
            raise ValueError(f"Oops, don't know representation: representation")

        return [box_t, box_x, box_y, box_z]

    def check_representations(self: QH, q_2: QH) -> bool:
        """
        Checks if self and q_2 have the same representation.

        Args:
            q_2: QH

        Returns: bool

        """

        if self.representation == q_2.representation:
            return True

        else:
            raise Exception(f"Oops, 2 have different representations: {self.representation} {q_2.representation}")

    def display_q(self: QH, label: str = "") -> QH:
        """
        Prints LaTeX-like output, one line for each of th 4 terms.

        Args:
            label: str  an additional bit of text.

        Returns: QH

        """

        if label:
            print(label)
        display(self.t)
        display(self.x)
        display(self.y)
        display(self.z)
        return

    def simple_q(self: QH) -> QH:
        """
        Runs symboy.simplify() on each term, good for symbolic expression.

        Returns: QH

        """

        self.t = sp.simplify(self.t)
        self.x = sp.simplify(self.x)
        self.y = sp.simplify(self.y)
        self.z = sp.simplify(self.z)
        return self

    def expand_q(self) -> QH:
        """
        Runs expand on each term, good for symbolic expressions.

        Returns: QH

        """
        """Expand each term."""

        self.t = sp.expand(self.t)
        self.x = sp.expand(self.x)
        self.y = sp.expand(self.y)
        self.z = sp.expand(self.z)
        return self

    def subs(self: QH, symbol_value_dict: annotations.Dict) -> QH:
        """
        Evaluates a quaternion using sympy values and a dictionary {t:1, x:2, etc}.

        Args:
            symbol_value_dict: Dict

        Returns: QH

        """

        t1 = self.t.subs(symbol_value_dict)
        x1 = self.x.subs(symbol_value_dict)
        y1 = self.y.subs(symbol_value_dict)
        z1 = self.z.subs(symbol_value_dict)

        q_txyz = QH(
            [t1, x1, y1, z1], q_type=self.q_type, representation=self.representation
        )

        return q_txyz

    def scalar(self: QH) -> QH:
        """
        Returns the scalar part of a quaternion as a quaternion.

        $ \rm{scalar(q)} = (q + q^*)/2 = (t, 0) $

        Returns: QH

        """

        end_q_type = f"scalar({self.q_type})"

        s = QH([self.t, 0, 0, 0], q_type=end_q_type, representation=self.representation)
        return s

    def vector(self: QH) -> QH:
        """
        Returns the vector part of a quaternion.
        $ \rm{vector(q)} = (q - q^*)/2 = (0, R) $

        Returns: QH

        """

        end_q_type = f"vector({self.q_type})"

        v = QH(
            [0, self.x, self.y, self.z],
            q_type=end_q_type,
            representation=self.representation,
        )
        return v

    def t(self: QH) -> np.array:
        """
        Returns the scalar t as an np.array.

        Returns: np.array

        """

        return np.array([self.t])

    def xyz(self: QH) -> np.array:
        """
        Returns the vector x, y, z as an np.array.

        Returns: np.array

        """

        return np.array([self.x, self.y, self.z])

    @staticmethod
    def q_0(q_type: str = "0", representation: str = "") -> QH:
        """
        Return a zero quaternion.

        $ q\_0() = 0 = (0, 0) $

        Returns: QH

        """

        q0 = QH([0, 0, 0, 0], q_type=q_type, representation=representation)
        return q0

    @staticmethod
    def q_1(n: float = 1.0, q_type: str = "1", representation: str = "") -> QH:
        """
        Return a real-valued quaternion multiplied by n.

        $ q\_1(n) = n = (n, 0) $

        Returns: QH

        """

        q_2 = QH([n, 0, 0, 0], q_type=q_type, representation=representation)
        return q_2

    @staticmethod
    def q_i(n: float = 1.0, q_type: str = "i", representation: str = "") -> QH:
        """
        Return a quaternion with $ i * n $.

        $ q\_i(n) = n i = (0, n i) $

        Returns: QH

        """

        qi = QH([0, n, 0, 0], q_type=q_type, representation=representation)
        return qi

    @staticmethod
    def q_j(n: float = 1.0, q_type: str = "j", representation: str = "") -> QH:
        """
        Return a quaternion with $ j * n $.

        $ q\_j(n) = n j = (0, n j) $

        Returns: QH

        """

        qj = QH([0, 0, n, 0], q_type=q_type, representation=representation)
        return qj

    @staticmethod
    def q_k(n: float = 1, q_type: str = "k", representation: str = "") -> QH:
        """
        Return a quaternion with $ k * n $.

        $ q\_k(n) = n k =(0, n k) $

        Returns: QH

        """

        qk = QH([0, 0, 0, n], q_type=q_type, representation=representation)
        return qk

    @staticmethod
    def q_random(low: float = -1.0, high: float = 1.0, distribution: str = "uniform", q_type: str = "?",
                 representation: str = "") -> QH:
        """
        Return a random-valued quaternion.
        The distribution is uniform, but one could add to options.
        It would take some work to make this clean so will skip for now.

        Args:
            low:
            high:
            distribution: str:    hove only implemented uniform distribution
            q_type:               ?
            representation:       Cartesian by default

        Returns: QH

        """

        random_distributions = Bunch()
        random_distributions.uniform = np.random.uniform

        qr = QH(
            [
                random_distributions[distribution](low=low, high=high),
                random_distributions[distribution](low=low, high=high),
                random_distributions[distribution](low=low, high=high),
                random_distributions[distribution](low=low, high=high),
            ],
            q_type=q_type,
            representation=representation,
        )
        return qr

    def dupe(self: QH) -> QH:
        """
        Return a duplicate copy.

        Returns: QH

        """

        du = QH(
            [self.t, self.x, self.y, self.z],
            q_type=self.q_type,
            representation=self.representation,
        )
        return du

    def equals(self: QH, q_2: QH) -> QH:
        """
        Tests if self and q_2 quaternions are close to equal.

        $ q.equals(q\_2) = q == q\_2 = True $

        Args:
            q_2: QH

        Returns: QH

        """

        self.check_representations(q_2)

        self_t, self_x, self_y, self_z = (
            sp.expand(self.t),
            sp.expand(self.x),
            sp.expand(self.y),
            sp.expand(self.z),
        )
        q_2_t, q_2_x, q_2_y, q_2_z = (
            sp.expand(q_2.t),
            sp.expand(q_2.x),
            sp.expand(q_2.y),
            sp.expand(q_2.z),
        )

        if (
                math.isclose(self_t, q_2_t)
                and math.isclose(self_x, q_2_x)
                and math.isclose(self_y, q_2_y)
                and math.isclose(self_z, q_2_z)
        ):
            return True

        else:
            return False

    def conj(self: QH, conj_type: int = 0) -> QH:
        """
        There are 4 types of conjugates.

        $ q.conj(0) = q^* =(t, -x, -y, -z) $
        $ q.conj(1) = (i q i)^* =(-t, x, -y, -z) $
        $ q.conj(2) = (j q j)^* =(-t, -x, y, -z) $
        $ q.conj(3) = (k q k)^* =(-t, -x, -y, z) $

        Args:
            conj_type: int:   0-3 depending on who stays positive.

        Returns: QH

        """

        end_q_type = f"{self.q_type}*"
        c_t, c_x, c_y, c_z = self.t, self.x, self.y, self.z
        conj_q = QH()

        if conj_type % 4 == 0:
            conj_q.t = c_t
            if c_x != 0:
                conj_q.x = -1 * c_x
            if c_y != 0:
                conj_q.y = -1 * c_y
            if c_z != 0:
                conj_q.z = -1 * c_z

        elif conj_type % 4 == 1:
            if c_t != 0:
                conj_q.t = -1 * c_t
            conj_q.x = c_x
            if c_y != 0:
                conj_q.y = -1 * c_y
            if c_z != 0:
                conj_q.z = -1 * c_z
            end_q_type += "1"

        elif conj_type % 4 == 2:
            if c_t != 0:
                conj_q.t = -1 * c_t
            if c_x != 0:
                conj_q.x = -1 * c_x
            conj_q.y = c_y
            if c_z != 0:
                conj_q.z = -1 * c_z
            end_q_type += "2"

        elif conj_type % 4 == 3:
            if c_t != 0:
                conj_q.t = -1 * c_t
            if c_x != 0:
                conj_q.x = -1 * c_x
            if c_y != 0:
                conj_q.y = -1 * c_y
            conj_q.z = c_z
            end_q_type += "3"

        conj_q.q_type = end_q_type
        conj_q.representation = self.representation

        return conj_q

    def conj_q(self: QH, q_2: QH) -> QH:
        """
        Given a quaternion with 0s or 1s, will do the standard conjugate, first conjugate
        second conjugate, sign flip, or all combinations of the above.

        q.conj(q(1, 1, 1, 1)) = q.conj(0).conj(1).conj(2).conj(3)

        Args:
            q_2: QH    Use a quaternion to do one of 4 conjugates in combinations

        Returns: QH

        """

        _conj = deepcopy(self)

        if q_2.t:
            _conj = _conj.conj(conj_type=0)

        if q_2.x:
            _conj = _conj.conj(conj_type=1)

        if q_2.y:
            _conj = _conj.conj(conj_type=2)

        if q_2.z:
            _conj = _conj.flip_signs()

        return _conj

    def flip_signs(self: QH) -> QH:
        """
        Flip the signs of all terms.

        $ q.flip\_signs() = -q = (-t, -R) $

        Returns: QH

        """

        end_q_type = f"-{self.q_type}"

        flip_t, flip_x, flip_y, flip_z = self.t, self.x, self.y, self.z

        flip_q = QH(q_type=end_q_type, representation=self.representation)
        if flip_t != 0:
            flip_q.t = -1 * flip_t
        if flip_x != 0:
            flip_q.x = -1 * flip_x
        if flip_y != 0:
            flip_q.y = -1 * flip_y
        if flip_z != 0:
            flip_q.z = -1 * flip_z

        return flip_q

    def vahlen_conj(self: QH, conj_type: str = "-", q_type: str = "vc") -> QH:
        """
        Three types of conjugates dash, apostrophe, or star as done by Vahlen in 1901.

        q.vahlen_conj("-") = q^* = (t, -x, -y, -z)

        q.vahlen_conj("'") = (k q k)^* = (t, -x, -y, z)

        q.vahlen_conj("*") = -(k q k)^* = (t, x, y, -z)

        Args:
            conj_type: str:    3 sorts, dash apostrophe,
            q_type: str:

        Returns:

        """

        vc_t, vc_x, vc_y, vc_z = self.t, self.x, self.y, self.z
        conj_q = QH()

        if conj_type == "-":
            conj_q.t = vc_t
            if vc_x != 0:
                conj_q.x = -1 * vc_x
            if vc_y != 0:
                conj_q.y = -1 * vc_y
            if vc_z != 0:
                conj_q.z = -1 * vc_z
            q_type += "*-"

        if conj_type == "'":
            conj_q.t = vc_t
            if vc_x != 0:
                conj_q.x = -1 * vc_x
            if vc_y != 0:
                conj_q.y = -1 * vc_y
            conj_q.z = vc_z
            q_type += "*'"

        if conj_type == "*":
            conj_q.t = vc_t
            conj_q.x = vc_x
            conj_q.y = vc_y
            if vc_z != 0:
                conj_q.z = -1 * vc_z
            q_type += "*"

        conj_q.q_type = self.q_type + q_type
        conj_q.representation = self.representation

        return conj_q

    def _commuting_products(self: QH, q_2: QH) -> Dict:
        """
        Returns a dictionary with the commuting products. For internal use.

        Args:
            q_2: QH

        Returns: Dict

        """

        s_t, s_x, s_y, s_z = self.t, self.x, self.y, self.z
        q_2_t, q_2_x, q_2_y, q_2_z = q_2.t, q_2.x, q_2.y, q_2.z

        products = {
            "tt": s_t * q_2_t,
            "xx+yy+zz": s_x * q_2_x + s_y * q_2_y + s_z * q_2_z,
            "tx+xt": s_t * q_2_x + s_x * q_2_t,
            "ty+yt": s_t * q_2_y + s_y * q_2_t,
            "tz+zt": s_t * q_2_z + s_z * q_2_t,
        }

        return products

    def _anti_commuting_products(self: QH, q_2: QH) -> Dict:
        """
        Returns a dictionary with the three anti-commuting products. For internal use.

        Args:
            q_2: QH

        Returns: Dict

        """

        s_x, s_y, s_z = self.x, self.y, self.z
        q_2_x, q_2_y, q_2_z = q_2.x, q_2.y, q_2.z

        products = {
            "yz-zy": s_y * q_2_z - s_z * q_2_y,
            "zx-xz": s_z * q_2_x - s_x * q_2_z,
            "xy-yx": s_x * q_2_y - s_y * q_2_x,
            "zy-yz": -s_y * q_2_z + s_z * q_2_y,
            "xz-zx": -s_z * q_2_x + s_x * q_2_z,
            "yx-xy": -s_x * q_2_y + s_y * q_2_x,
        }

        return products

    def _all_products(self: QH, q_2: QH) -> Dict:
        """
        All products, commuting and anti-commuting products as a dictionary. For internal use.

        Args:
            q_2: QH

        Returns: Dict

        """

        products = self._commuting_products(q_2)
        products.update(self._anti_commuting_products(q_2))

        return products

    def square(self: QH) -> QH:
        """
        Square a quaternion.

        $ q.square() = q^2 = (t^2 - R.R, 2 t R) $

        Returns:
            QH

        """

        end_q_type = f"{self.q_type}²"

        qxq = self._commuting_products(self)

        sq_q = QH(q_type=end_q_type, representation=self.representation)
        sq_q.t = qxq["tt"] - qxq["xx+yy+zz"]
        sq_q.x = qxq["tx+xt"]
        sq_q.y = qxq["ty+yt"]
        sq_q.z = qxq["tz+zt"]

        return sq_q

    def norm_squared(self: QH) -> QH:
        """
        The norm_squared of a quaternion.

        $ q.norm\_squared() = q q^* = (t^2 + R.R, 0) $

        Returns: QH

        """

        end_q_type = f"||{self.q_type}||²"

        qxq = self._commuting_products(self)

        n_q = QH(q_type=end_q_type, representation=self.representation)
        n_q.t = qxq["tt"] + qxq["xx+yy+zz"]

        return n_q

    def norm_squared_of_vector(self: QH):
        """
        The norm_squared of the vector of a quaternion.

        $ q.norm\_squared\_of\_vector() = ((q - q^*)(q - q^*)^*)/4 = (R.R, 0) $

        Returns: QH
        """

        end_q_type = f"|V({self.q_type})|²"

        qxq = self._commuting_products(self)

        nv_q = QH(q_type=end_q_type, representation=self.representation)
        nv_q.t = qxq["xx+yy+zz"]

        return nv_q

    def abs_of_q(self: QH) -> QH:
        """
        The absolute value, the square root of the norm_squared.

        $ q.abs_of_q() = \sqrt{q q^*} = (\sqrt{t^2 + R.R}, 0) $

        Returns: QH

        """

        end_q_type = f"|{self.q_type}|"

        a = self.norm_squared()
        sqrt_t = a.t ** (1 / 2)
        a.t = sqrt_t
        a.q_type = end_q_type
        a.representation = self.representation

        return a

    def normalize(self: QH, n: float = 1.0, q_type: str = "U") -> QH:
        """
        Normalize a quaternion to a given value n.

        $ q.normalized(n) = q (q q^*)^{-1} = q (n/\sqrt{q q^*}, 0) $

        Args:
            n:        Make the norm equal to n.
            q_type:

        Returns: QH

        """

        end_q_type = f"{self.q_type} {q_type}"

        abs_q_inv = self.abs_of_q().inverse()
        n_q = self.product(abs_q_inv).product(QH([n, 0, 0, 0]))
        n_q.q_type = end_q_type
        n_q.representation = self.representation

        return n_q

    def abs_of_vector(self: QH) -> QH:
        """
        The absolute value of the vector, the square root of the norm_squared of the vector.

        $ q.abs_of_vector() = \sqrt{(q - q^*)(q - q^*)/4} = (\sqrt{R.R}, 0) $

        Returns: QH

        """

        end_q_type = f"|V({self.q_type})|"

        av = self.norm_squared_of_vector()
        sqrt_t = av.t ** (1 / 2)
        av.t = sqrt_t
        av.representation = self.representation
        av.q_type = end_q_type

        return av

    def add(self: QH, q_2: QH) -> QH:
        """
        Add two quaternions.

        $ q.add(q\_2) = q + q\_2 = (t + t\_2, R + R\_2) $

        Args:
            q_2: QH

        Returns: QH

        """

        self.check_representations(q_2)

        add_q_type = f"{self.q_type}+{q_2.q_type}"

        t_1, x_1, y_1, z_1 = self.t, self.x, self.y, self.z
        t_2, x_2, y_2, z_2 = q_2.t, q_2.x, q_2.y, q_2.z

        add_q = QH(q_type=add_q_type, representation=self.representation)
        add_q.t = t_1 + t_2
        add_q.x = x_1 + x_2
        add_q.y = y_1 + y_2
        add_q.z = z_1 + z_2

        return add_q

    def dif(self: QH, q_2: QH) -> QH:
        """
        Takes the difference of 2 quaternions.

        $ q.dif(q\_2) = q - q\_2 = (t - t\_2, R - R\_2) $

        Args:
            q_2: QH

        Returns: QH

        """

        self.check_representations(q_2)

        end_dif_q_type = f"{self.q_type}-{q_2.q_type}"

        t_2, x_2, y_2, z_2 = q_2.t, q_2.x, q_2.y, q_2.z
        t_1, x_1, y_1, z_1 = self.t, self.x, self.y, self.z

        dif_q = QH(q_type=end_dif_q_type, representation=self.representation)
        dif_q.t = t_1 - t_2
        dif_q.x = x_1 - x_2
        dif_q.y = y_1 - y_2
        dif_q.z = z_1 - z_2

        return dif_q

    def product(self: QH, q_2: QH, kind: str = "", reverse: bool = False):
        """
        Form a product given 2 quaternions. Kind of product can be '' aka standard, even, odd, or even_minus_odd.
        Setting reverse=True is like changing the order.

        $ q.product(q_2) = q q\_2 = (t t_2 - R.R_2, t R_2 + R t_2 + RxR_2 ) $

        $ q.product(q_2, kind="even") = (q q\_2 + (q q\_2)^*)/2 = (t t_2 - R.R_2, t R_2 + R t_2 ) $

        $ q.product(q_2, kind="odd") = (q q\_2 - (q q\_2)^*)/2 = (0, RxR_2 ) $

        $ q.product(q_2, kind="even_minus_odd") = q\_2 q = (t t_2 - R.R_2, t R_2 + R t_2 - RxR_2 ) $

        $ q.product(q_2, reverse=True) = q\_2 q = (t t_2 - R.R_2, t R_2 + R t_2 - RxR_2 ) $

        Args:
            q_2: QH:
            kind: str:    can be blank, even, odd, or even_minus_odd
            reverse: bool:  if true, returns even_minus_odd

        Returns: QH

        """

        self.check_representations(q_2)

        commuting = self._commuting_products(q_2)
        q_even = QH()
        q_even.t = commuting["tt"] - commuting["xx+yy+zz"]
        q_even.x = commuting["tx+xt"]
        q_even.y = commuting["ty+yt"]
        q_even.z = commuting["tz+zt"]

        anti_commuting = self._anti_commuting_products(q_2)
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
            times_symbol = "xE-xO"
        else:
            raise Exception(
                "Four 'kind' values are known: '', 'even', 'odd', and 'even_minus_odd'."
            )

        if reverse:
            times_symbol = times_symbol.replace("x", "xR")

        result.q_type = f"{self.q_type}{times_symbol}{q_2.q_type}"
        result.representation = self.representation

        return result

    def inverse(self: QH, additive: bool = False) -> QH:
        """
        The additive or multiplicative inverse of a quaternion. Defaults to 1/q, not -q.

        $ q.inverse() = q^* (q q^*)^{-1} = (t, -R) / (t^2 + R.R) $

        $ q.inverse(additive=True) = -q = (-t, -R) $

        Args:
            additive: bool

        Returns: QH

        """

        if additive:
            end_q_type = f"-{self.q_type}"
            q_inv = self.flip_signs()
            q_inv.q_type = end_q_type

        else:
            end_q_type = f"{self.q_type}⁻¹"

            q_conj = self.conj()
            q_norm_squared = self.norm_squared()

            if (not self.is_symbolic()) and (q_norm_squared.t == 0):
                return self.q_0()

            q_norm_squared_inv = QH([1.0 / q_norm_squared.t, 0, 0, 0])
            q_inv = q_conj.product(q_norm_squared_inv)
            q_inv.q_type = end_q_type
            q_inv.representation = self.representation

        return q_inv

    def divide_by(self: QH, q_2: QH) -> QH:
        """
        Divide one quaternion by another. The order matters unless one is using a norm_squared (real number).

        $ q.divided_by(q_2) = q q_2^{-1} = (t t\_2 + R.R\_2, -t R\_2 + R t\_2 - RxR\_2) $

        Args:
            q_2:  QH

        Returns: QH

        """

        self.check_representations(q_2)

        end_q_type = f"{self.q_type}/{q_2.q_type}"

        q_div = self.product(q_2.inverse())
        q_div.q_type = end_q_type
        q_div.representation = self.representation

        return q_div

    def triple_product(self: QH, q_2: QH, q_3: QH) -> QH:
        """
        Form a triple product given 3 quaternions, in left-to-right order: self, q_2, q_3.

        $ q.triple_product(q_2, q_3) = q q_2 q_3 $

        $ = (t t\_2 t\_3 - R.R\_2 t\_3 - t R\_2.R|_3 - t\_2 R.R\_3 - (RxR_2).R\_3, $

        $ ... t t\_2 R\_3 - (R.R\_2) R\_3 + t t\_3 R\_2 + t\_2 t\_3 R $

        $ ... + t\_3 RxR\_2 + t R_2xR\_3 + t_2 RxR\_3 + RxR\_2xR\_3) $

        Args:
            q_2: QH:
            q_3: QH:

        Returns: QH

        """

        self.check_representations(q_2)
        self.check_representations(q_3)

        triple = self.product(q_2).product(q_3)
        triple.representation = self.representation

        return triple

    # Quaternion rotation involves a triple product:  u R 1/u
    def rotate(self: QH, u: QH) -> QH:
        """
        Do a rotation using a triple product: u R 1/u.

        $ q.rotate(u) = u q u^{-1} $

        $ = (u^2 t - u V.R + u R.V + t V.V, $
        $ ... - u t V + (V.R) V + u^2 R + V t u + VxR u - u RxV - VxRxV) $

        Args:
            u: QH    pre-multiply by u, post-multiply by $u^{-1}$.

        Returns: QH

        """

        self.check_representations(u)
        end_q_type = f"{self.q_type}*rot"

        q_rot = u.triple_product(self, u.inverse())
        q_rot.q_type = end_q_type
        q_rot.representation = self.representation

        return q_rot

    def rotation_and_or_boost(self: QH, h: QH) -> QH:
        """
        The method for doing a rotation in 3D space discovered by Rodrigues in the 1840s used a quaternion triple
        product. After Minkowski characterized Einstein's work in special relativity as a 4D rotation, efforts were
        made to do the same with one quaternion triple product. That obvious goal was not achieved until 2010 by
        D. Sweetser and indpendently by M. Kharinov. Two other triple products need to be used like so:

        $ b.rotation_and_or_boost(h) = h b h^* + 1/2 ((hhb)^* -(h^* h^* b)^*) $

        The parameter h is NOT free from constraints. There are two constraints. If the parameter h is to do a
        rotation, it must have a norm of unity and have the first term equal to zero.

        $ h = (0, R), scalar(h) = 0, scalar(h h^*) = 1 $

        To do a boost which may or may not also do a rotation, then the parameter h must have a square whose first
        term is equal to zero:

        $ h = (\cosh(a), \sinh(a)), scalar(h^2) = 1 $

        There has been no issue about the ability of this function to do boosts. There has been a spirited debate
        as to whether the function can do rotations. Notice that the form reduces to the Rodrigues triple product.
        I consider this so elementary that I cannot argue the other side. Please see the wiki page or use this code
        to see for yourself.

        Args:
            h: QH

        Returns: QH

        """
        self.check_representations(h)
        end_q_type = f"{self.q_type}rotation/boost"

        # if not h.is_symbolic():
        #     if math.isclose(h.t, 0):
        #         if not math.isclose(h.norm_squared().t, 1):
        #             h = h.normalize()
        #             h.print_state("To do a 3D rotation, h adjusted value so scalar(h h^*) = 1")

        #     else:
        #         if not math.isclose(h.square().t, 1):
        #             h = QH.Lorentz_next_boost(h, QH.q_1())
        #             h.print_state("To do a Lorentz boost, h adjusted value so scalar(h²) = 1")

        triple_1 = h.triple_product(self, h.conj())
        triple_2 = h.triple_product(h, self).conj()
        triple_3 = h.conj().triple_product(h.conj(), self).conj()

        triple_23 = triple_2.dif(triple_3)
        half_23 = triple_23.product(QH([0.5, 0, 0, 0], representation=self.representation))
        triple_123 = triple_1.add(half_23)
        triple_123.q_type = end_q_type
        triple_123.representation = self.representation

        return triple_123

    @staticmethod
    def Lorentz_next_rotation(q: QH, q_2: QH) -> QH:
        """
        Given 2 quaternions, creates a new quaternion to do a rotation
        in the triple triple quaternion function by using a normalized cross product.

        $ Lorentz_next_rotation(q, q_2) = (q q\_2 - q\_2 q) / 2|(q q\_2 - (q\_2 q)^*)| = (0, QxQ\_2)/|(0, QxQ\_2)| $

        Args:
            q:      any quaternion
            q_2:    any quaternion whose first term equals the first term of q and
                    for the first terms of each squared.

        Returns: QH

        """
        q.check_representations(q_2)

        if not math.isclose(q.t, q_2.t):
            raise ValueError(f"Oops, to be a rotation, the first values must be the same: {q.t} != {q_2.t}")

        if not math.isclose(q.square().t, q_2.square().t):
            raise ValueError(f"Oops, the squares of these two are not equal: {q.square().t} != {q_2.square().t}")

        next_rotation = q.product(q_2, kind="odd").normalize()

        # If the 2 quaternions point in exactly the same direction, the result is zoro.
        # That is unacceptable for closure, so return the normalized vector of one input.
        # This does create some ambiguity since q and q_2 could point in exactly opposite
        # directions. In that case, the first quaternion is always chosen.
        v_norm = next_rotation.norm_squared_of_vector()

        if v_norm.t == 0:
            next_rotation = q.vector().normalize()

        return next_rotation

    @staticmethod
    def Lorentz_next_boost(q: QH, q_2: QH) -> QH:
        """
        Given 2 quaternions, creates a new quaternion to do a boost/rotation
        using the triple triple quaternion product
        by using the scalar of an even product to form (cosh(x), i sinh(x)).

        $ Lorentz_next_boost(q, q_2) = q q\_2 + q\_2 q

        Args:
            q: QH
            q_2: QH

        Returns: QH

        """
        q.check_representations(q_2)

        if not (q.t >= 1.0 and q_2.t >= 1.0):
            raise ValueError(f"Oops, to be a boost, the first values must both be greater than one: {q.t},  {q_2.t}")

        if not math.isclose(q.square().t, q_2.square().t):
            raise ValueError(f"Oops, the squares of these two are not equal: {q.square().t} != {q_2.square().t}")

        q_even = q.product(q_2, kind="even")
        q_s = q_even.scalar()
        q_v = q_even.vector().normalize()

        if np.abs(q_s.t) > 1:
            q_s = q_s.inverse()

        exp_sum = q_s.exp().add(q_s.flip_signs().exp()).product(QH().q_1(1.0 / 2.0))
        exp_dif = q_s.exp().dif(q_s.flip_signs().exp()).product(QH().q_1(1.0 / 2.0))

        boost = exp_sum.add(q_v.product(exp_dif))

        return boost

    # Lorentz transformations are not exclusively about special relativity.
    # The most general case is B->B' such that the first term of scalar(B²)
    # is equal to scalar(B'²). Since there is just one constraint yet there
    # are 4 degrees of freedom, rescaling
    def Lorentz_by_rescaling(self: QH, op, h: QH = None, quiet: bool = True) -> QH:

        end_q_type = f"{self.q_type} Lorentz-by-rescaling"

        # Use h if provided.
        unscaled = op(h) if h is not None else op()

        self_interval = self.square().t
        unscaled_interval = unscaled.square().t

        # Figure out if the interval is time-like, space-like, or light-like (+, -, or 0)
        # if self_interval:
        #    if self_interval > 0:
        #        self_interval_type = "time-like"
        #    else:
        #        self_interval_type = "space-like"
        # else:
        #    self_interval_type = "light-like"

        # if unscaled_interval:
        #    if unscaled_interval > 0:
        #        unscaled_interval_type = "time-like"
        #    else:
        #        unscaled_interval_type = "space-like"
        # else:
        #    unscaled_interval_type = "light-like"

        self_interval_type = (
            "light-like" if unscaled_interval == 0 else "not_light-like"
        )
        unscaled_interval_type = (
            "light-like" if unscaled_interval == 0 else "not_light-like"
        )

        # My house rules after thinking about this rescaling stuff.
        # A light-like interval can go to a light-like interval.
        # Only a light-like interval can transform to the origin.
        # A light-like interval cannot go to a time- or space-like interval or visa versa.
        # If any of these exceptions are met, then an identity transformaton is returned - deepcopy(self).
        # A time-like interval can rescale to a time-like or space-like (via an 'improper rescaling') interval.
        # A space-like interval can rescale to a time-like or space-like interval interval.

        # For light-like to light-like, no scaling is required. I don't think boosting makes sense to return self
        if (self_interval_type == "light-like") and (
                unscaled_interval_type == "light-like"
        ):
            return unscaled

        # When one is light-like but the other is not, return a copy of the
        # starting value (an identity transformation).

        if (self_interval_type == "light-like") and (
                unscaled_interval_type != "light-like"
        ):
            return deepcopy(self)

        if (self_interval_type != "light-like") and (
                unscaled_interval_type == "light-like"
        ):
            return deepcopy(self)

        # The remaining case is to handle is if time-like goes to space-like
        # or visa-versa. Use a sign flip to avoid an imaginary value from the square root.
        if self.is_symbolic():
            sign_flip = False
        else:
            sign_flip = True if self_interval * unscaled_interval < 0 else False

        if sign_flip:
            scaling = np.sqrt(-1 * self_interval / unscaled_interval)
        else:
            scaling = np.sqrt(self_interval / unscaled_interval)

        if unscaled.equals(QH().q_0()):
            print("zero issue") if not quiet else 0
            return deepcopy(self)

        if not self.is_symbolic() and not np.isclose(scaling, 1):
            print(f"scaling needed: {scaling}") if not quiet else 0

        scaled = unscaled.product(QH([scaling, 0, 0, 0]))
        scaled.print_state("final scaled") if not quiet else 0
        scaled.square().print_state("scaled square") if not quiet else 0
        scaled.q_type = end_q_type

        return scaled

    # g_shift is a function based on the space-times-time invariance proposal for gravity,
    # which proposes that if one changes the distance from a gravitational source, then
    # squares a measurement, the observers at two different hieghts agree to their
    # space-times-time values, but not the intervals.
    # g_form is the form of the function, either minimal or exponential
    # Minimal is what is needed to pass all weak field tests of gravity
    def g_shift(self: QH, dimensionless_g, g_form="exp"):
        """Shift an observation based on a dimensionless GM/c^2 dR."""

        end_q_type = f"{self.q_type} gshift"

        if g_form == "exp":
            g_factor = sp.exp(dimensionless_g)
        elif g_form == "minimal":
            g_factor = 1 + 2 * dimensionless_g + 2 * dimensionless_g ** 2
        else:
            print("g_form not defined, should be 'exp' or 'minimal': {}".format(g_form))
            return self

        g_q = QH(q_type=end_q_type)
        g_q.t = self.t / g_factor
        g_q.x = self.x * g_factor
        g_q.y = self.y * g_factor
        g_q.z = self.z * g_factor
        g_q.q_type = end_q_type
        g_q.representation = self.representation

        return g_q

    def sin(self: QH) -> QH:
        """
        Take the sine of a quaternion

        $ q.sin() = (\sin(t) \cosh(|R|), \cos(t) \sinh(|R|) R/|R|)$

        Returns: QH

        """

        end_q_type = f"sin({self.q_type})"

        abs_v = self.abs_of_vector()

        if abs_v.t == 0:
            return QH([math.sin(self.t), 0, 0, 0], q_type=end_q_type, representation=self.representation)

        sint = math.sin(self.t)
        cost = math.cos(self.t)
        sinhR = math.sinh(abs_v.t)
        coshR = math.cosh(abs_v.t)

        k = cost * sinhR / abs_v.t

        q_sin = QH()
        q_sin.t = sint * coshR
        q_sin.x = k * self.x
        q_sin.y = k * self.y
        q_sin.z = k * self.z

        q_sin.q_type = end_q_type
        q_sin.representation = self.representation

        return q_sin

    def cos(self: QH) -> QH:
        """
        Take the cosine of a quaternion.
        $ q.cos() = (\cos(t) \cosh(|R|), \sin(t) \sinh(|R|) R/|R|) $

        Returns: QH

        """

        end_q_type = f"cos({self.q_type})"

        abs_v = self.abs_of_vector()

        if abs_v.t == 0:
            return QH([math.cos(self.t), 0, 0, 0], q_type=end_q_type, representation=self.representation)

        sint = math.sin(self.t)
        cost = math.cos(self.t)
        sinhR = math.sinh(abs_v.t)
        coshR = math.cosh(abs_v.t)

        k = -1 * sint * sinhR / abs_v.t

        q_cos = QH()
        q_cos.t = cost * coshR
        q_cos.x = k * self.x
        q_cos.y = k * self.y
        q_cos.z = k * self.z

        q_cos.q_type = end_q_type
        q_cos.representation = self.representation

        return q_cos

    def tan(self: QH) -> QH:
        """
        Take the tan of a quaternion.

         $ q.tan() = \sin(q) \cos(q)^{-1} $

         Returns: QH

         """

        end_q_type = f"tan({self.q_type})"

        abs_v = self.abs_of_vector()

        if abs_v.t == 0:
            return QH([math.tan(self.t), 0, 0, 0], q_type=end_q_type, representation=self.representation)

        sinq = self.sin()
        cosq = self.cos()

        q_tan = sinq.divide_by(cosq)
        q_tan.q_type = end_q_type
        q_tan.representation = self.representation

        return q_tan

    def sinh(self: QH) -> QH:
        """
        Take the sinh of a quaternion.

        $ q.sinh() = (\sinh(t) \cos(|R|), \cosh(t) \sin(|R|) R/|R|) $

        Returns: QH

        """

        end_q_type = f"sinh({self.q_type})"

        abs_v = self.abs_of_vector()

        if abs_v.t == 0:
            return QH([math.sinh(self.t), 0, 0, 0], q_type=end_q_type, representation=self.representation)

        sinh_t = math.sinh(self.t)
        cos_r = math.cos(abs_v.t)
        cosh_t = math.cosh(self.t)
        sin_r = math.sin(abs_v.t)

        k = cosh_t * sin_r / abs_v.t

        q_sinh = QH(q_type=end_q_type, representation=self.representation)
        q_sinh.t = sinh_t * cos_r
        q_sinh.x = k * self.x
        q_sinh.y = k * self.y
        q_sinh.z = k * self.z

        return q_sinh

    def cosh(self: QH) -> QH:
        """
        Take the cosh of a quaternion.

        $ (\cosh(t) \cos(|R|), \sinh(t) \sin(|R|) R/|R|) $

        Returns: QH

        """

        end_q_type = f"cosh({self.q_type})"

        abs_v = self.abs_of_vector()

        if abs_v.t == 0:
            return QH([math.cosh(self.t), 0, 0, 0], q_type=end_q_type, representation=self.representation)

        cosh_t = math.cosh(self.t)
        cos_r = math.cos(abs_v.t)
        sinh_t = math.sinh(self.t)
        sin_r = math.sin(abs_v.t)

        k = sinh_t * sin_r / abs_v.t

        q_cosh = QH(q_type=end_q_type, representation=self.representation)
        q_cosh.t = cosh_t * cos_r
        q_cosh.x = k * self.x
        q_cosh.y = k * self.y
        q_cosh.z = k * self.z

        return q_cosh

    def tanh(self: QH) -> QH:
        """
        Take the tanh of a quaternion.

        $ q.tanh() = \sin(q) \cos(q)^{-1} $

        Returns: QH

        """

        end_q_type = f"tanh({self.q_type})"

        abs_v = self.abs_of_vector()

        if abs_v.t == 0:
            return QH([math.tanh(self.t), 0, 0, 0], q_type=end_q_type, representation=self.representation)

        sinhq = self.sinh()
        coshq = self.cosh()

        q_tanh = sinhq.divide_by(coshq)
        q_tanh.q_type = end_q_type
        q_tanh.representation = self.representation

        return q_tanh

    def exp(self: QH) -> QH:
        """
        Take the exponential of a quaternion.

        $ q.exp() = (\exp(t) \cos(|R|, \exp(t) \sin(|R|) R/|R|) $

        Returns: QH
        """

        end_q_type = f"exp({self.q_type})"

        abs_v = self.abs_of_vector()
        et = math.exp(self.t)

        if abs_v.t == 0:
            return QH([et, 0, 0, 0], q_type=end_q_type, representation=self.representation)

        cosR = math.cos(abs_v.t)
        sinR = math.sin(abs_v.t)
        k = et * sinR / abs_v.t

        q_exp = QH(
            [et * cosR, k * self.x, k * self.y, k * self.z],
            q_type=end_q_type,
            representation=self.representation,
        )

        return q_exp

    def ln(self: QH) -> QH:
        """
        Take the natural log of a quaternion.

        $ q.ln() = (0.5 \ln t^2 + R.R, \atan2(|R|, t) R/|R|) $

        Returns: QH

        """
        end_q_type = f"ln({self.q_type})"

        abs_v = self.abs_of_vector()

        if abs_v.t == 0:

            if self.t > 0:
                return QH([math.log(self.t), 0, 0, 0], q_type=end_q_type, representation=self.representation)
            else:
                # I don't understand this, but Mathematica does the same thing.
                return QH([math.log(-self.t), math.pi, 0, 0], q_type=end_q_type, representation=self.representation)

        t_value = 0.5 * math.log(self.t * self.t + abs_v.t * abs_v.t)
        k = math.atan2(abs_v.t, self.t) / abs_v.t

        q_ln = QH(
            [t_value, k * self.x, k * self.y, k * self.z],
            q_type=end_q_type,
            representation=self.representation,
        )

        return q_ln

    def q_2_q(self: QH, q_2: QH) -> QH:
        """Take the natural log of a quaternion.

        $ q.q_2_q(p) = \exp(\ln(q) * p) $

        Returns: QH

        """

        self.check_representations(q_2)
        end_q_type = f"{self.q_type}^{q_2.q_type}"

        q2q = self.ln().product(q_2).exp()
        q2q.q_type = end_q_type
        q2q.representation = self.representation
        q2q.q_type = end_q_type

        return q2q

    def trunc(self: QH) -> QH:
        """
        Truncates values.

        Returns: QH

        """

        if not self.is_symbolic():
            self.t = math.trunc(self.t)
            self.x = math.trunc(self.x)
            self.y = math.trunc(self.y)
            self.z = math.trunc(self.z)

        return self


class QHStates(QH):
    """
    A class made up of many quaternions. It also includes values for rows * columns = dimension(QHStates).
    To mimic language already in wide use in linear algebra, there are qs_types of scalar, bra, ket, op/operator
    depending on the rows and column numbers.

    Quaternion states are a semi-group with inverses. A semi-group has more than one possible identity element. For
    quaternion states, there are $2^{dim}$ possible identities.
    """
    columns: int

    QS_TYPES = ["scalar", "bra", "ket", "op", "operator"]

    def __init__(self, qs=None, qs_type: str = "ket", rows: int = 0, columns: int = 0):

        super().__init__()
        self.qs = qs
        self.qs_type = qs_type
        self.rows = rows
        self.columns = columns

        if qs_type not in self.QS_TYPES:
            print(
                "Oops, only know of these quaternion series types: {}".format(
                    self.QS_TYPES
                )
            )

        if qs is None:
            self.d, self.dim, self.dimensions = 0, 0, 0
        else:
            self.d, self.dim, self.dimensions = int(len(qs)), int(len(qs)), int(len(qs))

        self.set_qs_type(qs_type, rows, columns, copy=False)

    def set_qs_type(self: QHStates, qs_type: str = "", rows: int = 0, columns: int = 0, copy: bool = True) -> QHStates:
        """
        Set the qs_type to something sensible.

        Args:
            qs_type: str:    can be scalar, ket, bra, op or operator
            rows: int        number of rows
            columns:         number of columns
            copy:

        Returns: QHStates

        """

        # Checks.
        if rows and columns and rows * columns != self.dim:
            raise ValueError(
                f"Oops, check those values again for rows:{rows} columns:{columns} dim:{self.dim}"
            )

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
            raise Exception(
                "Oops, please set rows and columns for this quaternion series operator. Thanks."
            )

        if new_q.dim == 1:
            qs_type = "scalar"

        new_q.qs_type = qs_type

        return new_q

    def bra(self: QHStates) -> QHStates:
        """
        Quickly set the qs_type to bra by calling set_qs_type() with rows=1, columns=dim and taking a conjugate.

        Returns: QHStates

        """

        if self.qs_type == "bra":
            return self

        bra = deepcopy(self).conj()
        bra.rows = 1
        bra.columns = self.dim

        bra.qs_type = "bra" if self.dim > 1 else "scalar"

        return bra

    def ket(self: QHStates) -> QHStates:
        """
        Quickly set the qs_type to ket by calling set_qs_type() with rows=dim, columns=1 and taking a conjugate.

        Returns: QHStates

        """

        if self.qs_type == "ket":
            return self

        ket = deepcopy(self).conj()
        ket.rows = self.dim
        ket.columns = 1

        ket.qs_type = "ket" if self.dim > 1 else "scalar"

        return ket

    def op(self: QHStates, rows: int, columns: int) -> QHStates:
        """
        Quickly set the qs_type to op by calling set_qs_type().

        Args:
            rows: int:
            columns: int:

        Returns: QHStates

        """

        if rows * columns != self.dim:
            raise Exception(
                f"Oops, rows * columns != dim: {rows} * {columns}, {self.dimensions}"
            )

        op_q = deepcopy(self)

        op_q.rows = rows
        op_q.columns = columns

        if self.dim > 1:
            op_q.qs_type = "op"

        return op_q

    def __str__(self: QHStates, quiet: bool = False) -> str:
        """
        Print out all the states.

        Args:
            quiet: bool   Suppress printing the qtype.

        Returns: str

        """

        states = ""

        for n, q in enumerate(self.qs, start=1):
            states = states + f"n={n}: {q.__str__(quiet)}\n"

        return states.rstrip()

    def print_state(self: QHStates, label: str = "", spacer: bool = True, quiet: bool = True) -> None:
        """
        Utility for printing states as a quaternion series.

        Returns: None

        """

        print(label)

        # Warn if empty.
        if self.qs is None or len(self.qs) == 0:
            raise ValueError("Oops, no quaternions in the series.")

        for n, q in enumerate(self.qs):
            print(f"n={n + 1}: {q.__str__(quiet)}")

        print(f"{self.qs_type}: {self.rows}/{self.columns}")

        if spacer:
            print("")

    def equals(self: QHStates, q_2: QHStates) -> bool:
        """
        Test if two states are equal, state by state.

        Args:
            q_2: QHStates   A quaternion state to compare with self.

        Returns: QHStates

        """

        if self.dim != q_2.dim:
            return False

        result = True

        for selfq, q_2q in zip(self.qs, q_2.qs):
            if not selfq.equals(q_2q):
                result = False

        return result

    def conj(self: QHStates, conj_type: int = 0) -> QHStates:
        """
        Take the conjugates of states, default is zero, but also can do 1 or 2.

        Args:
            conj_type: int   0-3 for which one remains positive.

        Returns: QHStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.conj(conj_type))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def conj_q(self: QHStates, q_2: QHStates) -> QHStates:
        """
        Does four conjugate operators on each state.

        Returns: QHStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.conj_q(q_2))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def display_q(self: QHStates, label: str = "") -> None:
        """
        Try to display algebra in a pretty LaTeX way.

        Args:
            label: str   Text to decorate printout.

        Returns: None

        """

        if label:
            print(label)

        for i, ket in enumerate(self.qs, start=1):
            print(f"n={i}")
            ket.display_q()
            print("")

    def simple_q(self: QHStates) -> QHStates:
        """
        Simplify the states using sympy.

        Returns: QHStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.simple_q())

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def subs(self: QHStates, symbol_value_dict) -> QHStates:
        """
        Substitutes values into a symbolic expresion.

        Args:
            symbol_value_dict: Dict   {t: 3, x: 4}

        Returns: QHStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.subs(symbol_value_dict))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def scalar(self: QHStates) -> QHStates:
        """
        Returns the scalar part of a quaternion as a quaternion.

        Returns: QHStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.scalar())

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def vector(self: QHStates) -> QHStates:
        """
        Returns the vector part of a quaternion.

        Returns: QHStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.vector())

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def xyz(self: QHStates) -> List:
        """
        Returns the 3-vector for each state.

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.xyz())

        return new_states

    @staticmethod
    def q_0(dim: int = 1, qs_type: str = "ket") -> QHStates:
        """
        Return zero dim quaternion states.

        print(q_0(3))
        n=1: (0, 0, 0, 0) 0
        n=2: (0, 0, 0, 0) 0
        n=3: (0, 0, 0, 0) 0

        Args:
            dim: int
            qs_type: str

        Returns: QHStates

        """

        new_states = []

        for _ in range(dim):
            new_states.append(QH().q_0())

        q0 = QHStates(new_states, qs_type=qs_type)

        return q0

    @staticmethod
    def q_1(n: float = 1.0, dim: int = 1, qs_type: str = "ket") -> QHStates:
        """
        Return n * 1 dim quaternion states.

        print(q_1(n, 3))
        n=1: (n, 0, 0, 0) 1
        n=2: (n, 0, 0, 0) 1
        n=3: (n, 0, 0, 0) 1
        Args:
            n: float    real valued
            dim: int
            qs_type: str

        Returns: QHStates

        """

        new_states = []

        for _ in range(dim):
            new_states.append(QH().q_1(n))

        q1 = QHStates(new_states, qs_type=qs_type)

        return q1

    @staticmethod
    def q_i(n: float = 1.0, dim: int = 1, qs_type: str = "ket") -> QHStates:
        """
        Return n * i dim quaternion states.

        print(q_i(3))
        n=1: (0, n, 0, 0) i
        n=2: (0, n, 0, 0) i
        n=3: (0, n, 0, 0) i

        Args:
            n: float    n times i
            dim: int
            qs_type: str

        Returns: QHStates

        """

        new_states = []

        for _ in range(dim):
            new_states.append(QH().q_i(n))

        qi = QHStates(new_states, qs_type=qs_type)

        return qi

    @staticmethod
    def q_j(n: float = 1.0, dim: int = 1, qs_type: str = "ket") -> QHStates:
        """
        Return n * j dim quaternion states.

        print(q_j(3))
        n=1: (0, 0, n, 0) j
        n=2: (0, 0, n, 0) j
        n=3: (0, 0, n, 0) j

        Args:
            n: float
            dim: int
            qs_type: str

        Returns: QHStates

        """

        new_states = []

        for _ in range(dim):
            new_states.append(QH().q_j(n))

        qj = QHStates(new_states, qs_type=qs_type)

        return qj

    @staticmethod
    def q_k(n: float = 1, dim: int = 1, qs_type: str = "ket") -> QHStates:
        """
        Return n * k dim quaternion states.

        print(q_k(3))
        n=1: (0, 0, 0, n) 0
        n=2: (0, 0, 0, n) 0
        n=3: (0, 0, 0, n) 0

        Args:
            n: float
            dim: int
            qs_type: str

        Returns: QHStates

        """

        new_states = []

        for _ in range(dim):
            new_states.append(QH().q_k(n))

        q0 = QHStates(new_states, qs_type=qs_type)

        return q0

    @staticmethod
    def q_random(low: float = -1.0, high: float = 1.0, distribution: str = "uniform", dim: int = 1,
                 qs_type: str = "ket", q_type: str = "?", representation: str = "") -> QHStates:
        """
        Return a random-valued quaternion.
        The distribution is uniform, but one could add to options.
        It would take some work to make this clean so will skip for now.

        Args:
            low: float
            high: float
            distribution: str     have only implemented uniform distribution
            dim: int              number of states
            qs_type: str          bra/ket/op
            q_type: str           ?
            representation:       Cartesian by default

        Returns: QHState

        """

        new_states = []

        for _ in range(dim):
            new_states.append(QH().q_random(low=low, high=high, distribution=distribution, q_type=q_type,
                                            representation=representation))

        qr = QHStates(new_states, qs_type=qs_type)

        return qr

    def flip_signs(self: QHStates) -> QHStates:
        """
        Flip signs of all states.

        Returns: QHStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.flip_signs())

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def norm(self: QHStates) -> QHStates:
        """
        Norm of states.

        Returns: QHStates

        """

        new_states = []

        for bra in self.qs:
            new_states.append(bra.norm())

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def normalize(self: QHStates, n: float = 1.0, **kwargs) -> QHStates:
        """
        Normalize all states.

        Args:
            **kwargs:
            n: float   number to normalize to, default is 1.0

        Returns: QHStates

        """

        new_states = []

        zero_norm_count = 0

        for bra in self.qs:
            if bra.norm_squared().t == 0:
                zero_norm_count += 1
                new_states.append(QH().q_0())
            else:
                new_states.append(bra.normalize(n, ))

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

    def orthonormalize(self: QHStates) -> QHStates:
        """
        Given a quaternion series, returns an orthonormal basis.

        Returns: QHStates

        """

        last_q = self.qs.pop(0).normalize(math.sqrt(1 / self.dim), )
        orthonormal_qs = [last_q]

        for q in self.qs:
            qp = q.conj().product(last_q)
            orthonormal_q = q.dif(qp).normalize(math.sqrt(1 / self.dim), )
            orthonormal_qs.append(orthonormal_q)
            last_q = orthonormal_q

        return QHStates(
            orthonormal_qs, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def determinant(self: QHStates) -> QHStates:
        """
        Calculate the determinant of a 'square' quaternion series.

        Returns: QHStates

        """

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
            raise ValueError("Oops, don't know how to calculate the determinant of this one.")

        return q_det

    def add(self: QHStates, ket: QHStates) -> QHStates:
        """
        Add two states.

        Args:
            ket: QHStates

        Returns: QHStates

        """

        if (self.rows != ket.rows) or (self.columns != ket.columns):
            error_msg = "Oops, can only add if rows and columns are the same.\n"
            error_msg += f"rows are {self.rows}/{ket.rows}, col: {self.columns}/{ket.columns}"
            raise ValueError(error_msg)

        new_states = []

        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(bra.add(ket))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def summation(self: QHStates) -> QHStates:
        """
        Add them all up, return one quaternion. Not sure if this ever is meaningful.

        Returns: QHStates

        """

        result = None

        for q in self.qs:
            if result is None:
                result = q
            else:
                result = result.add(q)

        return result

    def dif(self: QHStates, ket: QHStates) -> QHStates:
        """
        Take the difference of two states.

        Args:
            ket: QHStates

        Returns: QHStates

        """

        new_states = []

        for bra, ket in zip(self.qs, ket.qs):
            new_states.append(bra.dif(ket))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def diagonal(self: QHStates, dim: int) -> QHStates:
        """
        Make a state dim * dim with q or qs along the 'diagonal'. Always returns an operator.

        Args:
            dim: int

        Returns: QHStates

        """

        diagonal = []

        if len(self.qs) == 1:
            q_values = [self.qs[0]] * dim
        elif len(self.qs) == dim:
            q_values = self.qs
        elif self.qs is None:
            raise ValueError("Oops, the qs here is None.")
        else:
            raise ValueError("Oops, need the length to be equal to the dimensions.")

        for i in range(dim):
            for j in range(dim):
                if i == j:
                    diagonal.append(q_values.pop(0))
                else:
                    diagonal.append(QH().q_0())

        return QHStates(diagonal, qs_type="op", rows=dim, columns=dim)

    def trace(self: QHStates) -> QHStates:
        """
        Return the trace as a scalar quaternion series.

        Returns: QHStates

        """

        if self.rows != self.columns:
            raise ValueError(f"Oops, not a square quaternion series: {self.rows}/{self.columns}")

        else:
            trace = self.qs[0]

        for i in range(1, self.rows):
            trace = trace.add(self.qs[i * (self.rows + 1)])

        return QHStates([trace])

    @staticmethod
    def identity(dim: int = 1, operator: bool = False, additive: bool = False, non_zeroes=None, qs_type: str = "ket") \
            -> QHStates:
        """
        Identity operator for states or operators which are diagonal.

        Args:
            dim: int
            operator: bool
            additive: bool
            non_zeroes:
            qs_type: str

        Returns: QHStates

        """

        if additive:
            id_q = [QH().q_0() for _ in range(dim)]

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
            id_q = [QH().q_1() for _ in range(dim)]

        if operator:
            q_1 = QHStates(id_q)
            ident = QHStates.diagonal(q_1, dim)

        else:
            ident = QHStates(id_q, qs_type=qs_type)

        return ident

    def product(self: QHStates, q_2: QHStates, kind: str = "", reverse: bool = False) -> QHStates:
        """
        Forms the quaternion product for each state.

        Args:
            q_2: QHStates
            kind: str
            reverse: bool

        Returns: QHStates

        """

        self_copy = deepcopy(self)
        q_2_copy = deepcopy(q_2)
        qs_left, qs_right = QHStates(), QHStates()

        # Diagonalize if need be.
        if ((self.rows == q_2.rows) and (self.columns == q_2.columns)) or (
                "scalar" in [self.qs_type, q_2.qs_type]
        ):

            if self.columns == 1:
                qs_right = q_2_copy
                qs_left = self_copy.diagonal(qs_right.rows)

            elif q_2.rows == 1:
                qs_left = self_copy
                qs_right = q_2_copy.diagonal(qs_left.columns)

            else:
                qs_left = self_copy
                qs_right = q_2_copy

        # Typical matrix multiplication criteria.
        elif self.columns == q_2.rows:
            qs_left = self_copy
            qs_right = q_2_copy

        else:
            print(
                "Oops, cannot multiply series with row/column dimensions of {}/{} to {}/{}".format(
                    self.rows, self.columns, q_2.rows, q_2.columns
                )
            )

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
            [QH().q_0(q_type="") for _i in range(outer_column_max)]
            for _j in range(outer_row_max)
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

    def inverse(self: QHStates, additive: bool = False) -> QHStates:
        """
        Inversing bras and kets calls inverse() once for each.
        Inversing operators is more tricky as one needs a diagonal identity matrix.

        Args:
            additive: bool

        Returns: QHStates

        """

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
                    q_2 = self.qs[1].flip_signs().product(detinv)
                    q2 = self.qs[2].flip_signs().product(detinv)
                    q3 = self.qs[0].product(detinv)

                    q_inv = QHStates(
                        [q0, q_2, q2, q3],
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
                    q_2 = (
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
                        [q0, q_2, q2, q3, q4, q5, q6, q7, q8],
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

    def divide_by(self: QHStates, ket: QHStates, additive: bool = False) -> QHStates:
        """
        Take a quaternion and divide it by another using an inverse. Can only handle up to 3 states.

        Args:
            ket: QHStates
            additive: bool

        Returns: QHStates

        """

        new_states = []

        ket_inv = ket.inverse(additive)

        for bra, k in zip(self.qs, ket_inv.qs):
            new_states.append(bra.product(k))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def triple_product(self: QHStates, ket: QHStates, ket_2: QHStates) -> QHStates:
        """
        A quaternion triple product of states.

        Args:
            ket: QHStates
            ket_2: QHStates

        Returns: QHStates

        """

        new_states = []

        for bra, k, k2 in zip(self.qs, ket.qs, ket_2.qs):
            new_states.append(bra.product(k).product(k2))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def rotate(self: QHStates, ket: QHStates) -> QHStates:
        """
        Rotate one state by another.

        Args:
            ket: QHStates

        Returns: QHStates

        """

        new_states = []

        for bra, k in zip(self.qs, ket.qs):
            new_states.append(bra.rotate(k))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def rotation_and_or_boost(self: QHStates, ket: QHStates) -> QHStates:
        """
        Do state-by-state rotations or boosts.

        Args:
            ket: QHStates

        Returns: QHStates

        """

        new_states = []

        for bra, k in zip(self.qs, ket.qs):
            new_states.append(bra.rotation_and_or_boost(k))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    @staticmethod
    def Lorentz_next_rotation(q: QHStates, q_2: QHStates) -> QHStates:
        """
        Does multiple rotations of a QHState given another QHState of equal dimensions.

        Args:
            q: QHStates
            q_2: QHStaes

        Returns:

        """

        if q.dim != q_2.dim:
            raise ValueError(
                "Oops, this tool requires 2 quaternion states with the same number of dimensions."
            )

        new_states = []

        for ket, q2 in zip(q.qs, q_2.qs):
            new_states.append(QH.Lorentz_next_rotation(ket, q2))

        return QHStates(
            new_states, qs_type=q.qs_type, rows=q.rows, columns=q.columns
        )

    @staticmethod
    def Lorentz_next_boost(q: QHStates, q_2: QHStates) -> QHStates:
        """
        Does multiple boosts of a QHState given another QHState of equal dimensions.

        Args:
            q: QHStates
            q_2: QHStates

        Returns: QHStates

        """

        if q.dim != q_2.dim:
            raise ValueError(
                "Oops, this tool requires 2 quaternion states with the same number of dimensions."
            )

        new_states = []

        for ket, q2 in zip(q.qs, q_2.qs):
            new_states.append(QH.Lorentz_next_boost(ket, q2))

        return QHStates(
            new_states, qs_type=q.qs_type, rows=q.rows, columns=q.columns
        )

    def g_shift(self: QHStates, g_factor: float = 1.0, g_form="exp") -> QHStates:
        """
        Do the g_shift to each state.

        Args:
            g_factor: float
            g_form: str

        Returns: QHStates

        """

        new_states = []

        for bra in self.qs:
            new_states.append(bra.g_shift(g_factor, g_form))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    @staticmethod
    def bracket(bra: QHStates, op: QHStates, ket: QHStates) -> QHStates:
        """
        Forms <bra|op|ket>. Note: if fed 2 kets, will take a conjugate.

        Args:
            bra: QHStates
            op: QHStates
            ket: QHStates

        Returns: QHStates

        """

        flip = 0

        if bra.qs_type == "ket":
            bra = bra.bra()
            flip += 1

        if ket.qs_type == "bra":
            ket = ket.ket()
            flip += 1

        if flip == 1:
            print("fed 2 bras or kets, took a conjugate. Double check.")

        b = bra.product(op).product(ket)

        return b

    @staticmethod
    def braket(bra: QHStates, ket: QHStates) -> QHStates:
        """
        Forms <bra|ket>, no operator. Note: if fed 2 kets, will take a conjugate.

        Args:
            bra: QHStates
            ket: QHStates

        Returns: QHStates

        """

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

    def op_q(self: QHStates, q: QH, first: bool = True, kind: str = "", reverse: bool = False) -> QHStates:
        """
        Multiply an operator times a quaternion, in that order. Set first=false for n * Op

        Args:
            q: QH
            first: bool
            kind: str
            reverse: bool

        Returns: QHStates

        """

        new_states = []

        for op in self.qs:

            if first:
                new_states.append(op.product(q, kind, reverse))

            else:
                new_states.append(q.product(op, kind, reverse))

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def square(self: QHStates) -> QHStates:
        """
        The square of each state.

        Returns: QHStates

        """

        new_states = []

        for bra in self.qs:
            new_states.append(bra.square())

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def norm_squared(self: QHStates) -> QHStates:
        """
        Take the inner product, returning a scalar series.

        Returns: QHStates

        """

        norm_scalar = self.set_qs_type("bra").conj().product(self.set_qs_type("ket"))

        return norm_scalar

    def norm_squared_of_vector(self: QHStates) -> QHStates:
        """
        Take the inner product of the vector, returning a scalar.

        Returns: QHStates

        """

        vector_norm_scalar: QHStates = self.set_qs_type("bra").vector().conj().product(self.set_qs_type("ket").vector())

        return vector_norm_scalar

    def transpose(self: QHStates, m: int = None, n: int = None) -> QHStates:
        """
        Transposes a series.

        Args:
            m: int
            n: int

        Returns: QHStates

        """

        if m is None:
            # test if it is square.
            if math.sqrt(self.dim).is_integer():
                m = int(sp.sqrt(self.dim))
                n = m

        if n is None:
            n = int(self.dim / m)

        matrix = [[0 for _x in range(m)] for _y in range(n)]

        for mi in range(m):
            for ni in range(n):
                matrix[ni][mi] = self.qs[mi * n + ni]

        qs_t = []

        for t in matrix:
            for q in t:
                qs_t.append(q)

        # Switch rows and columns.
        return QHStates(qs_t, rows=self.columns, columns=self.rows)

    def Hermitian_conj(self: QHStates, m: int = None, n: int = None, conj_type: int = 0) -> QHStates:
        """
        Returns the Hermitian conjugate.

        Args:
            m: int
            n: int
            conj_type: int    0-3

        Returns: QHStates

        """

        return self.transpose(m, n).conj(conj_type)

    def dagger(self: QHStates, m: int = None, n: int = None, conj_type: int = 0) -> QHStates:
        """
        Just calls Hermitian_conj()

        Args:
            m: int
            n: int
            conj_type: 0-3

        Returns: QHStates

        """

        return self.Hermitian_conj(m, n, conj_type)

    def is_square(self: QHStates) -> bool:
        """
        Tests if a quaternion series is square, meaning the dimenion is n^2.

        Returns: bool

        """

        return math.sqrt(self.dim).is_integer()

    def is_Hermitian(self: QHStates) -> bool:
        """
        Tests if a series is Hermitian.

        Returns: bool

        """

        hc = self.Hermitian_conj()

        return self.equals(hc)

    @staticmethod
    def sigma(kind: str = "x", theta: float = None, phi: float = None) -> QHStates:
        """
        Returns a sigma when given a type like, x, y, z, xy, xz, yz, xyz, with optional angles theta and phi.

        Args:
            kind: str  x, y, z, xy, etc
            theta: float   an angle
            phi: float     an angle

        Returns:

        """

        q0, q_2, qi = QH().q_0(), QH().q_1(), QH().q_i()

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

        x_factor = q_2.product(QH([sin_theta * cos_phi, 0, 0, 0]))
        y_factor = qi.product(QH([sin_theta * sin_phi, 0, 0, 0]))
        z_factor = q_2.product(QH([cos_theta, 0, 0, 0]))

        sigma = Bunch()
        sigma.x = QHStates([q0, x_factor, x_factor, q0], "op")
        sigma.y = QHStates([q0, y_factor, y_factor.flip_signs(), q0], "op")
        sigma.z = QHStates([z_factor, q0, q0, z_factor.flip_signs()], "op")

        sigma.xy = sigma.x.add(sigma.y)
        sigma.xz = sigma.x.add(sigma.z)
        sigma.yz = sigma.y.add(sigma.z)
        sigma.xyz = sigma.x.add(sigma.y).add(sigma.z)

        if kind not in sigma:
            raise ValueError("Oops, I only know about x, y, z, and their combinations.")

        return sigma[kind].normalize()

    def sin(self: QHStates) -> QHStates:
        """
        sine of states.

        Returns: QHStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.sin())

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def cos(self: QHStates) -> QHStates:
        """
        cosine of states.

        Returns:

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.cos())

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def tan(self: QHStates) -> QHStates:
        """
        tan() of states.

        Returns: QHStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.tan())

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def sinh(self: QHStates) -> QHStates:
        """
        sinh() of states.

        Returns:

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.sinh())

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def cosh(self: QHStates) -> QHStates:
        """
        cosh() of states.

        Returns: QHStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.cosh())

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def tanh(self: QHStates) -> QHStates:
        """
        tanh() of states.

        Returns: QHStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.tanh())

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def exp(self: QHStates) -> QHStates:
        """
        exponential of states.

        Returns: QHStates

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.exp())

        return QHStates(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )
