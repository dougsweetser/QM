#!/usr/bin/env python
# coding: utf-8
"""
Developing Quaternions for iPython

Define a class Q to manipulate quaternions as Hamilton would have done it so many years ago.
The "q_type" is a little bit of text to leave a trail of breadcrumbs about how a particular quaternion was generated.

The class Qs is a semi-group with inverses, that has a row * column = dimensions as seen in
quantum mechanics. The importance of Qs is that one can deal with a great number of quaternions together. One quaternion
cannot tell a story but a few tens of thousands can.

This library was recently refactored so that functions are on the same level as the classes Q and Qs.

The function calls for Q and Qs are meant to be very similar.
"""

from __future__ import annotations
import math
from copy import deepcopy
from types import FunctionType

import itertools
import numpy as np
import pandas as pd
import sympy as sp
from typing import Dict, List, Union
from IPython.display import display
from bunch import Bunch


class Q(object):
    """
    Quaternions as Hamilton would have defined them, on the manifold R^4.
    Different representations are possible.
    """

    def __init__(self, values: List = None, q_type: object = "Q", representation: str = ""):

        if values is None:
            self.df = pd.DataFrame(data=[0, 0, 0, 0], index=["t", "x", "y", "z"])
        elif len(values) == 4:
            self.df = pd.DataFrame(data=values, index=["t", "x", "y", "z"])

        elif len(values) == 8:
            self.df = pd.DataFrame(data=[values[0] - values[1], values[2] - values[3],
                                         values[4] - values[5], values[6] - values[7]], index=["t", "x", "y", "z"])

        else:
            raise ValueError(f"The program accepts lists/arrays of 4 or 8 dimensions, not {len(values)}")

        self.t, self.x, self.y, self.z = self.df.at["t", 0], self.df.at["x", 0], self.df.at["y", 0], self.df.at["z", 0]

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

    def print_state(self: Q, label: str = "", spacer: bool = True, quiet: bool = True) -> None:
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

    def is_symbolic(self: Q) -> bool:
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

    def txyz_2_representation(self: Q, representation: str = "") -> List:
        """
        Given a quaternion in Cartesian coordinates
        returns one in another representation.
        Only 'polar' and 'spherical' are done so far.

        Args:
            representation: bool

        Return: Q

        """

        symbolic = self.is_symbolic()

        if representation == "":
            rep = [self.t, self.x, self.y, self.z]

        elif representation == "polar":
            amplitude = (self.t ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2) ** (
                    1 / 2
            )

            abs_v = abs_of_vector(self).t

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

    def representation_2_txyz(self: Q, representation: str = "") -> List:
        """
        Converts something in a representation such as
        polar, spherical
        and returns a Cartesian representation.

        Args:
            representation: str   can be polar or spherical

        Return: Q

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

    def check_representations(self: Q, q_2: Q) -> bool:
        """
        Checks if self and q_2 have the same representation.

        Args:
            q_2: Q

        Returns: bool

        """

        if self.representation == q_2.representation:
            return True

        else:
            raise Exception(f"Oops, 2 have different representations: {self.representation} {q_2.representation}")

    def display_q(self: Q, label: str = ""):
        """
        Prints LaTeX-like output, one line for each of th 4 terms.

        Args:
            label: str  an additional bit of text.

        Returns:

        """

        if label:
            print(label)
        display(self.t)
        display(self.x)
        display(self.y)
        display(self.z)

    def simple_q(self: Q) -> Q:
        """
        Runs symboy.simplify() on each term, good for symbolic expression.

        Returns: Q

        """

        self.t = sp.simplify(self.t)
        self.x = sp.simplify(self.x)
        self.y = sp.simplify(self.y)
        self.z = sp.simplify(self.z)
        return self

    def expand_q(self) -> Q:
        """
        Runs expand on each term, good for symbolic expressions.

        Returns: Q

        """
        """Expand each term."""

        self.t = sp.expand(self.t)
        self.x = sp.expand(self.x)
        self.y = sp.expand(self.y)
        self.z = sp.expand(self.z)
        return self

    def subs(self: Q, symbol_value_dict: annotations.Dict) -> Q:
        """
        Evaluates a quaternion using sympy values and a dictionary {t:1, x:2, etc}.

        Args:
            symbol_value_dict: Dict

        Returns: Q

        """

        t1 = self.t.subs(symbol_value_dict)
        x1 = self.x.subs(symbol_value_dict)
        y1 = self.y.subs(symbol_value_dict)
        z1 = self.z.subs(symbol_value_dict)

        q_txyz = Q(
            [t1, x1, y1, z1], q_type=self.q_type, representation=self.representation
        )

        return q_txyz

    def t(self: Q) -> np.array:
        """
        Returns the t as an np.array.

        Returns: np.array

        """

        return np.array([self.t])

    def xyz(self: Q) -> np.array:
        """
        Returns the vector_q x, y, z as an np.array.

        Returns: np.array

        """

        return np.array([self.x, self.y, self.z])


class Qs(object):
    """
    A class made up of many quaternions. It also includes values for rows * columns = dimension(Qs).
    To mimic language already in wide use in linear algebra, there are qs_types of scalar, bra, ket, op/operator
    depending on the rows and column numbers.

    Quaternion states are a semi-group with inverses. A semi-group has more than one possible identity element. For
    quaternion states, there are $2^{dim}$ possible identities.
    """

    QS_TYPES = ["scalar_q", "bra", "ket", "op", "operator"]

    def __init__(self, qs=None, qs_type: str = "ket", rows: int = 0, columns: int = 0):

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
            self.qs = [q0()]
            self.d, self.dim, self.dimensions = 0, 0, 0
            self.df = pd.DataFrame([[0, 0, 0, 0]],  columns=('t', 'x', 'y', 'z'))
        else:
            self.d, self.dim, self.dimensions = int(len(qs)), int(len(qs)), int(len(qs))
            self.df = pd.DataFrame([[q.t, q.x, q.y, q.z] for q in qs], columns=('t', 'x', 'y', 'z'))

        if not self.qs[0].is_symbolic():
            mins = self.df.min()
            self.min = Bunch()
            self.min.t = float(math.floor(mins.t))
            self.min.x = float(math.floor(mins.x))
            self.min.y = float(math.floor(mins.y))
            self.min.z = float(math.floor(mins.z))

            maxs = self.df.max()
            self.max = Bunch()
            self.max.t = float(math.ceil(maxs.t))
            self.max.x = float(math.ceil(maxs.x))
            self.max.y = float(math.ceil(maxs.y))
            self.max.z = float(math.ceil(maxs.z))

        self.set_qs_type(qs_type, rows, columns, copy=False)

    def set_qs_type(self: Qs, qs_type: str = "", rows: int = 0, columns: int = 0, copy: bool = True) -> Qs:
        """
        Set the qs_type to something sensible.

        Args:
            qs_type: str:    can be scalar_q, ket, bra, op or operator
            rows: int        number of rows
            columns:         number of columns
            copy:

        Returns: Qs

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
                qs_type = "scalar_q"
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
            qs_type = "scalar_q"

        new_q.qs_type = qs_type

        return new_q

    def bra(self: Qs) -> Qs:
        """
        Quickly set the qs_type to bra by calling set_qs_type() with rows=1, columns=dim and taking a conjugate.

        Returns: Qs

        """

        if self.qs_type == "bra":
            return self

        bra = conjs(deepcopy(self))
        bra.rows = 1
        bra.columns = self.dim
        bra.qs_type = "bra" if self.dim > 1 else "scalar_q"

        return bra

    def ket(self: Qs) -> Qs:
        """
        Quickly set the qs_type to ket by calling set_qs_type() with rows=dim, columns=1 and taking a conjugate.

        Returns: Qs

        """

        if self.qs_type == "ket":
            return self

        ket = conjs(deepcopy(self))
        ket.rows = self.dim
        ket.columns = 1

        ket.qs_type = "ket" if self.dim > 1 else "scalar_q"

        return ket

    def op(self: Qs, rows: int, columns: int) -> Qs:
        """
        Quickly set the qs_type to op by calling set_qs_type().

        Args:
            rows: int:
            columns: int:

        Returns: Qs

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

    def __str__(self: Qs, quiet: bool = False) -> str:
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

    def print_state(self: Qs, label: str = "", spacer: bool = True, quiet: bool = True) -> None:
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

    def display_q(self: Qs, label: str = "") -> None:
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

    def simple_q(self: Qs) -> Qs:
        """
        Simplify the states using sympy.

        Returns: Qs

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.simple_q())

        return Qs(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def subs(self: Qs, symbol_value_dict) -> Qs:
        """
        Substitutes values into a symbolic expresion.

        Args:
            symbol_value_dict: Dict   {t: 3, x: 4}

        Returns: Qs

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.subs(symbol_value_dict))

        return Qs(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )

    def t(self: Qs) -> List:
        """
        Returns the t for each state.

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.t())

        return new_states

    def xyz(self: Qs) -> List:
        """
        Returns the 3-vector_q for each state.

        """

        new_states = []

        for ket in self.qs:
            new_states.append(ket.xyz())

        return new_states

    @staticmethod
    def bracket(bra: Qs, op: Qs, ket: Qs) -> Qs:
        """
        Forms <bra|op|ket>. Note: if fed 2 kets, will take a conjugate.

        Args:
            bra: Qs
            op: Qs
            ket: Qs

        Returns: Qs

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

        b = products(bra, products(op, ket))

        return b

    @staticmethod
    def braket(bra: Qs, ket: Qs) -> Qs:
        """
        Forms <bra|ket>, no operator. Note: if fed 2 kets, will take a conjugate.

        Args:
            bra: Qs
            ket: Qs

        Returns: Qs

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

        b = products(bra, ket)

        return b

    def op_q(self: Qs, q: Qs, first: bool = True, kind: str = "", reverse: bool = False) -> Qs:
        """
        Multiply an operator times a quaternion, in that order. Set first=false for n * Op

        Args:
            q: Qs
            first: bool
            kind: str
            reverse: bool

        Returns: Qs

        """

        new_states = []

        for op in self.qs:

            if first:
                new_states.append(product(op, q, kind, reverse))

            else:
                new_states.append(product(q, op, kind, reverse))

        return Qs(
            new_states, qs_type=self.qs_type, rows=self.rows, columns=self.columns
        )


# # Aids to transform Q functions to Qs functions.

def q_to_qs_function(func, q_1):
    """
    Utility to transform quaternion functions to quaternion state function
    that operate separately on each qs state.

    Args:
        func:    pointer to a function
        q_1: Q   a quaternion argument

    Returns: Qs

    """

    return Qs([func(q) for q in q_1.qs], qs_type=q_1.qs_type, rows=q_1.rows, columns=q_1.columns)


def qq_to_qs_function(func, q_1, q_2):
    """
    Utility to transform quaternion functions to quaternion state function
    that operate separately on each qs state.

    Args:
        func:    pointer to a function
        q_1: Qs   a quaternion state
        q_2: Qs   a quaternion state

    Returns: Qs

    """

    return Qs([func(q, r) for q, r in zip(q_1.qs, q_2.qs)], qs_type=q_1.qs_type, rows=q_1.rows, columns=q_1.columns)


def qqq_to_qs_function(func, q_1, q_2, q_3):
    """
    Utility to transform quaternion functions to quaternion series function
    that operate separately on each qs state.

    Args:
        func:    pointer to a function
        q_1: Q   a quaternion argument
        q_2: Q   a quaternion argument
        q_3: Q   a quaternion argument

    Returns: Qs

    """

    return Qs([func(q, r, s) for q, r, s in zip(q_1.qs, q_2.qs, q_3.qs)], qs_type=q_1.qs_type, rows=q_1.rows,
              columns=q_1.columns)


def qs_to_q_function(func: FunctionType, q_1: Qs) -> Q:
    """
    Utility to transform quaternion series functions to a quaternion function.

    Args:
        func:     Pointer to a function
        q_1: Qs   A quaternion series

    Returns: Q

    """

    scalar = func(q_1)

    if scalar.qs_type != "scalar_q":
        raise Exception(f"Oops, does not evaluate to a scalar: {scalar}")

    return scalar.qs[0]


def qs_qs_to_q_function(func: FunctionType, q_1: Qs, q_2: Qs) -> Q:
    """
    Utility to transform quaternion series functions to a quaternion function.

    Args:
        func:     Pointer to a function
        q_1: Qs   A quaternion series
        q_2: Qs   A quaternion series

    Returns: Q

    """

    scalar = func(q_1, q_2)

    if scalar.qs_type != "scalar_q":
        raise Exception(f"Oops, does not evaluate to a scalar: {scalar}")

    return scalar.qs[0]


# # Parts of quaternions

def scalar_q(q_1: Q) -> Q:
    """
    Returns the scalar_q part of a quaternion as a quaternion.

    $ \rm{scalar_q(q)} = (q + q^*)/2 = (t, 0) $

    Returns: Q

    Args:
        q_1: Q

    Returns: Q

    """

    end_q_type = f"scalar_q({q_1.q_type})"
    s = Q([q_1.t, 0, 0, 0], q_type=end_q_type, representation=q_1.representation)
    return s


def scalar_qs(q_1: Qs) -> Qs:
    f"""{scalar_q.__doc__}""".replace("Q", "Qs")
    return q_to_qs_function(scalar_q, q_1)


def vector_q(q_1: Q) -> Q:
    """
    Returns the vector_q part of a quaternion.
    $ \rm{vector_q(q)} = (q\_1 - q\_1^*)/2 = (0, R) $

    Returns: Q

    """

    end_q_type = f"vector_q({q_1.q_type})"

    v = Q(
        [0, q_1.x, q_1.y, q_1.z],
        q_type=end_q_type,
        representation=q_1.representation,
    )
    return v


def vector_qs(q_1: Qs) -> Qs:
    f"""{vector_q.__doc__}""".replace("Q", "Qs")
    return q_to_qs_function(vector_q, q_1)


def q0(q_type: str = "0", representation: str = "") -> Q:
    """
    Return a zero quaternion.

    $ q\_0() = 0 = (0, 0) $

    Returns: Q

    """

    return Q([0, 0, 0, 0], q_type=q_type, representation=representation)


def q0s(dim: int = 1, qs_type: str = "ket") -> Qs:
    f"""{q0.__doc__}""".replace("Q", "Qs")
    return Qs([q0() for _ in range(dim)], qs_type=qs_type)


def q1(n: float = 1.0, q_type: str = "1", representation: str = "") -> Q:
    """
    Return a real-valued quaternion multiplied by n.

    $ q\_1(n) = n = (n, 0) $

    Returns: Q

    """

    return Q([n, 0, 0, 0], q_type=q_type, representation=representation)


def q1s(n: float = 1.0, dim: int = 1, qs_type: str = "ket") -> Qs:
    f"""{q1.__doc__}""".replace("Q", "Qs")
    return Qs([q1(n) for _ in range(dim)], qs_type=qs_type)


def qi(n: float = 1.0, q_type: str = "i", representation: str = "") -> Q:
    """
    Return a quaternion with $ i * n $.

    $ q\_i(n) = n i = (0, n i) $

    Returns: Q

    """

    return Q([0, n, 0, 0], q_type=q_type, representation=representation)


def qis(n: float = 1.0, dim: int = 1, qs_type: str = "ket") -> Qs:
    f"""{qi.__doc__}""".replace("Q", "Qs")
    return Qs([qi(n) for _ in range(dim)], qs_type=qs_type)


def qj(n: float = 1.0, q_type: str = "j", representation: str = "") -> Q:
    """
    Return a quaternion with $ j * n $.

    $ q\_j(n) = n j = (0, n j) $

    Returns: Q

    """

    return Q([0, 0, n, 0], q_type=q_type, representation=representation)


def qjs(n: float = 1.0, dim: int = 1, qs_type: str = "ket") -> Qs:
    f"""{qj.__doc__}""".replace("Q", "Qs")
    return Qs([qj(n) for _ in range(dim)], qs_type=qs_type)


def qk(n: float = 1, q_type: str = "k", representation: str = "") -> Q:
    """
    Return a quaternion with $ k * n $.

    $ q\_k(n) = n k =(0, n k) $

    Returns: Q

    """

    return Q([0, 0, 0, n], q_type=q_type, representation=representation)


def qks(n: float = 1.0, dim: int = 1, qs_type: str = "ket") -> Qs:
    f"""{qk.__doc__}""".replace("Q", "Qs")
    return Qs([qk(n) for _ in range(dim)], qs_type=qs_type)


def qrandom(low: float = -1.0, high: float = 1.0, distribution: str = "uniform", q_type: str = "?",
            representation: str = "") -> Q:
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

    Returns: Q

    """

    random_distributions = Bunch()
    random_distributions.uniform = np.random.uniform

    qr = Q(
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


def qrandoms(low: float = -1.0, high: float = 1.0, distribution: str = "uniform", dim: int = 1,
             qs_type: str = "ket") -> Qs:
    f"""{qrandom.__doc__}""".replace("Q", "Qs")
    return Qs([qrandom(low, high, distribution) for _ in range(dim)], qs_type=qs_type)


def dupe(q_1: Q) -> Q:
    """
    Return a duplicate copy.

    Returns: Q

    """

    du = Q(
        [q_1.t, q_1.x, q_1.y, q_1.z],
        q_type=q_1.q_type,
        representation=q_1.representation,
    )
    return du


def equal(q_1: Q, q_2: Q, scalar: bool = True, vector: bool = True) -> bool:
    """
    Tests if q1 and q_2 quaternions are close to equal. If vector_q is set to False, will compare
    only the scalar_q. If scalar_q is set to False, will compare 3-vectors.

    $ q.equal(q\_2) = q == q\_2 = True $

    Args:
        q_1: Q
        q_2: Q
        scalar: bool    Will compare quaternion scalars
        vector: bool    Will compare quaternion 3-vectors

    Returns: bool

    """

    q_1.check_representations(q_2)

    q_1_t, q_1_x, q_1_y, q_1_z = (
        sp.expand(q_1.t),
        sp.expand(q_1.x),
        sp.expand(q_1.y),
        sp.expand(q_1.z),
    )
    q_2_t, q_2_x, q_2_y, q_2_z = (
        sp.expand(q_2.t),
        sp.expand(q_2.x),
        sp.expand(q_2.y),
        sp.expand(q_2.z),
    )

    if not scalar and not vector:
        raise ValueError("Equals needs either scalar_q or vector_q to be set to True")

    t_equals = math.isclose(q_1_t, q_2_t)
    x_equals = math.isclose(q_1_x, q_2_x)
    y_equals = math.isclose(q_1_y, q_2_y)
    z_equals = math.isclose(q_1_z, q_2_z)

    result = False

    if scalar and not vector and t_equals:
        result = True

    elif not scalar and vector and x_equals and y_equals and z_equals:
        result = True

    elif scalar and vector and t_equals and x_equals and y_equals and z_equals:
        result = True

    return result


def equals(q_1: Qs, q_2: Qs, scalar: bool = True, vector: bool = True) -> bool:
    f"""{equal.__doc__}""".replace("Q", "Qs")

    if q_1.dim != q_2.dim:
        return False

    result = True

    for q_1q, q_2q in zip(q_1.qs, q_2.qs):
        if not equal(q_1q, q_2q, scalar, vector):
            result = False

    return result


def conj(q_1: Q, conj_type: int = 0) -> Q:
    """
    There are 4 types of conjugates.

    $ q.conj(0) = q^* =(t, -x, -y, -z) $
    $ q.conj(1) = (i q i)^* =(-t, x, -y, -z) $
    $ q.conj(2) = (j q j)^* =(-t, -x, y, -z) $
    $ q.conj(3) = (k q k)^* =(-t, -x, -y, z) $

    Args:
        q_1: Q
        conj_type: int:   0-3 depending on who stays positive.

    Returns: Q

    """

    end_q_type = f"{q_1.q_type}*"
    c_t, c_x, c_y, c_z = q_1.t, q_1.x, q_1.y, q_1.z
    cq = Q()

    if conj_type % 4 == 0:
        cq.t = c_t
        if c_x != 0:
            cq.x = -1 * c_x
        if c_y != 0:
            cq.y = -1 * c_y
        if c_z != 0:
            cq.z = -1 * c_z

    elif conj_type % 4 == 1:
        if c_t != 0:
            cq.t = -1 * c_t
        cq.x = c_x
        if c_y != 0:
            cq.y = -1 * c_y
        if c_z != 0:
            cq.z = -1 * c_z
        end_q_type += "1"

    elif conj_type % 4 == 2:
        if c_t != 0:
            cq.t = -1 * c_t
        if c_x != 0:
            cq.x = -1 * c_x
        cq.y = c_y
        if c_z != 0:
            cq.z = -1 * c_z
        end_q_type += "2"

    elif conj_type % 4 == 3:
        if c_t != 0:
            cq.t = -1 * c_t
        if c_x != 0:
            cq.x = -1 * c_x
        if c_y != 0:
            cq.y = -1 * c_y
        cq.z = c_z
        end_q_type += "3"

    cq.q_type = end_q_type
    cq.representation = q_1.representation

    return cq


def conjs(q_1: Qs, conj_type: int = 0) -> Qs:
    f"""{conj.__doc__}""".replace("Q", "Qs")
    return Qs([conj(q, conj_type) for q in q_1.qs], qs_type=q_1.qs_type)


def conj_q(q_1: Q, q_2: Q) -> Q:
    """
    Given a quaternion with 0s or 1s, will do the standard conjugate, first conjugate
    second conjugate, sign flip, or all combinations of the above.

    q.conj(q(1, 1, 1, 1)) = q.conj(0).conj(1).conj(2).conj(3)

    Args:
        q_1: Q
        q_2: Q    Use a quaternion to do one of 4 conjugates in combinations

    Returns: Q

    """

    _conj = deepcopy(q_1)

    if q_2.t:
        _conj = conj(_conj, conj_type=0)

    if q_2.x:
        _conj = conj(_conj, conj_type=1)

    if q_2.y:
        _conj = conj(_conj, conj_type=2)

    if q_2.z:
        _conj = flip_sign(_conj)

    return _conj


def conj_qs(q_1: Qs, q_2: Q) -> Qs:
    f"""{conj_q.__doc__}""".replace("Q", "Qs")
    return Qs([conj_q(q, q_2) for q in q_1.qs], qs_type=q_1.qs_type)


def flip_sign(q_1: Q) -> Q:
    """
    Flip the signs of all terms.

    $ q.flip\_sign() = -q = (-t, -R) $

    Args:
        q_1: Q

    Returns: Q
    """

    end_q_type = f"-{q_1.q_type}"

    flip_t, flip_x, flip_y, flip_z = q_1.t, q_1.x, q_1.y, q_1.z

    flip_q = Q(q_type=end_q_type, representation=q_1.representation)
    if flip_t != 0:
        flip_q.t = -1 * flip_t
    if flip_x != 0:
        flip_q.x = -1 * flip_x
    if flip_y != 0:
        flip_q.y = -1 * flip_y
    if flip_z != 0:
        flip_q.z = -1 * flip_z

    return flip_q


def flip_signs(q_1: Qs) -> Qs:
    f"""{flip_sign.__doc__}""".replace("Q", "Qs")
    return q_to_qs_function(flip_sign, q_1)


def vahlen_conj(q_1: Q, conj_type: str = "-", q_type: str = "vc") -> Q:
    """
    Three types of conjugates dash, apostrophe, or star as done by Vahlen in 1901.

    q.vahlen_conj("-") = q^* = (t, -x, -y, -z)

    q.vahlen_conj("'") = (k q k)^* = (t, -x, -y, z)

    q.vahlen_conj("*") = -(k q k)^* = (t, x, y, -z)

    Args:
        q_1: Q
        conj_type: str:    3 sorts, dash apostrophe,
        q_type: str:

    Returns:

    """

    vc_t, vc_x, vc_y, vc_z = q_1.t, q_1.x, q_1.y, q_1.z
    c_q = Q()

    if conj_type == "-":
        c_q.t = vc_t
        if vc_x != 0:
            c_q.x = -1 * vc_x
        if vc_y != 0:
            c_q.y = -1 * vc_y
        if vc_z != 0:
            c_q.z = -1 * vc_z
        q_type += "*-"

    if conj_type == "'":
        c_q.t = vc_t
        if vc_x != 0:
            c_q.x = -1 * vc_x
        if vc_y != 0:
            c_q.y = -1 * vc_y
        c_q.z = vc_z
        q_type += "*'"

    if conj_type == "*":
        c_q.t = vc_t
        c_q.x = vc_x
        c_q.y = vc_y
        if vc_z != 0:
            c_q.z = -1 * vc_z
        q_type += "*"

    c_q.q_type = f"{q_1.q_type}{q_type}"
    c_q.representation = q_1.representation

    return c_q


def valhen_conjs(q_1: Qs, conj_type: str = "-") -> Qs:
    f"""{vahlen_conj.__doc__}""".replace("Q", "Qs")
    return Qs([vahlen_conj(q, conj_type) for q in q_1.qs], qs_type=q_1.qs_type)


def _commuting_products(q_1: Q, q_2: Q) -> Dict:
    """
    Returns a dictionary with the commuting products. For internal use.

    Args:
        q_1: Q
        q_2: Q

    Returns: Dict

    """

    s_t, s_x, s_y, s_z = q_1.t, q_1.x, q_1.y, q_1.z
    q_2_t, q_2_x, q_2_y, q_2_z = q_2.t, q_2.x, q_2.y, q_2.z

    product_dict = {
        "tt": s_t * q_2_t,
        "xx+yy+zz": s_x * q_2_x + s_y * q_2_y + s_z * q_2_z,
        "tx+xt": s_t * q_2_x + s_x * q_2_t,
        "ty+yt": s_t * q_2_y + s_y * q_2_t,
        "tz+zt": s_t * q_2_z + s_z * q_2_t,
    }

    return product_dict


def _anti_commuting_products(q_1: Q, q_2: Q) -> Dict:
    """
    Returns a dictionary with the three anti-commuting products. For internal use.

    Args:
        q_1: Q
        q_2: Q

    Returns: Dict

    """

    s_x, s_y, s_z = q_1.x, q_1.y, q_1.z
    q_2_x, q_2_y, q_2_z = q_2.x, q_2.y, q_2.z

    dif_dict = {
        "yz-zy": s_y * q_2_z - s_z * q_2_y,
        "zx-xz": s_z * q_2_x - s_x * q_2_z,
        "xy-yx": s_x * q_2_y - s_y * q_2_x,
        "zy-yz": -s_y * q_2_z + s_z * q_2_y,
        "xz-zx": -s_z * q_2_x + s_x * q_2_z,
        "yx-xy": -s_x * q_2_y + s_y * q_2_x,
    }

    return dif_dict


def _all_products(q_1: Q, q_2: Q) -> Dict:
    """
    All products, commuting and anti-commuting products as a dictionary. For internal use.

    Args:
        q_1: Q
        q_2: Q

    Returns: Dict

    """

    all_dict = _commuting_products(q_1, q_2)
    all_dict.update(_anti_commuting_products(q_1, q_2))

    return all_dict


def square(q_1: Q) -> Q:
    """
    Square a quaternion.

    $ q.square() = q^2 = (t^2 - R.R, 2 t R) $

    Args:
        q_1: Q

    Returns:
        Q

    """

    end_q_type = f"{q_1.q_type}²"

    qxq = _commuting_products(q_1, q_1)

    sq_q = Q(q_type=end_q_type, representation=q_1.representation)
    sq_q.t = qxq["tt"] - qxq["xx+yy+zz"]
    sq_q.x = qxq["tx+xt"]
    sq_q.y = qxq["ty+yt"]
    sq_q.z = qxq["tz+zt"]

    return sq_q


def squares(q_1: Qs) -> Qs:
    f"""{square.__doc__}""".replace("Q", "Qs")
    return q_to_qs_function(square, q_1)


def norm_squared(q_1: Q) -> Q:
    """
    The norm_squared of a quaternion.

    $ q.norm\_squared() = q q^* = (t^2 + R.R, 0) $

    Returns: Q

    """

    end_q_type = f"||{q_1.q_type}||²"

    qxq = _commuting_products(q_1, q_1)

    n_q = Q(q_type=end_q_type, representation=q_1.representation)
    n_q.t = qxq["tt"] + qxq["xx+yy+zz"]

    return n_q


def norm_squareds(q_1: Qs) -> Qs:
    f"""{norm_squared.__doc__}""".replace("Q", "Qs")
    return products(conjs(q_1.set_qs_type("bra")), q_1.set_qs_type("ket"))


def norm_squared_of_vector(q_1: Q):
    """
    The norm_squared of the vector_q of a quaternion.

    $ q.norm\_squared\_of\_vector_q() = ((q - q^*)(q - q^*)^*)/4 = (R.R, 0) $

    Returns: Q
    """

    end_q_type = f"|V({q_1.q_type})|²"

    qxq = _commuting_products(q_1, q_1)

    nv_q = Q(q_type=end_q_type, representation=q_1.representation)
    nv_q.t = qxq["xx+yy+zz"]

    return nv_q


def norm_squared_of_vectors(q_1: Qs) -> Qs:
    f"""{norm_squared_of_vector.__doc__}""".replace("Q", "Qs")
    return products(vector_qs(conjs(q_1.set_qs_type("bra"))), vector_qs(q_1.set_qs_type("ket")))


def abs_of_q(q_1: Q) -> Q:
    """
    The absolute value, the square root of the norm_squared.

    $ q.abs_of_q() = \sqrt{q q^*} = (\sqrt{t^2 + R.R}, 0) $

    Returns: Q

    """

    end_q_type = f"|{q_1.q_type}|"

    a = norm_squared(q_1)
    sqrt_t = a.t ** (1 / 2)
    a.t = sqrt_t
    a.q_type = end_q_type
    a.representation = q_1.representation

    return a


def abs_of_qs(q_1: Qs) -> Qs:
    f"""{abs_of_q.__doc__}""".replace("Q", "Qs")
    squared_norm = norm_squareds(q_1)
    sqrt_t = squared_norm.qs[0].t ** (1 / 2)
    return q1s(sqrt_t, dim=1)


def normalize(q_1: Q, n: float = 1.0, q_type: str = "U") -> Q:
    """
    Normalize a quaternion to a given value n.

    $ q.normalized(n) = q (q q^*)^{-1} = q (n/\sqrt{q q^*}, 0) $

    Args:
        q_1: Q
        n: float       Make the norm equal to n.
        q_type: str

    Returns: Q

    """

    end_q_type = f"{q_1.q_type} {q_type}"

    abs_q_inv = inverse(abs_of_q(q_1))
    n_q = product(product(q_1, abs_q_inv), Q([n, 0, 0, 0]))
    n_q.q_type = end_q_type
    n_q.representation = q_1.representation

    return n_q


def normalizes(q_1: Qs, n: float = 1.0) -> Qs:
    """
    Normalize all states.

    Args:
        q_1: Qs
        n: float   number to normalize to, default is 1.0

    Returns: Qs

    """

    new_states = []

    zero_norm_count = 0

    for bra in q_1.qs:
        if norm_squared(bra).t == 0:
            zero_norm_count += 1
            new_states.append(q0())
        else:
            new_states.append(normalize(bra, n))

    new_states_normalized = []

    non_zero_states = q_1.dim - zero_norm_count

    for new_state in new_states:
        new_states_normalized.append(
            product(new_state, Q([math.sqrt(1 / non_zero_states), 0, 0, 0]))
        )

    return Qs(
        new_states_normalized,
        qs_type=q_1.qs_type,
        rows=q_1.rows,
        columns=q_1.columns,
    )


def orthonormalize(self: Qs) -> Qs:
    """
    Given a quaternion series, returns an orthonormal basis.

    Returns: Qs

    """

    last_q = self.qs.pop(0).normalize(math.sqrt(1 / self.dim), )
    orthonormal_qs = [last_q]

    for q in self.qs:
        qp = product(conj(q), last_q)
        orthonormal_q = normalize(dif(q, qp), math.sqrt(1 / self.dim), )
        orthonormal_qs.append(orthonormal_q)
        last_q = orthonormal_q

    return Qs(
        orthonormal_qs, qs_type=self.qs_type, rows=self.rows, columns=self.columns
    )


def determinant(self: Qs) -> Qs:
    """
    Calculate the determinant of a 'square' quaternion series.

    Returns: Qs

    """

    if self.dim == 1:
        q_det = self.qs[0]

    elif self.dim == 4:
        ad = product(self.qs[0], self.qs[3])
        bc = product(self.qs[1], self.qs[2])
        q_det = dif(ad, bc)

    elif self.dim == 9:
        aei = product(product(self.qs[0], self.qs[4]), self.qs[8])
        bfg = product(product(self.qs[3], self.qs[7]), self.qs[2])
        cdh = product(product(self.qs[6], self.qs[1]), self.qs[5])
        ceg = product(product(self.qs[6], self.qs[4]), self.qs[2])
        bdi = product(product(self.qs[3], self.qs[1]), self.qs[8])
        afh = product(product(self.qs[0], self.qs[7]), self.qs[5])

        sum_pos = add(aei, add(bfg, cdh))
        sum_neg = add(ceg, add(bdi, afh))

        q_det = dif(sum_pos, sum_neg)

    else:
        raise ValueError("Oops, don't know how to calculate the determinant of this one.")

    return Qs(
        [q_det], qs_type=self.qs_type, rows=1, columns=1
    )


def abs_of_vector(q_1: Q) -> Q:
    """
    The absolute value of the vector_q, the square root of the norm_squared of the vector_q.

    $ q.abs_of_vector() = \sqrt{(q\_1 - q\_1^*)(q\_1 - q\_1^*)/4} = (\sqrt{R.R}, 0) $

    Args:
        q_1: Q

    Returns: Q

    """

    end_q_type = f"|V({q_1.q_type})|"

    av = norm_squared_of_vector(q_1)
    sqrt_t = av.t ** (1 / 2)
    av.t = sqrt_t
    av.representation = q_1.representation
    av.q_type = end_q_type

    return av


def abs_of_vectors(q_1: Qs) -> Qs:
    f"""{abs_of_vector.__doc__}"""
    return q_to_qs_function(abs_of_vector, q_1)


def add(q_1: Q, q_2: Q) -> Q:
    """
    Add two quaternions.

    $ q.add(q\_2) = q_1 + q\_2 = (t + t\_2, R + R\_2) $

    Args:
        q_1: Q
        q_2: Q

    Returns: Q

    """

    q_1.check_representations(q_2)

    add_q_type = f"{q_1.q_type}+{q_2.q_type}"

    t_1, x_1, y_1, z_1 = q_1.t, q_1.x, q_1.y, q_1.z
    t_2, x_2, y_2, z_2 = q_2.t, q_2.x, q_2.y, q_2.z

    add_q = Q(q_type=add_q_type, representation=q_1.representation)
    add_q.t = t_1 + t_2
    add_q.x = x_1 + x_2
    add_q.y = y_1 + y_2
    add_q.z = z_1 + z_2

    return add_q


def adds(q_1: Qs, q_2: Qs) -> Qs:
    f"""{add.__doc__}""".replace("Q", "Qs")

    if (q_1.rows != q_2.rows) or (q_1.columns != q_2.columns):
        error_msg = "Oops, can only add if rows and columns are the same.\n"
        error_msg += f"rows are {q_1.rows}/{q_2.rows}, col: {q_1.columns}/{q_2.columns}"
        raise ValueError(error_msg)

    return qq_to_qs_function(add, q_1, q_2)


def dif(q_1: Q, q_2: Q) -> Q:
    """
    Takes the difference of 2 quaternions.

    $ q.dif(q\_2) = q_1 - q\_2 = (t - t\_2, R - R\_2) $

    Args:
        q_1: Q
        q_2: Q

    Returns: Q

    """

    q_1.check_representations(q_2)

    end_dif_q_type = f"{q_1.q_type}-{q_2.q_type}"

    t_2, x_2, y_2, z_2 = q_2.t, q_2.x, q_2.y, q_2.z
    t_1, x_1, y_1, z_1 = q_1.t, q_1.x, q_1.y, q_1.z

    dif_q = Q(q_type=end_dif_q_type, representation=q_1.representation)
    dif_q.t = t_1 - t_2
    dif_q.x = x_1 - x_2
    dif_q.y = y_1 - y_2
    dif_q.z = z_1 - z_2

    return dif_q


def difs(q_1: Qs, q_2: Qs) -> Qs:
    f"""{dif.__doc__}""".replace("Q", "Qs")

    if (q_1.rows != q_2.rows) or (q_1.columns != q_2.columns):
        error_msg = "Oops, can only dif if rows and columns are the same.\n"
        error_msg += f"rows are {q_1.rows}/{q_2.rows}, col: {q_1.columns}/{q_2.columns}"
        raise ValueError(error_msg)

    return qq_to_qs_function(dif, q_1, q_2)


def product(q_1: Q, q_2: Q, kind: str = "", reverse: bool = False) -> Q:
    """
    Form a product given 2 quaternions. Kind of product can be '' aka standard, even, odd, or even_minus_odd.
    Setting reverse=True is like changing the order.

    $ q.product(q_2) = q\_1 q\_2 = (t t_2 - R.R_2, t R_2 + R t_2 + RxR_2 ) $

    $ q.product(q_2, kind="even") = (q\_1 q\_2 + (q q\_2)^*)/2 = (t t_2 - R.R_2, t R_2 + R t_2 ) $

    $ q.product(q_2, kind="odd") = (q\_1 q\_2 - (q q\_2)^*)/2 = (0, RxR_2 ) $

    $ q.product(q_2, kind="even_minus_odd") = q\_2 q\_1 = (t t_2 - R.R_2, t R_2 + R t_2 - RxR_2 ) $

    $ q.product(q_2, reverse=True) = q\_2 q\_1 = (t t_2 - R.R_2, t R_2 + R t_2 - RxR_2 ) $

    Args:
        q_1: Q
        q_2: Q:
        kind: str:    can be '', even, odd, or even_minus_odd
        reverse: bool:  if true, returns even_minus_odd

    Returns: Q

    """

    q_1.check_representations(q_2)

    commuting = _commuting_products(q_1, q_2)
    q_even = Q()
    q_even.t = commuting["tt"] - commuting["xx+yy+zz"]
    q_even.x = commuting["tx+xt"]
    q_even.y = commuting["ty+yt"]
    q_even.z = commuting["tz+zt"]

    anti_commuting = _anti_commuting_products(q_1, q_2)
    q_odd = Q()

    if reverse:
        q_odd.x = anti_commuting["zy-yz"]
        q_odd.y = anti_commuting["xz-zx"]
        q_odd.z = anti_commuting["yx-xy"]

    else:
        q_odd.x = anti_commuting["yz-zy"]
        q_odd.y = anti_commuting["zx-xz"]
        q_odd.z = anti_commuting["xy-yx"]

    if kind == "":
        result = add(q_even, q_odd)
        times_symbol = "x"
    elif kind.lower() == "even":
        result = q_even
        times_symbol = "xE"
    elif kind.lower() == "odd":
        result = q_odd
        times_symbol = "xO"
    elif kind.lower() == "even_minus_odd":
        result = dif(q_even, q_odd)
        times_symbol = "xE-xO"
    else:
        raise Exception(
            "Four 'kind' values are known: '', 'even', 'odd', and 'even_minus_odd'."
        )

    if reverse:
        times_symbol = times_symbol.replace("x", "xR")

    result.q_type = f"{q_1.q_type}{times_symbol}{q_2.q_type}"
    result.representation = q_1.representation

    return result


def products(q_1: Qs, q_2: Qs, kind: str = "", reverse: bool = False) -> Qs:
    """
    Forms the quaternion product for each state. The details for handling the variety of cases for lengths
    of rows, states, and operators makes this code the most complex in this library.

    Args:
        q_1: Qs
        q_2: Qs
        kind: str    can be '', even, odd, or even_minus_odd
        reverse: bool

    Returns: Qs

    """

    q_1_copy = deepcopy(q_1)
    q_2_copy = deepcopy(q_2)
    qs_left, qs_right = Qs(), Qs()

    # Diagonalize if need be.
    if ((q_1.rows == q_2.rows) and (q_1.columns == q_2.columns)) or (
            "scalar_q" in [q_1.qs_type, q_2.qs_type]
    ):

        if q_1.columns == 1:
            qs_right = q_2_copy
            qs_left = diagonal(q_1_copy, qs_right.rows)

        elif q_2.rows == 1:
            qs_left = q_1_copy
            qs_right = diagonal(q_2_copy, qs_left.columns)

        else:
            qs_left = q_1_copy
            qs_right = q_2_copy

    # Typical matrix multiplication criteria.
    elif q_1.columns == q_2.rows:
        qs_left = q_1_copy
        qs_right = q_2_copy

    else:
        print(
            "Oops, cannot multiply series with row/column dimensions of {}/{} to {}/{}".format(
                q_1.rows, q_1.columns, q_2.rows, q_2.columns
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
        [q0(q_type="") for _i in range(outer_column_max)]
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

                result[outer_row][outer_column] = add(result[outer_row][outer_column],
                                                      product(qs_left.qs[left_index],
                                                              qs_right.qs[right_index], kind=kind, reverse=reverse
                                                              )
                                                      )

    # Flatten the list.
    new_qs = [item for sublist in result for item in sublist]
    new_states = Qs(new_qs, rows=outer_row_max, columns=outer_column_max)

    if projector_flag or operator_flag:
        return transpose(new_states)

    else:
        return new_states


def cross_q(q_1: Q, q_2: Q, reverse: bool = False) -> Q:
    """
    Convenience function, calling product with kind="odd".
    Called 'cross_q' to imply it returns 4 numbers, not the standard 3.

    Args:
        q_1: Q
        q_2: Q
        reverse: bool

    Returns: Q

    """
    return product(q_1, q_2, kind="odd", reverse=reverse)


def cross_qs(q_1: Qs, q_2: Qs) -> Qs:
    f""""{cross_q.__doc__}""".replace("Q", "Qs")
    return qq_to_qs_function(cross_q, q_1, q_2)


def dot_product(q_1: Qs, q_2: Qs) -> Q:
    f"""{products.__doc__}"""
    return  qs_qs_to_q_function(products, q_1, q_2)


def inverse(q_1: Q, additive: bool = False) -> Q:
    """
    The additive or multiplicative inverse of a quaternion. Defaults to 1/q, not -q.

    $ q.inverse() = q^* (q q^*)^{-1} = (t, -R) / (t^2 + R.R) $

    $ q.inverse(additive=True) = -q = (-t, -R) $

    Args:
        q_1: Q
        additive: bool

    Returns: Q

    """

    if additive:
        end_q_type = f"-{q_1.q_type}"
        q_inv = flip_sign(q_1)
        q_inv.q_type = end_q_type

    else:
        end_q_type = f"{q_1.q_type}⁻¹"

        q_conj = conj(q_1)
        q_norm_squared = norm_squared(q_1)

        if (not q_1.is_symbolic()) and (q_norm_squared.t == 0):
            q_inv = q0()

        else:
            q_norm_squared_inv = Q([1.0 / q_norm_squared.t, 0, 0, 0])
            q_inv = product(q_conj, q_norm_squared_inv)

        q_inv.q_type = end_q_type
        q_inv.representation = q_1.representation

    return q_inv


def inverses(q_1: Qs, additive: bool = False) -> Qs:
    """
    Inversing bras and kets calls inverse() once for each.
    Inversing operators is more tricky as one needs a diagonal identity matrix.

    Args:
        q_1: Qs
        additive: bool

    Returns: Qs

    """

    if q_1.qs_type in ["op", "operator"]:

        if additive:

            q_flip = q_1.inverse(additive=True)
            q_inv = q_flip.diagonal(q_1.dim)

        else:
            if q_1.dim == 1:
                q_inv = Qs(inverse(q_1.qs[0]))

            elif q_1.qs_type in ["bra", "ket"]:

                new_qs = []

                for q in q_1.qs:
                    new_qs.append(inverse(q))

                q_inv = Qs(
                    new_qs,
                    qs_type=q_1.qs_type,
                    rows=q_1.rows,
                    columns=q_1.columns,
                )

            elif q_1.dim == 4:
                det = determinant(q_1)
                detinv = inverse(det)

                q0 = product(q_1.qs[3], detinv)
                q_2 = product(flip_sign(q_1.qs[1]), detinv)
                q2 = product(flip_sign(q_1.qs[2]), detinv)
                q3 = product(q_1.qs[0], detinv)

                q_inv = Qs(
                    [q0, q_2, q2, q3],
                    qs_type=q_1.qs_type,
                    rows=q_1.rows,
                    columns=q_1.columns,
                )

            elif q_1.dim == 9:
                det = determinant(q_1)
                detinv = inverse(det)

                q0 = product(dif(q_1.qs[4], product(q_1.qs[8]), product(q_1.qs[5], q_1.qs[7])), detinv)

                q_2 = product(dif(q_1.qs[7], product(q_1.qs[2]), product(q_1.qs[8], q_1.qs[1])), detinv)

                q2 = product(dif(q_1.qs[1], product(q_1.qs[5]), product(q_1.qs[2], q_1.qs[4])), detinv)

                q3 = product(dif(q_1.qs[6], product(q_1.qs[5]), product(q_1.qs[8], q_1.qs[3])), detinv)

                q4 = product(dif(q_1.qs[0], product(q_1.qs[8]), product(q_1.qs[2], q_1.qs[6])), detinv)

                q5 = product(dif(q_1.qs[3], product(q_1.qs[2]), product(q_1.qs[5], q_1.qs[0])), detinv)

                q6 = product(dif(q_1.qs[3], product(q_1.qs[7]), product(q_1.qs[4], q_1.qs[6])), detinv)

                q7 = product(dif(q_1.qs[6], product(q_1.qs[1]), product(q_1.qs[7], q_1.qs[0])), detinv)

                q8 = product(dif(q_1.qs[0], product(q_1.qs[4]), product(q_1.qs[1], q_1.qs[3])), detinv)

                q_inv = Qs(
                    [q0, q_2, q2, q3, q4, q5, q6, q7, q8],
                    qs_type=q_1.qs_type,
                    rows=q_1.rows,
                    columns=q_1.columns,
                )

            else:
                raise ValueError("Oops, don't know how to invert.")

    else:
        new_states = []

        for bra in q_1.qs:
            new_states.append(inverse(bra, additive=additive))

        q_inv = Qs(
            new_states, qs_type=q_1.qs_type, rows=q_1.rows, columns=q_1.columns
        )

    return q_inv


def divide_by(q_1: Q, q_2: Q) -> Q:
    """
    Divide one quaternion by another. The order matters unless one is using a norm_squared (real number).

    $ q.divided_by(q_2) = q\_1 q_2^{-1} = (t t\_2 + R.R\_2, -t R\_2 + R t\_2 - RxR\_2) $

    Args:
        q_1: Q
        q_2: Q

    Returns: Q

    """

    q_1.check_representations(q_2)

    end_q_type = f"{q_1.q_type}/{q_2.q_type}"

    q_div = product(q_1, inverse(q_2))
    q_div.q_type = end_q_type
    q_div.representation = q_1.representation

    return q_div


def divide_bys(q_1: Qs, q_2: Qs) -> Qs:
    f"""{divide_by.__doc__}""".replace("Q", "Qs")
    return qq_to_qs_function(divide_by, q_1, q_2)


def triple_product(q_1: Q, q_2: Q, q_3: Q) -> Q:
    """
    Form a triple product given 3 quaternions, in left-to-right order: q1, q_2, q_3.

    $ q.triple_product(q_2, q_3) = q q_2 q_3 $

    $ = (t t\_2 t\_3 - R.R\_2 t\_3 - t R\_2.R|_3 - t\_2 R.R\_3 - (RxR_2).R\_3, $

    $ ... t t\_2 R\_3 - (R.R\_2) R\_3 + t t\_3 R\_2 + t\_2 t\_3 R $

    $ ... + t\_3 RxR\_2 + t R_2xR\_3 + t_2 RxR\_3 + RxR\_2xR\_3) $

    Args:
        q_1: Q
        q_2: Q:
        q_3: Q:

    Returns: Q

    """

    q_1.check_representations(q_2)
    q_1.check_representations(q_3)

    triple = product(product(q_1, q_2), q_3)
    triple.representation = q_1.representation

    return triple


def triple_products(q_1: Qs, q_2: Qs, q_3: Qs) -> Qs:
    f"""{triple_product.__doc__}""".replace("Q", "Qs")
    return qqq_to_qs_function(triple_product, q_1, q_2, q_3)


def rotation(q_1: Q, h: Q) -> Q:
    """
    Do a rotation using a triple product: u R 1/u.
    SPECIAL NOTE: q_1 = 0 MUST WORK! Zero is just another number.
    To make it work, view the rotation function as a 2-part function.
    If h=0, then return q_1.
    If h!=0, then form the Rodrigues triple product.
    Also note one needs to use the inverse to not rescale the result.

    $ rotation(q, h) = h q h^{-1} $

    Args:
        q_1: Q
        h: Q    pre-multiply by u, post-multiply by $u^{-1}$.

    Returns: Q

    """

    q_1.check_representations(h)
    end_q_type = f"{q_1.q_type}*rot"

    if equal(h, q0()):
        return q_1

    q_rot = triple_product(h, q_1, inverse(h))
    q_rot.q_type = end_q_type
    q_rot.representation = q_1.representation

    return q_rot


def rotations(q_1: Qs, u: Qs) -> Qs:
    f"""{rotation.__doc__}""".replace("Q", "Qs")
    return qq_to_qs_function(rotation, q_1, u)


def rotation_and_rescale(q_1: Q, h: Q) -> Q:
    """
    Do a rotation using a triple product: u R u^*.
    The rescaling will be by a factor of ||u||^2

    SPECIAL NOTE: q_1 = 0 MUST WORK! Zero is just another number.
    To make it work, view the rotation function as a 2-part function.
    If h=0, then return q_1.
    If h!=0, then form the Rodrigues triple product.

    $ rotation(q, h) = h q h^* $

    Args:
        q_1: Q
        h: Q    pre-multiply by u, post-multiply by $u^{-1}$.

    Returns: Q

    """

    q_1.check_representations(h)
    end_q_type = f"{q_1.q_type}*rot"

    if equal(h, q0()):
        return q_1

    q_rot = triple_product(u, q_1, inverse(h))
    q_rot.q_type = end_q_type
    q_rot.representation = q_1.representation

    return q_rot


def rotation_and_rescales(q_1: Qs, h: Qs) -> Qs:
    f"""{rotation_and_rescale.__doc__}""".replace("Q", "Qs")
    return qq_to_qs_function(rotation_and_rescale, q_1, h)


def rotation_angle(q_1: Q, q_2: Q, origin: Q = q0(), tangent_space_norm: float = 1.0, degrees: bool = False) -> Q:
    """
    Returns the spatial angle between the origin and 2 points.

    $$ scalar(normalize(vector(q_1) vector(q_2)^*)) = \cos(a) $$

    The product of the 3-vectors is a mix of symmetric and anti-symmetric terms.
    The normalized scalar is $\cos(a)$. Take the inverse cosine to get the angle
    for the angle in the plane between q_1, the origin and q_2.

    I have a radical view of space-time. It is where-when everything happens. In space-time, all algebra
    operations are the same: $ 2 + 3 = 5 $ and $ 2 * 3 = 6 $. The same cannot be said about the tangent
    space of space-time because it is tangent space that can be 'curved'. My proposal for gravity is that
    changes in the tangent space measurements of time, dt, exactly cancel those of the tangent space
    measurements of space, dR. When one is making a measurement that involves gravity, the tangent space
    norm will not be equal to unity, but greater or lesser than unity.

    There are a number of bothersome qualities about this function. The scalar term doesn't matter in the
    slightest way. As a consequence, this is a purely spatial function.

    Args:
        q_1: Q
        q_2: Q
        origin: Q    default is zero.
        tangent_space_norm: float   Will be different from unity in 'curved' tangent spaces
        degrees: float    Use degrees instead of radians

    Returns: Q    only the scalar is possibly non-zero

    """

    q_1_shifted_vector = vector_q(dif(q_1, origin))
    q_2_shifted_vector = vector_q(dif(q_2, origin))

    q_1__q_2 = normalize(product(q_1_shifted_vector, conj(q_2_shifted_vector)), n=tangent_space_norm)
    angle = math.acos(q_1__q_2.t)

    if degrees:
        angle = angle * 180 / math.pi

    return Q([angle, 0, 0, 0])


def rotation_and_or_boost(q_1: Q, h: Q, verbose=False) -> Q:
    """
    The method for doing a rotation in 3D space discovered by Rodrigues in the 1840s used a quaternion triple
    product. After Minkowski characterized Einstein's work in special relativity as a 4D rotation, efforts were
    made to do the same with one quaternion triple product. Two people were able to do the trick with complex-valued
    quaternions in 1910-1911, but complex-valued quaternion are not a division altebra. The goal to do the
    transformation with a division algebra took a century (not of work, but ignoring the issue). In 2010 D. Sweetser
    and independently by M. Kharinov (year unknown to me) the same algebra was found. Two other triple products need to
    be used like so:

    $ b.rotation_and_or_boost(h) = h b h^* + 1/2 ((hhb)^* -(h^* h^* b)^*) $

    The parameter h is NOT free from constraints. There are two constraints. If the parameter h is to do a
    rotation, it must have a norm of unity and have the first term equal to zero.

    $ h = (0, R), scalar_q(h) = 0, scalar_q(h h^*) = 1 $

    For such a value of h, the second and third terms cancel leaving

    To do a boost which may or may not also do a rotation, then the parameter h must have a square whose first
    term is equal to zero:

    $ h = (\cosh(a), \sinh(a)), scalar_q(h^2) = 1 $

    There has been no issue about the ability of this function to do boosts. There has been a spirited debate
    as to whether the function can do rotations. Notice that the form reduces to the Rodrigues triple product.
    I consider this so elementary that I cannot argue the other side. Please see the wiki page or use this code
    to see for yourself.

    Args:
        q_1: Q
        h: Q

    Returns: Q

    """
    q_1.check_representations(h)
    end_q_type = f"{q_1.q_type}rotation/boost"

    if not h.is_symbolic():

        if (not math.isclose(h.t, 0) and not equal(q0(), vector_q(h))) or equal(h, q0()):

            if not math.isclose(square(h).t, 1):
                # The scalar part of h will be used to calculate cosh(h.t) and sinh(h.t)
                # The normalized vector part will point sinh(t) in the direction of vector_q(h)
                h_scalar = scalar_q(h)
                h_nomralized_vector = normalize(vector_q(h))

                if np.abs(h_scalar.t) > 1:
                    h_scalar = inverse(h_scalar)

                h_cosh = product(add(exp(h_scalar), exp(flip_sign(h_scalar))), q1(1.0 / 2.0))
                h_sinh = product(dif(exp(h_scalar), exp(flip_sign(h_scalar))), q1(1.0 / 2.0))

                h = add(h_cosh, product(h_nomralized_vector, h_sinh))

                if verbose:
                    h.print_state("To do a Lorentz boost, adjusted value of h so scalar_q(h²) = 1")

        else:
            if not math.isclose(norm_squared(h).t, 1):
                h = normalize(h)
                if verbose:
                    h.print_state("To do a 3D rotation, adjusted value of h so scalar_q(h h^*) = 1")

    triple_1 = triple_product(h, q_1, conj(h))
    triple_2 = conj(triple_product(h, h, q_1))
    triple_3 = conj(triple_product(conj(h), conj(h), q_1))

    triple_23 = dif(triple_2, triple_3)
    half_23 = product(triple_23, Q([0.5, 0, 0, 0], representation=q_1.representation))
    triple_123 = add(triple_1, half_23)
    triple_123.q_type = end_q_type
    triple_123.representation = q_1.representation

    return triple_123


def rotation_and_or_boosts(q_1: Qs, h: Qs) -> Qs:
    f"""{rotation_and_or_boost.__doc__}""".replace("Q", "Qs")
    return qq_to_qs_function(rotation_and_or_boost, q_1, h)


def rotation_only(q_1: Q, h: Q) -> Q:
    """
    The function calls another function, rotations_and_or_boost() but does so hy constraining the parameter h
    to have a scalar of zero.

    $ b.rotation_and_or_boost(h) = h b h^* + 1/2 ((hhb)^* -(h^* h^* b)^*) $

    The second and third terms drop out, leaving the Rodrigues 3D spatial rotation formula.

    Args:
        q_1: Q
        h: Q

    Returns: Q

    """
    h_4_rotation = vector_q(h)
    return rotation_and_or_boost(q_1, h_4_rotation)


def rotation_onlys(q_1: Qs, h: Qs) -> Qs:
    f"""{rotation_only.__doc__}""".replace("Q", "Qs")
    return qq_to_qs_function(rotation_only, q_1, h)


def next_rotation(q_1: Q, q_2: Q) -> Q:
    """
    Given 2 quaternions, creates a new quaternion to do a rotation
    in the triple triple quaternion function by using a normalized cross product.

    $ next_rotation(q, q_2) = (q q\_2 - q\_2 q) / 2|(q q\_2 - (q\_2 q)^*)| = (0, QxQ\_2)/|(0, QxQ\_2)| $

    Args:
        q_1: Q   any quaternion
        q_2: Q   any quaternion whose first term equal the first term of q and
                  for the first terms of each squared.

    Returns: Q

    """
    q_1.check_representations(q_2)

    if not math.isclose(q_1.t, q_2.t):
        raise ValueError(f"Oops, to be a rotation, the first values must be the same: {q_1.t} != {q_2.t}")

    if not math.isclose(norm_squared(q_1).t, norm_squared(q_2).t):
        raise ValueError(f"Oops, the norm squared of these two are not equal: {norm_squared(q_1).t} != {norm_squared(q_2).t}")

    next_rot = product(q_1, q_2)
    v_abs_q_1 = abs_of_vector(q_1).t
    next_vector_normalized = normalize(vector_q(next_rot), v_abs_q_1)
    next_vector_normalized.t = q_1.t

    return next_vector_normalized


def next_rotations(q_1: Qs, q_2: Qs) -> Qs:
    f"""{next_rotation.__doc__}""".replace("Q", "Qs")
    return qq_to_qs_function(next_rotation, q_1, q_2)


def next_rotation_randomized(q_1: Q, q_2: Q) -> Q:
    """
    Given 2 quaternions, creates a new quaternion to do a rotation
    in the triple triple quaternion function by using a normalized cross product.

    To assure that repeated calls cover the sphere, multiply by a random factor.

    Args:
        q_1: Q   any quaternion
        q_2: Q   any quaternion whose first term equal the first term of q and
                  for the first terms of each squared.

    Returns: Q

    """
    q_1.check_representations(q_2)

    if not math.isclose(q_1.t, q_2.t):
        raise ValueError(f"Oops, to be a rotation, the first values must be the same: {q_1.t} != {q_2.t}")

    if not math.isclose(norm_squared(q_1).t, norm_squared(q_2).t):
        raise ValueError(f"Oops, the norm squared of these two are not equal: {norm_squared(q_1).t} != {norm_squared(q_2).t}")

    next_rot = product(product(q_1, q_2), qrandom())
    v_abs_q_1 = abs_of_vector(q_1).t
    next_vector_normalized = normalize(vector_q(next_rot), v_abs_q_1)
    next_vector_normalized.t = q_1.t

    return next_vector_normalized


def next_rotation_randomizeds(q_1: Qs, q_2: Qs) -> Qs:
    f"""{next_rotation_randomized.__doc__}""".replace("Q", "Qs")
    return qq_to_qs_function(next_rotation_randomized, q_1, q_2)


def next_boost(q_1: Q, q_2: Q) -> Q:
    """
    Given 2 quaternions, creates a new quaternion to do a boost/rotation
    using the triple triple quaternion product
    by using the scalar_q of an even product to form (cosh(x), i sinh(x)).

    $ next_boost(q, q_2) = q q\_2 + q\_2 q

    Args:
        q_1: Q
        q_2: Q

    Returns: Q

    """
    q_1.check_representations(q_2)

    if not (q_1.t >= 1.0 and q_2.t >= 1.0):
        raise ValueError(f"Oops, to be a boost, the first values must both be greater than one: {q_1.t},  {q_2.t}")

    if not math.isclose(square(q_1).t, square(q_2).t):
        raise ValueError(f"Oops, the squares of these two are not equal: {square(q_1).t} != {square(q_2).t}")

    q_even = product(q_1, q_2, kind="even")
    q_s = scalar_q(q_even)
    q_v = normalize(vector_q(q_even))

    if np.abs(q_s.t) > 1:
        q_s = inverse(q_s)

    exp_sum = product(add(exp(q_s), exp(flip_sign(q_s))), q1(-1.0 / 2.0))
    exp_dif = product(dif(exp(q_s), exp(flip_sign(q_s))), q1(-1.0 / 2.0))

    boost = add(exp_sum, product(q_v, exp_dif))

    return boost


def next_boosts(q_1: Qs, q_2: Qs) -> Qs:
    f"""{next_boost.__doc__}""".replace("Q", "Qs")
    return qq_to_qs_function(next_boost, q_1, q_2)


def permutation(q_1: Q, perm: str = "txyz") -> Q:
    """
    All possible permutations can be set with the 4 character perm string, variations on t, x, y, z.

    Args:
        q_1: Q     The quaternion to permute
        perm:      A shorthand for the 12 permutations

    Returns: Q

    """

    if len(perm) != 4:
        raise ValueError(f"The perm string must be 4 letters long: {perm}")

    result = {}

    result[f"{perm[0]}"] = q_1.t
    result[f"{perm[1]}"] = q_1.x
    result[f"{perm[2]}"] = q_1.y
    result[f"{perm[3]}"] = q_1.z

    rearranged = []

    for letter in tuple("txyz"):
        rearranged.append(result[letter])

    return Q(rearranged)


def all_permutations(q_1: Q) -> Qs:
    """
    Returns all permutations as a quaternion series. Can be made unique.

    Args:
        q_1: Q        The quaternion to permute

    Returns: Q

    """

    results = []

    for perm in itertools.permutations("txyz"):
        results.append(permutation(q_1, perm=perm))

    return Qs(results)


# g_shift is a function based on the space-times-time invariance proposal for gravity,
# which proposes that if one changes the distance from a gravitational source, then
# squares a measurement, the observers at two different hieghts agree to their
# space-times-time values, but not the intervals.
# g_form is the form of the function, either minimal or exponential
# Minimal is what is needed to pass all weak field tests of gravity
def g_shift(q_1: Q, dimensionless_g, g_form="exp"):
    """Shift an observation based on a dimensionless GM/c^2 dR."""

    end_q_type = f"{q_1.q_type} gshift"

    if g_form == "exp":
        g_factor = sp.exp(dimensionless_g)
    elif g_form == "minimal":
        g_factor = 1 + 2 * dimensionless_g + 2 * dimensionless_g ** 2
    else:
        print("g_form not defined, should be 'exp' or 'minimal': {}".format(g_form))
        return q_1

    g_q = Q(q_type=end_q_type)
    g_q.t = q_1.t / g_factor
    g_q.x = q_1.x * g_factor
    g_q.y = q_1.y * g_factor
    g_q.z = q_1.z * g_factor
    g_q.q_type = end_q_type
    g_q.representation = q_1.representation

    return g_q

def g_shifts(q_1: Qs, g: float, g_form="exp") -> Qs:
    f"""{g_shift.__doc__}""".replace("Q", "Qs")
    return Qs([g_shift(q, g, g_form) for q in q_1.qs], qs_type=q_1.qs_type)


def sin(q_1: Q) -> Q:
    """
    Take the sine of a quaternion

    $ q.sin() = (\sin(t) \cosh(|R|), \cos(t) \sinh(|R|) R/|R|)$

    Returns: Q

    """

    end_q_type = f"sin({q_1.q_type})"

    abs_v = abs_of_vector(q_1)

    if abs_v.t == 0:
        return Q([math.sin(q_1.t), 0, 0, 0], q_type=end_q_type, representation=q_1.representation)

    sint = math.sin(q_1.t)
    cost = math.cos(q_1.t)
    sinhR = math.sinh(abs_v.t)
    coshR = math.cosh(abs_v.t)

    k = cost * sinhR / abs_v.t

    q_sin = Q()
    q_sin.t = sint * coshR
    q_sin.x = k * q_1.x
    q_sin.y = k * q_1.y
    q_sin.z = k * q_1.z

    q_sin.q_type = end_q_type
    q_sin.representation = q_1.representation

    return q_sin


def sins(q_1: Qs) -> Qs:
    f"""{sin.__doc__}""".replace("Q", "Qs")
    return q_to_qs_function(sin, q_1)


def cos(q_1: Q) -> Q:
    """
    Take the cosine of a quaternion.
    $ q.cos() = (\cos(t) \cosh(|R|), \sin(t) \sinh(|R|) R/|R|) $

    Returns: Q

    """

    end_q_type = f"cos({q_1.q_type})"

    abs_v = abs_of_vector(q_1)

    if abs_v.t == 0:
        return Q([math.cos(q_1.t), 0, 0, 0], q_type=end_q_type, representation=q_1.representation)

    sint = math.sin(q_1.t)
    cost = math.cos(q_1.t)
    sinhR = math.sinh(abs_v.t)
    coshR = math.cosh(abs_v.t)

    k = -1 * sint * sinhR / abs_v.t

    q_cos = Q()
    q_cos.t = cost * coshR
    q_cos.x = k * q_1.x
    q_cos.y = k * q_1.y
    q_cos.z = k * q_1.z

    q_cos.q_type = end_q_type
    q_cos.representation = q_1.representation

    return q_cos


def coss(q_1: Qs) -> Qs:
    f"""{cos.__doc__}""".replace("Q", "Qs")
    return q_to_qs_function(cos, q_1)


def tan(q_1: Q) -> Q:
    """
    Take the tan of a quaternion.

     $ q.tan() = \sin(q) \cos(q)^{-1} $

     Returns: Q

    Args:
        q_1: Q

     """

    end_q_type = f"tan({q_1.q_type})"

    abs_v = abs_of_vector(q_1)

    if abs_v.t == 0:
        return Q([math.tan(q_1.t), 0, 0, 0], q_type=end_q_type, representation=q_1.representation)

    sinq = sin(q_1)
    cosq = cos(q_1)

    q_tan = divide_by(sinq, cosq)
    q_tan.q_type = end_q_type
    q_tan.representation = q_1.representation

    return q_tan


def tans(q_1: Qs) -> Qs:
    f"""{tan.__doc__}""".replace("Q", "Qs")
    return q_to_qs_function(tan, q_1)


def sinh(q_1: Q) -> Q:
    """
    Take the sinh of a quaternion.

    $ q.sinh() = (\sinh(t) \cos(|R|), \cosh(t) \sin(|R|) R/|R|) $

    Returns: Q

    """

    end_q_type = f"sinh({q_1.q_type})"

    abs_v = abs_of_vector(q_1)

    if abs_v.t == 0:
        return Q([math.sinh(q_1.t), 0, 0, 0], q_type=end_q_type, representation=q_1.representation)

    sinh_t = math.sinh(q_1.t)
    cos_r = math.cos(abs_v.t)
    cosh_t = math.cosh(q_1.t)
    sin_r = math.sin(abs_v.t)

    k = cosh_t * sin_r / abs_v.t

    q_sinh = Q(q_type=end_q_type, representation=q_1.representation)
    q_sinh.t = sinh_t * cos_r
    q_sinh.x = k * q_1.x
    q_sinh.y = k * q_1.y
    q_sinh.z = k * q_1.z

    return q_sinh


def sinhs(q_1: Qs) -> Qs:
    f"""{sinh.__doc__}""".replace("Q", "Qs")
    return q_to_qs_function(sinh, q_1)


def cosh(q_1: Q) -> Q:
    """
    Take the cosh of a quaternion.

    $ (\cosh(t) \cos(|R|), \sinh(t) \sin(|R|) R/|R|) $

    Returns: Q

    """

    end_q_type = f"cosh({q_1.q_type})"

    abs_v = abs_of_vector(q_1)

    if abs_v.t == 0:
        return Q([math.cosh(q_1.t), 0, 0, 0], q_type=end_q_type, representation=q_1.representation)

    cosh_t = math.cosh(q_1.t)
    cos_r = math.cos(abs_v.t)
    sinh_t = math.sinh(q_1.t)
    sin_r = math.sin(abs_v.t)

    k = sinh_t * sin_r / abs_v.t

    q_cosh = Q(q_type=end_q_type, representation=q_1.representation)
    q_cosh.t = cosh_t * cos_r
    q_cosh.x = k * q_1.x
    q_cosh.y = k * q_1.y
    q_cosh.z = k * q_1.z

    return q_cosh


def coshs(q_1: Qs) -> Qs:
    f"""{cosh.__doc__}""".replace("Q", "Qs")
    return q_to_qs_function(cosh, q_1)


def tanh(q_1: Q) -> Q:
    """
    Take the tanh of a quaternion.

    $ q.tanh() = \sin(q) \cos(q)^{-1} $

    Returns: Q

    """

    end_q_type = f"tanh({q_1.q_type})"

    abs_v = abs_of_vector(q_1)

    if abs_v.t == 0:
        return Q([math.tanh(q_1.t), 0, 0, 0], q_type=end_q_type, representation=q_1.representation)

    sinhq = sinh(q_1)
    coshq = cosh(q_1)

    q_tanh = divide_by(sinhq, coshq)
    q_tanh.q_type = end_q_type
    q_tanh.representation = q_1.representation

    return q_tanh


def tanhs(q_1: Qs) -> Qs:
    f"""{tanh.__doc__}""".replace("Q", "Qs")
    return q_to_qs_function(tanh, q_1)


def exp(q_1: Q) -> Q:
    """
    Take the exponential of a quaternion.

    $ q.exp() = (\exp(t) \cos(|R|, \exp(t) \sin(|R|) R/|R|) $

    Returns: Q
    """

    end_q_type = f"exp({q_1.q_type})"

    abs_v = abs_of_vector(q_1)
    et = math.exp(q_1.t)

    if abs_v.t == 0:
        return Q([et, 0, 0, 0], q_type=end_q_type, representation=q_1.representation)

    cosR = math.cos(abs_v.t)
    sinR = math.sin(abs_v.t)
    k = et * sinR / abs_v.t

    q_exp = Q(
        [et * cosR, k * q_1.x, k * q_1.y, k * q_1.z],
        q_type=end_q_type,
        representation=q_1.representation,
    )

    return q_exp


def exps(q_1: Qs) -> Qs:
    f"""{exp.__doc__}""".replace("Q", "Qs")
    return q_to_qs_function(exp, q_1)


def ln(q_1: Q) -> Q:
    """
    Take the natural log of a quaternion.

    $ q.ln() = (0.5 \ln t^2 + R.R, \atan2(|R|, t) R/|R|) $

    Returns: Q

    """
    end_q_type = f"ln({q_1.q_type})"

    abs_v = abs_of_vector(q_1)

    if abs_v.t == 0:

        if q_1.t > 0:
            return Q([math.log(q_1.t), 0, 0, 0], q_type=end_q_type, representation=q_1.representation)
        else:
            # I don't understand this, but Mathematica does the same thing.
            return Q([math.log(-q_1.t), math.pi, 0, 0], q_type=end_q_type, representation=q_1.representation)

    t_value = 0.5 * math.log(q_1.t * q_1.t + abs_v.t * abs_v.t)
    k = math.atan2(abs_v.t, q_1.t) / abs_v.t

    q_ln = Q(
        [t_value, k * q_1.x, k * q_1.y, k * q_1.z],
        q_type=end_q_type,
        representation=q_1.representation,
    )

    return q_ln


def lns(q_1: Qs) -> Qs:
    f"""{ln.__doc__}""".replace("Q", "Qs")
    return q_to_qs_function(ln, q_1)


def q_2_q(q_1: Q, q_2: Q) -> Q:
    """Take the natural log of a quaternion.

    $ q.q_2_q(p) = \exp(\ln(q) * p) $

    Returns: Q

    """

    q_1.check_representations(q_2)
    end_q_type = f"{q_1.q_type}^{q_2.q_type}"

    q2q = exp(product(ln(q_1), q_2))
    q2q.q_type = end_q_type
    q2q.representation = q_1.representation
    q2q.q_type = end_q_type

    return q2q


def q_2_qs(q_1: Qs, q_2: Qs) -> Qs:
    f"""{q_2_q.__doc__}""".replace("Q", "Qs")
    return qq_to_qs_function(q_2_q, q_1, q_2)


def trunc(q_1: Q) -> Q:
    """
    Truncates values.

    Returns: Q

    """

    if not q_1.is_symbolic():
        q_1.t = math.trunc(q_1.t)
        q_1.x = math.trunc(q_1.x)
        q_1.y = math.trunc(q_1.y)
        q_1.z = math.trunc(q_1.z)

    return q_1


def truncs(q_1: Qs) -> Qs:
    f"""{trunc.__doc__}""".replace("Q", "Qs")
    return q_to_qs_function(trunc, q_1)


def transpose(q_1: Qs, m: int = None, n: int = None) -> Qs:
    """
    Transposes a series.

    Args:
        q_1: Qs
        m: int
        n: int

    Returns: Qs

    """

    if m is None:
        # test if it is square.
        if math.sqrt(q_1.dim).is_integer():
            m = int(sp.sqrt(q_1.dim))
            n = m

    if n is None:
        n = int(q_1.dim / m)

    matrix = [[0 for _x in range(m)] for _y in range(n)]

    for mi in range(m):
        for ni in range(n):
            matrix[ni][mi] = q_1.qs[mi * n + ni]

    qs_t = []

    for t in matrix:
        for q in t:
            qs_t.append(q)

    # Switch rows and columns.
    return Qs(qs_t, rows=q_1.columns, columns=q_1.rows)


def Hermitian_conj(q_1: Qs, m: int, n: int, conj_type: object = 0) -> Qs:
    """
    Returns the Hermitian conjugate.

    Args:
        q_1, Qs
        m: int
        n: int
        conj_type: int    0-3

    Returns: Qs

    """

    # return q_1.transpose(m, n).conj(conj_type)
    return conjs(transpose(q_1, m, n), conj_type)


def dagger(q_1: Qs, m: int, n: int, conj_type: int = 0) -> Qs:
    """
    Just calls Hermitian_conj()

    Args:
        q_1: Qs
        m: int
        n: int
        conj_type: 0-3

    Returns: Qs

    """

    return Hermitian_conj(q_1, m, n, conj_type)


def is_square(q_1: Qs) -> bool:
    """
    Tests if a quaternion series is square, meaning the dimenion is n^2.

    Returns: bool

    """

    return math.sqrt(q_1.dim).is_integer()


def is_Hermitian(q_1: Qs) -> bool:
    """
    Tests if a series is Hermitian.

    Returns: bool

    """

    hc = Hermitian_conj(q_1, q_1.rows, q_1.columns)

    return equals(q_1, hc)


def diagonal(q_1: Qs, dim: int) -> Qs:
    """
    Make a state dim * dim with q or qs along the 'diagonal'. Always returns an operator.

    Args:
        q_1: Qs
        dim: int

    Returns: Qs

    """

    the_diagonal = []

    if len(q_1.qs) == 1:
        q_values = [q_1.qs[0]] * dim
    elif len(q_1.qs) == dim:
        q_values = q_1.qs
    elif q_1.qs is None:
        raise ValueError("Oops, the qs here is None.")
    else:
        raise ValueError("Oops, need the length to be equal to the dimensions.")

    for i in range(dim):
        for j in range(dim):
            if i == j:
                the_diagonal.append(q_values.pop(0))
            else:
                the_diagonal.append(q0())

    return Qs(the_diagonal, qs_type="op", rows=dim, columns=dim)


def identity(dim: int = 1, operator: bool = False, additive: bool = False, non_zeroes=None, qs_type: str = "ket") \
        -> Qs:
    """
    Identity operator for states or operators which are diagonal.

    Args:
        dim: int
        operator: bool
        additive: bool
        non_zeroes:
        qs_type: str

    Returns: Qs

    """

    if additive:
        id_q = [Q() for _ in range(dim)]

    elif non_zeroes is not None:
        id_q = []

        if len(non_zeroes) != dim:
            raise ValueError(f"Oops, len(non_zeroes)={len(non_zeroes)}, should be: {dim}")

        else:
            for non_zero in non_zeroes:
                if non_zero:
                    id_q.append(Q([1, 0, 0, 0]))
                else:
                    id_q.append(Q())

    else:
        id_q = [q1() for _ in range(dim)]

    if operator:
        q_1 = Qs(id_q)
        ident = diagonal(q_1, dim)

    else:
        ident = Qs(id_q, qs_type=qs_type)

    return ident


def trace(q_1: Qs) -> Qs:
    """
    Return the trace as a scalar_q quaternion series.

    Returns: Qs

    Args:
        q_1: Qs

    Returns: Qs

    """

    if q_1.rows != q_1.columns:
        raise ValueError(f"Oops, not a square quaternion series: {q_1.rows}/{q_1.columns}")

    else:
        tr = q_1.qs[0]

    for i in range(1, q_1.rows):
        tr = add(tr, q_1.qs[i * (q_1.rows + 1)])

    return Qs([tr])


def sigma(kind: str = "x", theta: float = None, phi: float = None) -> Qs:
    """
    Returns a sigma when given a type like, x, y, z, xy, xz, yz, xyz, with optional angles theta and phi.

    Args:
        kind: str  x, y, z, xy, etc
        theta: float   an angle
        phi: float     an angle

    Returns:

    """

    q0, q_2, qi = Q([0, 0, 0, 0]), Q([1, 0, 0, 0]), Q([0, 1, 0, 0])

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

    x_factor = products(q_2, Qs([sin_theta * cos_phi, 0, 0, 0]))
    y_factor = products(qi, Qs([sin_theta * sin_phi, 0, 0, 0]))
    z_factor = products(q_2, Qs([cos_theta, 0, 0, 0]))

    sigma_bunch = Bunch()
    sigma_bunch.x = Qs([q0, x_factor, x_factor, q0], "op")
    sigma_bunch.y = Qs([q0, y_factor, flip_sign(y_factor), q0], "op")
    sigma_bunch.z = Qs([z_factor, q0, q0, flip_sign(z_factor)], "op")

    sigma_bunch.xy = adds(sigma_bunch.x, sigma_bunch.y)
    sigma_bunch.xz = adds(sigma_bunch.x, sigma_bunch.z)
    sigma_bunch.yz = adds(sigma_bunch.y, sigma_bunch.z)
    sigma_bunch.xyz = adds(adds(sigma_bunch.x, sigma_bunch.y), sigma_bunch.z)

    if kind not in sigma_bunch:
        raise ValueError("Oops, I only know about x, y, z, and their combinations.")

    return normalizes(sigma_bunch[kind])


def zero_out(q_1: Q, t: bool = False, x: bool = False, y: bool = False, z: bool = False) -> Q:
    """
    Puts a zero in one or more of the four places.

    Args:
        q_1 Q
        t: bool    zero out t
        x: bool    zero out x
        y: bool    zero out y
        z: bool    zero out z

    Returns: Qs
    """

    new_q = deepcopy(q_1)

    if t:
        new_q.t = 0

    if x:
        new_q.x = 0

    if y:
        new_q.y = 0

    if z:
        new_q.z = 0

    return new_q


def zero_outs(q_1: Qs, t: bool = False, x: bool = False, y: bool = False, z: bool = False) -> Qs:
    f"""{zero_out.__doc__}""".replace("Q", "Qs")

    return Qs(
        [zero_out(q, t, x, y, z) for q in q_1.qs],
        qs_type=q_1.qs_type,
        rows=q_1.rows,
        columns=q_1.columns,
    )


# Generators of quaternion series.
def generate_Qs(func: FunctionType, q_1: Union[Q, Qs, FunctionType], dim: int = 10, qs_type: str = "ket") -> Qs:
    """
    One quaternion cannot tell a story. generate_Qs provides a general way to create a
    quaternion series given a function and one quaternion/another function. The function
    is applied to each subsequent value of the function. If q_1 is itself a function, it
    will be called each time.

    Args:
        func: FunctionType   a function that generates an instance of the class Q
        q_1: Q, FunctionType  Either an instance of Q, Qs, or a Q function
        dim: int    The dimensions of the quaternion series
        qs_type:    bra/ket/operator  Only works for a square operator at this time

    Returns: Qs

    """

    if type(q_1) == Q:
        new_qs = [func(q_1)]

        for _ in range(dim - 1):
            new_qs.append(func(new_qs[-1]))

    elif type(q_1) == Qs:
        new_qs = q_1.qs

    elif type(q_1) == FunctionType:
        new_qs = [func(q_1())]

        for _ in range(dim - 1):
            new_qs.append(func(q_1()))

    else:
        raise ValueError(f"Cannot work with q_1's type: {type(q_1)}")

    return Qs(new_qs, qs_type=qs_type)


def generate_QQs(func, q_1: Union[Q, Qs, FunctionType], q_2: Union[Q, Qs, FunctionType], dim: int = 10, qs_type: str = "ket") -> Qs:
    """
    One quaternion cannot tell a story. generate_QQs provides a general way to create a
    quaternion series given a function and two other quaternions/functions. The function
    is applied to each subsequent value of the function. If q_1 or q_2 is itself a function, it
    will be called each time.

    This function was written for the function add to be
    able to represent inertial motion, adding the same value over and over again.

    Args:
        func: FunctionType   a function that generates an instance of the class Q
        q_1: Q, Qs, FunctionType  Either an instance of Q, Qs, or a Q function
        q_2: Q, Qs, FunctionType  Either an instance of Q, Qs, or a Q function
        dim: int    The dimensions of the quaternion series
        qs_type:    bra/ket/operator  Only works for a square operator at this time

    Returns: Qs

    """

    if (type(q_1) == Q) and (type(q_2) == Q):

        new_qs = [func(q_1, q_2)]

        for _ in range(dim - 1):
            new_qs.append(func(new_qs[-1], q_2))

    elif ((type(q_1) == Q) and (type(q_2) == FunctionType)):
        new_qs = [func(q_1, q_2())]

        for _ in range(dim - 1):
            new_qs.append(func(new_qs[-1], q_2()))

    elif ((type(q_1) == FunctionType) and (type(q_2) == Q)):
        new_qs = [func(q_1(), q_2)]

        for _ in range(dim - 1):
            new_qs.append(func(q_1(), new_qs[-1]))

    elif type(q_1) == FunctionType and type(q_2) == FunctionType:
        new_qs = [func(q_1(), q_2())]

        for _ in range(dim - 1):
            new_qs.append(func(q_1(), q_2()))

    else:
        raise ValueError(f"Cannot work with q_1's type: {type(q_1)}")

    return Qs(new_qs, qs_type=qs_type)

