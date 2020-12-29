#!/usr/bin/env python
# coding: utf-8

# # Developing Quaternions for iPython
import pytest
from Qs import *

Q1: Q = Q([1, -2, -3, -4], q_type="Q")
P: Q = Q([0, 4, -3, 0], q_type="P")
R: Q = Q([3, 0, 0, 0], q_type="R")
C: Q = Q([2, 4, 0, 0], q_type="C")
t, x, y, z = sp.symbols("t x y z")
q_sym: Q = Q([t, x, y, x * y * z])
q22: Q = Q([2, 2, 0, 0])
q44: Q = Q([4, 4, 0, 0])
q4321: Q = Q([4, 3, 2, 1])
q1324: Q = Q([1, 3, 2, 4])
q2244: Q = Q([2, 2, 4, 4])

q_0 = q0()
q_1 = q1()
q_i = qi()
q_n1 = Q([-1, 0, 0, 0])
q_2 = Q([2, 0, 0, 0])
q_n2 = Q([-2, 0, 0, 0])
q_3 = Q([3, 0, 0, 0])
q_n3 = Q([-3, 0, 0, 0])
q_4 = Q([4, 0, 0, 0])
q_5 = Q([5, 0, 0, 0])
q_6 = Q([6, 0, 0, 0])
q_10 = Q([10, 0, 0, 0])
q_n5 = Q([-5, 0, 0, 0])
q_7 = Q([7, 0, 0, 0])
q_8 = Q([8, 0, 0, 0])
q_9 = Q([9, 0, 0, 0])
q_n11 = Q([-11, 0, 0, 0])
q_21 = Q([21, 0, 0, 0])
q_n34 = Q([-34, 0, 0, 0])

Q12: Q = Q([1, 2, 0, 0])
Q1123: Q = Q([1, 1, 2, 3])
Q11p: Q = Q([1, 1, 0, 0], representation="polar")
Q12p: Q = Q([1, 2, 0, 0], representation="polar")
Q12np: Q = Q([1, -2, 0, 0], representation="polar")
Q21p: Q = Q([2, 1, 0, 0], representation="polar")
Q23p: Q = Q([2, 3, 0, 0], representation="polar")
Q13p: Q = Q([1, 3, 0, 0], representation="polar")
Q5p: Q = Q([5, 0, 0, 0], representation="polar")

v3: Qs = Qs([q_3])
v1123: Qs = Qs([q_1, q_1, q_2, q_3])
v3n1n21: Qs = Qs([q_3, q_n1, q_n2, q_1])
v9: Qs = Qs([q_1, q_1, q_2, q_3, q_1, q_1, q_2, q_3, q_2])
v9i: Qs = Qs(
    [
        Q([0, 1, 0, 0]),
        Q([0, 2, 0, 0]),
        Q([0, 3, 0, 0]),
        Q([0, 4, 0, 0]),
        Q([0, 5, 0, 0]),
        Q([0, 6, 0, 0]),
        Q([0, 7, 0, 0]),
        Q([0, 8, 0, 0]),
        Q([0, 9, 0, 0]),
    ]
)
vv9 = adds(v9, v9i)
q_1d0 = Q([1.0, 0, 0, 0])
q12: Qs = Qs([q_1d0, q_1d0])
q14: Qs = Qs([q_1d0, q_1d0, q_1d0, q_1d0])
q19: Qs = Qs([q_1d0, q_0, q_1d0, q_1d0, q_1d0, q_1d0, q_1d0, q_1d0, q_1d0])
qn627 = Q([-6, 27, 0, 0])
v33 = Qs([q_7, q_0, q_n3, q_2, q_3, q_4, q_1, q_n1, q_n2])
v33inv: Qs = Qs([q_n2, q_3, q_9, q_8, q_n11, q_n34, q_n5, q_7, q_21])
q_i3: Qs = Qs([q_1, q_1, q_1])
q_i2d: Qs = Qs([q_1, q_0, q_0, q_1])
q_i3_bra = Qs([q_1, q_1, q_1], "bra")
q_6_op: Qs = Qs([q_1, q_0, q_0, q_1, q_i, q_i], "op")
q_6_op_32: Qs = Qs([q_1, q_0, q_0, q_1, q_i, q_i], "op", rows=3, columns=2)
q_i2d_op: Qs = Qs([q_1, q_0, q_0, q_1], "op")
q_i4 = Q([0, 4, 0, 0])
q_0_q_1: Qs = Qs([q_0, q_1])
q_1_q_0: Qs = Qs([q_1, q_0])
q_1_q_i: Qs = Qs([q_1, q_i])
q_0_q_i: Qs = Qs([q_0, q_i])
A: Qs = Qs([Q([4, 0, 0, 0]), Q([0, 1, 0, 0])], "bra")
B: Qs = Qs([Q([0, 0, 1, 0]), Q([0, 0, 0, 2]), Q([0, 3, 0, 0])])
Op: Qs = Qs(
    [
        Q([3, 0, 0, 0]),
        Q([0, 1, 0, 0]),
        Q([0, 0, 2, 0]),
        Q([0, 0, 0, 3]),
        Q([2, 0, 0, 0]),
        Q([0, 4, 0, 0]),
    ],
    "op",
    rows=2,
    columns=3,
)
Op4i = Qs([q_i4, q_0, q_0, q_i4, q_2, q_3], "op", rows=2, columns=3)
Op_scalar = Qs([q_i4], "scalar")
q_1234 = Qs(
    [Q([1, 1, 0, 0]), Q([2, 1, 0, 0]), Q([3, 1, 0, 0]), Q([4, 1, 0, 0])]
)
sigma_y = Qs(
    [Q([1, 0, 0, 0]), Q([0, -1, 0, 0]), Q([0, 1, 0, 0]), Q([-1, 0, 0, 0])]
)
qn = Qs([Q([3, 0, 0, 4])])
# TODO test exception like so: q_bad = Qs([q_1], rows=2, columns=3)

b = Qs([q_1, q_2, q_3], qs_type="bra")
k = Qs([q_4, q_5, q_6], qs_type="ket")
o = Qs([q_10], qs_type="op")

Q_states = Qs([Q1])
P_states = Qs([P])
t, x, y, z = sp.symbols("t x y z")
qs_22 = Qs([Q([2, 2, 0, 0])])
qs_44 = Qs([Q([4, 4, 0, 0])])

q1234 = Q([1, 2, 3, 4])
q2222 = Q([2, 2, 2, 2])
qsmall = Q([0.04, 0.2, 0.1, -0.3])
q2_states = Qs([q1234, qsmall], "ket")
qs_1234 = Qs([q1324, q1234])
qs_1324 = Qs([q1324, q1324])


def test__1000_qt():
    assert Q1.t == 1


def test__1010_txyz_2_representation():
    qr = Q(Q12.txyz_2_representation(""))
    assert equal(qr, Q12)
    qr = Q(Q12.txyz_2_representation("polar"))
    assert equal(qr, Q([2.23606797749979, 1.10714871779409, 0, 0]))
    qr = Q(Q1123.txyz_2_representation("spherical"))
    assert equal(qr, Q([1.0, 3.7416573867739413, 0.640522312679424, 1.10714871779409]))


def test__1011_representation_2_txyz():
    qr = Q(Q12.representation_2_txyz(""))
    assert equal(qr, Q12)
    qr = Q(Q12.representation_2_txyz("polar"))
    assert equal(qr, Q([-0.4161468365471424, 0.9092974268256817, 0, 0]))
    qr = Q(Q1123.representation_2_txyz("spherical"))
    assert equal(qr,
                 Q(
                     [
                         1.0,
                         -0.9001976297355174,
                         0.12832006020245673,
                         -0.4161468365471424,
                     ]
                 )
                 )


def test__1020_polar_products():
    qr = product(Q11p, Q12p)
    print("polar 1 1 0 0 * 1 2 0 0: ", qr)
    assert equal(qr, Q13p)
    qr = product(Q12p, Q21p)
    print("polar 1 2 0 0 * 2 1 0 0: ", qr)
    assert equal(qr, Q23p)


def test__1030_polar_conj():
    qr = conj(Q12p)
    print("polar conj of 1 2 0 0: ", qr)
    assert equal(qr, Q12np)


def test__1040_subs():
    q_z = q_sym.subs({t: 1, x: 2, y: 3, z: 4})
    print("t x y xyz sub 1 2 3 4: ", q_z)
    assert equal(q_z, Q([1, 2, 3, 24]))


def test__1031_subs():
    t1, x1, y1, z1 = sp.symbols("t x y z")
    q_syms = Qs([Q([t1, x1, y1, x1 * y1 * z1])])

    q_z = q_syms.subs({t1: 1, x1: 2, y1: 3, z1: 4})
    print("t x y xyz sub 1 2 3 4: ", q_z)
    assert equals(q_z, Qs([Q([1, 2, 3, 24])]))


def test__1040_xyz():
    q_z = Q1.xyz()
    print("q.xyz()): ", q_z)
    assert q_z[0] == -2
    assert q_z[1] == -3
    assert q_z[2] == -4


def test__1042_q_to_qs_function():
    assert type(q_to_qs_function(sin, q1s())) == Qs


def test__1043_qq_to_qs_function():
    assert type(qq_to_qs_function(add, q1s(), q1s())) == Qs


def test__1044_qqq_to_qs_function():
    assert type(qqq_to_qs_function(triple_product, q1s(), q1s(), q1s())) == Qs


def test__1050_scalar_q():
    q_z = scalar_q(Q1)
    print("scalar(q): ", q_z)
    assert q_z.t == 1
    assert q_z.x == 0
    assert q_z.y == 0
    assert q_z.z == 0


def test__1051_scalar_qs():
    qs = scalar_qs(q_1_q_i)
    print("scalar(q_1_q_i)", qs)
    assert equals(qs, q_1_q_0)


def test__1060_vector():
    q_z = vector_q(Q1)
    print("vector(q): ", q_z)
    assert q_z.t == 0
    assert q_z.x == -2
    assert q_z.y == -3
    assert q_z.z == -4


def test__1061_vector_qs():
    qv = vector_qs(q_1_q_i)
    print("vector(q_1_q_i)", qv)
    assert equals(qv, q_0_q_i)


def test__1070_q0():
    q_z = q0()
    print("q_0: ", q_z)
    assert q_z.t == 0
    assert q_z.x == 0
    assert q_z.y == 0
    assert q_z.z == 0


def test__1071_q0s():
    q_0s = q0s(dim=3)
    print("q0(3): ", q_0s)
    assert q_0s.dim == 3


def test__1080_q1():
    q_z: Q = q1()
    print("q_1: ", q_z)
    assert q_z.t == 1
    assert q_z.x == 0
    assert q_z.y == 0
    assert q_z.z == 0


def test__1081_q1s():
    q_1s = q1s(2.0, 3, "bra")
    print("q1(3): ", q_1s)
    assert q_1s.dim == 3
    assert q_1s.qs[0].t == 2.0
    assert q_1s.qs_type == "bra"


def test__1090_qi():
    q_z = qi()
    print("qi: ", q_z)
    assert q_z.t == 0
    assert q_z.x == 1
    assert q_z.y == 0
    assert q_z.z == 0


def test__1091_qis():
    q_is = qis(2.0, 3)
    print("qi(3): ", q_i)
    assert q_is.dim == 3
    assert q_is.qs[0].x == 2.0
    assert q_is.qs_type == "ket"


def test__1100_q_j():
    q_z = qj()
    print("q_j: ", q_z)
    assert q_z.t == 0
    assert q_z.x == 0
    assert q_z.y == 1
    assert q_z.z == 0


def test__1101_qjs():
    q_j = qjs(2.0, 3)
    print("qj(3): ", qj)
    assert q_j.dim == 3
    assert q_j.qs[0].y == 2.0
    assert q_j.qs_type == "ket"


def test__1110_qk():
    q_z = qk()
    print("q_k: ", q_z)
    assert q_z.t == 0
    assert q_z.x == 0
    assert q_z.y == 0
    assert q_z.z == 1


def test__1111_qks():
    q_k = qks(2.0, 3)
    print("qk(3): ", q_k)
    assert q_k.dim == 3
    assert q_k.qs[0].z == 2.0
    assert q_k.qs_type == "ket"


def test__1120_qrandom():
    q_z = qrandom()
    print("q_random():", q_z)
    assert -1 <= q_z.t <= 1
    assert -1 <= q_z.x <= 1
    assert -1 <= q_z.y <= 1
    assert -1 <= q_z.z <= 1


def test__1121_qrandomis():
    qr = qrandoms(-2, 2, dim=3)
    print("qk(3): ", qr)
    assert qr.dim == 3
    assert qr.qs[0].z != qr.qs[0].t


def test__1130_equals():
    assert equal(Q1, Q1)
    assert not equal(Q1, P)
    assert equal(C, q44, scalar=False)
    assert equal(q44, q4321, vector=False)


def test__1131_equals():
    assert equals(A, A)
    assert not equals(A, B)


def test__1140_conj():
    q_z = conj(Q1)
    print("q_conj 0: ", q_z)
    assert q_z.t == 1
    assert q_z.x == 2
    assert q_z.y == 3
    assert q_z.z == 4


def test__1141_conjs():
    qc = conjs(q_1_q_i)
    qc1 = conjs(q_1_q_i, 1)
    print("q_1_q_i*: ", qc)
    print("q_1_qc*1: ", qc1)
    assert qc.qs[1].x == -1
    assert qc1.qs[1].x == 1


def test__1141_conj_1():
    q_z = conj(Q1, 1)
    print("q_conj 1: ", q_z)
    assert q_z.t == -1
    assert q_z.x == -2
    assert q_z.y == 3
    assert q_z.z == 4


def test__1141_conjs_1():
    q_z = conjs(Qs([Q1]), 1)
    print("q_conj 1: ", q_z)
    assert q_z.qs[0].t == -1
    assert q_z.qs[0].x == -2
    assert q_z.qs[0].y == 3
    assert q_z.qs[0].z == 4


def test__1150_conj_2():
    q_z = conj(Q1, 2)
    print("q_conj 2: ", q_z)
    assert q_z.t == -1
    assert q_z.x == 2
    assert q_z.y == -3
    assert q_z.z == 4


def test__1151_conj_2():
    q_z = conjs(Qs([Q1]), 2)
    print("q_conj 2: ", q_z)
    assert q_z.qs[0].t == -1
    assert q_z.qs[0].x == 2
    assert q_z.qs[0].y == -3
    assert q_z.qs[0].z == 4


def test__1155_conj_q():
    q_z = conj_q(Q1, Q1)
    print("conj_q(conj_q): ", q_z)
    assert q_z.t == -1
    assert q_z.x == 2
    assert q_z.y == 3
    assert q_z.z == -4


def test__1156_conj_qs():
    qc = conj_qs(q_1_q_i, q_1)
    qc1 = conj_qs(q_1_q_i, q_1)
    print("q_1_q_i conj_q: ", qc)
    print("q_1_qc*1 conj_q: ", qc1)
    assert qc.qs[1].x == -1
    assert qc1.qs[1].x == -1


def test__1160_flip_sign():
    q_z = flip_sign(Q1)
    print("sign_flips: ", q_z)
    assert q_z.t == -1
    assert q_z.x == 2
    assert q_z.y == 3
    assert q_z.z == 4


def test__1161_flip_signs():
    qf = flip_signs(q_1_q_i)
    print("-q_1_q_i: ", qf)
    assert qf.qs[1].x == -1


def test__1170_vahlen_conj_minus():
    q_z = vahlen_conj(Q1)
    print("q_vahlen_conj -: ", q_z)
    assert q_z.t == 1
    assert q_z.x == 2
    assert q_z.y == 3
    assert q_z.z == 4


def test__1180_vahlen_conj_star():
    q_z = vahlen_conj(Q1, "*")
    print("q_vahlen_conj *: ", q_z)
    assert q_z.t == 1
    assert q_z.x == -2
    assert q_z.y == -3
    assert q_z.z == 4


def test__1190_vahlen_conj_prime():
    q_z = vahlen_conj(Q1, "'")
    print("q_vahlen_conj ': ", q_z)
    assert q_z.t == 1
    assert q_z.x == 2
    assert q_z.y == 3
    assert q_z.z == -4


def test__1200_square():
    q_z = square(Q1)
    print("square: ", q_z)
    assert q_z.t == -28
    assert q_z.x == -4
    assert q_z.y == -6
    assert q_z.z == -8


def test__1201_squares():
    ns = squares(q_1_q_i)
    ns.print_state("q_1_q_i squares")
    assert equals(ns, Qs([q_1, q_n1]))


def test__1210_norm_squared():
    q_z = norm_squared(Q1)
    print("norm_squared: ", q_z)
    assert q_z.t == 30
    assert q_z.x == 0
    assert q_z.y == 0
    assert q_z.z == 0


def test__1211_norm_squareds():
    ns = norm_squareds(q_1_q_i)
    ns.print_state("q_1_q_i norm squareds")
    assert equals(ns, Qs([Q([2, 0, 0, 0])]))


def test__1220_norm_squared_of_vector():
    q_z = norm_squared_of_vector(Q1)
    print("norm_squared_of_vector: ", q_z)
    assert q_z.t == 29
    assert q_z.x == 0
    assert q_z.y == 0
    assert q_z.z == 0


def test__1221_norm_squared_of_vector():
    ns = norm_squared_of_vectors(q_1_q_i)
    ns.print_state("q_1_q_i norm squared of vectors")
    assert equals(ns, Qs([q_1]))


def test__1230_abs_of_q():
    q_z = abs_of_q(P)
    print("abs_of_q: ", q_z)
    assert q_z.t == 5
    assert q_z.x == 0
    assert q_z.y == 0
    assert q_z.z == 0


def test__1231_abs_of_qs():
    ns = abs_of_qs(q_1_q_i)
    ns.print_state("q_1_q_i abs_of_qs")
    assert equals(ns, Qs([Q([2 ** (1 / 2), 0, 0, 0])]))


def test__1240_normalize():
    q_z = normalize(P)
    print("q_normalized: ", q_z)
    assert q_z.t == 0
    assert q_z.x == 0.8
    assert math.isclose(q_z.y, -0.6)
    assert q_z.z == 0


def test__1241_normalize():
    qn_test = normalizes(qn)
    print("Op normalized: ", qn_test)
    assert math.isclose(qn_test.qs[0].t, 0.6)
    assert qn_test.qs[0].z == 0.8


def test__1250_determinant():
    det_v3 = determinant(v3)
    assert equals(det_v3, v3)
    v_1123: Qs = Qs([q_1, q_1, q_2, q_3])
    det_v1123 = determinant(v_1123)
    print("det v1123", det_v1123)
    assert equals(det_v1123, Qs([q_1]))
    v9.print_state("v9")
    det_v9 = determinant(v9)
    print("det_v9", det_v9)
    assert equals(det_v9, Qs([q_9]))
    det_vv9 = determinant(vv9)
    print("det_vv9", det_vv9)
    assert equals(det_vv9, Qs([qn627]))


def test__1260_abs_of_vector():
    q_z = abs_of_vector(P)
    print("abs_of_vector: ", q_z)
    assert q_z.t == 5
    assert q_z.x == 0
    assert q_z.y == 0
    assert q_z.z == 0


def test__1261_abs_of_vectors():
    q_z = abs_of_vectors(P_states)
    print("abs_of_vector: ", q_z)
    assert q_z.qs[0].t == 5
    assert q_z.qs[0].x == 0
    assert q_z.qs[0].y == 0
    assert q_z.qs[0].z == 0


def test__1270_add():
    q_z = add(Q1, P)
    print("add: ", q_z)
    assert q_z.t == 1
    assert q_z.x == 2
    assert q_z.y == -6
    assert q_z.z == -4


def test__1271_adds():
    q_0110_add = adds(q_0_q_1, q_1_q_0)
    print("add 01 10: ", q_0110_add)
    assert q_0110_add.qs[0].t == 1
    assert q_0110_add.qs[1].t == 1


def test__1280_dif():
    q_z = dif(Q1, P)
    print("dif: ", q_z)
    assert q_z.t == 1
    assert q_z.x == -6
    assert q_z.y == 0
    assert q_z.z == -4


def test__1281_difs():
    q_0110_dif = difs(q_0_q_1, q_1_q_0)
    print("dif 01 10: ", q_0110_dif)
    assert q_0110_dif.qs[0].t == -1
    assert q_0110_dif.qs[1].t == 1


def test__1290_product():
    q_z = product(Q1, P)
    print("product: ", q_z)
    assert q_z.t == -1
    assert q_z.x == -8
    assert q_z.y == -19
    assert q_z.z == 18


def test__1300_product_even():
    q_z = product(Q1, P, kind="even")
    print("product, kind even: ", q_z)
    assert q_z.t == -1
    assert q_z.x == 4
    assert q_z.y == -3
    assert q_z.z == 0


def test__1310_product_odd():
    q_z = product(Q1, P, kind="odd")
    print("product, kind odd: ", q_z)
    assert q_z.t == 0
    assert q_z.x == -12
    assert q_z.y == -16
    assert q_z.z == 18


def test__1320_product_even_minus_odd():
    q_z = product(Q1, P, kind="even_minus_odd")
    print("product, kind even_minus_odd: ", q_z)
    assert q_z.t == -1
    assert q_z.x == 16
    assert q_z.y == 13
    assert q_z.z == -18


def test__1330_product_reverse():
    q1q2_rev = product(Q1, P, reverse=True)
    q2q1 = product(P, Q1)
    assert equal(q1q2_rev, q2q1)


def test__1340_product():
    assert equals(products(b, o), Qs([Q([10, 0, 0, 0]), Q([20, 0, 0, 0]), Q([30, 0, 0, 0])]))
    assert equals(products(b, k), Qs([Q([32, 0, 0, 0])]))
    assert equals(products(products(b, o), k), Qs([Q([320, 0, 0, 0])]))
    assert equals(products(b, b), Qs([Q([1, 0, 0, 0]), Q([4, 0, 0, 0]), Q([9, 0, 0, 0])]))
    assert equals(products(o, k), Qs([Q([40, 0, 0, 0]), Q([50, 0, 0, 0]), Q([60, 0, 0, 0])]))
    assert equals(products(o, o), Qs([Q([100, 0, 0, 0])]))
    assert equals(products(k, k), Qs([Q([16, 0, 0, 0]), Q([25, 0, 0, 0]), Q([36, 0, 0, 0])]))
    assert equals(products(k, b), Qs(
        [
            Q([4, 0, 0, 0]),
            Q([5, 0, 0, 0]),
            Q([6, 0, 0, 0]),
            Q([8, 0, 0, 0]),
            Q([10, 0, 0, 0]),
            Q([12, 0, 0, 0]),
            Q([12, 0, 0, 0]),
            Q([15, 0, 0, 0]),
            Q([18, 0, 0, 0]),
        ]
    )
                  )


def test__1350_product_AA():
    Aket = deepcopy(A).ket()
    AA = products(A, Aket)
    print("<A|A>: ", AA)
    assert equals(AA, Qs([Q([17, 0, 0, 0])]))


def test__1360_product_AOp():
    AOp: Qs = products(A, Op)
    print("A Op: ", AOp)
    assert equal(AOp.qs[0], Q([11, 0, 0, 0]))
    assert equal(AOp.qs[1], Q([0, 0, 5, 0]))
    assert equal(AOp.qs[2], Q([4, 0, 0, 0]))


def test__1370_product_AOp4i():
    AOp4i: Qs = products(A, Op4i)
    print("A Op4i: ", AOp4i)
    assert equal(AOp4i.qs[0], Q([0, 16, 0, 0]))
    assert equal(AOp4i.qs[1], Q([-4, 0, 0, 0]))


def test__1380_product_OpB():
    OpB: Qs = products(Op, B)
    print("Op B: ", OpB)
    assert equal(OpB.qs[0], Q([0, 10, 3, 0]))
    assert equal(OpB.qs[1], Q([-18, 0, 0, 1]))


def test__1390_product_AOpB():
    AOpB = products(A, products(Op, B))
    print("A Op B: ", AOpB)
    assert equals(AOpB, Qs([Q([0, 22, 11, 0])]))


def test__1400_product_AOp4i():
    AOp4i = products(A, Op4i)
    print("A Op4i: ", AOp4i)
    assert equal(AOp4i.qs[0], Q([0, 16, 0, 0]))
    assert equal(AOp4i.qs[1], Q([-4, 0, 0, 0]))


def test__1410_product_Op4iB():
    Op4iB = products(Op4i, B)
    print("Op4i B: ", Op4iB)
    assert equal(Op4iB.qs[0], Q([0, 6, 0, 4]))
    assert equal(Op4iB.qs[1], Q([0, 9, -8, 0]))


def test__1420_product_AOp4iB():
    AOp4iB = products(A, products(Op4i, B))
    print("A* Op4i B: ", AOp4iB)
    assert equals(AOp4iB, Qs([Q([-9, 24, 0, 8])]))


def test__1425_dot_product():
    q_dot = dot_product(qs_1234.bra(), qs_1234)
    print("qs_1234 dot product", q_dot)
    assert equal(q_dot, q0(), scalar=False)
    assert q_dot.t == 60


def test__1430_inverse():
    q_z = inverse(P)
    print("inverse: ", q_z)
    assert q_z.t == 0
    assert q_z.x == -0.16
    assert q_z.y == 0.12
    assert q_z.z == 0


def test__1431_inverses():
    q_z = inverses(P_states)
    print("inverse: ", q_z)
    assert equals(q_z, Qs([Q([0, -0.16, 0.12, 0])]))


def test__1440_divide_by():
    q_z = divide_by(Q1, Q1)
    print("divide_by: ", q_z)
    assert q_z.t == 1
    assert q_z.x == 0
    assert q_z.y == 0
    assert q_z.z == 0


def test__1441_divide_bys():
    q_z = divide_bys(Q_states, Q_states)
    print("divide_by: ", q_z)
    assert equals(q_z, Qs([q_1]))


def test__1450_triple_product():
    q_z = triple_product(Q1, P, Q1)
    print("triple product: ", q_z)
    assert q_z.t == -2
    assert q_z.x == 124
    assert q_z.y == -84
    assert q_z.z == 8


def test__1451_triple_products():
    q_z = triple_products(Q_states, P_states, Q_states)
    print("triple product: ", q_z)
    assert equals(q_z, Qs([Q([-2, 124, -84, 8])]))


def test__1460_rotate():
    q_z = rotation(Q1, Q([0, 1, 0, 0]))
    print("rotation: ", q_z)
    assert q_z.t == 1
    assert q_z.x == -2
    assert q_z.y == 3
    assert q_z.z == 4


def test__1460_rotates():
    q_z = rotations(Q_states, Qs([q_i]))
    print("rotation: ", q_z)
    assert equals(q_z, Qs([Q([1, -2, 3, 4])]))


def test__1470_rotation_angle():
    q_ij = add(qi(), qj())

    assert rotation_angle(qi(), qi(), degrees=True).t == 0.0
    assert math.isclose(rotation_angle(qi(), q_ij, degrees=True).t, 45.0)
    assert rotation_angle(qi(), qj(), degrees=True).t == 90.0

    # all 3 add up to 180
    r123 = rotation_angle(qi(), q_ij, origin=q0(), degrees=True).t
    r312 = rotation_angle(q0(), qi(), origin=q_ij, degrees=True).t
    r231 = rotation_angle(q_ij, q0(), origin=qi(), degrees=True).t
    assert math.isclose(r123 + r312 + r231, 180)


def _test_the_square(function_to_test: FunctionType, numbers_dict: Dict = None, quiet_pass: bool = False,
                     quiet_fail: bool = False) -> bool:
    """
    Pick out a few classes for q to test - real, imaginary, complex,
    small norm quaternion, large norm quaternion

    Pick out the same category for h to test.

    Test the Cartesian product of q and h to see if the first term of
    the squares are the same before and after the mapping.

    Args:
        function_to_test: FunctionType    The function to test
        quiet_pass: bool   quiets all passes
        quiet_fail: bool   quiets the failed tests

    Returns: bool   passed all tests or not

    Will print results to stdout.

    """

    if numbers_dict is None:
        numbers_dict = {"zero": q0(), "one": q1()}
        numbers_dict["real"], numbers_dict["imaginary"] = Q([2.0, 0, 0, 0]), Q([0, 3.0, 0, 0])
        numbers_dict["complex"] = Q([2.0, 4.0, 0, 0])
        numbers_dict["small"], numbers_dict["big"] = Q([0.1, 0.2, 0.3, 0.4]), Q([3.0, 4.0 , 2.0, 5.0])

    good, evil = 0, 0

    print(f"testing Cartesian product of these numbers: {', '.join(numbers_dict.keys())}, total={len(numbers_dict.keys()) ** 2}")

    for q_name, q_number in numbers_dict.items():

        q_square = square(q_number)

        for h_name, h_number in numbers_dict.items():

            applied = function_to_test(q_number, h_number)

            applied_square = square(applied)

            truth = equal(q_square, applied_square, vector=False)

            if truth:

                good += 1

                if not quiet_pass:
                    print(f"PASS: q={q_name} {q_number}, h={h_name} {h_number}")

            else:

                evil += 1

                if not quiet_fail:
                    print(f"FAIL: q={q_name} {q_number}, h={h_name} {h_number}")

    print(f"Pass/Fail: {good}/{evil}")

    return bool(not evil)

def test__1470_rotation_and_or_boost():
    q1_sq = square(Q1)
    beta: float = 0.003
    gamma = 1 / math.sqrt(1 - beta ** 2)
    h = Q([gamma, gamma * beta, 0, 0])
    q_z = rotation_and_or_boost(Q1, h)
    q_z2 = square(q_z)
    print("q1_sq: ", q1_sq)
    print("boosted: ", q_z)
    print("boosted squared: ", q_z2)
    assert round(q_z2.t, 5) == round(q1_sq.t, 5)
    assert _test_the_square(rotation_and_or_boost)

def test__1471_rotation_and_or_boosts():
    q1_sq = squares(Q_states)
    beta = 0.003
    gamma = 1 / math.sqrt(1 - beta ** 2)
    h = Qs([Q([gamma, gamma * beta, 0, 0])])
    q_z = rotation_and_or_boosts(Q_states, h)
    q_z2 = squares(q_z)
    print("q1_sq: ", q1_sq)
    print("boosted: ", q_z)
    print("boosted squared: ", q_z2)
    assert round(q_z2.qs[0].t, 5) == round(q1_sq.qs[0].t, 5)


def test_1472_rotation_only():
    Q1123_rot = rotation_only(Q1123, Q12)
    print("Q1123_rot", Q1123_rot)
    assert equal(Q1123, Q1123_rot, vector=False)
    assert not equal(Q1123, Q1123_rot, scalar=False)
    assert equal(norm_squared(Q1123), norm_squared(Q1123_rot))


def test_1473_rotations_onlys():
    Q12s, Q1123s = Qs([Q12]), Qs([Q1123])
    Q1123s_rot = rotation_onlys(Q1123s, Q12s)
    print("Q1123s_rot", Q1123s_rot)
    assert equals(Q1123s, Q1123s_rot, vector=False)
    assert not equals(Q1123s, Q1123s_rot, scalar=False)
#   TODO: Why doesn't this work?
#   assert equals(norm_squareds(Q1123s), norm_squareds(Q1123s_rot))


def test__1471_next_rotation():
    with pytest.raises(ValueError):
        next_rotation(Q1, q4321)
    next_rot = next_rotation(Q1, q1324)
    print("next_rotation: ", next_rot)
    assert next_rot.t == Q1.t
    rot = rotation_and_or_boost(q2244, vector_q(next_rot))
    assert math.isclose(rot.t, 2)
    assert math.isclose(square(rot).t, square(q2244).t)
    next_rot = next_rotation(Q1, Q1)
    assert equal(next_rot, Q1)


def test__1471_next_randomized_rotation():
    with pytest.raises(ValueError):
        next_rotation_randomized(Q1, q4321)
    next_randomized_rot = next_rotation_randomized(Q1, q1324)
    print("next_rotation: ", next_randomized_rot)
    assert next_randomized_rot.t == Q1.t
    rot = rotation_and_or_boost(q2244, vector_q(next_randomized_rot))
    assert math.isclose(rot.t, 2)
    assert math.isclose(square(rot).t, square(q2244).t)
    next_rot = next_rotation(Q1, Q1)
    assert equal(next_rot, Q1)


def test__1472_next_boost():
    with pytest.raises(ValueError):
        next_boost(Q1, q4321)
    next_boo = next_boost(Q1, q1324)
    print(f"next_boost: {next_boo}")
    assert next_boo.t != 0
    boost = rotation_and_or_boost(q2244, next_boo)
    assert math.isclose(square(boost).t, square(q2244).t)


def test__1475_permutation():
    with pytest.raises(ValueError):
        permutation(Q1123, "tx")
    new_p = permutation(q1234, "tyxz")
    assert equal(new_p, q1324)


def test__1476_all_permutations():
    new_ps = all_permutations(q1234)
    assert new_ps.dim == 24


def test__1480_g_shift():
    q1_sq = square(Q1)
    q_z = g_shift(Q1, 0.003)
    q_z2 = square(q_z)
    q_z_minimal = g_shift(Q1, 0.003, g_form="minimal")
    q_z2_minimal = square(q_z_minimal)
    print("q1_sq: ", q1_sq)
    print("g_shift: ", q_z)
    print("g squared: ", q_z2)
    assert q_z2.t != q1_sq.t
    assert q_z2.x == q1_sq.x
    assert q_z2.y == q1_sq.y
    assert q_z2.z == q1_sq.z
    assert q_z2_minimal.t != q1_sq.t
    assert q_z2_minimal.x == q1_sq.x
    assert q_z2_minimal.y == q1_sq.y
    assert q_z2_minimal.z == q1_sq.z


def test__1500_sin():
    assert equal(sin(q0()), q0())
    assert equal(sin(Q1), Q(
        [
            91.7837157840346691,
            -21.8864868530291758,
            -32.8297302795437673,
            -43.7729737060583517,
        ]
    )
                 )
    assert equal(sin(P), Q([0, 59.3625684622310033, -44.5219263466732542, 0]))
    assert equal(sin(R), Q([0.1411200080598672, 0, 0, 0]))
    assert equal(sin(C), Q([24.8313058489463785, -11.3566127112181743, 0, 0]))


def test__1501_sins():
    assert equals(sins(q0s()), q0s())
    assert equals(sins(Qs([Q1])), Qs([Q(
        [
            91.7837157840346691,
            -21.8864868530291758,
            -32.8297302795437673,
            -43.7729737060583517,
        ])]))
    assert equals(sins(Qs([P])), Qs([Q([0, 59.3625684622310033, -44.5219263466732542, 0])]))
    assert equals(sins(Qs([R])), Qs([Q([0.1411200080598672, 0, 0, 0])]))
    assert equals(sins(Qs([C])), Qs([Q([24.8313058489463785, -11.3566127112181743, 0, 0])]))


def test__1510_cos():
    assert equal(cos(q0()), q1())
    assert equal(cos(Q1), Q([
        58.9336461679439481,
        34.0861836904655959,
        51.1292755356983974,
        68.1723673809311919,
    ]))
    assert equal(cos(P), Q([74.2099485247878476, 0, 0, 0]))
    assert equal(cos(R), Q([-0.9899924966004454, 0, 0, 0]))
    assert equal(cos(C), Q([-11.3642347064010600, -24.8146514856341867, 0, 0]))


def test__1511_coss():
    assert equals(coss(q0s()), q1s())
    assert equals(coss(Qs([Q1])), Qs([Q([
        58.9336461679439481,
        34.0861836904655959,
        51.1292755356983974,
        68.1723673809311919,
    ])]))
    assert equals(coss(Qs([P])), Qs([Q([74.2099485247878476, 0, 0, 0])]))
    assert equals(coss(Qs([R])), Qs([Q([-0.9899924966004454, 0, 0, 0])]))
    assert equals(coss(Qs([C])), Qs([Q([-11.3642347064010600, -24.8146514856341867, 0, 0])]))


def test__1520_tan():
    assert equal(tan(q0()), q0())
    assert equal(tan(Q1), Q([0.0000382163172501,
                             -0.3713971716439372,
                             -0.5570957574659058,
                             -0.7427943432878743, ]))
    assert equal(tan(P), Q([0, 0.7999273634100760, -0.5999455225575570, 0]))
    assert equal(tan(R), Q([-0.1425465430742778, 0, 0, 0]))
    assert equal(tan(C), Q([-0.0005079806234700, 1.0004385132020521, 0, 0]))


def test__1521_tans():
    assert equals(tans(q0s()), q0s())
    assert equals(tans(Qs([Q1])), Qs([Q([0.0000382163172501,
                                         -0.3713971716439372,
                                         -0.5570957574659058,
                                         -0.7427943432878743, ])]))
    assert equals(tans(Qs([P])), Qs([Q([0, 0.7999273634100760, -0.5999455225575570, 0])]))
    assert equals(tans(Qs([R])), Qs([Q([-0.1425465430742778, 0, 0, 0])]))
    assert equals(tans(Qs([C])), Qs([Q([-0.0005079806234700, 1.0004385132020521, 0, 0])]))


def test__1530_sinh():
    assert equal(sinh(q0()), q0())
    assert equal(sinh(Q1),
                 Q(
                     [
                         0.7323376060463428,
                         0.4482074499805421,
                         0.6723111749708131,
                         0.8964148999610841,
                     ]
                 )
                 )
    assert equal(sinh(P), Q([0, -0.7671394197305108, 0.5753545647978831, 0]))
    assert equal(sinh(R), Q([10.0178749274099026, 0, 0, 0]))
    assert equal(sinh(C), Q([-2.3706741693520015, -2.8472390868488278, 0, 0]))


def test__1531_sinhs():
    assert equals(sinhs(q0s()), q0s())
    assert equals(sinhs(Qs([Q1])),
                  Qs([Q(
                      [
                          0.7323376060463428,
                          0.4482074499805421,
                          0.6723111749708131,
                          0.8964148999610841,
                      ])]))
    assert equals(sinhs(Qs([P])), Qs([Q([0, -0.7671394197305108, 0.5753545647978831, 0])]))
    assert equals(sinhs(Qs([R])), Qs([Q([10.0178749274099026, 0, 0, 0])]))
    assert equals(sinhs(Qs([C])), Qs([Q([-2.3706741693520015, -2.8472390868488278, 0, 0])]))


def test__1540_cosh():
    assert equal(cosh(q0()), q1())
    assert equal(cosh(Q1), Q(
        [
            0.9615851176369565,
            0.3413521745610167,
            0.5120282618415251,
            0.6827043491220334,
        ]
    )
                 )
    assert equal(cosh(P), Q([0.2836621854632263, 0, 0, 0]))
    assert equal(cosh(R), Q([10.0676619957777653, 0, 0, 0]))
    assert equal(cosh(C), Q([-2.4591352139173837, -2.7448170067921538, 0, 0]))


def test__1541_coshs():
    assert equals(coshs(q0s()), q1s())
    assert equals(coshs(Qs([Q1])), Qs([Q(
        [
            0.9615851176369565,
            0.3413521745610167,
            0.5120282618415251,
            0.6827043491220334,
        ])]))
    assert equals(coshs(Qs([P])), Qs([Q([0.2836621854632263, 0, 0, 0])]))
    assert equals(coshs(Qs([R])), Qs([Q([10.0676619957777653, 0, 0, 0])]))
    assert equals(coshs(Qs([C])), Qs([Q([-2.4591352139173837, -2.7448170067921538, 0, 0])]))


def test__1550_tanh():
    assert equal(tanh(q0()), q0())
    assert equal(tanh(Q1),
                 Q(
                     [
                         1.0248695360556623,
                         0.1022956817887642,
                         0.1534435226831462,
                         0.2045913635775283,
                     ]
                 )
                 )
    assert equal(tanh(P), Q([0, -2.7044120049972684, 2.0283090037479505, 0]))
    assert equal(tanh(R), Q([0.9950547536867305, 0, 0, 0]))
    assert equal(tanh(C), Q([1.0046823121902353, 0.0364233692474038, 0, 0]))


def test__1551_tanhs():
    assert equals(tanhs(q0s()), q0s())
    assert equals(tanhs(Qs([Q1])),
                  Qs([Q(
                      [
                          1.0248695360556623,
                          0.1022956817887642,
                          0.1534435226831462,
                          0.2045913635775283,
                      ])]))
    assert equals(tanhs(Qs([P])), Qs([Q([0, -2.7044120049972684, 2.0283090037479505, 0])]))
    assert equals(tanhs(Qs([R])), Qs([Q([0.9950547536867305, 0, 0, 0])]))
    assert equals(tanhs(Qs([C])), Qs([Q([1.0046823121902353, 0.0364233692474038, 0, 0])]))


def test__1560_exp():
    assert equal(exp(q0()), q1())
    assert equal(exp(Q1), Q(
        [
            1.6939227236832994,
            0.7895596245415588,
            1.1843394368123383,
            1.5791192490831176,
        ]
    )
                 )
    assert equal(exp(P), Q([0.2836621854632263, -0.7671394197305108, 0.5753545647978831, 0]))
    assert equal(exp(R), Q([20.0855369231876679, 0, 0, 0]))
    assert equal(exp(C), Q([-4.8298093832693851, -5.5920560936409816, 0, 0]))


def test__1561_exps():
    assert equals(exps(q0s()), q1s())
    assert equals(exps(Qs([Q1])), Qs([Q(
        [
            1.6939227236832994,
            0.7895596245415588,
            1.1843394368123383,
            1.5791192490831176,
        ])]))
    assert equals(exps(Qs([P])), Qs([Q([0.2836621854632263, -0.7671394197305108, 0.5753545647978831, 0])]))
    assert equals(exps(Qs([R])), Qs([Q([20.0855369231876679, 0, 0, 0])]))
    assert equals(exps(Qs([C])), Qs([Q([-4.8298093832693851, -5.5920560936409816, 0, 0])]))


def test__1570_ln():
    assert equal(exp(ln(Q1)), Q1)
    assert equal(ln(Q1), Q(
        [
            1.7005986908310777,
            -0.5151902926640850,
            -0.7727854389961275,
            -1.0303805853281700,
        ]
    )
                 )
    assert equal(ln(P), Q([1.6094379124341003, 1.2566370614359172, -0.9424777960769379, 0]))
    assert equal(ln(R), Q([1.0986122886681098, 0, 0, 0]))
    assert equal(ln(C), Q([1.4978661367769954, 1.1071487177940904, 0, 0]))


def test__1571_lns():
    assert equals(exps(lns(Qs([Q1]))), Qs([Q1]))
    assert equals(lns(Qs([Q1])), Qs([Q(
        [
            1.7005986908310777,
            -0.5151902926640850,
            -0.7727854389961275,
            -1.0303805853281700,
        ])]))
    assert equals(lns(Qs([P])), Qs([Q([1.6094379124341003, 1.2566370614359172, -0.9424777960769379, 0])]))
    assert equals(lns(Qs([R])), Qs([Q([1.0986122886681098, 0, 0, 0])]))
    assert equals(lns(Qs([C])), Qs([Q([1.4978661367769954, 1.1071487177940904, 0, 0])]))


def test__1580_q_2_q():
    assert equal(q_2_q(Q1, P), Q(
        [
            -0.0197219653530713,
            -0.2613955437374326,
            0.6496281248064009,
            -0.3265786562423951,
        ]
    )
                 )


def test__1581_q_2_qs():
    assert equals(q_2_qs(Qs([Q1]), Qs([P])), Qs([Q(
        [
            -0.0197219653530713,
            -0.2613955437374326,
            0.6496281248064009,
            -0.3265786562423951,
        ])]))


def test__1000_init():
    assert q_0_q_1.dim == 2


def test__1010_set_qs_type():
    bk = b.set_qs_type("ket")
    assert bk.rows == 3
    assert bk.columns == 1
    assert bk.qs_type == "ket"


def test__1020_set_rows_and_columns():
    assert q_i3.rows == 3
    assert q_i3.columns == 1
    assert q_i3_bra.rows == 1
    assert q_i3_bra.columns == 3
    assert q_i2d_op.rows == 2
    assert q_i2d_op.columns == 2
    assert q_6_op_32.rows == 3
    assert q_6_op_32.columns == 2


def test__1034_xyz():
    qxyz = q_1_q_i.xyz()
    print("q_1_q_i.xyz()", qxyz)
    assert qxyz[0][0] == 0
    assert qxyz[1][0] == 1


def test__1060_inverse():
    inv_v1123 = inverses(v1123)
    print("inv_v1123 operator", inv_v1123)
    vvinv = products(inv_v1123, v1123)
    vvinv.print_state("vinvD x v")
    assert equals(vvinv, q14)

    inv_v33 = inverses(v33)
    print("inv_v33 operator", inv_v33)
    vv33 = products(inv_v33, v33)
    vv33.print_state("inv_v33D x v33")
    assert equals(vv33, q19)

    Ainv = inverses(A)
    print("A ket inverse, ", Ainv)
    AAinv = products(A, Ainv)
    AAinv.print_state("A x AinvD")
    assert equals(AAinv, q12)


def test__1120_diagonal():
    Op4iDiag2 = diagonal(Op_scalar, 2)
    print("Op4i on a diagonal 2x2", Op4iDiag2)
    assert equal(Op4iDiag2.qs[0], q_i4)
    assert equal(Op4iDiag2.qs[1], q0())


def test__1125_trace():
    tr = trace(v1123.op(2, 2))
    print("trace: ", tr)
    assert equals(tr, Qs([q_4]))


def test__1130_identity():
    I2 = identity(2, operator=True)
    print("Operator Identity, diagonal 2x2", I2)
    assert equal(I2.qs[0], q1())
    assert equal(I2.qs[1], q0())
    I2 = identity(2)
    print("Identity on 2 state ket", I2)
    assert equal(I2.qs[0], q1())
    assert equal(I2.qs[1], q1())


def test__1305_next_rotations():
    with pytest.raises(ValueError):
        next_rotations(qs_1234, q2_states)
    next_rot: Qs = next_rotations(qs_1234, qs_1324)
    print("next_rotation: ", next_rot)
    assert math.isclose(next_rot.qs[0].t, 1)
    assert math.isclose(next_rot.qs[1].t, 1)
    assert math.isclose(norm_squareds(next_rot).qs[0].t, 60)
    assert not equal(next_rot.qs[0], next_rot.qs[1])


def test__1305_next_boost():
    with pytest.raises(ValueError):
        next_boosts(qs_1234, q2_states)
    next_boost: Qs = next_boosts(qs_1234, qs_1324)
    print("next_boost: ", next_boost)
    assert next_boost.qs[0].t != 0
    assert next_boost.qs[1].t != 0
    assert norm_squareds(next_boost).qs[0].t != 1
    assert not equal(next_boost.qs[0], next_boost.qs[1])
    boosted_square = squares(rotation_and_or_boosts(q2_states, next_boost))
    q2_states_square = squares(q2_states)
    assert math.isclose(q2_states_square.qs[0].t, boosted_square.qs[0].t)


def test__1306_g_shifts():
    qs1_sq = squares(Q_states)
    qs_z = g_shifts(Q_states, 0.003)
    qs_z2 = squares(qs_z)
    qs_z_minimal = g_shifts(Q_states, 0.003, g_form="minimal")
    qs_z2_minimal = squares(qs_z_minimal)
    print("q1_sq: ", qs1_sq)
    print("g_shift: ", qs_z)
    print("g squared: ", qs_z2)
    assert qs_z2.qs[0].t != qs1_sq.qs[0].t
    assert qs_z2.qs[0].x == qs1_sq.qs[0].x
    assert qs_z2.qs[0].y == qs1_sq.qs[0].y
    assert qs_z2.qs[0].z == qs1_sq.qs[0].z
    assert qs_z2_minimal.qs[0].t != qs1_sq.qs[0].t
    assert qs_z2_minimal.qs[0].x == qs1_sq.qs[0].x
    assert qs_z2_minimal.qs[0].y == qs1_sq.qs[0].y
    assert qs_z2_minimal.qs[0].z == qs1_sq.qs[0].z


def test__1305_bracket():
    bracket1234 = Qs().bracket(
        q_1234, identity(4, operator=True), q_1234
    )
    print("bracket <1234|I|1234>: ", bracket1234)
    assert equals(bracket1234, Qs([Q([34, 0, 0, 0])]))


def test__1310_op_q():
    opn = Op.op_q(q=q_i)
    print("op_q: ", opn)
    assert opn.qs[0].x == 3


def test__1320_transpose():
    opt = transpose(q_1234)
    print("op1234 transposed: ", opt)
    assert opt.qs[0].t == 1
    assert opt.qs[1].t == 3
    assert opt.qs[2].t == 2
    assert opt.qs[3].t == 4
    optt = transpose(transpose(q_1234))
    assert equals(optt, q_1234)


def test__1330_Hermitian_conj():
    q_hc = Hermitian_conj(q_1234, 2, 2)
    print("op1234 Hermtian_conj: ", q_hc)
    assert q_hc.qs[0].t == 1
    assert q_hc.qs[1].t == 3
    assert q_hc.qs[2].t == 2
    assert q_hc.qs[3].t == 4
    assert q_hc.qs[0].x == -1
    assert q_hc.qs[1].x == -1
    assert q_hc.qs[2].x == -1
    assert q_hc.qs[3].x == -1


def test__1340_is_Hermitian():
    assert not is_Hermitian(sigma_y)
    assert not is_Hermitian(q_1234)


def test__1350_is_square():
    assert not is_square(Op)
    assert is_square(Op_scalar)


def test__1360_zero_out():
    qz = zero_out(q1234, x=True)
    q1234.print_state("qz, x=0: ", qz)
    assert q1234.x == 2
    assert qz.x == 0


def test__1360_zero_outs():
    qz = zero_outs(qs_1234, x=True)
    qz.print_state("qz, x=0: ", qz)
    assert qs_1234.qs[1].x == 2
    assert qz.qs[0].x == 0


def test__1600_generate_Qs():
    q_10 = generate_Qs(scalar_q, Q1123)
    assert q_10.dim == 10
    assert equals(q_10, q1s(dim=10))


def test__1610_generate_QQs():
    q_10s = generate_QQs(add, Q1123, q0())
    assert q_10s.dim == 10
    assert equal(q_10s.qs[9], Q1123)

