import tensorflow as tf
from Dif_op import Dif

def PINN(fu1, fv1, fu2, fv2, cost1, cost2, j):

    ### Material properties
    mu = 0.3
    E = 1e2

    ### Declare inputs of PINN
    x1 = tf.keras.layers.Input(shape=(2,))
    x1_cont = tf.keras.layers.Input(shape=(2,))

    x2 = tf.keras.layers.Input(shape=(2,))
    x2_cont = tf.keras.layers.Input(shape=(2,))

    ### Position of ring boundaries
    U1_cont = fu1(x1_cont) + x1_cont[...,0,tf.newaxis]+0.35
    V1_cont = fv1(x1_cont) + x1_cont[..., 1, tf.newaxis]+0.35

    U2_cont = fu2(x2_cont) + x2_cont[..., 0, tf.newaxis]+0.9
    V2_cont = fv2(x2_cont) + x2_cont[..., 1, tf.newaxis]+0.8

    ### Distances
    dV_a = (1.15 - 0.05 * j) - V1_cont
    dV_b = (1.15 - 0.05 * j) - V2_cont
    dV_c = V1_cont
    dV_d = V2_cont
    dU_e = U1_cont
    dU_f = 1.25 - U2_cont

    dU_r = U1_cont * cost1 - tf.transpose(U2_cont * cost2)
    dV_r = V1_cont * cost1 - tf.transpose(V2_cont * cost2)
    r = tf.sqrt(tf.square(dU_r) + tf.square(dV_r))

    ### Contact energy
    a = 1e3
    b = 1e-2
    # exp type energy
    e_a = b * tf.exp(-a * dV_a)
    e_b = b * tf.exp(-a * dV_b)
    e_c = b * tf.exp(-a * dV_c)
    e_d = b * tf.exp(-a * dV_d)
    e_e = b * tf.exp(-a * dU_e)
    e_f = b * tf.exp(-a * dU_f)
    e_r = b * tf.exp(-a * r)

    ### Strain energy
    Difx1 = Dif(fu1)
    Dify1 = Dif(fv1)

    Difx2 = Dif(fu2)
    Dify2 = Dif(fv2)

    ### Obtain partial derivatives with respect to x and y
    U1_x, U1_y = Difx1(x1)
    V1_x, V1_y = Dify1(x1)

    U2_x, U2_y = Difx2(x2)
    V2_x, V2_y = Dify2(x2)

    ### plain stress
    la = mu / (1 + mu) / (1 - 2 * mu) * E
    nu = 1 / (1 + mu) / 2 * E

    F11_1 = (1 + U1_x)
    F22_1 = (1 + V1_y)
    F12_1 = U1_y
    F21_1 = V1_x

    F11_2 = (1 + U2_x)
    F22_2 = (1 + V2_y)
    F12_2 = U2_y
    F21_2 = V2_x

    J1 = F11_1 * F22_1 - F12_1 * F21_1
    I1 = (F11_1 ** 2 + F12_1 ** 2 + F21_1 ** 2 + F22_1 ** 2)
    e1 = 0.25 * la* (J1 ** 2 - 1) - (la / 2 + nu) * tf.math.log(J1) + 0.5 * nu * (I1 - 2)

    J2 = F11_2 * F22_2 - F12_2 * F21_2
    I2 = (F11_2 ** 2 + F12_2 ** 2 + F21_2 ** 2 + F22_2 ** 2)
    e2 = 0.25 * la * (J2 ** 2 - 1) - (la / 2 + nu) * tf.math.log(J2) + 0.5 * nu * (I2 - 2)

    ### build up the PINN
    pinn = tf.keras.models.Model(
        inputs = [x1, x1_cont, x2, x2_cont], \
            outputs = [e1, e2, e_a, e_b, e_c, e_d, e_e, e_f, e_r])
        
    return pinn