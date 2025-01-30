import tensorflow as tf
from Dif_op import Dif

def PINN(fu1, fv1, fu2, fv2, cost1, cost2):
    """
    ====================================================================================================================

    This function is to initialize a PINN.

    ====================================================================================================================
    """

    E1 = 300
    E2 = 100
    mu = 0.3

    ### declare PINN's inputs
    x1 = tf.keras.layers.Input(shape=(2,))
    x1_cont = tf.keras.layers.Input(shape=(2,))
    x1_t = tf.keras.layers.Input(shape=(2,))

    x2 = tf.keras.layers.Input(shape=(2,))
    x2_cont = tf.keras.layers.Input(shape=(2,))

    ### Initialize the differential operators
    U1_cont = fu1(x1_cont) + x1_cont[...,0,tf.newaxis]
    V1_cont = fv1(x1_cont) + x1_cont[..., 1, tf.newaxis] + 2.0
    V1_t = fv1(x1_t)
    U1_t = fu1(x1_t)

    U2_cont = fu2(x2_cont) + x2_cont[..., 0, tf.newaxis]
    V2_cont = fv2(x2_cont) + x2_cont[..., 1, tf.newaxis] + 1.5

    dU = U1_cont * cost1 - tf.transpose(U2_cont * cost2)
    dV = V1_cont * cost1 - tf.transpose(V2_cont * cost2)
    r = tf.sqrt(tf.square(dU) + tf.square(dV))

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
    la1 = mu / (1 + mu) / (1 - 2 * mu) * E1
    nu1 = 1 / (1 + mu) / 2 * E1

    la2 = mu / (1 + mu) / (1 - 2 * mu) * E2
    nu2 = 1 / (1 + mu) / 2 * E2

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
    Ein1 = 0.25 * la1 * (J1 ** 2 - 1) - (la1 / 2 + nu1) * tf.math.log(J1) + 0.5 * nu1 * (I1 - 2)

    J2 = F11_2 * F22_2 - F12_2 * F21_2
    I2 = (F11_2 ** 2 + F12_2 ** 2 + F21_2 ** 2 + F22_2 ** 2)
    Ein2 = 0.25 * la2 * (J2 ** 2 - 1) - (la2 / 2 + nu2) * tf.math.log(J2) + 0.5 * nu2 * (I2 - 2)

    ### repulsive force
    a = 1e2
    b = 1e2
    # exp type energy
    Erepulsive = b * tf.exp(-a * r)

    ### build up the PINN
    pinn = tf.keras.models.Model(
        inputs = [x1, x1_cont, x1_t, x2, x2_cont], \
            outputs = [Ein1, Ein2, Erepulsive, V1_t, U1_t])
        
    return pinn