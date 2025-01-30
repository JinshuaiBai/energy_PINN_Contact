import tensorflow as tf
from Dif_op import Dif
import scipy.io

def vis_out_data(fu1, fv1, fu2, fv2, x1, x2, i, j, result, training_time):
    pinn_1 = out_PINN(fu1, fv1)
    pinn_2 = out_PINN(fu2, fv2)

    u1=fu1(x1)
    v1=fv1(x1)

    u2 = fu2(x2)
    v2 = fv2(x2)

    t1 = pinn_1([x1])
    F11_1=t1[1]
    F12_1=t1[2]
    F21_1=t1[3]
    F22_1=t1[4]

    t2 = pinn_2([x2])
    F11_2 = t2[1]
    F12_2 = t2[2]
    F21_2 = t2[3]
    F22_2 = t2[4]

    weights_u1 = fu1.get_weights()
    weights_v1 = fv1.get_weights()
    weights_u2 = fu2.get_weights()
    weights_v2 = fv2.get_weights()

    scipy.io.savemat('out_loadingstep%d_loop%d.mat' % (j,i),
                     {'x1': x1, 'x2': x2, 'u1': u1.numpy(), 'v1': v1.numpy(), 'u2': u2.numpy(),
                      'v2': v2.numpy(), 'F11_1': F11_1.numpy(), 'F12_1': F12_1.numpy(),
                      'F21_1': F21_1.numpy(), 'F22_1': F22_1.numpy(), 'F11_2': F11_2.numpy(),
                      'F12_2': F12_2.numpy(), 'F21_2': F21_2.numpy(), 'F22_2': F22_2.numpy(),
                      'L': result,'t': training_time, 'w1_u': weights_u1, 'w1_v': weights_v1,
                      'w2_u': weights_u2, 'w2_v': weights_v2})

def out_PINN(fu, fv):

    ### Material properties
    mu = 0.3
    E = 1e2


    ### Declare inputs of PINN
    x = tf.keras.layers.Input(shape=(2,))

    ### Strain energy
    Difx = Dif(fu)
    Dify = Dif(fv)

    ### Obtain partial derivatives with respect to x and y
    U_x, U_y = Difx(x)
    V_x, V_y = Dify(x)

    ### plain stress
    la = mu / (1 + mu) / (1 - 2 * mu) * E
    nu = 1 / (1 + mu) / 2 * E

    F11 = (1 + U_x)
    F22 = (1 + V_y)
    F12 = U_y
    F21 = V_x


    J = F11 * F22 - F12 * F21
    I = (F11 ** 2 + F12 ** 2 + F21 ** 2 + F22 ** 2)
    e = 0.25 * la * (J ** 2 - 1) - (la / 2 + nu) * tf.math.log(J) + 0.5 * nu * (I - 2)

    ### build up the PINN
    pinn = tf.keras.models.Model(inputs=[x], outputs=[e, F11, F12, F21, F22, J, I])

    return pinn