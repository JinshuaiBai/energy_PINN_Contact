import tensorflow as tf
from Dif_op import Dif
import scipy.io

def vis_func(fu1, fv1, fu2, fv2, cost1, cost2, x, i):

    pinn = Vis_PINN(fu1, fv1, fu2, fv2, cost1, cost2)
    tmp = pinn(x)
    F11_1 = tmp[0]
    F12_1 = tmp[1]
    F21_1 = tmp[2]
    F22_1 = tmp[3]
    F11_2 = tmp[4]
    F12_2 = tmp[5]
    F21_2 = tmp[6]
    F22_2 = tmp[7]

    u1 = fu1(x[0])
    v1 = fv1(x[0])

    u2 = fu2(x[3])
    v2 = fv2(x[3])

    scipy.io.savemat('out_cont_%d.mat' % i, {'x1': x[0], 'x2': x[3], 'u1': u1.numpy(), 'v1': v1.numpy(), 'u2': u2.numpy(),
                                 'v2': v2.numpy(), 'F11_1': F11_1.numpy(), 'F12_1': F12_1.numpy(),
                                 'F21_1': F21_1.numpy(), 'F22_1': F22_1.numpy()
        , 'F11_2': F11_2.numpy(), 'F12_2': F12_2.numpy(), 'F21_2': F21_2.numpy(), 'F22_2': F22_2.numpy()})


def vis_func_full(fu1, fv1, fu2, fv2, cost1, cost2, x, i):

    pinn = Vis_PINN(fu1, fv1, fu2, fv2, cost1, cost2)
    tmp = pinn(x)
    F11_1 = tmp[0]
    F12_1 = tmp[1]
    F21_1 = tmp[2]
    F22_1 = tmp[3]
    F11_2 = tmp[4]
    F12_2 = tmp[5]
    F21_2 = tmp[6]
    F22_2 = tmp[7]

    u1 = fu1(x[0])
    v1 = fv1(x[0])

    u2 = fu2(x[3])
    v2 = fv2(x[3])

    scipy.io.savemat('out_vis_%d.mat' % i, {'x1': x[0], 'x2': x[3], 'u1': u1.numpy(), 'v1': v1.numpy(), 'u2': u2.numpy(),
                                 'v2': v2.numpy(), 'F11_1': F11_1.numpy(), 'F12_1': F12_1.numpy(),
                                 'F21_1': F21_1.numpy(), 'F22_1': F22_1.numpy()
        , 'F11_2': F11_2.numpy(), 'F12_2': F12_2.numpy(), 'F21_2': F21_2.numpy(), 'F22_2': F22_2.numpy()})
def Vis_PINN(fu1, fv1, fu2, fv2, cost1, cost2):

    mu1 = 0.3
    E1 = 3e2

    mu2 = 0.3
    E2 = 1e2

    ### declare PIRBN's inputs
    x1 = tf.keras.layers.Input(shape=(2,))
    x1_cont = tf.keras.layers.Input(shape=(2,))
    x1_t = tf.keras.layers.Input(shape=(2,))

    x2 = tf.keras.layers.Input(shape=(2,))
    x2_cont = tf.keras.layers.Input(shape=(2,))

    ### Initialize the differential operators

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
    la1 = mu1 / (1 + mu1) / (1 - 2 * mu1) * E1
    nu1 = 1 / (1 + mu1) / 2 * E1

    la2 = mu2 / (1 + mu2) / (1 - 2 * mu2) * E2
    nu2 = 1 / (1 + mu2) / 2 * E2

    F11_1 = (1 + U1_x)
    F22_1 = (1 + V1_y)
    F12_1 = U1_y
    F21_1 = V1_x

    F11_2 = (1 + U2_x)
    F22_2 = (1 + V2_y)
    F12_2 = U2_y
    F21_2 = V2_x


    ### build up the PINN
    pirbn = tf.keras.models.Model(
        inputs=[x1, x1_cont, x1_t, x2, x2_cont], \
        outputs=[F11_1, F12_1, F21_1, F22_1, F11_2, F12_2, F21_2, F22_2])

    return pirbn