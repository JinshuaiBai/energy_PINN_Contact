import tensorflow as tf
from Dif_op import Dif
import numpy as np

def Relax_PINN(fu1, fv1, fu2, fv2):
    E1 = 300
    E2 = 100
    mu = 0.3

    ### declare PINN's inputs
    x1 = tf.keras.layers.Input(shape=(2,))
    x1_t = tf.keras.layers.Input(shape=(2,))
    x2 = tf.keras.layers.Input(shape=(2,))

    ### Initialize the differential operators
    V1_t = fv1(x1_t)

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

    ### build up the PINN
    pinn = tf.keras.models.Model(
        inputs=[x1, x1_t, x2], \
        outputs=[Ein1, Ein2, V1_t])

    return pinn

class Relax_Adam:

    def __init__(self, pinn, x_train, y_train, learning_rate = 0.001, maxiter=10000):
        # set attributes
        self.pinn = pinn
        self.learning_rate = learning_rate
        self.x_train = [ tf.constant(x, dtype=tf.float32) for x in x_train ]
        self.y_train = [ tf.constant(y, dtype=tf.float32) for y in y_train ]
        self.maxiter = maxiter
        self.iter = 0

    def set_weights(self, flat_weights):
        # get model weights
        shapes = [ w.shape for w in self.pinn.get_weights() ]
        # compute splitting indices
        split_ids = np.cumsum([ np.prod(shape) for shape in [0] + shapes ])
        # reshape weights
        weights = [ flat_weights[from_id:to_id].reshape(shape)
            for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes) ]
        # set weights to the model
        self.pinn.set_weights(weights)

    @tf.function
    def Loss(self, x, y):
        with tf.GradientTape() as g:
            tmp = self.pinn(x)
            l1 = tf.reduce_sum(tmp[0] * y[0])
            l2 = tf.reduce_sum(tmp[1] * y[1])
            l3 = tf.reduce_mean(tf.square(tmp[2]-0.1))
            loss = l1 + l2 + l3 * 1e4
        grads = g.gradient(loss, self.pinn.trainable_variables)
        return loss, grads, l1, l2, l3

    def evaluate(self, weights):
        # update weights
        self.set_weights(weights)
        # compute loss and gradients for weights

        loss, grads, l1, l2, l3 = self.Loss(self.x_train, self.y_train)
        if self.iter % 100 == 0:
            print('Iter: ', self.iter, '\te1 =', l1.numpy(), '\te2 = ', l2.numpy(), '\tl1 = ', l3.numpy())
        self.iter = self.iter+1
        # convert tf.Tensor to flatten ndarray
        loss = loss.numpy().astype('float64')
        grads = np.concatenate([ g.numpy().flatten() for g in grads ]).astype('float64')

        return loss, grads

    def fit(self):
        print('Optimizer: Adam')
        print('Initializing ...')
        # get initial weights as a flat vector
        initial_weights = np.concatenate(
            [ w.flatten() for w in self.pinn.get_weights() ])
        print('Optimizer: Adam (maxiter={})'.format(self.maxiter))
        beta1 = 0.9
        beta2 = 0.999
        learning_rate = self.learning_rate
        eps=1e-8
        x0=initial_weights
        x=x0
        m=np.zeros_like(x)
        v=np.zeros_like(x)
        for i in range(0,self.maxiter):
            loss, g = self.evaluate(x)
            m = (1 - beta1) * g + beta1 * m
            v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
            mhat = m / (1 - beta1**(i + 1))  # bias correction.
            vhat = v / (1 - beta2**(i + 1))
            x = x - learning_rate * mhat / (np.sqrt(vhat) + eps)
