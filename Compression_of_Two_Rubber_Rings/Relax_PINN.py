import tensorflow as tf
from Dif_op import Dif
import numpy as np

def Relax_PINN(fu1, fv1):
    """
    ====================================================================================================================

    This function is to initialize a PINN.

    ====================================================================================================================
    """

    ### Material properties
    mu = 0.3
    E = 1e2

    ### Declare inputs of PINN
    x1 = tf.keras.layers.Input(shape=(2,))

    ### Position of ring boundaries
    U1 = fu1(x1)
    V1 = fv1(x1)

    ### Strain energy
    Difx1 = Dif(fu1)
    Dify1 = Dif(fv1)

    ### Obtain partial derivatives with respect to x and y
    U1_x, U1_y = Difx1(x1)
    V1_x, V1_y = Dify1(x1)

    ### plain stress
    la = mu / (1 + mu) / (1 - 2 * mu) * E
    nu = 1 / (1 + mu) / 2 * E

    F11_1 = (1 + U1_x)
    F22_1 = (1 + V1_y)
    F12_1 = U1_y
    F21_1 = V1_x

    J1 = F11_1 * F22_1 - F12_1 * F21_1
    I1 = (F11_1 ** 2 + F12_1 ** 2 + F21_1 ** 2 + F22_1 ** 2)
    e1 = 0.25 * la * (J1 ** 2 - 1) - (la / 2 + nu) * tf.math.log(J1) + 0.5 * nu * (I1 - 2)

    ### build up the PINN
    pinn = tf.keras.models.Model(
        inputs=[x1], \
        outputs=[e1, U1, V1])

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
            l2 = tf.reduce_sum(tf.square(tmp[1])) + \
                tf.reduce_sum(tf.square(tmp[2]))
            loss = l1 + l2 * 1e6
        grads = g.gradient(loss, self.pinn.trainable_variables)
        return loss, grads, l1, l2

    def evaluate(self, weights):
        # update weights
        self.set_weights(weights)
        # compute loss and gradients for weights

        loss, grads, l1, l2 = self.Loss(self.x_train, self.y_train)
        if self.iter % 100 == 0:
            print('Iter: ', self.iter, '\te1 =', l1.numpy(), '\tl1 = ', l2.numpy())
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
        t=0
        b_w = 0
        for i in range(0,self.maxiter):
            loss, g = self.evaluate(x)
            m = (1 - beta1) * g + beta1 * m
            v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
            mhat = m / (1 - beta1**(i + 1))  # bias correction.
            vhat = v / (1 - beta2**(i + 1))
            x = x - learning_rate * mhat / (np.sqrt(vhat) + eps)
