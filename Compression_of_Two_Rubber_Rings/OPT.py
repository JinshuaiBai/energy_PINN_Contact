import numpy as np
import tensorflow as tf

class Adam:

    def __init__(self, pirbn, x_train, y_train, learning_rate = 0.001, maxiter=10000):
        # set attributes
        self.pirbn = pirbn
        self.learning_rate = learning_rate
        self.x_train = [ tf.constant(x, dtype=tf.float32) for x in x_train ]
        self.y_train = [ tf.constant(y, dtype=tf.float32) for y in y_train ]
        self.maxiter = maxiter
        self.his_l1 = []
        self.his_l2 = []
        self.his_l3 = []
        self.his_l4 = []
        self.his_l5 = []
        self.his_l6 = []
        self.his_l7 = []
        self.his_l8 = []
        self.his_l9 = []

    def set_weights(self, flat_weights):
        # get model weights
        shapes = [ w.shape for w in self.pirbn.get_weights() ]
        # compute splitting indices
        split_ids = np.cumsum([ np.prod(shape) for shape in [0] + shapes ])
        # reshape weights
        weights = [ flat_weights[from_id:to_id].reshape(shape)
            for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes) ]
        # set weights to the model
        self.pirbn.set_weights(weights)

    @tf.function
    def Loss(self, x, y):
        with tf.GradientTape() as g:
            tmp = self.pirbn(x)
            l1 = tf.reduce_sum(tmp[0] * y[0])
            l2 = tf.reduce_sum(tmp[1] * y[1])
            l3 = tf.reduce_sum(tmp[2]) * y[2]
            l4 = tf.reduce_sum(tmp[3]) * y[3]
            l5 = tf.reduce_sum(tmp[4]) * y[2]
            l6 = tf.reduce_sum(tmp[5]) * y[3]
            l7 = tf.reduce_sum(tmp[6]) * y[2]
            l8 = tf.reduce_sum(tmp[7]) * y[3]
            l9 = tf.reduce_sum(tmp[8]) * 1e-3
            loss = l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9
        grads = g.gradient(loss, self.pirbn.trainable_variables)
        return loss, grads, l1, l2, l3, l4, l5, l6, l7, l8, l9

    def evaluate(self, weights):
        # update weights
        self.set_weights(weights)
        # compute loss and gradients for weights

        loss, grads, l1, l2, l3, l4, l5, l6, l7, l8, l9 = self.Loss(self.x_train, self.y_train)
        self.his_l1.append(l1.numpy())
        self.his_l2.append(l2.numpy())
        self.his_l3.append(l3.numpy())
        self.his_l4.append(l4.numpy())
        self.his_l5.append(l5.numpy())
        self.his_l6.append(l6.numpy())
        self.his_l7.append(l7.numpy())
        self.his_l8.append(l8.numpy())
        self.his_l9.append(l9.numpy())
        self.iter = self.iter + 1
        if self.iter % 500 == 0:
            print('Iter: ', self.iter, '\te1 =', l1.numpy(), '\te2 =', l2.numpy(), '\tea = ', l3.numpy(), '\teb = '
                  , l4.numpy(), '\tec = ', l5.numpy(), '\ted = ', l6.numpy(), '\tee = ', l7.numpy(),
                  '\tef = ', l8.numpy(), '\ter = ', l9.numpy())
        # convert tf.Tensor to flatten ndarray
        loss = loss.numpy().astype('float64')
        grads = np.concatenate([ g.numpy().flatten() for g in grads ]).astype('float64')

        return loss, grads

    def fit(self):
        print('Optimizer: Adam')
        print('Initializing ...')
        self.iter = 0
        # get initial weights as a flat vector
        initial_weights = np.concatenate(
            [ w.flatten() for w in self.pirbn.get_weights() ])
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

        return [self.his_l1, self.his_l2, self.his_l3, self.his_l4, self.his_l5, self.his_l6, self.his_l7, self.his_l8,
                self.his_l9]