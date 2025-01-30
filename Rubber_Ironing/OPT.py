# import scipy.optimize
import numpy as np
import tensorflow as tf
# import scipy.io
import math
import matplotlib.pyplot as plt
# from Cal_jac import cal_adapt

class Adam:

    def __init__(self, pinn, x_train, y_train, i, j, learning_rate = 0.001, maxiter=10000, t_or_r=0):
        # set attributes
        self.pinn = pinn
        self.learning_rate = learning_rate
        self.x_train = [ tf.constant(x, dtype=tf.float32) for x in x_train ]
        self.y_train = [ tf.constant(y, dtype=tf.float32) for y in y_train ]
        self.maxiter = maxiter
        self.i = i
        self.his_l1 = []
        self.his_l2 = []
        self.his_l3 = []
        self.his_l4 = []
        self.his_l5 = []
        self.iter = 0
        self.j = j
        ### t_or_r = 0 means loading
        ### t_or_r = 1 means relax
        self.t_or_r=t_or_r

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
    def Loss(self, x, y, iter):
        with tf.GradientTape() as g:
            tmp = self.pinn(x)
            l1 = tf.reduce_sum(tmp[0]*y[0])
            l2 = tf.reduce_sum(tmp[1]*y[1])
            l3 = tf.reduce_sum(tmp[2]*y[2]*y[2])
            l4 = tf.reduce_mean(tf.square(tmp[3]+iter))
            l5 = tf.reduce_mean(tf.square(tmp[4]))
            loss = l1+l2+l3+l4*1e4+l5*1e4 #**self.j
        grads = g.gradient(loss, self.pinn.trainable_variables)
        return loss, grads, l1, l2, l3, l4, l5

    def evaluate(self, weights):
        # update weights
        self.set_weights(weights)
        # compute loss and gradients for weights
        if self.t_or_r==0:
            ttt = tf.constant((self.i - 1) * 0.1 + self.iter * 0.1 * 0.5e-3)
        else:
            ttt = tf.constant((self.i - 0) * 0.1)
        loss, grads, l1, l2, l3, l4, l5 = self.Loss(self.x_train, self.y_train, ttt)
        self.his_l1.append(l1.numpy())
        self.his_l2.append(l2.numpy())
        self.his_l3.append(l3.numpy())
        self.his_l4.append(l4.numpy())
        self.his_l5.append(l5.numpy())
        if self.iter % 100 == 0:
            print('Iter: ',self.iter,'\tL1 =',l1.numpy(),'\tL2 =',l2.numpy(),'\tL3 = ',l3.numpy(),'\tL4 = ',l4.numpy()
                  ,'\tL5 = ',l5.numpy())
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

        return loss, [self.his_l1, self.his_l2, self.his_l3, self.his_l4, self.his_l5]

        # return np.array(self.his_l1)+np.array(self.his_l2)+np.array(self.his_l3)+np.array(self.his_l4), \
        #        [self.his_l1, self.his_l2, self.his_l3, self.his_l4]