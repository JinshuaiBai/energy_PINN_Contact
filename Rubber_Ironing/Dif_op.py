import tensorflow as tf

class Dif(tf.keras.layers.Layer):

    def __init__(self, fnn, **kwargs):
        self.fnn = fnn
        super().__init__(**kwargs)

    def call(self, xy):
        x, y = (xy[..., i, tf.newaxis] for i in range(xy.shape[-1]))
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            g.watch(y)
            U = self.fnn(tf.concat([x, y], axis=-1))
        U_x = g.gradient(U, x)
        U_y = g.gradient(U, y)
        del g

        return U_x, U_y