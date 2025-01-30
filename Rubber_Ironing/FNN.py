import tensorflow as tf

class FNN:

    def __init__(self, n_in, n_out, layers, acti_fun='tanh', k_init='LecunNormal', time_layer = False, t_in = 1,
                t_variable = 0, trainable_plus_layer = False, p_variable = 0):
        # super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.layers = layers
        self.acti_fun = acti_fun
        self.k_init = k_init
        self.time_layer = time_layer
        self.t_in = t_in
        self.t_variable = t_variable
        self.trainable_pl = trainable_plus_layer
        self.p_variable = p_variable

    def net(self):
        x = tf.keras.layers.Input(shape=(self.n_in,))
        temp = x
        for l in self.layers:
            temp = tf.keras.layers.Dense(l, activation=self.acti_fun, kernel_initializer=self.k_init)(temp)
        y = tf.keras.layers.Dense(self.n_out, kernel_initializer=self.k_init, use_bias=False)(temp)

        if self.time_layer:
            t_l = time_layer(self.t_variable)
            y = t_l(y, x[..., self.t_in-1, tf.newaxis],)

        if self.trainable_pl:
            p_l = plus_trainable_layer()
            y = p_l(y)
        else:
            p_l = plus_untrainable_layer(self.p_variable)
            y = p_l(y)

        nn = tf.keras.models.Model(inputs=x, outputs=y)

        return nn

    def init_pl(self, nn):
        weights = nn.get_weights()
        weights[-1][0] = self.p_variable
        nn.set_weights(weights)
        return nn

    def build(self):
        nn = self.net()
        if self.trainable_pl:
            nn = self.init_pl(nn)

        return nn

class plus_trainable_layer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.unit = 1

    def build(self, input_shape):
        self.bias=self.add_weight(
            shape = (self.unit,),
            initializer="zero",
            trainable = True,
            name="bias",
        )

    def call(self, inputs):
        return inputs + self.bias

class plus_untrainable_layer(tf.keras.layers.Layer):
    def __init__(self, variable):
        super().__init__()
        self.variable = variable

    def call(self, inputs):
        return inputs + self.variable

class time_layer(tf.keras.layers.Layer):
    def __init__(self, variable):
        super().__init__()
        self.unit = 1
        self.variable = variable

    def call(self, inputs, x):
        return inputs * (x - self.variable)