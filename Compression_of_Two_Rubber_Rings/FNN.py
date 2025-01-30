import tensorflow as tf

def FNN(n_input, n_output, layers, f = 1., acti_fun='tanh', k_init='LecunNormal'):

    ### Setup the input layer of the FNN
    x = tf.keras.layers.Input(shape=(n_input))
    
    ### Setup the hidden layers of the FNN
    temp = x
    for l in layers:
        temp = tf.keras.layers.Dense(l, activation = acti_fun, kernel_initializer=k_init)(temp)
    
    ### Setup the output layers of the FNN
    y = tf.keras.layers.Dense(n_output, kernel_initializer=k_init, use_bias = False)(temp) * f

    ### Combine the input, hidden, and output layers to build up a FNN
    net = tf.keras.models.Model(inputs=x, outputs=y)

    return net
