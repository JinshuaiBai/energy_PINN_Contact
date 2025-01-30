import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PINN import PINN
from OPT import Adam
from Vis_func import vis_func, vis_func_full
from Relax_PINN import Relax_PINN, Relax_Adam
import scipy.io
from FNN import FNN
import time
import random

random.seed(2024)

def load_nn_data(fnn_u1, fnn_v1, fnn_u2, fnn_v2, i):
    C = scipy.io.loadmat('NN_weights_%d.mat' % i)
    weights_u1 = C['w_u1']
    weights_v1 = C['w_v1']
    weights_u2 = C['w_u2']
    weights_v2 = C['w_v2']

    ### u
    shapes_u1 = [weights_u1.shape for weights_u1 in fnn_u1.get_weights()]
    weights_u1 = [tmp.reshape(shape)
                  for tmp, shape in zip(weights_u1[0], shapes_u1)]
    shapes_u2 = [weights_u2.shape for weights_u2 in fnn_u2.get_weights()]
    weights_u2 = [tmp.reshape(shape)
                  for tmp, shape in zip(weights_u2[0], shapes_u2)]

    ### v
    shapes_v1 = [weights_v1.shape for weights_v1 in fnn_v1.get_weights()]
    weights_v1 = [tmp.reshape(shape)
                  for tmp, shape in zip(weights_v1[0], shapes_v1)]
    shapes_v2 = [weights_v2.shape for weights_v2 in fnn_v2.get_weights()]
    weights_v2 = [tmp.reshape(shape)
                  for tmp, shape in zip(weights_v2[0], shapes_v2)]

    fnn_u1.set_weights(weights_u1)
    fnn_v1.set_weights(weights_v1)
    fnn_u2.set_weights(weights_u2)
    fnn_v2.set_weights(weights_v2)
    return fnn_u1, fnn_v1, fnn_u2, fnn_v2


### Define the number of sample points
C = scipy.io.loadmat('Coord.mat')

x1=C['x1']
x1_cont=C['x1_cont']
x1_t=C['x_t']
w1 = C['w1']
cost1 = C['cost1']
x1_out = C['x1_out']
dr1=C['dr1']

x2=C['x2']
x2_cont=C['x2_cont']
w2 = C['w2']
cost2 = C['cost2']
x2_out = C['x2_out']
dr2=C['dr2']

x_train = [ x1, x1_cont, x1_t, x2, x2_cont ]
y_train=[w1, w2, dr1]
x_relax=[x1, x1_t, x2]
y_relax=[w1, w2]

n_in = 2
n_out = 1
layer = [30,30,30]

### time_layer           : incorporate hard boundary condition
### time_t               : specify x (1) or y (2) (if time_layer = False, this is meaningless)
### time_variable        : incorporate the position of the hard boundary condition (if time_layer = False, this is meaningless)
### trainable_plus_layer : incorporate exact disp on the hard boundary condition
### p_variable           : the exact disp on the hard boundary condition

### example       : If the displacement on the boundary x = 2.5 is u = -1, then:
###                 time_layer = True
###                 time_t = 1
###                 time_variable = 2.5
###                 trainable_plus_layer = False
###                 p_variable = -1

fu1 = FNN(n_in, n_out, layer,time_layer = True, t_in = 2, t_variable = 1, trainable_plus_layer = False,
          p_variable = 0).build()
fv1 = FNN(n_in, n_out, layer,time_layer = True, t_in = 2, t_variable = 1, trainable_plus_layer = True,
          p_variable = 0.1).build()
fu2 = FNN(n_in, n_out, layer,time_layer = True, t_in = 2, t_variable = -1.5).build()
fv2 = FNN(n_in, n_out, layer,time_layer = True, t_in = 2, t_variable = -1.5).build()

pinn_relax = Relax_PINN(fu1, fv1, fu2, fv2)
opt_relax = Relax_Adam(pinn_relax, x_relax, y_relax, learning_rate = 1e-4, maxiter=2001)
opt_relax.fit()

# try:
#     fu1, fv1, fu2, fv2 = load_nn_data(fu1, fv1, fu2, fv2, 5)
#     print('Loaded!\n')
# except:
#     print('Error when loading!\n')

t_or_r = 0
for i in range(0,6):
    # fv1 = FNN(n_in, n_out, layer, time_layer=True, t_in=2, t_variable=1, trainable_plus_layer=False,
    #           p_variable=-i*0.1).build()
    # fu1 = FNN(n_in, n_out, layer, time_layer=True, t_in=2, t_variable=1, trainable_plus_layer=False,
    #           p_variable=(i-5)*0.1).build()

    for j in range(10):

        if j==1:
            ### t_or_r = 0 means loading
            ### t_or_r = 1 means relax
            t_or_r = 1

        pinn = PINN(fu1, fv1, fu2, fv2, cost1, cost2)
        t1 = time.time()
        opt = Adam(pinn, x_train, y_train, i, j, learning_rate = 1e-5, maxiter=2000, t_or_r = t_or_r)
        result=opt.fit()
        t2 = time.time()
        print(t2-t1)

        loss_hist = result[1]

        tmp=pinn([x1_out, x1_cont, x1_cont, x2_out, x2_cont])
        u1 = fu1(x1_out)
        v1 = fv1(x1_out)

        u2 = fu2(x2_out)
        v2 = fv2(x2_out)

        scipy.io.savemat('out_%d.mat' % i, {'x1': x1_out, 'x2': x2_out, 'u1': u1.numpy(), 'v1': v1.numpy(),
                                            'u2': u2.numpy(), 'v2': v2.numpy()})

        vis_func(fu1, fv1, fu2, fv2, cost1, cost2, [x1_cont, x1_cont, x1_cont, x2_cont, x2_cont], i)
        vis_func_full(fu1, fv1, fu2, fv2, cost1, cost2, [x1_out, x1_cont, x1_cont, x2_out, x2_cont], i)

        weights_u1 = fu1.get_weights()
        weights_v1 = fv1.get_weights()
        weights_u2 = fu2.get_weights()
        weights_v2 = fv2.get_weights()

        scipy.io.savemat('NN_weights_%d.mat' % i,{'w_u1': weights_u1, 'w_v1': weights_v1,
                                                  'w_u2': weights_u2, 'w_v2': weights_v2})
