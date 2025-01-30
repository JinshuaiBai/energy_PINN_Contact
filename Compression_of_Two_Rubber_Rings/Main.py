import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PINN import PINN
from Relax_PINN import Relax_PINN, Relax_Adam
from OPT import Adam
import scipy.io
from FNN import FNN
import time
from vis_out import vis_out_data

def load_nn_data(fnn_u_1, fnn_v_1, fnn_u_2, fnn_v_2, i):
    C = scipy.io.loadmat('NN_weights_%d.mat' % i)
    weights_1_u = C['w1_u']
    weights_1_v = C['w1_v']
    weights_2_u = C['w2_u']
    weights_2_v = C['w2_v']

    ### u
    shapes_1_u = [weights_1_u.shape for weights_1_u in fnn_u_1.get_weights()]
    weights_1_u = [tmp.reshape(shape)
                   for tmp, shape in zip(weights_1_u[0], shapes_1_u)]

    shapes_2_u = [weights_2_u.shape for weights_2_u in fnn_u_2.get_weights()]
    weights_2_u = [tmp.reshape(shape)
                   for tmp, shape in zip(weights_2_u[0], shapes_2_u)]

    ### v
    shapes_1_v = [weights_1_v.shape for weights_1_v in fnn_v_1.get_weights()]
    weights_1_v = [tmp.reshape(shape)
                   for tmp, shape in zip(weights_1_v[0], shapes_1_v)]
    shapes_2_v = [weights_2_v.shape for weights_2_v in fnn_v_2.get_weights()]
    weights_2_v = [tmp.reshape(shape)
                   for tmp, shape in zip(weights_2_v[0], shapes_2_v)]

    fnn_u_1.set_weights(weights_1_u)
    fnn_v_1.set_weights(weights_1_v)
    fnn_u_2.set_weights(weights_2_u)
    fnn_v_2.set_weights(weights_2_v)
    return fnn_u_1, fnn_v_1, fnn_u_2, fnn_v_2


### Define the number of sample points
C = scipy.io.loadmat('Coord.mat')

x1=C['x1']
x1_cont=C['x1_cont']
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

x_relax=[x1]
y_relax=[w1]

x_train = [ x1, x1_cont, x2, x2_cont ]
y_train=[w1, w2, dr1[0,0], dr2[0,0]]

n_in = 2
n_out = 1

fu1 = FNN(n_in, n_out, [50, 50, 50], f = 1., acti_fun='tanh', k_init='LecunNormal')
fv1 = FNN(n_in, n_out, [50, 50, 50], f = 1., acti_fun='tanh', k_init='LecunNormal')

fu2 = FNN(n_in, n_out, [50, 50, 50], f = 1., acti_fun='tanh', k_init='LecunNormal')
fv2 = FNN(n_in, n_out, [50, 50, 50], f = 1., acti_fun='tanh', k_init='LecunNormal')

pinn_relax = Relax_PINN(fu1, fv1)
opt_relax = Relax_Adam(pinn_relax, x_relax, y_relax, learning_rate = 1e-4, maxiter=10001)
opt_relax.fit()

w_u = fu1.get_weights()
w_v = fv1.get_weights()
fu2.set_weights(w_u)
fv2.set_weights(w_v)

# try:
#     fu1, fv1, fu2, fv2 = load_nn_data(fu1, fv1, fu2, fv2, 9 )
# except:
#     print('Error when loading!\n')

for j in range(0,12):

    t1 = time.time()
    pinn = PINN(fu1, fv1, fu2, fv2, cost1, cost2, j)
    opt = Adam(pinn, x_train, y_train, learning_rate = 1e-5, maxiter=2000)

    for i in range(20):
        result=opt.fit()

        t2 = time.time()
        training_time = t2-t1
        print(training_time)
        vis_out_data(fu1, fv1, fu2, fv2, x1_out, x2_out, i, j, result, training_time)

    weights_u1 = fu1.get_weights()
    weights_v1 = fv1.get_weights()
    weights_u2 = fu2.get_weights()
    weights_v2 = fv2.get_weights()
    scipy.io.savemat('NN_weights_%d.mat' % j,{'w1_u': weights_u1, 'w1_v': weights_v1,
                                                'w2_u': weights_u2, 'w2_v': weights_v2})