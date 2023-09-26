import tensorflow as tf
import os
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

tf.config.list_physical_devices('CPU')

def loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype = tf.float32)
    y_pred = tf.cast(y_pred, dtype = tf.float32)
    return tf.reduce_mean((y_true - y_pred)**2)

class DenseLayer(tf.Module):
    def __init__(self, input_dim, output_dim, activation):
        super().__init__()
        self.w = tf.Variable(tf.random.normal([input_dim, output_dim]), dtype = tf.float32)
        self.b = tf.Variable(tf.zeros([output_dim]), dtype = tf.float32)
        self.activation = activation
    
    def __call__(self, x):
        x = tf.cast(x, dtype = tf.float32)
        return self.activation(tf.matmul(x, self.w) + self.b) if self.activation else tf.matmul(x, self.w) + self.b

class UntrainedModelError(Exception):
    def __init__(self):
        super().__init__()
    def __call__(self):
        print("Model is untrained")


class NNmodel(tf.Module):
    def __init__(self):
        super().__init__()
        self.hiddenlayers = []
        self.hiddenlayers.append(DenseLayer(1, 64, tf.nn.tanh))
        self.hiddenlayers.append(DenseLayer(64, 64, tf.nn.tanh))
        self.hiddenlayers.append(DenseLayer(64, 64, tf.nn.tanh))
        self.hiddenlayers.append(DenseLayer(64, 1, tf.nn.tanh))


        self.trained = False
        self.trainable_variables

    def forward(self, x):
        for hl in self.hiddenlayers:
            x = hl(x)
        return x
    
    
    def backprop(self, x, y, x_phys, y_phys, m, b, k):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            y_pred_data = self.forward(x)
            with tf.GradientTape(persistent=True) as tape_deriv:
                tape_deriv.watch(x_phys)    
                y_pred_phys = self.forward(x_phys)
                y_x = tape_deriv.gradient(y_pred_phys, x_phys)
            y_xx = tape_deriv.gradient(y_x, x_phys)
            f = m*y_xx + b*y_x + k*y_pred_phys
            loss_data = loss(y, y_pred_data)
            loss_phys = loss(f, tf.zeros_like(f))
            mseloss = loss_data + 1e-4*loss_phys
        grad = tape.gradient(mseloss, self.trainable_variables)
        return mseloss, grad
    
    def train(self, x, y, x_phys, y_phys, m, b, k, epoch = 1000, lr = 1e-4):
        optim = tf.optimizers.Adam(learning_rate = lr)
        for i in range(epoch):
            loss, grad = self.backprop(x, y, x_phys, y_phys, m, b, k)
            if i % 1000 == 0:
                print(loss)

            optim.apply_gradients(zip(grad, self.trainable_variables))
        self.trained = True

    def predict(self, x):
        try:
            if self.trained:
                return tf.squeeze(self.forward(x), axis = -1)
            else:
                raise UntrainedModelError
        except UntrainedModelError as e:
            print("UntrainedModelError: Model is untrained")


nn = NNmodel()
m, b, k = tf.constant(1, dtype = tf.float32) , tf.constant(4, dtype = tf.float32), tf.constant(400, dtype = tf.float32)
omega = tf.cast(tf.math.sqrt((4*m*k - b**2)/(4*m**2)), dtype = tf.float32)

t = tf.cast(tf.expand_dims(tf.linspace(0, 1, 500), axis = -1), dtype = tf.float32)
x = tf.exp(-b*t/2*m) * tf.cos(omega*t)

t_data = t[0:200:20]
x_data = x[0:200:20]

t_phys = tf.cast(tf.expand_dims(tf.linspace(0, 1, 30), axis = -1), dtype = tf.float32)
x_phys= tf.exp(-b*t_phys/2*m) * tf.cos(omega*t_phys)

nn.train(t_data, x_data, t_phys, x_phys, m, b, k, epoch = 50000, lr = 1e-4)
x_pred = nn.predict(t)

plt.plot(t, x, alpha = 0.4)
plt.scatter(t_data, x_data, color = 'green')
plt.scatter(t_phys, tf.zeros_like(t_phys))
plt.plot(t, x_pred, alpha = 0.5, color = 'red')
plt.savefig('harmonic_oscillator.png')
plt.show()