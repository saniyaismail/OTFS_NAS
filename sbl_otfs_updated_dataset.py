from functions_OTFS import update_mu_Sigma_OTFS, circular_padding_1d
from tensorflow.keras.layers import Input, Conv2D, Lambda
from tensorflow.keras import backend as K
import os
# import matplotlib.pyplot as plt
import argparse
from scipy import io
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import mat73
import random
import h5py
from tensorflow.keras.models import load_model

# Keras Tuner is used for the NAS search space. We try both module names for compatibility.
try:
    import keras_tuner as kt
except ImportError:
    import kerastuner as kt  # type: ignore

# ma'am code

np.random.seed(2024)

tf.random.set_seed(2024)
random.seed(2024)
# %%


def custom_loss(y_true, y_pred):
    pruning_thrld = 0.1

    thresholded_pred = tf.where(tf.abs(y_pred) < pruning_thrld, tf.zeros_like(y_pred), y_pred)
    # thresholded_true = tf.where(tf.abs(y_true) < pruning_thrld, tf.zeros_like(y_true), y_true)

    # mse = K.mean(K.square(tf.abs(thresholded_true-thresholded_pred)))
    mse = K.mean(K.square(tf.abs(y_true-thresholded_pred)))

    return mse


# %% Enable GPU usage
parser = argparse.ArgumentParser()

parser.add_argument('-data_num')
parser.add_argument('-gpu_index')
parser.add_argument('--mode', choices=['train', 'nas'], default='train',
                    help='train = baseline training, nas = run NAS for hyperparams')
parser.add_argument('--nas_trials', type=int, default=15,
                    help='Number of NAS trials')
parser.add_argument('--nas_epochs', type=int, default=40,
                    help='Epochs per NAS trial')
parser.add_argument('--nas_project', default='SBL_OTFS_NAS',
                    help='Directory name under ./NAS_TEST for tuner outputs')
#
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))


args = parser.parse_args()

# Allow CPU fallback if no GPU is detected
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("No GPU detected, running on CPU.")


# %% Building SBL_net model
def SBL_net_OTFS(Nt, Nr, Ms, Gs, G, num_layers, num_filters, kernel_size,
                alpha=0.0001 + 1j*0.0001, pruning_thrld=0.01, wt=0.1,
                layer_filters=None, layer_activations=None):
    # scale_factor = 0.01
    y_real_imag = Input(shape=(Np2, 2))
    Psi_real_imag = Input(shape=(Np2, Ms*Gs, 2))
    sigma_2 = Input(shape=(1, 1))

    # KerasTensor fix: Wrap Gamma_init creation
    def create_gamma_init(x):
        # x is Psi (BATCH, ...). Use dynamic shape for batch size.
        batch_size = tf.shape(x)[0]
        return tf.ones((batch_size, Ms*Gs, num_sc), dtype=x.dtype)
        
    Gamma_init = Lambda(create_gamma_init)(Psi_real_imag)

    # update mu and Sigma
    mu_real, mu_imag, diag_Sigma_real = Lambda(lambda x: update_mu_Sigma_OTFS(
        x, Np2))([Psi_real_imag, y_real_imag, Gamma_init, sigma_2])

    # Defaults if NAS is not supplying a search space
    kernel_sizes = [7, 5, 3, 3]
    default_filters = [64, 32, 16, 1]
    filters = layer_filters if layer_filters is not None else default_filters
    activations = layer_activations if layer_activations is not None else ['relu'] * 4

    # Helper function for feature extraction block
    def feature_extraction_block(args):
        mu_r, mu_i, diag_S = args
        
        # mu_square logic
        mu_sq = mu_r ** 2 + mu_i ** 2
        MU_sq = tf.transpose((tf.reshape(mu_sq, (-1, Gs, Ms, 1))), (0, 2, 1, 3))
        diag_S_exp = tf.transpose((tf.reshape(diag_S, (-1, Gs, Ms, 1))), (0, 2, 1, 3))
        temp_orig = tf.concat([MU_sq, diag_S_exp], axis=-1)
        
        # mu_complex logic
        mu_comp = tf.cast(mu_r, tf.complex64) + 1j * tf.cast(mu_i, tf.complex64)
        mu_comp_exp = tf.transpose((tf.reshape(mu_comp, (-1, Gs, Ms, 1))), (0, 2, 1, 3))
        
        # Thresholding
        thresholded_MU = tf.where(tf.abs(mu_comp_exp) < pruning_thrld, tf.zeros_like(mu_comp_exp), alpha)
        MU_r = tf.math.real(thresholded_MU)
        MU_i = tf.math.imag(thresholded_MU)
        
        F3 = tf.concat([MU_r, MU_i], axis=-1)
        temp_out = tf.concat([temp_orig, F3], axis=-1)
        return temp_out, F3

    for i in range(num_layers):

        mu_real_old = mu_real
        mu_imag_old = mu_imag

        # Pre-processing via Lambda
        # Note: Lambda can return multiple outputs if expected by destructuring
        temp, F3 = Lambda(feature_extraction_block)(
            [mu_real, mu_imag, diag_Sigma_real])

#
        # Convolution Filtering

        conv_layer1 = Conv2D(name='SBL_%d1' % i, filters=filters[0], kernel_size=kernel_sizes[0],
                             strides=1, padding='valid', activation=activations[0])
        temp_padded1 = circular_padding_1d(temp, kernel_size=kernel_sizes[0], strides=1)
        temp_conv1 = conv_layer1(temp_padded1)
        temp_conv1 = Lambda(lambda x: tf.concat(x, axis=-1))([temp_conv1, F3])

        conv_layer2 = Conv2D(name='SBL_%d2' % i, filters=filters[1], kernel_size=kernel_sizes[1],
                             strides=1, padding='valid', activation=activations[1])
        temp_padded2 = circular_padding_1d(
            temp_conv1, kernel_size=kernel_sizes[1], strides=1)
        temp_conv2 = conv_layer2(temp_padded2)
        temp_conv2 = Lambda(lambda x: tf.concat(x, axis=-1))([temp_conv2, F3])

        conv_layer3 = Conv2D(name='SBL_%d3' % i, filters=filters[2], kernel_size=kernel_sizes[2],
                             strides=1, padding='valid', activation=activations[2])
        temp_padded3 = circular_padding_1d(
            temp_conv2, kernel_size=kernel_sizes[2], strides=1)
        temp_conv3 = conv_layer3(temp_padded3)
        temp_conv3 = Lambda(lambda x: tf.concat(x, axis=-1))([temp_conv3, F3])

        conv_layer4 = Conv2D(name='SBL_%d4' % i, filters=filters[3], kernel_size=kernel_sizes[3],
                             strides=1, padding='valid', activation=activations[3])
        temp_padded4 = circular_padding_1d(
            temp_conv3, kernel_size=kernel_sizes[3], strides=1)
        temp_conv4 = conv_layer4(temp_padded4)
        # temp_conv4 = Lambda(lambda x: tf.concat(x, axis=-1))([temp_conv4, F3])

        Gamma = temp_conv4
        
        # KerasTensor fix: Wrap reshape of Gamma for update_mu_Sigma_OTFS
        Gamma_reshaped = Lambda(lambda x: tf.reshape(
            tf.transpose(x, (0, 2, 1, 3)), (-1, Ms*Gs, 1)))(Gamma)

        # Update mu and Sigma using new Gamma
        mu_real, mu_imag, diag_Sigma_real = Lambda(lambda x: update_mu_Sigma_OTFS(x, Np2))(
            [Psi_real_imag, y_real_imag, Gamma_reshaped, sigma_2])

        # KerasTensor fix: Weighted update
        # mu_real = wt*mu_real_old + (1-wt)*mu_real
        def weighted_update(args):
            old, new = args
            return wt * old + (1 - wt) * new
            
        mu_real = Lambda(weighted_update)([mu_real_old, mu_real])
        mu_imag = Lambda(weighted_update)([mu_imag_old, mu_imag])

    x_hat = Lambda(lambda x: tf.concat(
        [x[0], x[1]], axis=-1))([mu_real, mu_imag])

    model = Model(inputs=[Psi_real_imag, y_real_imag, sigma_2], outputs=x_hat)
    return model
# %% Load channel

# To investigate the impact of nn structure, use typical parameters


num_layers = 7
num_filters = 8
kernel_size = 8


# epochs = 50
# batch_size = 16
epochs = 1
batch_size = 4

resolution = 1  # resolution of angle grids
Nt = 32
Nr = 32

Ms = 16
Gs = 32
Np2 = 150
G = resolution*np.max([Ms, Gs])
G_A = 64
G_B = 64

div = 8

# SNRdb = 10
# sigma_2 = 1/10**(SNRdb/10)
num_sc = 1
mse_final = []
nmse_final = []

mode = ['LCConv3D']
data_num = int(args.data_num) if args.data_num else 1000
analyse_num = 1000

# h_list = io.loadmat('./hDL_10dB_40k_150pilots_ipjp.mat')['hDL'][-data_num:]
# y_list = io.loadmat('./yDL_10dB_40k_150pilots_ipjp.mat')['yDL'][-data_num:]
# Psi_list = mat73.loadmat('./PsiDL_10dB_40k_150pilots_ipjp.mat')['PsiDL'][-data_num:]
# sigma2_list = io.loadmat('./sigma2DL_10dB_40k_150pilots_ipjp.mat')['sigma2DL'][-data_num:]
# sigma2_list_new = sigma2_list.astype(np.complex64)

# h_list = io.loadmat('/home/saniya/Downloads/model_run/data/save_hDL.mat')['hDL'][-data_num:]
# y_list = io.loadmat('/home/saniya/Downloads/model_run/data/save_yDL.mat')['yDL'][-data_num:]
# # Load Psi using h5py to support standard HDF5 files generated
# with h5py.File('/home/saniya/Downloads/model_run/data/save_PsiDL.mat', 'r') as f:
#     # Need to handle potential transposition if saved from standard h5py vs mat73
#     # Our generator saved as (N, Np2, Ms*Gs, 2).
#     # SBL model expects this shape locally?
#     # Original code: Psi_list = mat73.loadmat(...)[...][-data_num:]
#     Psi_list = f['PsiDL'][-data_num:]

# sigma2_list = io.loadmat('/home/saniya/Downloads/model_run/data/save_wDL.mat')['sigma2DL'][-data_num:]
# sigma2_list_new = sigma2_list.astype(np.complex64)


h_list = io.loadmat('/home/saniya/Downloads/model_run/data/5db_3kiter/save_hDL.mat')['hDL'][-data_num:]
y_list = io.loadmat('/home/saniya/Downloads/model_run/data/5db_3kiter/save_yDL.mat')['yDL'][-data_num:]

# with h5py.File('/home/saniya/Downloads/model_run/data/5db_3kiter/save_PsiDL.mat', 'r') as f:
#     Psi_list = f['PsiDL'][-data_num:]

# Psi_list = np.array(Psi_list)


with h5py.File('/home/saniya/Downloads/model_run/data/5db_3kiter/save_PsiDL.mat', 'r') as f:
    Psi_raw = f['PsiDL'][:]

Psi_raw = np.array(Psi_raw)

# First transpose (MATLAB → Python)
Psi_list = np.transpose(Psi_raw, (3, 2, 0, 1))

# FIX: swap last two axes
Psi_list = np.transpose(Psi_list, (0, 1, 3, 2))

# Keep last samples
Psi_list = Psi_list[-data_num:]




sigma2_list = io.loadmat('/home/saniya/Downloads/model_run/data/5db_3kiter/save_sigma2DL.mat')['sigma2DL'][-data_num:]
# Reshape to (batch_size, 1, 1) to match model input shape
sigma2_list_new = sigma2_list.reshape(-1, 1, 1).astype(np.complex64)

print("Psi_list shape:", Psi_list.shape)
print("y_list shape:", y_list.shape)
print("h_list shape:", h_list.shape)



assert Psi_list.shape[1:] == (Np2, Ms*Gs, 2), "Psi_list shape mismatch"
assert y_list.shape[1:] == (Np2, 2), "y_list shape mismatch"
assert h_list.shape[1:] == (Ms*Gs, 2), "h_list shape mismatch"
assert sigma2_list_new.shape[1:] == (1, 1), "sigma2 shape mismatch"





# define callbacks
best_model_path = './NAS_TEST/TESTmodel_for_review_comments.weights.h5'
checkpointer = ModelCheckpoint(
    best_model_path, verbose=1, save_best_only=True, save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2,
                              cooldown=1, verbose=1, mode='auto', min_delta=1e-5, min_lr=1e-5)


def train_baseline():
    """Run the original training loop (no NAS)."""
    model = SBL_net_OTFS(Nt, Nr, Ms, Gs, G, num_layers, num_filters, kernel_size)
    model.summary()
    model.compile(loss=custom_loss, optimizer=Adam(learning_rate=1e-4))

    loss_history = model.fit([Psi_list, y_list, sigma2_list_new], h_list, epochs=epochs, batch_size=batch_size,
                             verbose=1, shuffle=True, validation_split=0.1, callbacks=[checkpointer, reduce_lr])

    # save loss history for plotting
    train_loss = np.squeeze(loss_history.history['loss'])
    val_loss = np.squeeze(loss_history.history['val_loss'])
    io.savemat('./NAS_TEST/TESTmodel_for_review_comments.mat',
                {'train_loss': train_loss, 'val_loss': val_loss})
    return model


def build_model_for_tuner(hp):
    """Build SBL model with NAS-searchable hyperparameters."""
    alpha_real = hp.Float('alpha_real', 1e-5, 1e-1, sampling='log')
    alpha_imag = hp.Float('alpha_imag', 1e-5, 1e-1, sampling='log')
    pruning_thrld = hp.Float('pruning_thrld', 1e-4, 5e-2, sampling='log')
    wt = hp.Float('wt', 0.0, 1.0, step=0.05)

    # Four Conv2D layers with flexible filters and activations
    # Four Conv2D layers with flexible filters and activations
    layer_filters = [
        hp.Int('filters_l1', min_value=16, max_value=128, step=16),
        hp.Int('filters_l2', min_value=8, max_value=96, step=16),
        hp.Int('filters_l3', min_value=8, max_value=64, step=8),
        1 # Force last layer to 1 filter to match Gamma shape requirements
    ]
    act_choices = ['relu', 'elu', 'tanh', 'gelu', 'selu']
    layer_activations = [
        hp.Choice('act_l1', act_choices),
        hp.Choice('act_l2', act_choices),
        hp.Choice('act_l3', act_choices),
        hp.Choice('act_l4', act_choices),
    ]

    lr = hp.Float('learning_rate', 5e-5, 5e-3, sampling='log')

    model = SBL_net_OTFS(
        Nt, Nr, Ms, Gs, G, num_layers, num_filters, kernel_size,
        alpha=alpha_real + 1j * alpha_imag,
        pruning_thrld=pruning_thrld,
        wt=wt,
        layer_filters=layer_filters,
        layer_activations=layer_activations
    )
    model.compile(loss=custom_loss, optimizer=Adam(learning_rate=lr))
    return model


def run_nas():
    """Random search NAS over alpha, pruning threshold, filters, activations, and wt."""
    tuner = kt.RandomSearch(
        build_model_for_tuner,
        objective='val_loss',
        max_trials=args.nas_trials,
        executions_per_trial=1,
        overwrite=True,
        directory='./NAS_TEST',
        project_name=args.nas_project
    )
    
    # Clean previous results if overwriting (though overwrite=True should handle it, sometimes file locks persist)
    import shutil
    try:
        shutil.rmtree(os.path.join('./NAS_TEST', args.nas_project))
    except:
        pass

    tuner.search(
        [Psi_list, y_list, sigma2_list_new],
        h_list,
        epochs=args.nas_epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[reduce_lr],
        verbose=1
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]

    # Persist best weights and hyperparameters
    best_model.save_weights('./NAS_TEST/best_nas_model.weights.h5')
    np.savez('./NAS_TEST/best_nas_hyperparameters.npz', **best_hp.values)
    return best_hp, best_model


if args.mode == 'nas':
    hp_cfg, best_model_initial = run_nas()
    print("NAS best hyperparameters:", hp_cfg.values)
    
    # Retrain with best hyperparameters to get full history and final plot data
    print("\nRetraining with best NAS hyperparameters to generate final loss history...")
    best_model = build_model_for_tuner(hp_cfg)
    best_model.summary()
    
    callbacks_list = [checkpointer, reduce_lr]
    
    # Add EarlyStopping to avoid long waits if converged
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, min_delta=1e-5)
    callbacks_list.append(early_stopping)
    
    loss_history = best_model.fit(
        [Psi_list, y_list, sigma2_list_new], h_list, 
        epochs=20, # Explicitly increased epochs for better visualization
        batch_size=batch_size,
        verbose=1, shuffle=True, validation_split=0.1, 
        callbacks=callbacks_list
    )
    
    # Save loss history for plotting
    train_loss = np.squeeze(loss_history.history['loss'])
    val_loss = np.squeeze(loss_history.history['val_loss'])
    io.savemat('./NAS_TEST/TESTmodel_for_review_comments.mat',
                {'train_loss': train_loss, 'val_loss': val_loss})
    print("Saved training history to ./NAS_TEST/TESTmodel_for_review_comments.mat")

else:
    train_baseline()