from interface import DDPGInterface

import tensorflow as tf

from keras.models import Model

from keras import layers
import keras

import numpy as np
import os
from pathlib import Path

import matplotlib.pyplot as plt



# --------- helper losses/metrics ----------
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# Use combined loss: BCE + Dice
def bce_dice_loss(y_true, y_pred):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# --------- 3D UNet (compact) ----------
def conv_block(x, filters, kernel_size=3, activation='relu'):
    x = layers.Conv3D(filters, kernel_size, padding='same', activation=activation)(x)
    x = layers.Conv3D(filters, kernel_size, padding='same', activation=activation)(x)
    return x

def build_task_predictor_3d(input_shape=(32,32,32,1), base_filters=16):
    """
    input_shape: (Z, Y, X, C)
    returns compiled model (uncompiled if you prefer to compile later)
    """
    inputs = layers.Input(shape=input_shape, name='ct_input')

    # Encoder
    c1 = conv_block(inputs, base_filters)
    p1 = layers.MaxPool3D((2,2,2))(c1)

    c2 = conv_block(p1, base_filters * 2)
    p2 = layers.MaxPool3D((2,2,2))(c2)

    c3 = conv_block(p2, base_filters * 4)
    p3 = layers.MaxPool3D((2,2,2))(c3)

    c4 = conv_block(p3, base_filters * 8)

    # Decoder
    u3 = layers.UpSampling3D((2,2,2))(c4)
    u3 = layers.Concatenate()([u3, c3])
    c5 = conv_block(u3, base_filters * 4)

    u2 = layers.UpSampling3D((2,2,2))(c5)
    u2 = layers.Concatenate()([u2, c2])
    c6 = conv_block(u2, base_filters * 2)

    u1 = layers.UpSampling3D((2,2,2))(c6)
    u1 = layers.Concatenate()([u1, c1])
    c7 = conv_block(u1, base_filters)

    # Output single-channel sigmoid (binary segmentation)
    outputs = layers.Conv3D(1, (1,1,1), activation='sigmoid', name='seg_out')(c7)

    model = Model(inputs=inputs, outputs=outputs, name='3D_UNet')

    # compile with combined loss and dice metric
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss=bce_dice_loss,
                  metrics=[dice_coef])
    return model



# --------- 3D Actor & Critic ----------
def build_actor_critic_3d(img_shape, action_shape=(1,)):
    """
    img_shape: (Z, Y, X, C) e.g. (32,32,32,1) -- same convention used in the rest of the repo
    action_shape: shape of action vector (default single scalar)
    Returns: actor, critic, action_input 
    """
    n_actions = action_shape[0]

    # Actor
    ## ------------------------------------------------------------------------------------------
    #repo convention
    #act_in = layers.Input((1,) + img_shape)               # matches convention used in repo
    #act_in_reshape = layers.Reshape(img_shape)(act_in)    # remove extra dim
    #a = layers.Conv3D(16, 3, activation='relu', padding='same')(act_in_reshape)

    # 3D actor input directly, NOW expects (Z,Y,X,C)
    act_in = layers.Input(shape=img_shape, name='observation_input_actor')
    a = layers.Conv3D(16, 3, activation='relu', padding='same')(act_in)
    ## ------------------------------------------------------------------------------------------
    
    a = layers.MaxPool3D(2)(a)
    a = layers.Conv3D(32, 3, activation='relu', padding='same')(a)
    a = layers.MaxPool3D(2)(a)
    a = layers.Conv3D(64, 3, activation='relu', padding='same')(a)
    a = layers.GlobalAveragePooling3D()(a)
    a = layers.Dense(64, activation='relu')(a)
    a = layers.Dense(32, activation='relu')(a)
    act_out = layers.Dense(n_actions, activation='sigmoid')(a)  # sigmoid between 0..1 if required
    actor = Model(inputs=act_in, outputs=act_out, name='3D_actor')

    # Critic: takes action input (vector) and observation input (volume)
    action_input = layers.Input(shape=(n_actions,), name='action_input')
    
    ## ------------------------------------------------------------------------------------------
    # repo convention
    #observation_input = layers.Input((1,) + img_shape, name='observation_input')
    #obs = layers.Reshape(img_shape)(observation_input)
    
    #3D critic input directly, NOW expects (Z,Y,X,C)
    observation_input = layers.Input(shape=img_shape, name='observation_input')
    obs = observation_input
    ## ------------------------------------------------------------------------------------------
    
    o = layers.Conv3D(16, 3, activation='relu', padding='same')(obs)
    o = layers.MaxPool3D(2)(o)
    o = layers.Conv3D(32, 3, activation='relu', padding='same')(o)
    o = layers.MaxPool3D(2)(o)
    o = layers.Conv3D(64, 3, activation='relu', padding='same')(o)
    o = layers.GlobalAveragePooling3D()(o)

    x = layers.Concatenate()([action_input, o])
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    crit_out = layers.Dense(1, name='q_value')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=crit_out, name='3D_critic')

    return actor, critic, action_input









img_shape = (32, 32, 32, 1)       # match your preprocessed data
x_train = np.load(Path(r"D:/liver_data_arrays_32/x_train.npy"))
y_train = np.load(Path(r"D:/liver_data_arrays_32/y_train.npy"))
x_val = np.load(Path(r"D:/liver_data_arrays_32/x_val.npy"))
y_val = np.load(Path(r"D:/liver_data_arrays_32/y_val.npy"))
x_holdout = np.load(Path(r"D:/liver_data_arrays_32/x_holdout.npy"))
y_holdout = np.load(Path(r"D:/liver_data_arrays_32/y_holdout.npy"))

# check shapes
print("img_shape variable:", img_shape)
print("x_train.shape:", x_train.shape)
print("x_val.shape:", x_val.shape)
print("x_holdout.shape:", x_holdout.shape)
# Inspect one sample that the agent uses:
sample = x_train[0:1]
print("sample.shape (one batch):", sample.shape)

# instantiate
task_predictor = build_task_predictor_3d(img_shape)
# it's already compiled in the helper; if you prefer different optimizer/lr change it there.

# actor/critic
actor, critic, action_input = build_actor_critic_3d(img_shape, action_shape=(1,))

print("actor.input_shape:", actor.input_shape)         # should show 
print("critic.inputs:", [t.shape for t in critic.inputs])  # check observation_input shape








controller_batch_size = 2
task_predictor_batch_size = 2

interface = DDPGInterface(x_train, y_train, x_val, y_val, x_holdout, y_holdout, task_predictor, img_shape,
                          custom_controller=True, actor=actor, critic=critic, action_input=action_input,
                          modify_env_params=True, modified_env_params_list=[controller_batch_size, task_predictor_batch_size])


num_train_episodes = 512

interface.train(num_train_episodes)
