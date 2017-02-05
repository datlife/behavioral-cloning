#
# MODEL ARCHITECTURE
# Let's build the main graph. Code is mostly self-explanatory.

# A few comments:
# 1) PNG images were used as the input only because this was the format for round1 test-set.
#    In practice, raw images should be fed directly from the ros-bags.

# 2) We define: `get_initial_state` and `deep_copy_initial_state` functions to be able to preserve the state of
#    our recurrent net between batches. The back-propagation is still truncated by SEQ_LEN.

# 3) The loss is composed of two components.
#    (1) MSE of the steering angle prediction in the autoregressive setting
#        -- that is exactly what interests us in the test time.
#    (2) Weighted by aux_cost_weight: sum of MSEs for all outputs both in autoregressive and ground truth settings.

# Note: if the saver definition doesn't work for you please make sure you are using TensorFlow 0.12rc0 or newer.

import tensorflow as tf
from constants import *
from helpers import apply_vision_simple, get_optimizer
from RNNCell import SamplingRNNCell

graph = tf.Graph()

with graph.as_default():
    # inputs
    learning_rate = tf.placeholder_with_default(input=1e-4, shape=())
    keep_prob = tf.placeholder_with_default(input=1.0, shape=())
    aux_cost_weight = tf.placeholder_with_default(input=0.1, shape=())

    # path to png files from the central camera
    inputs = tf.placeholder(shape=(BATCH_SIZE, LEFT_CONTEXT + SEQ_LEN), dtype=tf.string)

    # seq_len x batch_size x OUTPUT_DIM
    targets = tf.placeholder(shape=(BATCH_SIZE, SEQ_LEN, OUTPUT_DIM), dtype=tf.float32)
    # targets_normalized = (targets - mean) / std
    targets_normalized = targets/255

    # ================================= INPUT IMAGE ================================================ #
    input_images = tf.pack([tf.image.decode_png(tf.read_file(x))
                            for x in tf.unpack(tf.reshape(inputs,
                                                          shape=[(LEFT_CONTEXT + SEQ_LEN) * BATCH_SIZE]))])
    # Normalize image
    input_images = -1.0 + 2.0 * tf.cast(input_images, tf.float32) / 255.0
    input_images = tf.reshape(input_images, shape=[(LEFT_CONTEXT + SEQ_LEN) * BATCH_SIZE, HEIGHT, WIDTH, CHANNELS])

    # ================================== CNN BLOCK ================================================== #
    visual_conditions_reshaped = apply_vision_simple(image=input_images, keep_prob=keep_prob,
                                                     batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    visual_conditions = tf.reshape(visual_conditions_reshaped, [BATCH_SIZE, SEQ_LEN, -1])
    visual_conditions = tf.nn.dropout(x=visual_conditions, keep_prob=keep_prob)

    # ================================= LSTM BLOCK =================================================== #
    # Input retrieved from CNN will be feed into RNN
    rnn_inputs_with_ground_truth = (visual_conditions, targets_normalized)
    rnn_inputs_autoregressive = (visual_conditions, tf.zeros(shape=(BATCH_SIZE, SEQ_LEN, OUTPUT_DIM),
                                                             dtype=tf.float32))

    internal_cell = tf.nn.rnn_cell.LSTMCell(num_units=RNN_SIZE, num_proj=RNN_PROJ)
    cell_with_ground_truth = SamplingRNNCell(num_outputs=OUTPUT_DIM, use_ground_truth=True,
                                             internal_cell=internal_cell)
    cell_autoregressive = SamplingRNNCell(num_outputs=OUTPUT_DIM, use_ground_truth=False,
                                          internal_cell=internal_cell)


    def get_initial_state(complex_state_tuple_sizes):
        flat_sizes = tf.nn.rnn_cell.nest.flatten(complex_state_tuple_sizes)
        init_state_flat = [tf.tile(multiples=[BATCH_SIZE, 1], input=tf.get_variable("controller_initial_state_%d" % i,
                                                                                    initializer=tf.zeros_initializer,
                                                                                    shape=([1, s]), dtype=tf.float32))
                           for i, s in enumerate(flat_sizes)]
        init_state = tf.nn.rnn_cell.nest.pack_sequence_as(complex_state_tuple_sizes, init_state_flat)
        return init_state


    def deep_copy_initial_state(complex_state_tuple):
        flat_state = tf.nn.rnn_cell.nest.flatten(complex_state_tuple)
        flat_copy = [tf.identity(s) for s in flat_state]
        deep_copy = tf.nn.rnn_cell.nest.pack_sequence_as(complex_state_tuple, flat_copy)
        return deep_copy


    controller_initial_state_variables = get_initial_state(cell_autoregressive.state_size)
    controller_initial_state_autoregressive = deep_copy_initial_state(controller_initial_state_variables)
    controller_initial_state_gt = deep_copy_initial_state(controller_initial_state_variables)

    with tf.variable_scope("predictor"):
        out_gt, controller_final_state_gt = tf.nn.dynamic_rnn(cell=cell_with_ground_truth,
                                                              inputs=rnn_inputs_with_ground_truth,
                                                              sequence_length=[SEQ_LEN] * BATCH_SIZE,
                                                              initial_state=controller_initial_state_gt,
                                                              dtype=tf.float32,
                                                              swap_memory=True, time_major=False)
    with tf.variable_scope("predictor", reuse=True):
        out_autoregressive, controller_final_state_autoregressive = tf.nn.dynamic_rnn(cell=cell_autoregressive,
                                                                                      inputs=rnn_inputs_autoregressive,
                                                                                      sequence_length=[SEQ_LEN] * BATCH_SIZE,
                                                                                      initial_state = controller_initial_state_autoregressive,
                                                                                      dtype=tf.float32,
                                                                                      swap_memory=True,
                                                                                      time_major=False)

    mse_gt = tf.reduce_mean(tf.squared_difference(out_gt, targets_normalized))
    mse_autoregressive = tf.reduce_mean(tf.squared_difference(out_autoregressive, targets_normalized))
    mse_autoregressive_steering = tf.reduce_mean(tf.squared_difference(out_autoregressive[:, :, 0],
                                                                       targets_normalized[:, :, 0]))
    # steering_predictions = (out_autoregressive[:, :, 0] * std[0]) + mean[0]
    steering_predictions = (out_autoregressive[:, :, 0])


    total_loss = mse_autoregressive_steering + aux_cost_weight * (mse_gt + mse_autoregressive)

    optimizer = get_optimizer(total_loss, learning_rate)

    tf.scalar_summary("MAIN TRAIN METRIC: rmse_autoregressive_steering", tf.sqrt(mse_autoregressive_steering))
    tf.scalar_summary("rmse_gt", tf.sqrt(mse_gt))
    tf.scalar_summary("rmse_autoregressive", tf.sqrt(mse_autoregressive))

    summaries = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter('v3/train_summary', graph=graph)
    valid_writer = tf.train.SummaryWriter('v3/valid_summary', graph=graph)
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)