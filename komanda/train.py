import os
import numpy as np
from komanda.model import *
from komanda.BatchGenerator import BatchGenerator
from komanda.helpers import read_csv, process_csv

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
checkpoint_dir = os.getcwd() + "/v3"
global_train_step = 0
global_valid_step = 0
KEEP_PROB_TRAIN = 0.25
NUM_EPOCHS = 100
best_validation_score = None

# concatenated interpolated.csv from ros bags
(train_seq, valid_seq), (mean, std) = process_csv(filename="output/interpolated_concat.csv", val=5)
# interpolated.csv for TestSet filled with dummy values
test_seq = read_csv("challenge_2/exampleSubmissionInterpolatedFinal.csv")


def do_epoch(session, sequences, mode):
    global global_train_step, global_valid_step

    test_predictions = {}
    valid_predictions = {}

    batch_generator = BatchGenerator(sequence=sequences, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
    total_num_steps = 1 + (batch_generator.indices[1] - 1) / SEQ_LEN

    controller_final_state_gt_cur = None
    controller_final_state_autoregressive_cur = None

    acc_loss = np.float128(0.0)
    for step in range(total_num_steps):
        feed_inputs, feed_targets = batch_generator.next()
        feed_dict = {inputs: feed_inputs, targets: feed_targets}
        if controller_final_state_autoregressive_cur is not None:
            feed_dict.update({controller_initial_state_autoregressive: controller_final_state_autoregressive_cur})
        if controller_final_state_gt_cur is not None:
            feed_dict.update({controller_final_state_gt: controller_final_state_gt_cur})
        if mode == "train":
            feed_dict.update({keep_prob: KEEP_PROB_TRAIN})
            summary, _, loss, controller_final_state_gt_cur, controller_final_state_autoregressive_cur = \
                session.run([summaries, optimizer, mse_autoregressive_steering, controller_final_state_gt,
                             controller_final_state_autoregressive],
                            feed_dict=feed_dict)
            train_writer.add_summary(summary, global_train_step)
            global_train_step += 1
        elif mode == "valid":
            model_predictions, summary, loss, controller_final_state_autoregressive_cur = \
                session.run([steering_predictions, summaries, mse_autoregressive_steering,
                             controller_final_state_autoregressive],
                            feed_dict=feed_dict)
            valid_writer.add_summary(summary, global_valid_step)
            global_valid_step += 1
            feed_inputs = feed_inputs[:, LEFT_CONTEXT:].flatten()
            steering_targets = feed_targets[:, :, 0].flatten()
            model_predictions = model_predictions.flatten()
            stats = np.stack([steering_targets, model_predictions, (steering_targets - model_predictions) ** 2])
            for i, img in enumerate(feed_inputs):
                valid_predictions[img] = stats[:, i]
        elif mode == "test":
            model_predictions, controller_final_state_autoregressive_cur = \
                session.run([steering_predictions, controller_final_state_autoregressive],
                            feed_dict=feed_dict)
            feed_inputs = feed_inputs[:, LEFT_CONTEXT:].flatten()
            model_predictions = model_predictions.flatten()
            for i, img in enumerate(feed_inputs):
                test_predictions[img] = model_predictions[i]
        if mode != "test":
            acc_loss += loss
            print('\r', step + 1, "/", total_num_steps, np.sqrt(acc_loss / (step + 1)))
    return (np.sqrt(acc_loss / total_num_steps), valid_predictions) if mode != "test" else (None, test_predictions)


with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as session:
    session.run(tf.initialize_all_variables())
    print('Initialized')
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if ckpt:
        print("Restoring from", ckpt)
        saver.restore(sess=session, save_path=ckpt)
    for epoch in range(NUM_EPOCHS):
        print("Starting epoch %d" % epoch)
        print("Validation:")
        valid_score, valid_predictions = do_epoch(session=session, sequences=valid_seq, mode="valid")

        if best_validation_score is None:
            best_validation_score = valid_score

        if valid_score < best_validation_score:
            saver.save(session, 'v3/checkpoint-sdc-ch2')
            best_validation_score = valid_score
            print('\r', "SAVED at epoch %d" % epoch)
            with open("v3/valid-predictions-epoch%d" % epoch, "w") as out:
                result = np.float128(0.0)
                for img, stats in valid_predictions.items():
                    print(out, img, stats)
                    result += stats[-1]
            print("Validation un-normalized RMSE:", np.sqrt(result / len(valid_predictions)))
            with open("v3/test-predictions-epoch%d" % epoch, "w") as out:
                _, test_predictions = do_epoch(session=session, sequences=test_seq, mode="test")
                print(out, "frame_id,steering_angle")
                for img, pred in test_predictions.items():
                    img = img.replace("challenge_2/Test-final/center/", "")
                    print(out, "%s,%f" % (img, pred))
        if epoch != NUM_EPOCHS - 1:
            print("Training")
            do_epoch(session=session, sequences=train_seq, mode="train")
