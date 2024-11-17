import os
import sys
#os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices from 0 in the order of PCI_BUS_ID
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import numpy as np
import tensorflow as tf
import load_trace
import a3c_future
import fixed_env_future_bw_rl as env
# command
# python *.py
# -REBUF_PENALTY, float/int
# -tag, string, u*t depend on REBUF_PENALTY, trace
# -TEST_TRACES, string, eg, './cooked_test_traces/'
# -NN_MODEL, string, '.../nn_model_best_constrained.ckpt'
# -state_scale, default 1, float, 1 or 100
# -use_gumbel_noise, default 0, 1 or 0
# -rm_rtt, default 0,  1 or 0

S_INFO = 7 #6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
Penalty_LR_RATE = 0.001
VIDEO_BIT_RATE = [1500, 4900, 8200, 11700, 32800, 152400] # [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 120.0 # 48.0
M_IN_K = 1000.0
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 100#42
RAND_RANGE = 1000
REBUF_PENALTY = float(sys.argv[1]) # 4.3  # 1 sec rebuffering -> 3 Mbps
tag = sys.argv[2]
TEST_TRACES = sys.argv[3]  #'./cooked_test_traces/'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
NN_MODEL = sys.argv[4]
#LOG_FILE = './test_results_future_constrained/log_sim_rl'
#SUMMARY_DIR = './results_' + tag
#NN_MODEL = '../sim/' + SUMMARY_DIR + "/nn_model_best_constrained.ckpt"
SUMMARY_DIR = './results_' + tag
LOG_FILE = SUMMARY_DIR + '/log_sim_cons_rl_future'
#LOG_FILE = './test_results_' + tag +'/log_sim_rl'
state_scale = float(sys.argv[5]) # 100 in 5G, 1 in 3G original pensieve paper.
use_gumbel_noise = int(sys.argv[6])
rm_rtt = int(sys.argv[7])
future_steps = 4

def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    with tf.compat.v1.Session() as sess:

        actor = a3c_future.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE, future_step=future_steps)

        critic = a3c_future.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE, future_step=future_steps)

        penalty = a3c_future.PenaltyNetwork(sess,
                                     state_dim=[S_INFO, S_LEN],
                                     learning_rate=Penalty_LR_RATE, future_step=future_steps)

        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []
        entropy_ = 0.5

        video_count = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain, future_bw = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty - smoothness
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                               VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

            #reward /= state_scale
            r_batch.append(reward)

            last_bit_rate = bit_rate

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp / M_IN_K) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            # retrieve previous state
            if len(s_batch) == 0:
                state = np.zeros((S_INFO, S_LEN))
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            if rm_rtt > 0:
                delay = float(delay) - env.LINK_RTT
            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K / state_scale # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K / state_scale  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
            state[6] = [0] * S_LEN
            state[6, :future_steps] = np.array(future_bw) / state_scale

            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            if use_gumbel_noise > 0:
                noise = np.random.gumbel(size=len(action_prob))
                bit_rate = np.argmax(np.log(action_prob) + noise)
            else:
                action_cumsum = np.cumsum(action_prob)
                bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                # Note: we need to discretize the probability into 1/RAND_RANGE steps,
                # because there is an intrinsic discrepancy in passing single state and batch states

            s_batch.append(state)
            entropy_ = a3c_future.compute_entropy(action_prob[0])
            entropy_record.append(entropy_)

            if end_of_video:
                log_file.write('\n')
                log_file.close()

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)
                entropy_record = []

                video_count += 1

                if video_count >= len(all_file_names):
                    break

                log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
                log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()
