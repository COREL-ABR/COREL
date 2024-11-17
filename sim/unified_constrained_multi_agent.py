import datetime
import os, time
import logging
import numpy as np
import multiprocessing as mp

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import tensorflow as tf
import env_future_bandwidth as env
import a3c
import load_trace
import sys

S_INFO = 7 # 6 # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
Penalty_LR_RATE = 0.001
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 200  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [1500, 4900, 8200, 11700, 32800, 152400]  # [300,750,1200,1850,2850,4300]  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 120.0  # 48.0
M_IN_K = 1000.0
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
TRAIN_TRACES = './cooked_traces/'
# NN_MODEL = './results/pretrain_linear_reward.ckpt'
# NN_MODEL = None
NN_MODEL = sys.argv[1]
MAX_EPOCH = int(sys.argv[2])  # 20000  #
ENTROPY_WEIGHT = float(sys.argv[3])
REBUF_PENALTY = float(sys.argv[4])  # 4.3  # 1 sec rebuffering -> 3 Mbps
tag = sys.argv[5]
use_true_stall = 1
SUMMARY_DIR = './results_' + tag  # future_constrained
LOG_FILE = SUMMARY_DIR + '/log'
#LOG_FILE = './results_future_constrained/log'
TEST_LOG_FOLDER = './test_results_' + tag +'/'  #future_constrained/
PATIENCE = 20000
state_scale = 100  # 100 in 5G, 1 in 3G original pensieve paper.
use_gumbel_noise = 1
rm_rtt = 1
future_steps = 5
use_stall_duration = 1
if use_gumbel_noise < 1:
    SUMMARY_DIR = SUMMARY_DIR + '_noGumbel'
    LOG_FILE = SUMMARY_DIR + '/log'
    TEST_LOG_FOLDER = TEST_LOG_FOLDER[:-1] + '_noGumbel/'

# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)


def testing(epoch, nn_model, log_file, rebuf_penalty):
    # clean up the test results folder
    os.system('rm -r ' + TEST_LOG_FOLDER)
    if not os.path.exists(TEST_LOG_FOLDER):
        os.makedirs(TEST_LOG_FOLDER)
    # run test script
    os.system('python rl_test_constrained_future.py ' + nn_model + ' ' + str(ENTROPY_WEIGHT) + ' ' + str(rebuf_penalty) + ' ' + tag)

    # append test performance to the log
    rewards, entropies = [], []
    stalls = []
    bitrates = []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward, entropy = [], []
        stall = []
        bitrate = []
        with open(TEST_LOG_FOLDER + test_log_file, 'r') as f:
            for line in f:
                parse = line.split()
                try:
                    entropy.append(float(parse[-2]))
                    reward.append(float(parse[-1]))
                    rebuf = float(parse[-5])
                    bitrate.append(int(parse[1]))
                    if use_stall_duration > 0:
                        stall.append(rebuf)
                    else:
                        if rebuf > 0:
                            stall.append(1)
                        else:
                            stall.append(0)
                except IndexError:
                    break
        rewards.append(np.mean(reward[1:]))  # append the total rewards of a test file
        entropies.append(np.mean(entropy[1:]))
        stalls.append(np.sum(stall[1:]))
        bitrates.append(np.mean(bitrate[1:]))

    rewards = np.array(rewards)
    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\n')
    log_file.flush()

    return rewards_mean, np.mean(entropies), np.mean(stalls), np.mean(bitrates)


def central_agent(net_params_queues, exp_queues, shared_rebuf_penalty):
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)

    with tf.compat.v1.Session() as sess, open(LOG_FILE + '_test', 'w') as test_log_file:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE, entropy_weight=ENTROPY_WEIGHT)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)
        penalty = a3c.PenaltyNetwork(sess,
                                     state_dim=[S_INFO, S_LEN],
                                     learning_rate=Penalty_LR_RATE)

        summary_ops, summary_vars = a3c.build_summaries()

        sess.run(tf.compat.v1.global_variables_initializer())
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        TRAIN_SUMMARY_DIR = SUMMARY_DIR + '/train'
        # TEST_SUMMARY_DIR = './results/'+curr_time+'/test'
        writer = tf.compat.v1.summary.FileWriter(TRAIN_SUMMARY_DIR, sess.graph)  # training monitor
        # test_writer = tf.compat.v1.summary.FileWriter(TEST_SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.compat.v1.train.Saver()  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None and 'None' not in NN_MODEL:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        epoch = 0
        # test_avg_reward, test_avg_entropy, test_avg_td_loss = 0, 0.5, 0  # ????

        # assemble experiences from agents, compute the gradients
        best_valid_reward = -9999999999
        best_epoch = -1
        while True:
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            penalty_net_params = penalty.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params, penalty_net_params])
                # Note: this is synchronous version of the parallel training,
                # which is easier to understand and probe. The framework can be
                # fairly easily modified to support asynchronous training.
                # Some practices of asynchronous training (lock-free SGD at
                # its core) are nicely explained in the following two papers:
                # https://arxiv.org/abs/1602.01783
                # https://arxiv.org/abs/1106.5730

            # record average reward and td loss change
            # in the experiences from the agents
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0
            total_u = 0.0

            # assemble experiences from the agents
            actor_gradient_batch = []
            critic_gradient_batch = []
            penalty_gradient_batch = []

            for i in range(NUM_AGENTS):
                s_batch, a_batch, r_batch, stall_batch, terminal, info = exp_queues[i].get()

                actor_gradient, critic_gradient, td_batch, penalty_gradients, u_batch = \
                    a3c.compute_gradients_with_penalty(
                        s_batch=np.stack(s_batch, axis=0),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        stall_batch=np.vstack(stall_batch),
                        terminal=terminal, actor=actor, critic=critic, penalty=penalty)
                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)
                penalty_gradient_batch.append(penalty_gradients)

                total_reward += np.sum(r_batch)
                total_td_loss += np.sum(td_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0
                total_entropy += np.sum(info['entropy'])
                if use_true_stall > 0:
                    total_u += np.sum(stall_batch)
                else:
                    total_u += np.sum(u_batch) #u_batch[-1, 0]

            # compute aggregated gradient
            assert NUM_AGENTS == len(actor_gradient_batch)
            assert len(actor_gradient_batch) == len(critic_gradient_batch)
            for i in range(len(actor_gradient_batch)):
                actor.apply_gradients(actor_gradient_batch[i])
                critic.apply_gradients(critic_gradient_batch[i])
                penalty.apply_gradients(penalty_gradient_batch[i])

            # log training information
            epoch += 1
            avg_reward = total_reward / total_agents
            avg_td_loss = total_td_loss / total_batch_len
            avg_entropy = total_entropy / total_batch_len
            avg_u = total_u / total_agents #/ total_batch_len

            logging.info('Epoch: ' + str(epoch) +
                         ' TD_loss: ' + str(avg_td_loss) +
                         ' Avg_reward: ' + str(avg_reward) +
                         ' Avg_entropy: ' + str(avg_entropy) +
                         ' Avg_u: ' + str(avg_u))

            # Training summary
            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: avg_td_loss,
                summary_vars[1]: avg_reward,
                summary_vars[2]: avg_entropy
            })
            writer.add_summary(summary_str, epoch)
            writer.flush()
            # Testing summary ?????
            # summary_str = sess.run(summary_ops, feed_dict={
            #     summary_vars[0]: test_avg_td_loss,
            #     summary_vars[1]: test_avg_reward,
            #     summary_vars[2]: test_avg_entropy
            # })
            #
            # test_writer.add_summary(summary_str, epoch)
            # test_writer.flush()

            # if epoch % MODEL_SAVE_INTERVAL == 0:
            #     # Save the neural net parameters to disk.
            #     save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt")
            #     logging.info("Model saved in file: " + save_path)
            #     test_avg_reward, test_avg_entropy = testing(epoch, SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt", test_log_file)

            new_mu = max(shared_rebuf_penalty[-1] + 0.0001 * (avg_u - 0.05), 0)  # avg_u
            if (epoch - 1) % MODEL_SAVE_INTERVAL == 0:
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_latest.ckpt")
                test_avg_reward, test_avg_entropy, test_avg_stalls, test_avg_bitrate = testing(epoch, SUMMARY_DIR + "/nn_model_latest.ckpt",
                                                            test_log_file, shared_rebuf_penalty[-1])
                test_results = test_avg_bitrate/1000 - test_avg_stalls
                if test_results > best_valid_reward:
                    best_valid_reward = test_results
                    best_epoch = epoch
                    best_valid_stall = avg_u #test_avg_stalls
                    best_rebuf_penalty = shared_rebuf_penalty[-1]
                    # save the best model
                    save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_best_constrained.ckpt")
                    logging.info("Model saved in file: " + save_path)
                    logging.info("current rebuf penalty: " + str(shared_rebuf_penalty[-1]))
                    logging.info("Best test average reward, test total stalls, test average bitrate, test results: " +  str(test_avg_reward) + '\t' + str(test_avg_stalls) + '\t' + str(test_avg_bitrate) + '\t' + str(test_results))
                    logging.info("train average reward, train total stalls: " +  str(avg_reward) +'\t' + str(avg_u))
                    print('best_epoch: ', best_epoch, '. Saved best model')
            shared_rebuf_penalty.append(new_mu)
            if (epoch - best_epoch > PATIENCE or epoch >= MAX_EPOCH) and (abs(shared_rebuf_penalty[-1] - shared_rebuf_penalty[-2]) < 0.00005):
                np.savetxt(SUMMARY_DIR + '/mu_history.txt', shared_rebuf_penalty, delimiter='\n')
                with open(SUMMARY_DIR + '/best_mu.txt', 'w') as f:
                    f.write(str(best_rebuf_penalty))
                    f.close()
                break


def agent(agent_id, all_cooked_time, all_cooked_bw, net_params_queue, exp_queue, shared_rebuf_penalty):
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              random_seed=agent_id)

    with tf.compat.v1.Session() as sess, open(LOG_FILE + '_agent_' + str(agent_id), 'w') as log_file:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE, entropy_weight=ENTROPY_WEIGHT)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)
        penalty = a3c.PenaltyNetwork(sess,
                                     state_dim=[S_INFO, S_LEN],
                                     learning_rate=Penalty_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params, penalty_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)
        penalty.set_network_params(penalty_net_params)

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []
        stall_batch = []

        time_stamp = 0
        while True:  # experience video streaming forever

            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain, future_bw = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # -- linear reward --
            # reward is video quality - rebuffer penalty - smoothness
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - shared_rebuf_penalty[-1] * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                               VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
            reward /= state_scale
            # -- log scale reward --
            # log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[-1]))
            # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[-1]))

            # reward = log_bit_rate \
            #          - REBUF_PENALTY * rebuf \
            #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

            # -- HD reward --
            # reward = HD_REWARD[bit_rate] \
            #          - REBUF_PENALTY * rebuf \
            #          - SMOOTH_PENALTY * np.abs(HD_REWARD[bit_rate] - HD_REWARD[last_bit_rate])

            r_batch.append(reward)
            if use_stall_duration:
                stall_batch.append(rebuf)
            else:
                if rebuf > 0:
                    stall_batch.append(1)
                else:
                    stall_batch.append(0)

            last_bit_rate = bit_rate

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
            # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec, buffer_size
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K / state_scale  # kilo byte / ms bandwidth_measurement
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K / state_scale  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
            state[6, :future_steps] = np.array(future_bw) / state_scale
            # compute action probability vector
            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            if use_gumbel_noise > 0:
                # gumbel noise
                noise = np.random.gumbel(size=len(action_prob))
                bit_rate = np.argmax(np.log(action_prob) + noise)
            else:
                action_cumsum = np.cumsum(action_prob)
                bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                # Note: we need to discretize the probability into 1/RAND_RANGE steps,
                # because there is an intrinsic discrepancy in passing single state and batch states

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            # report experience to the coordinator
            if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:
                exp_queue.put([s_batch[1:],  # ignore the first chuck
                               a_batch[1:],  # since we don't have the
                               r_batch[1:],  # control over it
                               stall_batch[1:],
                               end_of_video,
                               {'entropy': entropy_record}])

                # synchronize the network parameters from the coordinator
                actor_net_params, critic_net_params, penalty_net_params = net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)
                penalty.set_network_params(penalty_net_params)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del entropy_record[:]
                del stall_batch[:]

                log_file.write('\n')  # so that in the log we know where video ends

            # store the state and action into batches
            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)

            else:
                s_batch.append(state)

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1
                a_batch.append(action_vec)


def main():
    np.random.seed(RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == A_DIM

    # create result directory
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    manager = mp.Manager()
    shared_rebuf_penalty = manager.list()
    shared_rebuf_penalty.append(REBUF_PENALTY)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues, shared_rebuf_penalty))
    coordinator.start()

    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)
    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i, all_cooked_time, all_cooked_bw,
                                       net_params_queues[i],
                                       exp_queues[i], shared_rebuf_penalty)))
    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()
    print('Central Agent finished')
    for i in range(NUM_AGENTS):
        agents[i].terminate()
        agents[i].join()
    print('Agents finished.')


if __name__ == '__main__':
    main()
