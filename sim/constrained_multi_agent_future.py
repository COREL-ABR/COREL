import datetime
import os, time
import logging
import numpy as np
import multiprocessing as mp

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import tensorflow as tf
import env_future_bandwidth as env
import a3c_future
import load_trace
import sys

# command
# python multi_agent.py
# -REBUF_PENALTY, float/int
# -tag, string, u*t*scale*gumbel*rtt* depend on REBUF_PENALTY, trace, state_scale, use_gumbel_noise, rm_rtt
# -TRAIN_TRACES, string, eg, './cooked_traces/'
# -NN_MODEL, string, first=None, others=SUMMARY_DIR+'/nn_model_best_constrained.ckpt'
# -MAX_EPOCH, int, 2000
# -ENTROPY_WEIGHT, float, [2, 1, 0.5, 0.25, 0.1, 0.05, 0.01] #[5, 4, 3, 2, 1, 0.5, 0.01]
# -state_scale, float, 1 or 100, default 1
# -use_gumbel_noise, 1 or 0, default 0
# -rm_rtt, 1 or 0, default 0
# -use_true_stall, 1 or 0, default 1
# -mu_update_step, int, mu will update every mu_update_step * MODEL_SAVE_INTERVAL(100) steps
# -epsilon, float, mu_history[-1] + 0.1 * (best_valid_stall - epsilon), range [0.01, 0.05, 0.1, 0.5, 1]

S_INFO = 7 # 6 # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
Penalty_LR_RATE = 0.001
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 100  # take as a train batch
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
REBUF_PENALTY = float(sys.argv[1])  # 4.3  # 1 sec rebuffering -> 3 Mbps
tag = sys.argv[2]
TRAIN_TRACES = sys.argv[3]  #'./cooked_traces/'
# NN_MODEL = './results/pretrain_linear_reward.ckpt'
# NN_MODEL = None
NN_MODEL = sys.argv[4]
MAX_EPOCH = int(sys.argv[5])  # 20000  #
ENTROPY_WEIGHT = float(sys.argv[6])
state_scale = float(sys.argv[7])  # 100 in 5G, 1 in 3G original pensieve paper.
use_gumbel_noise = int(sys.argv[8])
rm_rtt = int(sys.argv[9])
use_true_stall = int(sys.argv[10])
SUMMARY_DIR = './results_future_' + tag  # future_constrained
LOG_FILE = SUMMARY_DIR + '/entropy' + str(ENTROPY_WEIGHT) + 'log'
#LOG_FILE = './results_future_constrained/log'
TEST_LOG_FOLDER = './test_results_future_' + tag +'/'  #future_constrained/
future_steps = 4
use_stall_duration = 1
mu_update_step = int(sys.argv[11])
epsilon = float(sys.argv[12])
PATIENCE = 2 * mu_update_step * MODEL_SAVE_INTERVAL #1000


# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)


def testing(epoch, nn_model, log_file, rebuf_penalty):
    # clean up the test results folder
    os.system('rm -r ' + TEST_LOG_FOLDER)
    if not os.path.exists(TEST_LOG_FOLDER):
        os.makedirs(TEST_LOG_FOLDER)
    # run test script
    os.system('python rl_test_constrained_future_v2.py ' + str(rebuf_penalty) + ' ' + tag +
              ' ' + TRAIN_TRACES[:-1] + '_test/' + ' ' + nn_model + ' ' + str(ENTROPY_WEIGHT) +
              ' ' + str(state_scale) + ' ' + str(use_gumbel_noise) + ' ' + str(rm_rtt))

    # append test performance to the log
    rewards, entropies = [], []
    stalls = []
    normalized_stalls = []
    bitrates = []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward, entropy = [], []
        stall = []
        bitrate = []
        #start_time = None
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
                    #cur_time = float(parse[0])
                    #if len(entropy) == 1:
                    #    start_time = cur_time
                except IndexError:
                    break
        #trace_time = cur_time - start_time
        #normalized_stalls.append(np.sum(stall[1:]) / trace_time)
        normalized_stalls.append(np.sum(stall[1:]) / (CHUNK_TIL_VIDEO_END_CAP*2))
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

    stalls = np.array(stalls)
    stalls_min = np.min(stalls)
    stalls_5per = np.percentile(stalls, 5)
    stalls_mean = np.mean(stalls)
    stalls_median = np.percentile(stalls, 50)
    stalls_95per = np.percentile(stalls, 95)
    stalls_max = np.max(stalls)

    log_file.write('trace total stalls: ' + '\t' +
                   str(stalls_min) + '\t' +
                   str(stalls_5per) + '\t' +
                   str(stalls_mean) + '\t' +
                   str(stalls_median) + '\t' +
                   str(stalls_95per) + '\t' +
                   str(stalls_max) + '\n')

    normalized_stalls = np.array(normalized_stalls)
    normalized_stalls_min = np.min(normalized_stalls)
    normalized_stalls_5per = np.percentile(normalized_stalls, 5)
    normalized_stalls_mean = np.mean(normalized_stalls)
    normalized_stalls_median = np.percentile(normalized_stalls, 50)
    normalized_stalls_95per = np.percentile(normalized_stalls, 95)
    normalized_stalls_max = np.max(normalized_stalls)

    log_file.write('trace normalized total stall: ' + '\t' +
                   str(normalized_stalls_min) + '\t' +
                   str(normalized_stalls_5per) + '\t' +
                   str(normalized_stalls_mean) + '\t' +
                   str(normalized_stalls_median) + '\t' +
                   str(normalized_stalls_95per) + '\t' +
                   str(normalized_stalls_max) + '\n')

    bitrates = np.array(bitrates)
    bitrates_min = np.min(bitrates)
    bitrates_5per = np.percentile(bitrates, 5)
    bitrates_mean = np.mean(bitrates)
    bitrates_median = np.percentile(bitrates, 50)
    bitrates_95per = np.percentile(bitrates, 95)
    bitrates_max = np.max(bitrates)

    log_file.write('trace average bitrate: ' + '\t' +
                   str(bitrates_min) + '\t' +
                   str(bitrates_5per) + '\t' +
                   str(bitrates_mean) + '\t' +
                   str(bitrates_median) + '\t' +
                   str(bitrates_95per) + '\t' +
                   str(bitrates_max) + '\n')
    log_file.flush()

    return rewards_mean, np.mean(entropies), stalls_mean, bitrates_mean, normalized_stalls_mean


def central_agent(net_params_queues, exp_queues):
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)

    with tf.compat.v1.Session() as sess, open(LOG_FILE + '_test', 'w') as test_log_file:

        actor = a3c_future.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE, entropy_weight=ENTROPY_WEIGHT, future_step=future_steps)
        critic = a3c_future.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE, future_step=future_steps)
        penalty = a3c_future.PenaltyNetwork(sess,
                                     state_dim=[S_INFO, S_LEN],
                                     learning_rate=Penalty_LR_RATE, future_step=future_steps)

        summary_ops, summary_vars = a3c_future.build_summaries()

        sess.run(tf.compat.v1.global_variables_initializer())
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
        mu_history = [REBUF_PENALTY]
        while True:
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            penalty_net_params = penalty.get_network_params()
            #print("sending parameters %s to worker agents" % [actor_net_params, critic_net_params, penalty_net_params, mu_history[-1]])
            for i in range(NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params, penalty_net_params, mu_history[-1]])
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
                    a3c_future.compute_gradients_with_penalty(
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
            avg_u = total_u / total_batch_len  # fixme: divided by total_agents or total_batch_len

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

            if (epoch - 1) % MODEL_SAVE_INTERVAL == 0:
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_latest.ckpt")
                test_avg_reward, test_avg_entropy, test_avg_stalls, test_avg_bitrate, test_avg_normalized_stall = \
                    testing(epoch, SUMMARY_DIR + "/nn_model_latest.ckpt", test_log_file, mu_history[-1])
                valid_score = test_avg_reward #test_avg_bitrate + test_avg_stalls
                if valid_score > best_valid_reward:
                    best_valid_reward = valid_score #test_avg_reward
                    best_epoch = epoch
                    best_valid_stall = test_avg_stalls #test_avg_normalized_stall # test_avg_stalls fixme: use normalized one or not
                    # save the best model
                    save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_best_constrained.ckpt")
                    logging.info("Model saved in file: " + save_path)
                    logging.info("Best test average reward, average stall_sum, average normalized_stall, average bitrate: " +
                                 str(test_avg_reward) + '\t' + str(test_avg_stalls) + '\t' +
                                 str(test_avg_normalized_stall)+ '\t' + str(test_avg_bitrate))
                    logging.info("train average reward_sum, average stall: " +  str(avg_reward) +'\t' + str(avg_u))
                    print('best_epoch: ', best_epoch, '. Saved best model')

                if epoch > 1 and (epoch - 1) % (MODEL_SAVE_INTERVAL * mu_update_step) == 0:
                    new_mu = max(mu_history[-1] + 0.1 * (best_valid_stall - epsilon), 0)  # avg_u
                    print('mu_history[-1]: ', mu_history[-1], 'best_valid_stall: ', best_valid_stall, 'new_mu: ', new_mu)
                    mu_history.append(new_mu)
                    print(mu_history)
                    logging.info("new mu, old mu: " + str(new_mu) + '\t' + str(mu_history[-2]))
                    best_valid_reward = -99999999

            if (epoch - best_epoch > PATIENCE and abs(mu_history[-1] - mu_history[-2]) < 0.01 and len(mu_history) > 1) or epoch >= MAX_EPOCH:
                #if ENTROPY_WEIGHT == 0.01:
                #new_mu = max(mu_history[-1] + 0.1 * (best_valid_stall - 0.05), 0)  # avg_u
                #if new_mu != mu_history[-1]:
                #    mu_history.append(new_mu)
                print('mu_history', mu_history)
                with open(SUMMARY_DIR + '/updated_mu.txt', 'w') as f:
                    for u in mu_history:
                        print('u', u)
                        f.write(str(u) + '\n')
                f.close()
                break
            '''if len(mu_history)>=2 and abs(mu_history[-1] - mu_history[-2]) < 0.01:
                with open(SUMMARY_DIR + '/updated_mu.txt', 'w') as f:
                    for u in mu_history:
                        f.write(str(u) + '\n')
                        f.close()
                break'''


def agent(agent_id, all_cooked_time, all_cooked_bw, net_params_queue, exp_queue):
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              random_seed=agent_id)

    #with tf.compat.v1.Session() as sess, open(LOG_FILE + '_agent_' + str(agent_id), 'w') as log_file:
    with tf.compat.v1.Session() as sess:
        actor = a3c_future.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE, entropy_weight=ENTROPY_WEIGHT, future_step=future_steps)
        critic = a3c_future.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE, future_step=future_steps)
        penalty = a3c_future.PenaltyNetwork(sess,
                                     state_dim=[S_INFO, S_LEN],
                                     learning_rate=Penalty_LR_RATE, future_step=future_steps)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params, penalty_net_params, rebuf_penalty = net_params_queue.get()
        #print("getting parameters %s from central" % [actor_net_params, critic_net_params, penalty_net_params, rebuf_penalty])
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
                     - rebuf_penalty * rebuf \
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
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality, normalized (0,1)
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec, buffer_size
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K / state_scale  # kilo byte / ms bandwidth_measurement
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K / state_scale  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)  # normalized
            state[6] = [0] * S_LEN
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

            entropy_record.append(a3c_future.compute_entropy(action_prob[0]))

            # log time_stamp, bit_rate, buffer_size, reward
            #log_file.write(str(time_stamp) + '\t' +
            #               str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
            #              str(buffer_size) + '\t' +
            #               str(rebuf) + '\t' +
            #               str(video_chunk_size) + '\t' +
            #               str(delay) + '\t' +
            #               str(reward) + '\n')
            #log_file.flush()

            # report experience to the coordinator
            if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:
                exp_queue.put([s_batch[1:],  # ignore the first chuck
                               a_batch[1:],  # since we don't have the
                               r_batch[1:],  # control over it
                               stall_batch[1:],
                               end_of_video,
                               {'entropy': entropy_record}])

                # synchronize the network parameters from the coordinator
                actor_net_params, critic_net_params, penalty_net_params, rebuf_penalty = net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)
                penalty.set_network_params(penalty_net_params)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del entropy_record[:]
                del stall_batch[:]

                #log_file.write('\n')  # so that in the log we know where video ends

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
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)
    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i, all_cooked_time, all_cooked_bw,
                                       net_params_queues[i],
                                       exp_queues[i])))
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
