import numpy as np
import fixed_env as env
import load_trace
import itertools
import sys
import os


S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
# VIDEO_BIT_RATE = [1500, 4900, 8200, 11700, 32800, 152400] # [300,750,1200,1850,2850,4300]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 120.0 # 48.0
TOTAL_VIDEO_CHUNKS = 120 # 48
M_IN_K = 1000.0
REBUF_PENALTY = float(sys.argv[1]) #160.0 # 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
tag = sys.argv[2]
SUMMARY_DIR = './results_' + tag
#SUMMARY_DIR = './results'
#LOG_FILE = './results/log_sim_mpc'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = './models/nn_model_ep_5900.ckpt'
TRACES = sys.argv[3]  #'./cooked_traces/'
MPC_FUTURE_CHUNK_COUNT = int(sys.argv[4])  # default 5
ROBUST_MPC = int(sys.argv[5])
fiveG = int(sys.argv[6])


if fiveG == 0:
    # VIDEO_BIT_RATE = [1500, 4900, 8200, 11700, 32800, 152400] # [300,750,1200,1850,2850,4300]  # Kbps
    VIDEO_BIT_RATE = [2500, 5000, 8000, 75000, 150000, 300000]  # 5g Wei # Kbps
else:
    VIDEO_BIT_RATE = [1500, 2500, 40710, 152660, 280000, 585000]  # 5g Wei # Kbps

if ROBUST_MPC == 0:
    LOG_FILE = SUMMARY_DIR + '/log_sim_mpc'
else:
    LOG_FILE = SUMMARY_DIR + '/log_sim_robustmpc'

VIDEO_SIZE_MULTIPLIER = 1
# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)
    
CHUNK_COMBO_OPTIONS = []

# past errors in bandwidth
past_errors = []
past_bandwidth_ests = []

# pensieve original video chunk size
#size_video1 = [3155849, 2641256, 2410258, 2956927, 2593984, 2387850, 2554662, 2964172, 2541127, 2553367, 2641109, 2876576, 2493400, 2872793, 2304791, 2855882, 2887892, 2474922, 2828949, 2510656, 2544304, 2640123, 2737436, 2559198, 2628069, 2626736, 2809466, 2334075, 2775360, 2910246, 2486226, 2721821, 2481034, 3049381, 2589002, 2551718, 2396078, 2869088, 2589488, 2596763, 2462482, 2755802, 2673179, 2846248, 2644274, 2760316, 2310848, 2647013, 1653424]
#size_video1 = [2354772, 2123065, 2177073, 2160877, 2233056, 1941625, 2157535, 2290172, 2055469, 2169201, 2173522, 2102452, 2209463, 2275376, 2005399, 2152483, 2289689, 2059512, 2220726, 2156729, 2039773, 2176469, 2221506, 2044075, 2186790, 2105231, 2395588, 1972048, 2134614, 2164140, 2113193, 2147852, 2191074, 2286761, 2307787, 2143948, 1919781, 2147467, 2133870, 2146120, 2108491, 2184571, 2121928, 2219102, 2124950, 2246506, 1961140, 2155012, 1433658]
#size_video2 = [1728879, 1431809, 1300868, 1520281, 1472558, 1224260, 1388403, 1638769, 1348011, 1429765, 1354548, 1519951, 1422919, 1578343, 1231445, 1471065, 1491626, 1358801, 1537156, 1336050, 1415116, 1468126, 1505760, 1323990, 1383735, 1480464, 1547572, 1141971, 1498470, 1561263, 1341201, 1497683, 1358081, 1587293, 1492672, 1439896, 1139291, 1499009, 1427478, 1402287, 1339500, 1527299, 1343002, 1587250, 1464921, 1483527, 1231456, 1364537, 889412]
#size_video3 = [1034108, 957685, 877771, 933276, 996749, 801058, 905515, 1060487, 852833, 913888, 939819, 917428, 946851, 1036454, 821631, 923170, 966699, 885714, 987708, 923755, 891604, 955231, 968026, 874175, 897976, 905935, 1076599, 758197, 972798, 975811, 873429, 954453, 885062, 1035329, 1026056, 943942, 728962, 938587, 908665, 930577, 858450, 1025005, 886255, 973972, 958994, 982064, 830730, 846370, 598850]
#size_video4 = [668286, 611087, 571051, 617681, 652874, 520315, 561791, 709534, 584846, 560821, 607410, 594078, 624282, 687371, 526950, 587876, 617242, 581493, 639204, 586839, 601738, 616206, 656471, 536667, 587236, 590335, 696376, 487160, 622896, 641447, 570392, 620283, 584349, 670129, 690253, 598727, 487812, 575591, 605884, 587506, 566904, 641452, 599477, 634861, 630203, 638661, 538612, 550906, 391450]
#size_video5 = [450283, 398865, 350812, 382355, 411561, 318564, 352642, 437162, 374758, 362795, 353220, 405134, 386351, 434409, 337059, 366214, 360831, 372963, 405596, 350713, 386472, 399894, 401853, 343800, 359903, 379700, 425781, 277716, 400396, 400508, 358218, 400322, 369834, 412837, 401088, 365161, 321064, 361565, 378327, 390680, 345516, 384505, 372093, 438281, 398987, 393804, 331053, 314107, 255954]
#size_video6 = [181801, 155580, 139857, 155432, 163442, 126289, 153295, 173849, 150710, 139105, 141840, 156148, 160746, 179801, 140051, 138313, 143509, 150616, 165384, 140881, 157671, 157812, 163927, 137654, 146754, 153938, 181901, 111155, 153605, 149029, 157421, 157488, 143881, 163444, 179328, 159914, 131610, 124011, 144254, 149991, 147968, 161857, 145210, 172312, 167025, 160064, 137507, 118421, 112270]

# 8k video chunk size, chunk = 2s
size_video6 = [262817, 400734, 397064, 401798, 369983, 369873, 375448, 364376, 378591, 377157, 360160, 394026, 393535, 270281, 402376, 382453, 387391, 397870, 374480, 208523, 634440, 510330, 327017, 127764, 542900, 498035, 498624, 458184, 424074, 541938, 475109, 353325, 384065, 454134, 392263, 315173, 380243, 397310, 280338, 262359, 243411, 169887, 197710, 213123, 290828, 489686, 367753, 328924, 319876, 339525, 318190, 485665, 533490, 325468, 344625, 281055, 308912, 470691, 359137, 346232, 413002, 340597, 317155, 231420, 284253, 393112, 580962, 512512, 487212, 440417, 187305, 242343, 329516, 447220, 451921, 256076, 245142, 264121, 252169, 284180, 355468, 301002, 322548, 319631, 547116, 603846, 431043, 425477, 416456, 602670, 629000, 656659, 474975, 405676, 295521, 317058, 334698, 525832, 619681, 570372, 347494, 301221, 337753, 337969, 357082, 346907, 425302, 392750, 412569, 308786, 321014, 398512, 385015, 399189, 412255, 378621, 370250, 185288, 172824, 245954, 38882]
size_video5 = [735197, 1203704, 1229290, 1259341, 1139801, 1131324, 1126494, 1129385, 1190871, 1249216, 1133150, 1274488, 1204331, 808601, 1405479, 1551227, 1254031, 1286110, 1094465, 593126, 1692128, 1315001, 901377, 376360, 1679905, 1893581, 1985165, 1550514, 1495220, 1800960, 1627188, 1175038, 1170130, 1540976, 1338991, 1005869, 1213495, 1314564, 1076582, 994311, 930705, 625665, 684707, 686127, 939904, 1622601, 1092078, 1023769, 1035011, 1087587, 960727, 1516643, 1950108, 1024048, 1011933, 836152, 943007, 1594181, 1040452, 1129903, 1218384, 1070676, 961505, 815736, 926068, 1283877, 2093273, 1783052, 1708178, 1506493, 657866, 867872, 1016202, 1732912, 1764752, 797593, 746899, 819795, 817764, 945697, 1049335, 875829, 880824, 884645, 1686141, 2086134, 1294214, 1265797, 1215627, 1776673, 2228681, 2149181, 1651334, 1484672, 950579, 964600, 1052009, 1476309, 1849917, 1818499, 972892, 936929, 969347, 1085954, 1035947, 1024040, 1520432, 1690608, 1395704, 937697, 1012367, 1177124, 1114352, 1196283, 1255837, 1158803, 1189325, 541237, 594932, 861744, 119467]
size_video4 = [1289956, 2025117, 2045684, 2169702, 1877874, 1843058, 1897234, 1934209, 1962315, 1980790, 1909056, 2108450, 1947139, 1217782, 2262799, 2225243, 2425780, 2558437, 1852593, 879766, 2340266, 2186272, 1445741, 574526, 2715348, 3187700, 3493710, 2847028, 2572878, 3093270, 2951173, 2036939, 2102291, 2548768, 2316134, 1928347, 2383729, 2365118, 1711345, 1621790, 1506357, 951202, 1064916, 1088715, 1510170, 2573321, 1837128, 1595630, 1514114, 1490557, 1363470, 2417601, 2931779, 1555303, 1591609, 1361062, 1576800, 2622868, 1808580, 2036621, 2253757, 1795191, 1693847, 1432371, 1682148, 2350485, 2961252, 3228034, 2798041, 2571849, 1083179, 1467018, 1662920, 2986653, 3291238, 1442700, 1256182, 1395445, 1190383, 1476593, 1755638, 1454426, 1510527, 1502121, 2955604, 3413095, 2328663, 2236487, 2105225, 3122213, 3448620, 3766934, 2861073, 2486722, 1622763, 1696420, 1819262, 2943416, 3209741, 3130288, 1656615, 1550745, 1614772, 1711777, 1784260, 1690232, 2635236, 2674755, 2396840, 1537825, 1733607, 1993850, 1898355, 2041834, 2201832, 2022199, 1991892, 848232, 923309, 1322358, 181940]
size_video3 = [1946512, 2889240, 3028346, 3070290, 2714669, 2586229, 2705687, 2738530, 2779700, 2826967, 2743536, 3018472, 2806808, 1865012, 3307091, 3320404, 3631851, 3576542, 2607043, 1214754, 3132429, 2699727, 1936121, 799879, 3538827, 4605238, 4896400, 4100220, 3578331, 4179409, 4056373, 2696425, 2853370, 3633802, 3379094, 2870301, 3352527, 3378609, 2753762, 2580630, 2275159, 1335006, 1484066, 1587618, 2106563, 3725844, 2712274, 2387279, 2470263, 2316997, 2016922, 3501557, 4131772, 2309027, 2191060, 1860790, 2134703, 3375845, 2556217, 2795099, 2887381, 2452150, 2301224, 2109174, 2422929, 3216581, 4197397, 4512850, 4277383, 3686605, 1560235, 2085260, 2406806, 4338728, 4910069, 2108064, 1780962, 1971084, 1746740, 2164120, 2570670, 2122347, 2163455, 2191971, 3768908, 4417842, 3496910, 3279757, 3056907, 4147598, 5077690, 5538454, 4110195, 3496452, 2360957, 2424413, 2579838, 4018053, 4272792, 3948853, 2431165, 2323254, 2329191, 2563566, 2510654, 2469134, 3894561, 3763341, 3418127, 2261133, 2498503, 2858822, 2656424, 2923693, 3208824, 2859935, 2715223, 1191364, 1285882, 1854181, 265773]
size_video2 = [5007283, 8543637, 8294107, 8823886, 7654555, 7637199, 7839529, 7642655, 7855021, 8144988, 7654599, 8652859, 7639772, 4610473, 8008376, 7279045, 8857450, 9426226, 8570042, 3231189, 8229131, 6935382, 5254905, 2225616, 10381937, 12032554, 13708061, 12142878, 10968078, 13323640, 12301452, 9240614, 10277859, 10687865, 9581134, 8211424, 9171940, 9726924, 6183941, 5425945, 5380841, 4168746, 4693380, 4809905, 6048774, 9505452, 7366875, 6631037, 4892506, 5169674, 4862596, 9278682, 9863179, 6445680, 6244748, 5407083, 6263374, 9535347, 8037249, 8642264, 9126743, 7375437, 7088895, 6430021, 7233981, 9481897, 12123449, 12978969, 11234177, 10848013, 4798008, 5665508, 6298240, 11675904, 13133924, 6598465, 5631374, 5700609, 4583977, 5652490, 7468397, 6301897, 6786494, 6677949, 11552853, 12160676, 10775988, 10086219, 9719696, 13651813, 12595581, 13499480, 10776218, 9837772, 6901969, 7268474, 7497193, 12088196, 12519500, 13177110, 6413821, 5859334, 6832089, 6509972, 6753485, 7025637, 9772651, 10287012, 8850102, 5932128, 6258134, 7632822, 7747930, 8410029, 8907262, 8154223, 8139267, 3985336, 4257583, 5556092, 654840]
size_video1 = [28075703, 30713488, 32519503, 34598504, 37959591, 33371746, 34690382, 33920348, 35046614, 36230899, 34337803, 40027276, 35652095, 20874304, 36430531, 30046085, 38916899, 40671688, 37935589, 11374247, 37716295, 29597670, 21722711, 9375054, 56601860, 54670720, 62883390, 55860705, 51592603, 60104451, 62667759, 50471369, 53635187, 54019168, 51217287, 41528702, 41637734, 42930345, 25708060, 21677623, 22787362, 19128068, 20722432, 21110607, 27520784, 40747140, 34073715, 29817154, 16012521, 20114111, 20895763, 42107900, 40998957, 31215473, 34938519, 27169802, 29189029, 41253179, 45464491, 46173169, 48188667, 44290753, 43409381, 26439912, 33302024, 41324895, 51208250, 55792541, 58154273, 51904788, 23470572, 28687318, 27932254, 47826284, 43649804, 31113221, 30834371, 28565750, 21554534, 27043882, 34573938, 28354533, 29806255, 27786095, 46650094, 51484956, 44795804, 46948950, 46205507, 59294999, 48127482, 66402251, 49382975, 47167419, 32911240, 34149495, 35362513, 65160959, 64399196, 82844336, 33416465, 28687915, 32881638, 31117422, 31517538, 32375760, 44437624, 46361028, 40170609, 29353245, 27771915, 31861161, 33841065, 35454428, 39361285, 35771046, 38404194, 19332664, 21363585, 26739406, 2459417]

def get_chunk_size(quality, index):
    if ( index < 0 or index > TOTAL_VIDEO_CHUNKS ):
        return 0
    # note that the quality and video labels are inverted (i.e., quality 4 is highest and this pertains to video1)
    if fiveG == 0:
        sizes = {5: size_video1[index] * 1.968503937007874,
                 4: size_video2[index] * 4.573170731707317,
                 3: size_video3[index] * 6.410256410256411,
                 2: size_video4[index],
                 1: size_video5[index],
                 0: size_video6[index] * 1.6666666666666667}
    else:
        sizes = {5: size_video1[index]*3.838582677165354,
                 4: size_video2[index]*8.536585365853659,
                 3: size_video3[index]*13.047863247863248,
                 2: size_video4[index]*4.964634146341464,
                 1: size_video5[index]*0.51,
                 0:size_video6[index]*VIDEO_SIZE_MULTIPLIER}
    return sizes[quality]


def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time1, all_cooked_bw1, all_file_names1 = load_trace.load_trace(TRACES)
    variance_trace = {}
    varainces = []
    for i in range(len(all_file_names1)):
        variance = np.std(all_cooked_bw1[i])
        varainces.append(variance)
        variance_trace[variance] = i
    varainces = sorted(varainces, reverse=True)
    top50_traces = []
    all_cooked_time, all_cooked_bw, all_file_names = [], [], []
    for i in range(200):
        all_cooked_time.append(all_cooked_time1[variance_trace[varainces[i]]])
        all_cooked_bw.append(all_cooked_bw1[variance_trace[varainces[i]]])
        all_file_names.append(all_file_names1[variance_trace[varainces[i]]])

    if fiveG == 0:
        net_env = env.Environment(all_cooked_time=all_cooked_time,
                                  all_cooked_bw=all_cooked_bw, fiveG=False)
    else:
        net_env = env.Environment(all_cooked_time=all_cooked_time,
                                  all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []

    video_count = 0

    # make chunk combination options
    for combo in itertools.product(range(A_DIM), repeat=MPC_FUTURE_CHUNK_COUNT):
        CHUNK_COMBO_OPTIONS.append(combo)
    print('CHUNK_COMBO_OPTIONS: ', CHUNK_COMBO_OPTIONS)

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size,_, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                           VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

        # log scale reward
        # log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[0]))
        # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))

        # reward = log_bit_rate \
        #          - REBUF_PENALTY * rebuf \
        #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

        # reward = BITRATE_REWARD[bit_rate] \
        #          - 8 * rebuf - np.abs(BITRATE_REWARD[bit_rate] - BITRATE_REWARD[last_bit_rate])


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
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = rebuf
        state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        # state[5: 10, :] = future_chunk_sizes / M_IN_K / M_IN_K

        # ================== MPC =========================
        curr_error = 0 # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if ( len(past_bandwidth_ests) > 0 ):
            curr_error  = abs(past_bandwidth_ests[-1]-state[3,-1])/float(state[3,-1])
        past_errors.append(curr_error)

        # pick bitrate according to MPC           
        # first get harmonic mean of last 5 bandwidths
        past_bandwidths = state[3,-5:]
        while past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]
        #if ( len(state) < 5 ):
        #    past_bandwidths = state[3,-len(state):]
        #else:
        #    past_bandwidths = state[3,-5:]
        bandwidth_sum = 0
        for past_val in past_bandwidths:
            bandwidth_sum += (1/float(past_val))
        harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))

        # future bandwidth prediction
        # divide by 1 + max of last 5 (or up to 5) errors
        max_error = 0
        error_pos = -5
        if ( len(past_errors) < 5 ):
            error_pos = -len(past_errors)
        max_error = float(max(past_errors[error_pos:]))
        if ROBUST_MPC == 0:
            future_bandwidth = harmonic_bandwidth ## default MPC
        else:
            future_bandwidth = harmonic_bandwidth/(1+max_error)  # robustMPC here
        past_bandwidth_ests.append(harmonic_bandwidth)


        # future chunks length (try 4 if that many remaining)
        last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain)
        future_chunk_length = MPC_FUTURE_CHUNK_COUNT
        if ( TOTAL_VIDEO_CHUNKS - last_index < 5 ):
            future_chunk_length = TOTAL_VIDEO_CHUNKS - last_index

        # all possible combinations of 5 chunk bitrates (9^5 options)
        # iterate over list and for each, compute reward and store max reward combination
        max_reward = -100000000
        best_combo = ()
        start_buffer = buffer_size
        #start = time.time()
        for full_combo in CHUNK_COMBO_OPTIONS:
            combo = full_combo[0:future_chunk_length]
            # calculate total rebuffer time for this combination (start with start_buffer and subtract
            # each download time and add 2 seconds in that order)
            curr_rebuffer_time = 0
            curr_buffer = start_buffer
            bitrate_sum = 0
            smoothness_diffs = 0
            last_quality = int( bit_rate )
            for position in range(0, len(combo)):
                chunk_quality = combo[position]
                index = last_index + position + 1 # e.g., if last chunk is 3, then first iter is 3+0+1=4
                download_time = (get_chunk_size(chunk_quality, index)/1000000.)/future_bandwidth # this is MB/MB/s --> seconds
                if ( curr_buffer < download_time ):
                    curr_rebuffer_time += (download_time - curr_buffer)
                    curr_buffer = 0
                else:
                    curr_buffer -= download_time
                curr_buffer += net_env.VIDEO_CHUNCK_LEN / 1000
                bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
                smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                # bitrate_sum += BITRATE_REWARD[chunk_quality]
                # smoothness_diffs += abs(BITRATE_REWARD[chunk_quality] - BITRATE_REWARD[last_quality])
                last_quality = chunk_quality
            # compute reward for this combination (one reward per 5-chunk combo)
            # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s
            
            reward = (bitrate_sum/1000.) - (REBUF_PENALTY*curr_rebuffer_time) - (smoothness_diffs/1000.)
            # reward = bitrate_sum - (8*curr_rebuffer_time) - (smoothness_diffs)


            if ( reward >= max_reward ):
                if (best_combo != ()) and best_combo[0] < combo[0]:
                    best_combo = combo
                else:
                    best_combo = combo
                max_reward = reward
                # send data to html side (first chunk of best combo)
        send_data = 0 # no combo had reward better than -1000000 (ERROR) so send 0
        if ( best_combo != () ): # some combo was good
            send_data = best_combo[0]


        bit_rate = send_data
        # hack
        # if bit_rate == 1 or bit_rate == 2:
        #    bit_rate = 0

        # ================================================

        # Note: we need to discretize the probability into 1/RAND_RANGE steps,
        # because there is an intrinsic discrepancy in passing single state and batch states

        s_batch.append(state)

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

            print("video count", video_count)
            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()

