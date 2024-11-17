import os
import random

trace_folder='./sufficient_5G/'
trace_files = os.listdir(trace_folder)

def copy_files(train_dir, test_dir, test_dir1, train_files, test_files):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(test_dir1):
        os.makedirs(test_dir1)
    for train_file in train_files:
        os.system('cp ' + trace_folder + train_file + ' ' + train_dir)
    for test_file in test_files:
        os.system('cp ' + trace_folder + test_file + ' ' + test_dir)
        os.system('cp ' + trace_folder + test_file + ' ' + test_dir1)


# randomly split
train_dir = "../sim/5G_cooked_random/"
test_dir = train_dir[:-1]+ '_test/'
test_dir1 = "../test/5G_cooked_random_test/"
random_train = random.sample(trace_files, int(len(trace_files) * 0.8))
random_test = [item for item in trace_files if item not in random_train]
copy_files(train_dir, test_dir, test_dir1, random_train, random_test)

walking_files = [item for item in trace_files if 'walking' in item]
driving_files = [item for item in trace_files if 'driving' in item]
# train on driving, test on walking
train_dir = "../sim/5G_cooked_driving/"
test_dir = train_dir[:-1]+ '_test/'
test_dir1 = "../test/5G_cooked_driving_test/"
copy_files(train_dir, test_dir, test_dir1, driving_files, walking_files)
# train on walking, test on driving
train_dir = "../sim/5G_cooked_walking/"
test_dir = train_dir[:-1]+ '_test/'
test_dir1 = "../test/5G_cooked_walking_test/"
copy_files(train_dir, test_dir, test_dir1, walking_files, driving_files)

# test on high fluctuation: trace with std >= 413.2
hightSTD_files = []
with open('./sufficient_5G_statistics/high_variance_trace_names') as inf:
    for line in inf:
        hightSTD_files.append(line.strip())
others_files = [item for item in trace_files if item not in hightSTD_files]
train_dir = "../sim/5G_cooked_hightSTD/"
test_dir = train_dir[:-1]+ '_test/'
test_dir1 = "../test/5G_cooked_hightSTD_test/"
copy_files(train_dir, test_dir, test_dir1, others_files, hightSTD_files)


# test on small variation: traces with std <= 145
lowSTD_files = []
with open('./sufficient_5G_statistics/low_variance_trace_names') as inf:
    for line in inf:
        lowSTD_files.append(line.strip())
others_files = [item for item in trace_files if item not in lowSTD_files]
train_dir = "../sim/5G_cooked_lowSTD/"
test_dir = train_dir[:-1]+ '_test/'
test_dir1 = "../test/5G_cooked_lowSTD_test/"
copy_files(train_dir, test_dir, test_dir1, others_files, lowSTD_files)

# test on Deadzone traces
deadzone_files = []
with open('./sufficient_5G_statistics/deadzone_trace_names') as inf:
    for line in inf:
        deadzone_files.append(line.strip())
others_files = [item for item in trace_files if item not in deadzone_files]
train_dir = "../sim/5G_cooked_deadzone/"
test_dir = train_dir[:-1]+ '_test/'
test_dir1 = "../test/5G_cooked_deadzone_test/"
copy_files(train_dir, test_dir, test_dir1, others_files, deadzone_files)
