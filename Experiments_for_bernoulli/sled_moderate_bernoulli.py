import numpy as np
from pympler import asizeof
from drift_detection_for_bernoulli.cusum import CUSUM
from drift_detection_for_bernoulli.ddm import DDM
from drift_detection_for_bernoulli.eddm import EDDM
from drift_detection_for_bernoulli.ewma import EWMA
from drift_detection_for_bernoulli.fhddm import FHDDM
from drift_detection_for_bernoulli.hddm_a import HDDM_A_test
from drift_detection_for_bernoulli.hddm_w import HDDM_W_test
from drift_detection_for_bernoulli.page_hinkley import PH
from drift_detection_for_bernoulli.rddm import RDDM
from drift_detection_for_bernoulli.seq_drift2 import SeqDrift2ChangeDetector
from drift_detection_for_bernoulli.adwin import ADWINChangeDetector
from drift_detection.sled import Sled

# The function to normalize false positives and delay time

# Initialize base detectors
adwin = ADWINChangeDetector()
ddm = DDM()
hddm_w = HDDM_W_test()
cusum = CUSUM()
eddm = EDDM()
rddm = RDDM()
ewma = EWMA()
ph = PH()
hddm_a = HDDM_A_test()
fhddm = FHDDM()
seq_drift2 = SeqDrift2ChangeDetector()
d = [ddm, hddm_w, cusum, eddm, rddm, ewma, ph, hddm_a, seq_drift2, fhddm]
sled = Sled(d)

fusion_parameter = 0.5
smoothing_factor = 0.5
concept_length = 5000
drift_acceptance_interval = 1000

x = [0.2, 0.8]
mean_values = []
for i in range(100):
    mean_values.extend(x)
#
np.random.seed(1)
# Create training data stream
data_stream_training = []
for i in mean_values:
    d_s = np.random.binomial(n=1, p=i, size=5000)
    data_stream_training.extend(list(d_s))


true_change_training = [i for i in range(5000, 1000000, 10000)]
for c in true_change_training:
    for j in range(500):
        data_stream_training[c + j + 1] = np.random.binomial(n=1, p=0.2+0.6/500 * (j + 1), size=1)[0]


# Training

sled.train_ber(smoothing_factor=smoothing_factor, fusion_parameter=fusion_parameter,
               drift_acceptance_interval=drift_acceptance_interval, concept_length=concept_length,
               data_stream=data_stream_training)

# Detection

repeat_count = 30
average_fp_rate = []
mean_average_dt = []
mean_tp_rate = []
m_mem = []
m_runtime = []
mean_fn = []
mean_tn = []
mean_mcc = []
mean_pre = []
mean_re = []
true_change_testing = [i for i in range(5000, 300000, 10000)]
tolerance_length = 1000
for r_c in range(repeat_count):
    sled.TOTAL_RUNTIME = 0
    sled.reset()
    sled.current_time = -1
    data_stream_testing = []
    for i in mean_values[:60]:
        d_s = np.random.binomial(n=1, p=i, size=5000)
        data_stream_testing.extend(list(d_s))
    for c in true_change_testing:
        for j in range(500):
            data_stream_testing[c + j + 1] = np.random.binomial(n=1, p=0.2 + 0.6 / 500 * (j + 1), size=1)[0]
    start_point = 0
    total_false_positive = []
    average_time_delay = []
    average_mem = []
    average_runtime = []
    true_positive = 0
    for t_c in true_change_testing:
        sled.TOTAL_RUNTIME = 0
        sled.reset()
        flag = False
        false_positive = 0
        delay_time = 0
        for i in range(start_point, t_c + tolerance_length):
            drift_status = sled.detect(data_stream_testing[i])[1]
            if drift_status:
                average_mem.append(asizeof.asizeof(sled, limit=20))
                sled.reset()
                if i < t_c:
                    false_positive += 1
                else:
                    if flag is False:
                        delay_time = i - t_c
                        flag = True

        average_runtime.append(sled.TOTAL_RUNTIME)
        start_point = t_c + concept_length
        total_false_positive.append(false_positive)
        average_time_delay.append(delay_time)
        if flag is True:
            true_positive += 1

    false_negative = 30 - true_positive
    true_negative = 300000 - 30 - sum(total_false_positive)
    a = true_positive + sum(total_false_positive)
    b = true_positive + false_negative
    c = true_negative + sum(total_false_positive)
    d = true_negative + false_negative
    mcc = (true_positive * true_negative - sum(total_false_positive) * false_negative) / np.sqrt(a * b * c * d)
    average_time_delay = np.mean(average_time_delay)
    average_fp_rate.append(sum(total_false_positive))
    mean_average_dt.append(average_time_delay)
    mean_tp_rate.append(true_positive)
    mean_mcc.append(mcc)
    precision = true_positive / a
    mean_pre.append(precision)

std_fp_rate = np.std(average_fp_rate)
average_fp_rate = np.mean(average_fp_rate)
std_dt = np.std(mean_average_dt)
mean_average_dt = np.mean(mean_average_dt)
std_tp = np.std(mean_tp_rate)
mean_tp_rate = np.mean(mean_tp_rate)
std_mcc = np.std(mean_mcc)
m_mcc = np.mean(mean_mcc)
std_precision = np.std(mean_pre)
m_precision = np.mean(mean_pre)

f = open('sled_moderate_bernoulli', 'a')
f.write(f'sled: \n mean_fp {average_fp_rate}, std_fpr {std_fp_rate}\n mean_tp {mean_tp_rate}, std_tp {std_tp}\n mean_dt {mean_average_dt}, std_dt {std_dt}')
f.write(f'sled: \n mean_mcc {m_mcc}, std_mcc {std_mcc}\n')
f.write(f'sled: \n mean_precision {m_precision}, std_precision {std_precision}')
f.close()



