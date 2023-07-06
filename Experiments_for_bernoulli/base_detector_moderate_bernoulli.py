import numpy as np
from drift_detection_for_bernoulli.cusum import CUSUM
from drift_detection_for_bernoulli.ddm import DDM
from drift_detection_for_bernoulli.eddm import EDDM
from drift_detection_for_bernoulli.ewma import EWMA
from drift_detection_for_bernoulli.hddm_a import HDDM_A_test
from drift_detection_for_bernoulli.hddm_w import HDDM_W_test
from drift_detection_for_bernoulli.page_hinkley import PH
from drift_detection_for_bernoulli.rddm import RDDM
from drift_detection_for_bernoulli.seq_drift2 import SeqDrift2ChangeDetector
from drift_detection_for_bernoulli.fhddm import FHDDM
from collections import OrderedDict

# Initialize base detectors
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

repeat_count = 30
concept_length = 5000
m_fp = OrderedDict()
m_dt = OrderedDict()
m_tp = OrderedDict()
m_mcc = OrderedDict()
m_precision = OrderedDict()
std_fp = OrderedDict()
std_dt = OrderedDict()
std_tp = OrderedDict()
std_mcc = OrderedDict()
std_precision = OrderedDict()

x = [0.2, 0.8]
mean_values = []
for i in range(30):
    mean_values.extend(x)
np.random.seed(1)
for detector in d:
    mean_fp = []
    mean_tp = []
    mean_dt = []
    mean_mcc = []
    mean_precision = []
    mean_fn = []
    mean_tn = []
    for r_c in range(repeat_count):
        detector.reset()
        # Create data streams
        data_stream_testing = []
        for i in mean_values:
            d = np.random.binomial(n=1, p=i, size=5000)
            data_stream_testing.extend(list(d))
        true_change_testing = [i for i in range(5000, 300000, 10000)]
        for c in true_change_testing:
            for j in range(500):
                data_stream_testing[c + j + 1] = np.random.binomial(n=1, p=0.2 + 0.6 / 500 * (j + 1), size=1)[0]
        start_point = 0
        tolerance_length = 1000
        total_false_positive = []
        average_time_delay = []
        true_positive = 0
        for t_c in true_change_testing:
            detector.reset()
            flag = False
            false_positive = 0
            delay_time = 0
            for i in range(start_point, t_c + tolerance_length):
                drift_status = detector.run(data_stream_testing[i])[1]

                if drift_status:
                    detector.reset()
                    if i < t_c:
                        false_positive += 1
                    else:
                        if flag is False:
                            delay_time = i - t_c
                            print(i, t_c)
                            flag = True
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
        mean_fp.append(sum(total_false_positive))
        mean_dt.append(average_time_delay)
        mean_tp.append(true_positive)
        mean_mcc.append(mcc)
        precision = true_positive / a
        mean_precision.append(precision)
    std_fp[detector.DETECTOR_NAME] = np.std(mean_fp)
    std_tp[detector.DETECTOR_NAME] = np.std(mean_tp)
    std_dt[detector.DETECTOR_NAME] = np.std(mean_dt)
    std_mcc[detector.DETECTOR_NAME] = np.std(mean_mcc)
    std_precision[detector.DETECTOR_NAME] = np.std(mean_precision)
    mean_fp = np.mean(mean_fp)
    mean_tp = np.mean(mean_tp)
    mean_dt = np.mean(mean_dt)
    mean_mcc = np.mean(mean_mcc)
    mean_precision = np.mean(mean_precision)

    m_fp[detector.DETECTOR_NAME] = mean_fp
    m_tp[detector.DETECTOR_NAME] = mean_tp
    m_dt[detector.DETECTOR_NAME] = mean_dt
    m_precision[detector.DETECTOR_NAME] = mean_precision
    m_mcc[detector.DETECTOR_NAME] = mean_mcc


f = open('base_detector_moderate_bernoulli.txt','a')
for i,j in m_fp.items():
    f.write(f'\n{i}: mean_fp {j},\n')
for i,j in std_fp.items():
    f.write(f'\n{i}: std_fpr {j},\n')
for i,j in m_tp.items():
    f.write(f'\n{i}: mean_tp {j},\n')
for i,j in std_tp.items():
    f.write(f'\n{i}: std_tp {j},\n')
for i,j in m_dt.items():
    f.write(f'\n{i}: mean_dt {j},\n')
for i,j in std_dt.items():
    f.write(f'\n{i}: std_dt {j},\n')
for i,j in m_mcc.items():
    f.write(f'\n{i}: mean_tp {j},\n')
for i,j in std_mcc.items():
    f.write(f'\n{i}: std_tp {j},\n')
for i,j in m_precision.items():
    f.write(f'\n{i}: mean_dt {j},\n')
for i,j in std_precision.items():
    f.write(f'\n{i}: std_dt {j},\n')
f.close()

