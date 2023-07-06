"""
By Shuxiang Zhang
University of Auckland, Auckland, New Zealand
E-mail: szha861@aucklanduni.ac.nz

---
*** The DDE Implementation ***
Paper: B.I.F.Maciel, S.G.T.C.Santos and R.S.M. Barros. " A lightweight concept drift detection ensemble"
Published in: Proceedings of the IEEE 27th International Conference on Tools with Artificial Intelligence (ICTAI), 2015.
"""

import numpy as np
from collections import OrderedDict
from drift_detection.cusum import CUSUM
from drift_detection.ddm import DDM
from drift_detection.eddm import EDDM
from drift_detection.ewma import EWMA
from drift_detection.fhddm import FHDDM
from drift_detection.hddm_a import HDDM_A_test
from drift_detection.hddm_w import HDDM_W_test
from drift_detection.page_hinkley import PH
from drift_detection.rddm import RDDM
from drift_detection.seq_drift2 import SeqDrift2ChangeDetector
from drift_detection.adwin import ADWINChangeDetector
from drift_detection.dde import DDE

# Initializing detectors
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

d_1 = [hddm_a,hddm_w,ddm]
d_2 = [hddm_a,hddm_w,ewma]
dde_1 = DDE(base_detector=d_1,sens=1)
dde_2 = DDE(base_detector=d_2,sens=2)


d = [dde_1, dde_2]

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

for detector in d:
    mean_fp = []
    mean_tp = []
    mean_dt = []
    mean_mcc = []
    mean_precision = []
    for r_c in range(repeat_count):
        detector.reset()
        np.random.seed(r_c)
        # Create data streams
        data_stream = []
        for i in mean_values:
            d = np.random.binomial(n=1, p=i, size=5000)
            data_stream.extend(list(d))
        # Specify true change points
        true_change = [i for i in range(5000, 300000, 10000)]

        start_point = 0
        tolerance_length = 1000
        total_false_positive = []
        average_time_delay = []
        true_positive = 0
        for t_c in true_change:
            detector.reset()
            flag = False
            false_positive = 0
            delay_time = 0
            for i in range(start_point, t_c + tolerance_length):
                drift_status = detector.run(data_stream[i])[1]

                if drift_status:
                    print(f'Change was detected by {detector.DETECTOR_NAME} at index {i}')
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


f = open('dde_abrupt_bernoulli','a')
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

