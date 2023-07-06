"""
By Shuxiang Zhang
University of Auckland, Auckland, New Zealand
E-mail: szha861@aucklanduni.ac.nz
"""
import numpy as np
import os
from collections import OrderedDict
from data_structures.attribute_scheme import AttributeScheme
from classifier.__init__ import *
from filters.project_creator import Project
from streams.readers.arff_reader import ARFFReader
from tasks.__init__ import *
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

d_1 = [hddm_a, hddm_w, ddm]
d_2 = [hddm_a, hddm_w, ewma]
dde_1 = DDE(base_detector=d_1, sens=1)
dde_2 = DDE(base_detector=d_2, sens=2)
d = [dde_1, dde_2]

m_accuracy = OrderedDict()
m_fp = OrderedDict()
m_dt = OrderedDict()
m_tp = OrderedDict()
m_mcc = OrderedDict()
m_precision = OrderedDict()

stream_name = 'forest'
# 1. Creating a project
project = Project("projects/single", "dde_forest" + stream_name)

# 2. Generating training stream

project_path = "data_streams/dde_forest/" + stream_name + "/"
if not os.path.exists(project_path):
    os.makedirs(project_path)
actual_drift_points = []
raw_actual_drift_points = [365833, 366443, 366636, 367182, 367663, 370935, 371610, 372733, 373504, 373858, 374468, 375110, 375463, 375688, 376426, 376779, 377198, 377680, 378452, 379127, 379706, 379996, 380446, 380671, 381378, 382020, 382567, 383306, 383981, 384399, 385074, 385717, 386070, 386392, 387162, 387643, 388477, 389632, 390402, 391043, 391556, 392005, 392166, 392776, 393419, 394222, 394896, 396243, 396918, 397850, 398300, 398429, 398686, 398975, 399392, 399649, 399938, 400323, 400997, 401286, 401896, 402345, 402666, 403244, 404495, 404848, 406229, 406518, 406936, 407225, 407642, 408992, 409217, 409731, 410406, 411082, 411692, 423218, 424021, 424503, 424856, 438290, 460351, 461282, 462889, 464368, 464721, 466358, 466711, 469981, 471518, 474559, 476288, 476834, 485748, 487193, 488061, 488350, 488447, 488768, 489121, 489218, 489507, 489604, 489925, 490599, 491466, 491563, 492847, 493200, 493746, 494356, 494485, 494678, 495289, 495674, 495771, 496156, 496285, 496800, 497121, 497218, 497347, 497894, 498247, 499306, 500492, 501037, 501518, 501615, 502737, 505078, 505495, 506042, 506299, 506460, 506781, 507038, 507648, 508997, 509543, 510571, 510796, 513553, 514803, 515381, 516474, 516860, 517310, 524789, 535321, 552422, 552647, 553033, 572014, 572783]
for drift in raw_actual_drift_points:
    actual_drift_points.append(drift-365700)
print(actual_drift_points)
for detector in d:
    file_path = "C:/Users/szha861/Desktop/Real_world_datasets/"
    # 3. Loading an arff file
    labels, attributes, stream_records = ARFFReader.read(file_path+"covTesting.arff")
    attributes_scheme = AttributeScheme.get_scheme(attributes)

    # 4. Initializing a Learner
    learner = NaiveBayes(labels, attributes_scheme['nominal'])

    drift_acceptance_interval = 200

    # 5. Creating a Prequential Evaluation Process
    prequential = PrequentialDriftEvaluator(learner, detector, attributes, attributes_scheme,
                                            actual_drift_points, drift_acceptance_interval, project)

    result = prequential.run(stream_records, 1)
    accuracy = (100 - result[0])
    dt = (result[1])
    tp = (result[2])
    fp = (result[3])
    a = result[2] + result[3]
    b = result[2] + (157 - result[2])
    c = (215311 - 157 - result[3]) + result[3]
    d = (215311 - 157 - result[3]) + (12 - result[2])
    MCC = ((result[2] * (215311 - 157 - result[3])) - result[3] * (157 - result[2])) / np.sqrt(a * b * c * d)
    precision = result[2] / (result[2] + result[3])

    m_accuracy[detector.DETECTOR_NAME] = accuracy
    m_dt[detector.DETECTOR_NAME] = dt
    m_fp[detector.DETECTOR_NAME] = fp
    m_tp[detector.DETECTOR_NAME] = tp
    m_mcc[detector.DETECTOR_NAME] = MCC
    m_precision[detector.DETECTOR_NAME] = precision

f = open('dde_forest.txt','a')
for i,j in m_accuracy.items():
    f.write(f'{j},\n')
for i,j in m_fp.items():
    f.write(f'{j},\n')
for i,j in m_tp.items():
    f.write(f'{j},\n')
for i,j in m_dt.items():
    f.write(f'{j},\n')
for i,j in m_precision.items():
    f.write(f'{j},\n')
for i,j in m_mcc.items():
    f.write(f'{j},\n')
f.close()
