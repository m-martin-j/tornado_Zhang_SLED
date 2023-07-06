"""
By Shuxiang Zhang
University of Auckland, Auckland, New Zealand
E-mail: szha861@aucklanduni.ac.nz
"""
import numpy as np
import os
from collections import OrderedDict
from streams.generators.__init__ import *
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

d = [ddm,hddm_w,cusum,eddm,rddm,ewma,ph,hddm_a,seq_drift2,fhddm]

m_accuracy = OrderedDict()
m_kappa = OrderedDict()
m_fp = OrderedDict()
m_dt = OrderedDict()
m_tp = OrderedDict()
m_fn = OrderedDict()
m_tn = OrderedDict()
m_mcc = OrderedDict()
m_run_time = OrderedDict()
m_mem_usage = OrderedDict()
m_precision = OrderedDict()
m_recall = OrderedDict()

stream_name = 'electricity'
# 1. Creating a project
project = Project("projects/single", "base_detector_electricity"+ stream_name)

# 2. Generating training stream

project_path = "data_streams/base_detector_electricity/" + stream_name + "/"
if not os.path.exists(project_path):
    os.makedirs(project_path)
actual_drift_points = [1090, 2148, 3653, 4070, 4391, 4808, 5898, 8172, 9069, 10192, 11153, 11826, 14164, 15061, 20027, 20348, 22525, 24160, 24513, 25602, 27206,27559, 27976, 28809, 29642, 30283, 30572, 30861, 31054, 32943, 33264, 33906, 34388, 36246, 37847, 39992, 41433, 42010, 42811, 43164, 44157, 44350]
print(actual_drift_points)
for detector in d:
    file_path = "C:/Users/szha861/Desktop/Real_world_datasets/"
    # 3. Loading an arff file
    labels, attributes, stream_records = ARFFReader.read(file_path+"elecNormNew.arff")
    attributes_scheme = AttributeScheme.get_scheme(attributes)

    # 4. Initializing a Learner
    learner = NaiveBayes(labels, attributes_scheme['nominal'])

    drift_acceptance_interval = 200

    # 5. Creating a Prequential Evaluation Process
    prequential = PrequentialDriftEvaluator(learner, detector, attributes, attributes_scheme,
                                            actual_drift_points, drift_acceptance_interval, project)

    result = prequential.run(stream_records, 1)
    accuracy = (100-result[0])
    dt = (result[1])
    fn = (21-result[2])
    tn = (17912-21-result[3])
    tp =(result[2])
    fp =(result[3])
    a = result[2]+ result[3]
    b = result[2]+(21-result[2])
    c = (155000-21-result[3])+result[3]
    d = (155000-21-result[3])+(21-result[2])
    MCC = ((result[2]*(17912-21-result[3]))-result[3]*(21-result[2]))/np.sqrt(a*b*c*d)
    mem_usage=result[4]
    run_time=result[5]
    kappa=result[6]
    precision = result[2] / (result[2] + result[3])
    recall = result[2] / (result[2] + (21 - result[2]))

    m_accuracy[detector.DETECTOR_NAME] = accuracy
    m_kappa[detector.DETECTOR_NAME] = kappa
    m_dt[detector.DETECTOR_NAME] = dt
    m_fp[detector.DETECTOR_NAME] = fp
    m_tp[detector.DETECTOR_NAME] = tp

    m_fn[detector.DETECTOR_NAME] = fn
    m_tn[detector.DETECTOR_NAME] = tn
    m_mcc[detector.DETECTOR_NAME] = MCC

    m_run_time[detector.DETECTOR_NAME] = run_time
    m_mem_usage[detector.DETECTOR_NAME] = mem_usage
    m_precision[detector.DETECTOR_NAME] = precision
    m_recall[detector.DETECTOR_NAME] = recall

f = open('base_detector_electricity.txt','a')
for i,j in m_accuracy.items():
    f.write(f'\n{i}: mean_accuracy {j},\n')

for i,j in m_fp.items():
    f.write(f'\n{i}: mean_fp {j},\n')

for i,j in m_tp.items():
    f.write(f'\n{i}: mean_tp {j},\n')

for i,j in m_dt.items():
    f.write(f'\n{i}: mean_dt {j},\n')

for i,j in m_mem_usage.items():
    f.write(f'\n{i}: mean_mem {j},\n')

for i,j in m_run_time.items():
    f.write(f'\n{i}: mean_runtime {j},\n')

for i,j in m_kappa.items():
    f.write(f'\n{i}: mean_kappa {j},\n')

for i,j in m_fn.items():
    f.write(f'\n{i}: mean_fn {j},\n')

for i,j in m_tn.items():
    f.write(f'\n{i}: mean_tn {j},\n')

for i,j in m_mcc.items():
    f.write(f'\n{i}: mean_mcc {j},\n')
f.close()