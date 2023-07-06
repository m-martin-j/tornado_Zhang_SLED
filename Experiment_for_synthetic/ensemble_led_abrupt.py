"""
By Shuxiang Zhang
University of Auckland, Auckland, New Zealand
E-mail: szha861@aucklanduni.ac.nz
"""
import numpy as np
import os
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
from drift_detection.sled import Sled

# Initializing detectors
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

# 1. Creating a project for training
stream_name = "led"
project = Project("projects/single", "ensemble_led_abrupt_training/" + stream_name)

# 2. Generating training stream
project_path = "data_streams/ensemble_led_abrupt_training/" + stream_name + "/"
if not os.path.exists(project_path):
    os.makedirs(project_path)

file_path = project_path + stream_name
led_attr_drift = [0, 3, 1, 3] * 25
led_attr_drift.append(0)
stream_generator = LEDConceptDrift(concept_length=5000, noise_rate=0.1, transition_length=5,
                                   led_attr_drift=led_attr_drift)
stream_generator.generate(file_path)

# 2. Loading an arff file
labels, attributes, stream_records = ARFFReader.read(file_path+".arff")
attributes_scheme = AttributeScheme.get_scheme(attributes)

# 3. Initializing a Learner
learner = NaiveBayes(labels, attributes_scheme['nominal'])
drift_acceptance_interval = 1000
detected_drifts = []
for detector in d:

    '''The reason why we put this statement inside this for loop is
    because the variable actual_drift_points will be changed
    every time the following statements are executed'''

    actual_drift_points = [i for i in range(5000, 505000, 5000)]
    # 4. Creating a Prequential Evaluation Process
    prequential = PrequentialDriftEvaluator(learner, detector, attributes, attributes_scheme,
                                            actual_drift_points, drift_acceptance_interval, project)
    prequential.run(stream_records, 1)
    detected_drifts.append(prequential.located_drift_points)
    detector.reset()

# 5. Training

num_of_true_drifts = 100
concept_length = 5000
true_change = [i for i in range(concept_length, (num_of_true_drifts+1)*concept_length, concept_length)]
sled.train(smoothing_factor=0.5, fusion_parameter=0.5, drift_acceptance_interval=1000, detected_drifts=detected_drifts,
           concept_length=concept_length)

# 6. Evaluating SLED through detection
repeat_count = 30
accuracy = []
dt = []
tp = []
fp = []
mcc = []
mean_pre = []
# 1.Creating a project for detection
stream_name = "led"
project = Project("projects/single", "ensemble_led_abrupt_detection/"+ stream_name)

# 2.Generating detection stream
project_path = "data_streams/ensemble_led_abrupt_detection/" + stream_name + "/"
if not os.path.exists(project_path):
    os.makedirs(project_path)
for tm in range(repeat_count):
    led_attr_drift = [0, 3, 1, 3] * 7
    led_attr_drift.extend([0, 3, 1])
    stream_generator = LEDConceptDrift(concept_length=5000, noise_rate=0.1, transition_length=5,
                                       led_attr_drift=led_attr_drift, random_seed=tm)
    file_path = project_path+str(tm)
    stream_generator.generate(file_path)

    # 3. Loading an arff file
    labels, attributes, stream_records = ARFFReader.read(file_path+".arff")
    attributes_scheme = AttributeScheme.get_scheme(attributes)

    # 4. Initializing a Learner
    learner = NaiveBayes(labels, attributes_scheme['nominal'])

    # 5. Creating a Prequential Evaluation Process for detection
    actual_drift_points = [i for i in range(5000, 155000, 5000)]
    drift_acceptance_interval = 1000
    prequential = PrequentialDriftEvaluator(learner, sled, attributes, attributes_scheme,
                                            actual_drift_points, drift_acceptance_interval, project)

    result = prequential.run(stream_records, 1)
    accuracy.append(100 - result[0])
    dt.append(result[1])
    tp.append(result[2])
    fp.append(result[3])
    a = result[2] + result[3]
    b = result[2] + (30 - result[2])
    c = (155000 - 30 - result[3]) + result[3]
    d = (155000 - 30 - result[3]) + (30 - result[2])
    MCC = ((result[2] * (155000 - 30 - result[3])) - result[3] * (30 - result[2])) / np.sqrt(a * b * c * d)
    mcc.append(MCC)
    precision = result[2] / (result[2] + result[3])
    mean_pre.append(precision)
mean_accuracy = np.mean(accuracy)
std_accuracy = np.std(accuracy)
mean_dt = np.mean(dt)
std_dt = np.std(dt)
mean_tp = np.mean(tp)
std_tp = np.std(tp)
mean_fp = np.mean(fp)
std_fp = np.std(fp)
mean_mcc = np.mean(mcc)
std_mcc = np.std(mcc)
std_precision = np.std(mean_pre)
m_precision = np.mean(mean_pre)
f = open('ensemble_led_abrupt', 'a')
f.write(f'\n ensemble: mean_accuracy {mean_accuracy}, sd_accuracy {std_accuracy}\n')
f.write(f'\n ensemble: mean_dt {mean_dt}, sd_dt{std_dt}\n')
f.write(f'\n ensemble: mean_tp {mean_tp}, std_tp {std_tp}\n')
f.write(f'\n ensemble: mean_fp {mean_fp}, std_fp {std_fp}\n')
f.write(f'\n ensemble: mean_mcc {mean_mcc}, std_mcc{std_mcc}\n')
f.write(f'ensemble: \n mean_precision {m_precision}, std_precision {std_precision}\n')
f.close()
