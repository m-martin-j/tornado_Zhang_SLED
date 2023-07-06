"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""
import numpy as np
import os
import time
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

detector = adwin



stream_name = 'poker_hand'
# 1. Creating a project
project = Project("projects/single", "base_detector_circle_abrupt"+ stream_name)

# 2. Generating training stream

project_path = "data_streams/base_detector_circle_abrupt/" + stream_name + "/"
if not os.path.exists(project_path):
    os.makedirs(project_path)


# file_path = project_path
# # Specify the number of drifts
# thresholds = [[[0.2, 0.5], 0.15], [[0.4, 0.5], 0.2], [[0.6, 0.5], 0.25], [[0.8, 0.5], 0.3]]
# stream_generator = CIRCLES(concept_length=5000, transition_length=50, thresholds=thresholds, random_seed=1)
# stream_generator.generate(file_path)
t_1 = time.perf_counter()
# 3. Loading an arff file
file_path = 'C:/Users/szha861/Desktop/Real_world_datasets/'
labels, attributes, stream_records = ARFFReader.read(file_path+"poker-lsn.arff")
attributes_scheme = AttributeScheme.get_scheme(attributes)
learner = NaiveBayes(labels, attributes_scheme['nominal'])
prequential = PrequentialDriftEvaluator_real_world(learner, detector, attributes, attributes_scheme, project)
result_1 = prequential.run_2(stream_records,1)
located_drifts = result_1[0]
t_2 = time.perf_counter()
t = t_2-t_1
print(f'Time taken to run through the whole stream is {t}')
print(f'Drifts located are: {located_drifts}')
trained_error = result_1[1]
print(f'Error rates with adjustment are:{trained_error}')
untrained_error = []
true_drifts = []
for i in range(1,len(located_drifts)):
    # 4. Initializing a Learner
    learner = NaiveBayes(labels, attributes_scheme['nominal'])


    # 5. Creating a Prequential Evaluation Process
    prequential = PrequentialDriftEvaluator_real_world(learner, detector, attributes, attributes_scheme,project)
    #result_1 = prequential.run_2(stream_records,1)
    result_2 = prequential.run_1(stream_records,located_drifts,i,1)
    # prequential = PrequentialDriftEvaluator_real_world(learner, detector, attributes, attributes_scheme,project)
    # result_2 = prequential.run_2(stream_records, 1)
    untrained_error.append(result_2[1][-1])

for err in range(len(untrained_error)):
    if untrained_error[err] > trained_error[err]:
        true_drifts.append(located_drifts[err])
print(f'Error rates without adjustment are: {untrained_error}')
print(f'Identified true drifts are: {true_drifts}')
