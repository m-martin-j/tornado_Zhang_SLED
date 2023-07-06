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

d = [ddm, hddm_w, cusum, eddm, rddm, ewma, ph, hddm_a, seq_drift2, fhddm]

m_accuracy = OrderedDict()
m_fp = OrderedDict()
m_dt = OrderedDict()
m_tp = OrderedDict()
m_mcc = OrderedDict()
m_precision = OrderedDict()

stream_name = 'forest'
# 1. Creating a project
project = Project("projects/single", "base_detector_forest" + stream_name)

# 2. Generating training stream

project_path = "data_streams/base_detector_forest/" + stream_name + "/"
if not os.path.exists(project_path):
    os.makedirs(project_path)
actual_drift_points = []
raw_actual_drift_points = [503024, 503505, 504980, 509307, 510492, 513697, 516804, 517606, 521583, 522896, 525173, 528219, 529021, 529727, 530850, 532421, 533672, 534891, 535372, 539475, 540790, 541559, 542233, 544029, 545376, 546017, 546627, 547812, 550471, 552523, 554800, 555923, 557012, 561627, 568744, 569802, 570955, 571789, 572046, 572912, 573617, 576151, 577049, 577818, 578683, 579772, 580125, 581088, 581601, 581826, 582436, 584264, 585289, 585674, 586411, 589200, 591413, 592727, 593913, 601161, 602284, 603469, 606032, 607218, 608211, 611319, 612440, 613433, 614170, 615742, 616960, 618082, 620681, 622476, 623311, 624464, 625906, 627091, 629302, 629975, 630136, 630905, 631964, 632733, 633246, 636291, 638725, 640679, 642377, 642666, 643724, 643885, 644942, 645487, 647729, 648242, 648499, 650420, 650869, 651318, 657121, 661996, 662670, 663471, 666099, 667606, 668471, 671644, 674401, 676228, 678536, 679595, 680332, 681134, 682704, 684082, 684691, 685236, 693764, 695593, 697838, 699183, 699568, 700691, 704634, 705115, 711748, 713925, 718251, 719149, 719726, 719983, 720688, 721137, 724759, 725784, 726809, 730399, 731873, 736364, 737005, 737550, 738704, 743483, 744991, 746016, 746881, 748130, 751784, 752266, 756497, 757682, 758869, 759639, 760344, 762910, 764447, 765312, 767652, 768358, 772108, 772557, 773134, 775730, 776404, 777944, 779099, 780444, 780733, 784484, 784902, 785543, 786824, 787049, 790640, 791409, 791890, 794580, 795701, 796406, 799033, 799290, 800220, 801245, 803297, 803586, 804388, 805573, 807207, 808265, 810379, 811148, 811725, 813424, 814001, 814482, 815957, 818103, 818616, 819578, 820252, 821597, 822334, 822783]
for drift in raw_actual_drift_points:
    actual_drift_points.append(drift-502000)
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
    b = result[2] + (205 - result[2])
    c = (327200 - 205 - result[3]) + result[3]
    d = (327200 - 205 - result[3]) + (205 - result[2])
    MCC = ((result[2] * (327200 - 205 - result[3])) - result[3] * (205 - result[2])) / np.sqrt(a * b * c * d)
    precision = result[2] / (result[2] + result[3])

    m_accuracy[detector.DETECTOR_NAME] = accuracy
    m_dt[detector.DETECTOR_NAME] = dt
    m_fp[detector.DETECTOR_NAME] = fp
    m_tp[detector.DETECTOR_NAME] = tp
    m_mcc[detector.DETECTOR_NAME] = MCC
    m_precision[detector.DETECTOR_NAME] = precision

f = open('base_detector_forest.txt','a')
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
