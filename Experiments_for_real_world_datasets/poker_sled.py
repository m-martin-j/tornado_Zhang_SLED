"""
By Shuxiang Zhang
University of Auckland, Auckland, New Zealand
E-mail: szha861@aucklanduni.ac.nz
"""
import numpy as np
import os
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
from drift_detection.sled import Sled


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
sled = Sled(d)

# 1. Creating a project
stream_name = "poker"
project = Project("projects/single", "sled_poker_training/" + stream_name)

# 2. Generating training stream

project_path = "data_streams/sled_poker_training/" + stream_name + "/"
if not os.path.exists(project_path):
    os.makedirs(project_path)

file_path = os.path.join(os.getcwd(), 'Real_world_datasets\\')
# 2. Loading an arff file
labels, attributes, stream_records = ARFFReader.read(file_path+"pokerTraining.arff")
attributes_scheme = AttributeScheme.get_scheme(attributes)

# 3. Initializing a Learner
learner = NaiveBayes(labels, attributes_scheme['nominal'])
drift_acceptance_interval = 200
detected_drifts = []
for detector in d:

    '''The reason why we put this statement inside this for loop is
    because the variable actual_drift_points will be changed
    every time the following statements are executed'''

    actual_drift_points = [2563, 5511, 6665, 7402, 7595, 9454, 10320, 10897, 11186, 11860, 12341, 12534, 13015, 14136, 14489, 14650, 15515, 16188, 16957, 17822, 18559, 19296, 23943, 24264, 25001, 26123, 28976, 29841, 32630, 33432, 33817, 34074, 34587, 34844, 35325, 35646, 35903, 36192, 37249, 37538, 38531, 39204, 39717, 39814, 40841, 42603, 45999, 46384, 47185, 48339, 50647, 51544, 51705, 52314, 54045, 54270, 54591, 54816, 55329, 56099, 56612, 57190, 57479, 57961, 58314, 58571, 59628, 60493, 61935, 63281, 66325, 66614, 67287, 67929, 70620, 71421, 71614, 73729, 74338, 74467, 76101, 76454, 77448, 77609, 78186, 78731, 79884, 80397, 80942, 81135, 81712, 82514, 84695, 84856, 85369, 85626, 86139, 86428, 87165, 87614, 87871, 88160, 89730, 91108, 91429, 92198, 92583, 94090, 95212, 97777, 98514, 98771, 99316, 101399, 102072, 102681, 102938, 103740, 104221, 104606, 104799, 106083, 106308, 107141, 107398, 108071, 108296, 109161, 109514, 110315, 111245, 113681, 114546, 117109, 117398, 118264, 119131, 120029, 121343, 122656, 123521, 123618, 124003, 124388, 125125, 125671, 128235, 128652, 129133, 129454, 131473, 131986, 132275, 132628, 133750, 134712, 135321, 136026, 136411, 137596, 137981, 138558, 138815, 139489, 140226, 140868, 141382, 142761, 142890, 143211, 144174, 144687, 145264, 146193, 147475, 149848, 150297, 150618, 151292, 152030, 152767, 153088, 154593, 155619, 156228, 156645, 157062, 157639, 158088, 158249, 160652, 161486, 161903, 162705, 163315, 164885, 165334, 166359, 167320, 167737, 168378, 169212, 170815, 171264, 171617, 173413, 174856, 175818, 177003, 177324, 178414, 179215, 179632, 180657, 181202, 181811, 182100, 182645, 182870, 183768, 183993, 184796, 185277, 187232, 187873, 188674, 189731, 190725, 191014, 192716, 193101, 194384, 195089, 195571, 196628, 197077, 199226, 199515, 199804, 200797, 201727, 202048, 203265, 203810, 204772, 205701, 206982, 207239, 208618, 209067, 210060, 210957, 211854, 213168, 213361, 214226, 214963, 215476, 215605, 215958, 216343, 217529, 218619, 220732, 222333, 223231, 224288, 225025, 225635, 226084, 226629, 227270, 228585, 229770, 231435, 231852, 233454, 233871, 234160, 234833, 235378, 236211, 237940, 238837, 241498, 244863, 248774, 249447, 249736, 250121, 250539, 251213, 251694, 253779, 254068, 255254, 255799, 256344, 257626, 258203, 258556, 262017, 265733, 266503, 266920, 267177, 267915, 268364, 270576, 271410, 271635, 272564, 275577, 278942, 280097, 283591, 286956, 288944, 289393, 291189, 292503, 293400, 294940, 299171, 299524, 304266, 306928, 309045, 310456, 311513, 312890, 314076, 317731, 318180, 318373, 318854, 319079, 319817, 320940, 322383, 323825, 324850, 325779, 329850, 330267, 330780, 330909, 331647, 332128, 335271, 335976, 336809, 337355, 341938, 344886, 346905, 347386, 350306, 351043, 351812, 352646, 355883, 356749, 359120, 361333, 362968, 363897, 365178, 366172, 366717, 366974, 367680, 368290, 370181, 371687, 372616, 372745, 373738, 374348, 374990, 375696, 376370, 376627, 378456, 379226, 380411, 384383, 387013, 389032, 389993, 390826, 391275, 394384, 396694, 399259, 400540, 402784, 404098, 405219, 406756, 408070, 409417, 410026, 410571, 411116, 412977, 413490, 415222, 415895, 416856, 416953, 417658, 420095, 422758, 423879, 425579, 425868, 426669, 428880, 429873, 430899, 432213, 433432, 434041, 434874, 436444, 437470, 438047, 439360, 439873, 441218, 441859, 443525, 444902, 445191, 445929, 446122, 447307, 449676, 450125, 450894, 451599, 453040, 453361, 454130, 454707, 460797, 465029, 465639, 466409, 467179, 469809, 470866, 471475, 471796, 472405, 472566, 472727, 473112, 473465, 478786, 481640, 482442, 482827, 483116, 483854, 486227, 486644, 487701, 488470, 493792, 494561, 497670, 499177, 499658, 501422, 501839]

    prequential = PrequentialDriftEvaluator(learner, detector, attributes, attributes_scheme,
                                            actual_drift_points, drift_acceptance_interval, project)
    prequential.run(stream_records, 1)
    detected_drifts.append(prequential.located_drift_points)


# 5 Training
true_change = [2563, 5511, 6665, 7402, 7595, 9454, 10320, 10897, 11186, 11860, 12341, 12534, 13015, 14136, 14489, 14650, 15515, 16188, 16957, 17822, 18559, 19296, 23943, 24264, 25001, 26123, 28976, 29841, 32630, 33432, 33817, 34074, 34587, 34844, 35325, 35646, 35903, 36192, 37249, 37538, 38531, 39204, 39717, 39814, 40841, 42603, 45999, 46384, 47185, 48339, 50647, 51544, 51705, 52314, 54045, 54270, 54591, 54816, 55329, 56099, 56612, 57190, 57479, 57961, 58314, 58571, 59628, 60493, 61935, 63281, 66325, 66614, 67287, 67929, 70620, 71421, 71614, 73729, 74338, 74467, 76101, 76454, 77448, 77609, 78186, 78731, 79884, 80397, 80942, 81135, 81712, 82514, 84695, 84856, 85369, 85626, 86139, 86428, 87165, 87614, 87871, 88160, 89730, 91108, 91429, 92198, 92583, 94090, 95212, 97777, 98514, 98771, 99316, 101399, 102072, 102681, 102938, 103740, 104221, 104606, 104799, 106083, 106308, 107141, 107398, 108071, 108296, 109161, 109514, 110315, 111245, 113681, 114546, 117109, 117398, 118264, 119131, 120029, 121343, 122656, 123521, 123618, 124003, 124388, 125125, 125671, 128235, 128652, 129133, 129454, 131473, 131986, 132275, 132628, 133750, 134712, 135321, 136026, 136411, 137596, 137981, 138558, 138815, 139489, 140226, 140868, 141382, 142761, 142890, 143211, 144174, 144687, 145264, 146193, 147475, 149848, 150297, 150618, 151292, 152030, 152767, 153088, 154593, 155619, 156228, 156645, 157062, 157639, 158088, 158249, 160652, 161486, 161903, 162705, 163315, 164885, 165334, 166359, 167320, 167737, 168378, 169212, 170815, 171264, 171617, 173413, 174856, 175818, 177003, 177324, 178414, 179215, 179632, 180657, 181202, 181811, 182100, 182645, 182870, 183768, 183993, 184796, 185277, 187232, 187873, 188674, 189731, 190725, 191014, 192716, 193101, 194384, 195089, 195571, 196628, 197077, 199226, 199515, 199804, 200797, 201727, 202048, 203265, 203810, 204772, 205701, 206982, 207239, 208618, 209067, 210060, 210957, 211854, 213168, 213361, 214226, 214963, 215476, 215605, 215958, 216343, 217529, 218619, 220732, 222333, 223231, 224288, 225025, 225635, 226084, 226629, 227270, 228585, 229770, 231435, 231852, 233454, 233871, 234160, 234833, 235378, 236211, 237940, 238837, 241498, 244863, 248774, 249447, 249736, 250121, 250539, 251213, 251694, 253779, 254068, 255254, 255799, 256344, 257626, 258203, 258556, 262017, 265733, 266503, 266920, 267177, 267915, 268364, 270576, 271410, 271635, 272564, 275577, 278942, 280097, 283591, 286956, 288944, 289393, 291189, 292503, 293400, 294940, 299171, 299524, 304266, 306928, 309045, 310456, 311513, 312890, 314076, 317731, 318180, 318373, 318854, 319079, 319817, 320940, 322383, 323825, 324850, 325779, 329850, 330267, 330780, 330909, 331647, 332128, 335271, 335976, 336809, 337355, 341938, 344886, 346905, 347386, 350306, 351043, 351812, 352646, 355883, 356749, 359120, 361333, 362968, 363897, 365178, 366172, 366717, 366974, 367680, 368290, 370181, 371687, 372616, 372745, 373738, 374348, 374990, 375696, 376370, 376627, 378456, 379226, 380411, 384383, 387013, 389032, 389993, 390826, 391275, 394384, 396694, 399259, 400540, 402784, 404098, 405219, 406756, 408070, 409417, 410026, 410571, 411116, 412977, 413490, 415222, 415895, 416856, 416953, 417658, 420095, 422758, 423879, 425579, 425868, 426669, 428880, 429873, 430899, 432213, 433432, 434041, 434874, 436444, 437470, 438047, 439360, 439873, 441218, 441859, 443525, 444902, 445191, 445929, 446122, 447307, 449676, 450125, 450894, 451599, 453040, 453361, 454130, 454707, 460797, 465029, 465639, 466409, 467179, 469809, 470866, 471475, 471796, 472405, 472566, 472727, 473112, 473465, 478786, 481640, 482442, 482827, 483116, 483854, 486227, 486644, 487701, 488470, 493792, 494561, 497670, 499177, 499658, 501422, 501839]
sled.train(true_change, smoothing_factor=0.5, fusion_parameter=0.5, drift_acceptance_interval=200,
           detected_drifts=detected_drifts)
stream_name = 'poker'


# 1. Creating a project
project = Project("projects/single", "sled_poker" + stream_name)

# 2. Generating training stream

project_path = "data_streams/sled_poker/" + stream_name + "/"
if not os.path.exists(project_path):
    os.makedirs(project_path)
actual_drift_points_testing = []
raw_actual_drift_points = [503024, 503505, 504980, 509307, 510492, 513697, 516804, 517606, 521583, 522896, 525173, 528219, 529021, 529727, 530850, 532421, 533672, 534891, 535372, 539475, 540790, 541559, 542233, 544029, 545376, 546017, 546627, 547812, 550471, 552523, 554800, 555923, 557012, 561627, 568744, 569802, 570955, 571789, 572046, 572912, 573617, 576151, 577049, 577818, 578683, 579772, 580125, 581088, 581601, 581826, 582436, 584264, 585289, 585674, 586411, 589200, 591413, 592727, 593913, 601161, 602284, 603469, 606032, 607218, 608211, 611319, 612440, 613433, 614170, 615742, 616960, 618082, 620681, 622476, 623311, 624464, 625906, 627091, 629302, 629975, 630136, 630905, 631964, 632733, 633246, 636291, 638725, 640679, 642377, 642666, 643724, 643885, 644942, 645487, 647729, 648242, 648499, 650420, 650869, 651318, 657121, 661996, 662670, 663471, 666099, 667606, 668471, 671644, 674401, 676228, 678536, 679595, 680332, 681134, 682704, 684082, 684691, 685236, 693764, 695593, 697838, 699183, 699568, 700691, 704634, 705115, 711748, 713925, 718251, 719149, 719726, 719983, 720688, 721137, 724759, 725784, 726809, 730399, 731873, 736364, 737005, 737550, 738704, 743483, 744991, 746016, 746881, 748130, 751784, 752266, 756497, 757682, 758869, 759639, 760344, 762910, 764447, 765312, 767652, 768358, 772108, 772557, 773134, 775730, 776404, 777944, 779099, 780444, 780733, 784484, 784902, 785543, 786824, 787049, 790640, 791409, 791890, 794580, 795701, 796406, 799033, 799290, 800220, 801245, 803297, 803586, 804388, 805573, 807207, 808265, 810379, 811148, 811725, 813424, 814001, 814482, 815957, 818103, 818616, 819578, 820252, 821597, 822334, 822783]

for drift in raw_actual_drift_points:
    actual_drift_points_testing.append(drift-502000)
print(actual_drift_points_testing)

# 3. Loading an arff file
labels, attributes, stream_records = ARFFReader.read(file_path+"pokerTesting.arff")
attributes_scheme = AttributeScheme.get_scheme(attributes)

# 4. Initializing a Learner
learner = NaiveBayes(labels, attributes_scheme['nominal'])

# 5. Reset base drift detectors

for detector in d:
    detector.reset()

# 4. Initializing sled method
detector = sled
# 5. Creating a Prequential Evaluation Process
prequential = PrequentialDriftEvaluator(learner, detector, attributes, attributes_scheme,
                                        actual_drift_points_testing, drift_acceptance_interval, project)

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


f = open('sled_poker', 'a')
f.write(f'\n sled: mean_accuracy {accuracy}')
f.write(f'\n sled: mean_dt {dt}')
f.write(f'\n sled: mean_tp {tp}')
f.write(f'\n sled: mean_fp {fp}')
f.write(f'\n sled: mean_mcc {MCC}')
f.write(f'sled: \n mean_precision {precision}')
f.close()
