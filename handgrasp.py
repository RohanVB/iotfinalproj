import  numpy as np
from numpy.lib.stride_tricks import as_strided

import itertools as it

from scipy.io import loadmat
import scipy.stats as scipystats
import matplotlib.pyplot as plt

from PyEMD import EMD as Pyemd

from pyhht.emd import EMD
from pyhht.visualization import plot_imfs

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import linear_model
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
# from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from sklearn.feature_selection import VarianceThreshold

import pandas as pd

from sklearn import metrics as skm


data = loadmat('sEMG_Basic_Hand_movements_upatras-2/Database 1/female_1.mat')
data2 = loadmat('sEMG_Basic_Hand_movements_upatras-2/Database 1/female_2.mat')
data3 = loadmat('sEMG_Basic_Hand_movements_upatras-2/Database 1/female_3.mat')
data4 = loadmat('sEMG_Basic_Hand_movements_upatras-2/Database 1/male_1.mat')
data5 = loadmat('sEMG_Basic_Hand_movements_upatras-2/Database 1/male_2.mat')

data_set = [data, data2, data3, data4, data5]

hand_grasps = ['cyl_ch1', 'cyl_ch2', 'hook_ch1', 'hook_ch2', 'tip_ch1', 'tip_ch2', 'palm_ch1', 'palm_ch2',
               'spher_ch1', 'spher_ch2', 'lat_ch1', 'lat_ch2'] # 0 to 11

# Get all the grasps of a particular channel
subject_list = []

for n in data_set:
    all_subjects = n[hand_grasps[7]]  # change this from anything for 0 to 11
    subject_list.append(all_subjects)
    subjects = np.array(subject_list)

x = subjects.reshape(150, 3000)  # So instead of 30 trials for 5 people, we now have 150 trials for 1 person
x_emd = data['palm_ch1']

class PreProcess(object):

    def __init__(self, reshaped_x):
        self.reshaped_x = reshaped_x

    def sliding_window_plot(self, y):
        """
        Plots our data
        """
        x_axis = []
        a = 0.2

        for i in range(0, 300):
            x_axis.append(a)
            a = a + 0.2

        plt.plot(x_axis, np.reshape(y, (300, 11)))
        plt.show()

        # # create plot names
        # plt_names = []
        #
        # for i in range(1, 31):
        #     plt_name = 'Plot_'
        #     plt_name = plt_name + str(i)
        #     plt_names.append(plt_name)

    # print(x_[0])

    def emd_create(self):
        """
        Creates EMD
        """
        emd = Pyemd()
        IMFs = emd(self.reshaped_x[0])
        print(IMFs)

        # plot_imfs(reshaped_x[0], imfs)

        for i in IMFs:
            self.sliding_window_plot(i)

    def hilbert_huang(self):
        """
        Create EMD and Calculate Hilbert-Huang
        """

        imfs_list = []
        for i in self.reshaped_x: # This traverse (11, 300) 150 times.
            for j in i:
                decomposer = EMD(j)
                imfs = decomposer.decompose()
                imfs_list.append(imfs)
                # this contains 11(windows) * 150(trials) = 1650 lists of 3000 (points) * 6/7/8 (IMFs)
        return np.array(imfs_list)

        # for i in imfs_list:
        #     new_var = i[:4, :]
        #     pass_me = np.reshape(new_var, (4, 300))
        #     return pass_me
        #     # self.sliding_window_plot(pass_me)

    # emd_create(x_)
    # hilbert_huang(x_)

# preprocess = PreProcess(x)
# preprocess.sliding_window_plot(x)


class FeatExtract(object):

    def __init__(self, j):
        self.j = j

    def iemg(self): # Note: We're supposed to use Sliding windows with this but see how it works first
        absolute_val = abs(self.j)
        return np.mean(absolute_val)

    def calc_median(self):
        return np.median(self.j)

    def calc_std(self):
        return np.std(self.j)

    def calc_variance(self):
        return np.var(self.j)

    def skewness_calc(self):
        return scipystats.skew(self.j)

    def kurtosis(self):
        return scipystats.kurtosis(self.j)

    def zero_crossings(self):
        signs = np.sign(self.j)
        crossings = np.where(np.diff(signs), 1, 0)
        crossings = np.insert(crossings, 0, 0)
        return sum(crossings)

    def slope_sign_changes(self):
        pass
        # TODO https://stackoverflow.com/questions/47519626/using-numpy-scipy-to-identify-slope-changes-in-digital-signals

    def waveform_length(self):
        # TODO Find difference between consecutive elements of a list and then do the sum of all of these elements (300 sums)
        # TODO To give an output of (1650,)
        pass

    def willison_amp(self):
        # TODO Find difference, if they are greater than a threshold, "X" then return 1, else return 0.
        pass


# This is just used in feature_extraction()
def feat_window(arr, window, overlap):
    """
    Use either feature window (for every feature except iemg) or use iemg_window
    Note: This is overlapping windowing
    """
    arr = np.asarray(arr)
    window_step = window - overlap
    new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step,
                                  window)
    new_strides = (arr.strides[:-1] + (window_step * arr.strides[-1],) +
                   arr.strides[-1:])
    return as_strided(arr, shape=new_shape, strides=new_strides)

# process_me = feat_window(x_emd[0], 300, 30)
# preprocess = PreProcess(process_me)
# preprocess.sliding_window_plot(process_me)


def feature_extraction():
    """
    Performs Overlapping Window
    :return list
    """
    feat_extract_list = []
    for i in x: # x is (150, 3000)
        x_feat_extract = feat_window(i, 300, 30)  # 11 windows created
        feat_extract_list.append(x_feat_extract)
    x_ = np.array(feat_extract_list)
    # x_ = np.reshape(feat_extract_list, (150, 11, 300))

    pre_process = PreProcess(x_)
    hil_huang_pass = pre_process.hilbert_huang()

    # Feature extraction for non-hilbert_huang features
    feature_val = []
    for i in x_:
        # feature_val.append(obj.sliding_window_plot(i))
        new_shape = np.reshape(i, (11, 300))
        for j in new_shape:
            use_feat = FeatExtract(j)
            feat_val = use_feat.skewness_calc()  # TODO: Generate features again using this and hilbert huang
            feature_val.append(feat_val)

    # Feature extraction on IMFs (hilbert huang)
    feature_val_imf = []
    for m in hil_huang_pass:
        this_var = m[1:4, :]
        pass_me = np.reshape(this_var, (3, 300))
        for new_var in pass_me:
            use_feat_imf = FeatExtract(new_var)
            feat_val_imf = use_feat_imf.skewness_calc()
            feature_val_imf.append(feat_val_imf)

    feat_val_array = np.array(feature_val)
    feat_val_imf_array = np.array(feature_val_imf)
    return feat_val_array, feat_val_imf_array # Contain 1650 and 4950 features respectively
    # TODO Process them together for the first 8 features, then for the "special features" just do IMFs and just
    # TODO return feat_val_imf_array for the final features


feature_list = feature_extraction()

def create_list_of_features(list_of_features):
    # first_val, second_val = list_of_features
    # with open('features/palm/skewness_temp.txt', 'a') as f:
    #     for first in first_val:
    #         f.write("%s\n" % first)
    #     for second in second_val:
    #         f.write("%s\n" % second)

    print("creating list of features")
    with open('features/feat_list/variance.txt', 'r') as r:
        my_list1 = [line.strip() for line in r]

    with open('features/feat_list/iemg.txt', 'r') as r:
        my_list2 = [line.strip() for line in r]

    with open('features/feat_list/kurtosis.txt', 'r') as r:
        my_list3 = [line.strip() for line in r]

    with open('features/feat_list/skewness.txt', 'r') as r:
        my_list4 = [line.strip() for line in r]

    with open('features/feat_list/zero_crossings.txt', 'r') as r:
        my_list5 = [line.strip() for line in r]

    label_list = []
    for i in range(0, 13200):
        label_list.append('1') # This should be a class like: Cylinder, Hook, etc,. (Maybe have 1, 2,..6 instead)
    for i in range(13200, 26400):
        label_list.append('2')
    for i in range(26400, 39600):
        label_list.append('3')
    for i in range(39600, 52800):
        label_list.append('4')
    for i in range(52800, 66000):
        label_list.append('5')
    for i in range(66000, 79200):
        label_list.append('6')

    array_headings = ['median', 'std', 'var', 'iemg', 'kurt', 'skew', 'zero_cross', 'grasp']

    feature_array = []
    feature_array.append(my_list1)
    feature_array.append(my_list2)
    feature_array.append(my_list3)
    feature_array.append(my_list4)
    feature_array.append(my_list5)
    feature_array.append(label_list)

    final_array = np.array(feature_array)
    # final_array = np.insert(final_array, 0, array_headings)
    return final_array.transpose()


feature_list = None
pass_value = create_list_of_features(feature_list)

# obj.sliding_window_plot(i)

# def iemg_window(x, length, step=1):
#     """
#     Use either feature window (for every feature except iemg) or use iemg_window
#     """
#     streams = it.tee(x, length)
#     return zip(*[it.islice(stream, i, None, step*length) for stream, i in zip(streams, it.count(step=step))])


# x_=list(iemg_window(x, 15))
# x_=np.asarray(x_)
# print(x.shape)
# print(x_.shape)


# pandas_reformat(pass_to_pandas)

def loop_over_sample(value):
    """

    THIS FUNCTION MIGHT BE USELESS!
    """

    outer_list = []
    myobj = PreProcess(value)
    # for itr, val in enumerate(value): # val has 30 trials of 11 rows containing 300 samples
    # myobj.hilbert_huang()
    for i in range(0, 30):
        for j in range(0, 11):
            outer_list.append(myobj.hilbert_huang())
    return np.array(outer_list).shape


def feature_selection():
    from sklearn.feature_selection import VarianceThreshold
    pass


def give_feat_and_labels(the_final_list):

    X = the_final_list[:, :5]
    y = the_final_list[:, 5:]

    return X, y


the_x_and_y = give_feat_and_labels(pass_value)

class UseClassifier(object):

    def __init__(self, X):
        self.X, self.y = X
        try_this = np.array(self.X).astype(np.float)
        scaled_val = scale(try_this)
        # print(f'after...{try_this_new.mean(axis = 0)}')
        pca = PCA()
        self.X = pca.fit_transform(scaled_val)
        print(f'PCA variance ratio of features: {np.round_(pca.explained_variance_ratio_, decimals=2)}')

    def linear_classifier(self):
        clf = svm.SVC(kernel='sigmoid', verbose=3)

        # X's are a numpy array of features (list of list where first list is the row of features
        # corresponding to the first label, and subsequent rows correspond to subsequent features
        # x would have each feature generated by the initial signal hook_ch1[0], etc,.
        # y is the label (this would be the type of grasp - Hook, tip, etc,.)
        clf.fit(self.X, self.y.ravel())

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.30, random_state = 42)

        return clf.predict(X_test), y_test

    def random_forest_classifier(self):
        clf = ensemble.RandomForestClassifier(n_estimators=30, max_depth=30, random_state=40, verbose=3)
        clf.fit(self.X, self.y.ravel())
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.30, random_state = 42)

        # TODO: Mention in paper that we used KFold to see how good our model was.
        # kf = StratifiedKFold(n_splits=5)
        # kf.get_n_splits(self.X)
        # for train_index, test_index in kf.split(self.X, self.y):
        #     X_train, X_test = self.X[train_index], self.X[test_index]
        #     y_train, y_test = self.y[train_index], self.y[test_index]

        return clf.predict(X_test), y_test

    def adaboost_classifier(self):
        print('Running Adaboost Classifier with test_size = 0.30, random_state = 42')
        clf = ensemble.AdaBoostClassifier()
        clf.fit(self.X, self.y.ravel())
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.30, random_state = 42)

        return clf.predict(X_test), y_test

    # def centroid_classifier(self):
    #     clf = NearestCentroid()
    #     a = np.array(self.X).astype(np.float)
    #     b = np.array(self.y).astype(np.float)
    #     clf.fit(a, b.ravel())
    #     X_train, X_test, y_train, y_test = train_test_split(a, b, test_size = 0.30, random_state = 42)
    #
    #     return clf.predict(X_test), y_test

    def gaussian_naive_bayes(self):
        print('Running Gaussian Naive Bayes with test_size = 0.30, random_state = 42...')
        clf = naive_bayes.GaussianNB()
        a = np.array(self.X).astype(np.float)
        b = np.array(self.y).astype(np.float)
        clf.fit(a, b.ravel())
        X_train, X_test, y_train, y_test = train_test_split(a, b, test_size = 0.30, random_state = 42)

        return clf.predict(X_test), y_test

    # def ridge_classifier(self):
    #     clf = linear_model.RidgeClassifier()
    #     a = np.array(self.X).astype(np.float)
    #     b = np.array(self.y).astype(np.float)
    #     clf.fit(a, b.ravel())
    #     X_train, X_test, y_train, y_test = train_test_split(a, b, test_size = 0.30, random_state = 42)
    #
    #     return clf.predict(X_test), y_test

    def k_neighbors(self):
        print('Running KNeighborsClassifier with n = 2, test_size = 0.30, random_state = 42...')
        clf = neighbors.KNeighborsClassifier(n_neighbors = 1)  # see bias-variance tradeoff because n=1; acc=1
        a = np.array(self.X).astype(np.float)
        b = np.array(self.y).astype(np.float)
        clf.fit(a, b.ravel())
        X_train, X_test, y_train, y_test = train_test_split(a, b, test_size = 0.30, random_state = 42)

        return clf.predict(X_test), y_test

    # def perceptron(self):
    #     clf = linear_model.Perceptron()
    #     a = np.array(self.X).astype(np.float)
    #     b = np.array(self.y).astype(np.float)
    #     clf.fit(a, b.ravel())
    #     X_train, X_test, y_train, y_test = train_test_split(a, b, test_size = 0.30, random_state = 42)
    #     # print(clf.score(X_test, y_test))
    #     return clf.predict(X_test), y_test


def predict_metrics(x):
    y_pred, y_true = x
    classes = {'Cylinder': 0, 'Hook': 1, 'Tip': 2, 'Palm': 3, 'Sphere': 4, 'Lateral': 5}
    print(f'Precision, Recall and F1 score are... \n'
          f'{classification_report(y_true, y_pred, target_names=classes.keys())}')
    print('The accuracy is: ',skm.accuracy_score(y_true, y_pred))
    # print('The precision is: ',skm.precision_score(y_true, y_pred))
    # print('The recall is: ', skm.recall_score(y_true, y_pred))
    # print('The F1 score for each class is: ', skm.f1_score(y_true, y_pred, average=None))
    # skm.roc_auc_score(x)
    # plt = skm.roc_curve(y_true, y_pred)
    # plt.show()
    print('Confusion matrix:')
    print(skm.confusion_matrix(y_true, y_pred))

my_obj = UseClassifier(the_x_and_y)
metric_predict = my_obj.random_forest_classifier()
predict_metrics(metric_predict)
