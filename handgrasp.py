import  numpy as np
import scipy.stats as scipystats
import matplotlib.pyplot as plt

from numpy.lib.stride_tricks import as_strided
from scipy.io import loadmat
from PyEMD import EMD as Pyemd
from pyhht.emd import EMD

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale
from sklearn import metrics as skm


data = loadmat('sEMG_Basic_Hand_movements_upatras-2/Database 1/female_1.mat')
data2 = loadmat('sEMG_Basic_Hand_movements_upatras-2/Database 1/female_2.mat')
data3 = loadmat('sEMG_Basic_Hand_movements_upatras-2/Database 1/female_3.mat')
data4 = loadmat('sEMG_Basic_Hand_movements_upatras-2/Database 1/male_1.mat')
data5 = loadmat('sEMG_Basic_Hand_movements_upatras-2/Database 1/male_2.mat')

data_set = [data, data2, data3, data4, data5]

"""
Obtain all the grasps of a particular channel
"""
hand_grasps = ['cyl_ch1', 'cyl_ch2', 'hook_ch1', 'hook_ch2', 'tip_ch1', 'tip_ch2', 'palm_ch1', 'palm_ch2',
               'spher_ch1', 'spher_ch2', 'lat_ch1', 'lat_ch2'] # 0 to 11

subject_list = []

for n in data_set:
    all_subjects = n[hand_grasps[7]]
    subject_list.append(all_subjects)
    subjects = np.array(subject_list)

x = subjects.reshape(150, 3000)
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

    def emd_create(self):
        """
        Creates EMD
        """
        emd = Pyemd()
        IMFs = emd(self.reshaped_x[0])
        print(IMFs)

        for i in IMFs:
            self.sliding_window_plot(i)

    def hilbert_huang(self):
        """
        Create EMD and Calculate Hilbert-Huang
        """

        imfs_list = []
        for i in self.reshaped_x:
            for j in i:
                decomposer = EMD(j)
                imfs = decomposer.decompose()
                imfs_list.append(imfs)
        return np.array(imfs_list)


class FeatExtract(object):
    """
    Features that are to be extracted.
    """

    def __init__(self, j):
        self.j = j

    def iemg(self):
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

    def waveform_length(self):
        pass

    def willison_amp(self):
        pass


class FeatureExtract(object):

    def feat_window(self, arr, window, overlap):
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

    def feature_extraction(self):
        """
        Performs Overlapping Window
        :return list
        """
        feat_extract_list = []
        for i in x:
            x_feat_extract = feat_window(i, 300, 30)
            feat_extract_list.append(x_feat_extract)
        x_ = np.array(feat_extract_list)

        pre_process = PreProcess(x_)
        hil_huang_pass = pre_process.hilbert_huang()

        # Feature extraction for non-hilbert_huang features
        feature_val = []
        for i in x_:
            new_shape = np.reshape(i, (11, 300))
            for j in new_shape:
                use_feat = FeatExtract(j)
                feat_val = use_feat.skewness_calc()
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
        return feat_val_array, feat_val_imf_array

    feature_list = feature_extraction()

    def create_list_of_features(self, list_of_features):

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
        final_array = np.insert(final_array, 0, array_headings)
        return final_array.transpose()

    pass_value = create_list_of_features(feature_list)

    def feature_selection(self):
        pass

    def give_feat_and_labels(self, the_final_list):

        X = the_final_list[:, :5]
        y = the_final_list[:, 5:]

        return X, y


class UseClassifier(object):

    def __init__(self, X):
        self.X, self.y = X
        try_this = np.array(self.X).astype(np.float)
        scaled_val = scale(try_this)
        pca = PCA()
        self.X = pca.fit_transform(scaled_val)
        print(f'PCA variance ratio of features: {np.r   ound_(pca.explained_variance_ratio_, decimals=2)}')

    def linear_classifier(self):
        clf = svm.SVC(kernel='sigmoid', verbose=3)
        clf.fit(self.X, self.y.ravel())
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.30, random_state = 42)
        return clf.predict(X_test), y_test

    def random_forest_classifier(self):
        clf = ensemble.RandomForestClassifier(n_estimators=30, max_depth=30, random_state=40, verbose=3)
        clf.fit(self.X, self.y.ravel())
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.30, random_state = 42)

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


if __name__ == '__main__':
    initialize_obj = FeatExtract()
    classify_values = initialize_obj.give_feat_and_labels()
    my_obj = UseClassifier(classify_values)
    metric_predict = my_obj.random_forest_classifier()
    predict_metrics(metric_predict)
