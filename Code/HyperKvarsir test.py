#%% Import modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import dataframe_image as dfi
import matplotlib.pyplot as plt
import os, scipy, progressbar

from tqdm import tqdm
from scipy import ndimage
from sklearn import metrics
from contextlib import redirect_stdout
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Concatenate
from tensorflow.keras.applications import VGG16


#%% Define basic functions
def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def plot_confusion_matrix(true, pred, normalize=False, title=None, cmap=plt.cm.Blues,
                          savepath = os.getcwd(), fullname=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(true, pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(4,5))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.grid(False)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=['MES=0','MES=1'], yticklabels=['MES=0','MES=1'],
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_yticklabels(), rotation=90, ha="center",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    sns.set(font_scale=1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(savepath,fullname+'_confusion_matrix.png'))

def get_pr(true, pred):
    precision, recall, thresholds = precision_recall_curve(true.flatten(), pred.flatten())
    pr_auc = auc(recall, precision)
    return precision, recall, pr_auc

def get_roc(true, prob):
    fpr, tpr, thresh = roc_curve(true.flatten(), prob.flatten())
    roc_auc = roc_auc_score(true.flatten(), prob.flatten())
    return fpr, tpr, roc_auc

def AUC_PR(testY, pred_prob, temp_full_name, save_dir, fontsize=16):
    no_skill = testY.sum()/len(testY)
    plt.clf()
    plt.figure(figsize=(5, 5))
    lr_precision, lr_recall, _ = metrics.precision_recall_curve(testY.reshape(-1), pred_prob.reshape(-1))
    lr_auc = metrics.auc(lr_recall, lr_precision)
    plt.plot(lr_recall, lr_precision, lw=2, label='Ours: %0.4f'% (lr_auc))
    plt.plot([1, 0], [no_skill, no_skill], linestyle='--', lw=2, color='k', label='Baseline: %0.4f' % no_skill)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall', fontsize=fontsize)
    plt.ylabel('Precision', fontsize=fontsize)
    plt.title('AUPRC', fontsize=fontsize)
    plt.legend(loc="lower right", fontsize=fontsize)
    plt.savefig(os.path.join(save_dir, temp_full_name+'_PR-AUC.png'))
    plt.clf()
    return lr_auc

def AUC_ROC(testY, pred_prob, temp_full_name, save_dir, fontsize=16):
    plt.clf()
    plt.figure(figsize=(5, 5))
    fpr, tpr, thresh = metrics.roc_curve(testY.reshape(-1), pred_prob.reshape(-1))
    auc = metrics.roc_auc_score(testY.reshape(-1), pred_prob.reshape(-1))
    plt.plot(fpr, tpr, lw=2, label='Ours: %0.4f' % (auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Baseline: %0.4f' % 0.5)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.title('AUROC', fontsize=fontsize)
    plt.legend(loc="lower right", fontsize=fontsize)
    plt.savefig(os.path.join(save_dir, temp_full_name+'_ROC-AUC.png'))
    plt.clf()
    return auc


#%% Define model functions
class JoinedGen(tf.keras.utils.Sequence):
    def __init__(self, input_gen1, input_gen2):
        self.gen1 = input_gen1
        self.gen2 = input_gen2
        assert len(input_gen1) == len(input_gen2)

    def __len__(self):
        return len(self.gen1)

    def __getitem__(self, i):
        x1, y1 = self.gen1[i]
        if y1>0:
            y2=1
        else:
            y2=0
        return [x1, x1], [y1, y2]

    def on_epoch_end(self):
        self.gen1.on_epoch_end()
        self.gen2.on_epoch_end()

def Build_model(backbone):
    inputs1 = Input(shape=(256, 256, 3))
    inputs2 = Input(shape=(256, 256, 3))
    base_model = backbone(weights='imagenet', include_top=False)
    GAP = GlobalAveragePooling2D(name='gap')
    FC = Dense(2, activation='softmax')

    base_output1 = base_model(inputs1)
    gap = GAP(base_output1)
    aux1 = FC(gap)

    base_output2 = base_model(inputs2)
    gap2 = GAP(base_output2)
    aux2 = FC(gap2)

    concat_fmap = Concatenate()([base_output1, base_output2])
    gap3 = GlobalAveragePooling2D()(concat_fmap)
    out = Dense(2, activation='softmax', name='out')(gap3)

    model = Model(inputs=[inputs1, inputs2], outputs=[out, aux1, aux2])
    return base_model, model


#%% Basic hyper-parameters
classes = {0: 'MES 0', 1 : 'MES 1'}
backbone = VGG16


#%% Load dataset
root       = os.getcwd()
data_root  = os.path.join(root, 'Data')
save_root  = os.path.join(root, 'Result')
weight_dir = os.path.join(root, 'Weight')
data_dir   = os.path.join(data_root, 'img')
makedirs(save_root), makedirs(weight_dir)

col_names = ['Model', 'acc', 'F1-Score 0', 'F1-Score 1', 'precision 0', 'precision 1', 'recall 0', ' recall 1', 'AUROC', 'AUPRC']
writer    = pd.ExcelWriter(os.path.join(save_root, 'Results.xlsx'), engine='xlsxwriter')
all_results  =[]
mean_results =[]

lbl_df     = pd.read_csv(os.path.join(data_root, 'Hyperkvarsir_lbl.csv'))
target = lbl_df.loc[:,'label']


#%% Image Datagenerator
generator   = ImageDataGenerator(samplewise_std_normalization=True, fill_mode='nearest')

generator_1 = generator.flow_from_dataframe(dataframe=lbl_df, directory=data_dir, x_col='ID', y_col='label',
                                                 batch_size=1, seed=42, shuffle=False, class_mode="raw", target_size=(256, 256))

generator_2 = generator.flow_from_dataframe(dataframe=lbl_df, directory=data_dir, x_col='ID', y_col='label',
                                                 batch_size=1, seed=42, shuffle=False, class_mode="raw", target_size=(256, 256))

val= JoinedGen(generator_1, generator_2)


#%% Test
base_model, model = Build_model(backbone)
model_path = os.path.join(weight_dir, 'VGG16_based_best_model.hdf5')
model.load_weights(model_path)

true_ = []
pred_= []
for i in tqdm(range(len(val))):
    images, labels = val[i]
    pred_.append(model.predict(images)[2])
    true_.append(labels[1])
pred_ = np.concatenate(pred_)
true = np.array(true_)
pred   = pred_.argmax(axis=1)
pred_  = 1-pred_[:,0]


#%% Save results
plot_confusion_matrix(true, pred, normalize=False,
                      title='Confusion matrix', savepath=save_root, fullname='HyperKvarsir 0 and 1')

roc_auc = roc_auc_score(true, pred_)
precision , recall, thresholds = precision_recall_curve(true, pred_)
pr_auc = auc(recall, precision)
AUC_PR(true, pred_, 'HyperKvarsir 0 and 1', save_root)
AUC_ROC(true, pred_, 'HyperKvarsir 0 and 1', save_root)

report = classification_report(true, pred, target_names=classes, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df = report_df.round(decimals=3)
report_df_styled = report_df.style.background_gradient()
dfi.export(report_df_styled,os.path.join(save_root, 'HyperKvarsir 0 and 1.png'))

result = [VGG16.__name__, (true==pred).sum()/len(true), report_df['f1-score'][0], report_df['f1-score'][1],
          report_df['precision'][0], report_df['precision'][1], report_df['recall'][0], report_df['recall'][1],
          roc_auc, pr_auc]

df = pd.DataFrame([result], columns=col_names)
df.to_excel(writer)
writer.save()
