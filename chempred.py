# A script for implementing a QSAR model to predict the best aligning peptide chemistries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools as itt
from scipy.optimize import curve_fit as cf
from sklearn import linear_model as lm
sns.set(rc={'axes.labelsize':50, 'font.size':35, 'font.family':'serif', 'text.usetex':True, 'xtick.labelsize':25, 'ytick.labelsize':25, 'lines.markersize':6})
sns.set_style('whitegrid')



modparms = {'df2': [['MATS3c', 'MATS1c_aaWing'], ['MATS3c', 'ATSC4e_aaWing'], ['SpMax6_Bhs', 'piPC3'], ['AVP-5', 'piPC3']], 'df3': [['piPC3'], ['maxHBint10'], ['GATS2i']]}  # Descriptors to fit to training data. modparms[value] will give the model used to predict that value. Each list is a different model to average over. Each sublist is its own model. As written: fit df2 using MATS3c and MATS1c_aawing, and fit df2 using MATS3c and ATSC4e_aaWing, and fit df2 using SpMax6_Bhs and piPC3, and fit df2 using AVP-5 and piPC3. Average over each of these four separate fits to obtain the predicted df2 value. The names listed in these models must match the feature names in descsFnm.
predChems = []  # Chemistries to predict. If predAll is true, this will be replaced. If predAll is false and this is blank, it will default to all chemistries in the feature list.
selChems = ['dssw-pdi', 'dwcg-pdi', 'dyga-pdi', 'dygg-pdi', 'dwyw-ndi', 'dnww-pdi', 'dtct-ndi', 'davg-pdi', 'dmmp-pdi', 'dwww-ndi', 'daia-pdi']  # Chemistries to print to screen. 

trainAaList = ['a', 'f', 'g', 'i', 'v']
nonPolAaList = ['g', 'a', 'v', 'l', 'm', 'i', 'f', 'w', 'd', 'e']  # d and e are included as nonpolar since they will be fully protonated at low pH
nonPosAaList = ['g', 'a', 'v', 'l', 'm', 'i', 'f', 'w', 'd', 'e', 's', 't', 'c', 'p', 'n', 'q']  # all aas that aren't formally charged at low pH.
allAaList = ['g', 'a', 'v', 'l', 'm', 'i', 'f', 'w', 'd', 'e', 's', 't', 'c', 'p', 'n', 'q', 'k', 'h', 'r']  # all aas
cores = ['ndi', 'pdi']

predAll = True # If true, predChems will be replaced with all possible chemistries matching dxxx-pi, depending on following options
trainAas = True # If true and predAll is true, only chemistries entirely comprised of trainAaList will be predicted
nonpolAas = True # If true and predAll is true and trainAas is false, all chemistries entirely comprised of nonpolar amino acids will be predicted
wingMin = 3  # If predAll == True, this will give the wing sizes to be considered; eg. dxxx would be size 3, while dxx would be 2.
wingMax = 3
aalist = nonPolAaList  # If predChems is not set, or if predAll is True, the peptide chemistries that will be utilized are all possible combinations of dxxx-pi where each x is an amino acid from aalist (eg. if aalist = ['a'] and cores = ['ndi', 'pdi'] then only 'daaa-ndi' and 'daaa-pdi' would be looked at. If 'f' was also included so aalist = ['a', 'f'], then daaf, dafa, dfaa, daff, dfaf, dffa, and dfff would also be looked at.)

pltscatter = True  # Plot a scatter plot of predictions
figname = 'dfScatter'  # Save that scatter plot with this base filename (will save both a .png and a .pdf)
xlims = [-40, 0]  # Limits for x on scatter plot
ylims = [-40, 0]  # Limits for y on scatter plot

trainAlignFnm = 'align.csv'  # Filename containing alignments
trainDataFnm = 'ergs.csv'  # Filename containing free energies
descsFnm = 'features_cut.csv'  # Filename containing descriptors

savepreds = True  # Save the predictions for all chemistries in predChems to a saveCsv
saveCsv = 'predictions.csv'
savekeys = ['df2_pred', 'df3_pred', 'align_pred', 'align_prederr', 'align_predSamp', 'align_predSamperr']  # df2_pred is the predicted dimerization free energy, df3_pred is the predicted trimerization free energy, align_pred is the alignment predicted directly from df2_pred and df3_pred, align_predSamp is the alignment predicted from random sampling of df2_pred and df3_pred using df2err and df3err

samplePreds = True  # Use random sampling of free energies to obtain mean and error estimates for alignments.
df2err = 3.0  # Estimated error in predicted value of df2. 3.0 +/- 1.3 (test rmse over 15 train test splits of this model)
df3err = 3.9  # Estimated error in predicted value of df3. 3.9 +/- 1.6 (test rmse  "")
alSamps = 10000  # Number of different samples of free energy to make to obtain mean sampled estimates and error estimates in aligments
topn = 50  # Print the best topn chemistries


def calcmod(desclist, data):
    '''
    desclist is a list of dfs of descriptors to serve to predict data. Each element will be fit separately and the resulting models will be returned.
    '''
    mods = []
    for descs in desclist:
        mod = lm.LinearRegression()
        mod.fit(descs, data)
        mods.append(mod)

    return(mods)


def getxyz(yfnm='align.csv', xfnm='train_data.csv', ykey='Align', xkeys=['df2', 'df3'], err=True):
    '''
    Gets a series of 3 lists from .csv files
    '''
    df1 = pd.read_csv(yfnm, index_col=0)
    df2 = pd.read_csv(xfnm, index_col=0)
    dat = pd.concat([df1, df2], axis=1).dropna()
    x = dat[xkeys[0]]
    y = dat[xkeys[1]]
    z = dat[ykey]
    if err:
        xerr = dat[xkeys[0]+'err']
        yerr = dat[xkeys[1]+'err']
        zerr = dat[ykey+'err']
        return(x, y, z, xerr, yerr, zerr)

    return(x, y, z)


def getfit(fn, guess, yfnm='align.csv', xfnm='train_data.csv', ykey='Align', xkeys=['df2', 'df3'], err=False, bound=[]):
    x, y, z = getxyz(yfnm=yfnm, xfnm=xfnm, ykey=ykey, xkeys=xkeys, err=err)
    x = x.values
    y = y.values
    z = z.values
    xy = np.row_stack([x,y])

    if bound:
        fit = cf(fn, xy, z, p0=guess, bounds=bound)
    else:
        fit = cf(fn, xy, z, p0=guess)

    return(fit[0])


def gauss(xy, amp=1, mux=0, muy=0, a=.1):
    x = xy[0]
    y = xy[1]
    xterm = a*(x-mux)**2
    yterm = a*(y-muy)**2
    return amp*np.exp(-(xterm + yterm))


if __name__ == '__main__':
    # Make list of chemistries to predict free energies and alignments for.
    if predAll:
        if trainAas:
            aalist = trainAaList
        elif nonpolAas:
            predChems = nonPolAaList

        predChems = ['d{}-{}'.format(''.join(aas), core) for it in [itt.product(aalist, repeat=num) for num in range(wingMin, wingMax+1)] for aas in it for core in cores]

    #Fit gaussian model (other models could be used)
    guess = [.6, -23, -27, .1]
    bound = ([0, -27, -33, 0], [1, -17, -21, 1])
    amp, mux, muy, a = getfit(gauss, guess, bound=bound, yfnm=trainAlignFnm, xfnm=trainDataFnm, ykey='Align', xkeys=['df2', 'df3'])
    sig = np.sqrt(1/(2*a))
    fitparms = [amp, mux, muy, a]
    print('-'*20)
    print('Gaussian parameters: ')
    print('Factor: {}, df2 mean: {}, df3 mean: {}, sigma value: {}\n'.format(amp, mux, muy, sig))

    #Get various data frames read in and zscored
    allfeatNames = list(set([desc for parm in modparms for num in modparms[parm] for desc in num]))
    traindata = pd.read_csv(trainDataFnm, index_col=0)

    print('-'*20)
    print('z-scoring features\n')
    align = pd.read_csv('align.csv', index_col=0)
    descs = pd.read_csv(descsFnm, index_col=0)[allfeatNames]

    descs.dropna(inplace=True)
    descs = (descs-descs.loc[traindata.index].mean())/descs.loc[traindata.index].std()

    if predChems == []:
        predChems = list(descs.index)


    # Form DataFrame for training purposes
    trainDf = descs.loc[traindata.index].copy()
    trainDf['df2'] = traindata['df2']
    trainDf['df3'] = traindata['df3']
    trainDf['al'] = align['Align']

    #Fit models for training features agains training data
    df2feats = [trainDf[parm] for parm in modparms['df2']]
    df3feats = [trainDf[parm] for parm in modparms['df3']]
    df2mod = calcmod(df2feats, trainDf['df2'])
    df3mod = calcmod(df3feats, trainDf['df3'])
    print('-'*20)
    print('Calculated model:\n')
    print('df2 ensembles\n')
    for ensemb,mod in zip(modparms['df2'], df2mod):
        modstr = ' + '.join(['{:.3f}*{}'.format(mod.coef_[i], ensemb[i]) for i in range(len(ensemb))])
        modstr += ' + {:.3f}'.format(mod.intercept_)
        print('df2 = {}'.format(modstr))

    print('\n\ndf3 ensembles\n')
    for ensemb,mod in zip(modparms['df3'], df3mod):
        modstr = ' + '.join(['{:.3f}*{}'.format(mod.coef_[i], ensemb[i]) for i in range(len(ensemb))])
        modstr += ' + {:.3f}'.format(mod.intercept_)
        print('df3 = {}'.format(modstr))

    #Predict free energies of training data based on models
    print('\n'+'-'*20)
    print('Predicting free energies and alignments for chemistries\n')
    preds = []
    for i,mod in enumerate(df2mod):
        preds.append(mod.predict(df2feats[i]))
    trainDf['df2_pred'] = np.mean(preds, axis=0)

    for i,mod in enumerate(df3mod):
        preds.append(mod.predict(df3feats[i]))
    trainDf['df3_pred'] = np.mean(preds, axis=0)


    #Predict free energies for all chemistries
    testDf = descs.copy()

    df2preds = [testDf[parm] for parm in modparms['df2']]
    df3preds = [testDf[parm] for parm in modparms['df3']]

    preds = []
    for i,mod in enumerate(df2mod):
        preds.append(mod.predict(df2preds[i]))
    testDf['df2_pred'] = np.mean(preds, axis=0)

    preds = []
    for i,mod in enumerate(df3mod):
        preds.append(mod.predict(df3preds[i]))
    testDf['df3_pred'] = np.mean(preds, axis=0)

    preds = []

    testDf['align_pred'] = [gauss([testDf['df2_pred'].loc[chem], testDf['df3_pred'].loc[chem]], *fitparms) for chem in testDf.index]

    # Scatter plot of predicted free energies
    if pltscatter:
        print('-'*20)
        print('Creating scatter plot\n')
        testDf.plot.scatter('df2_pred', 'df3_pred', figsize=(12,12))
        plt.xlabel('$\Delta F_2$ / $k_B T$', labelpad=10)
        plt.ylabel('$\Delta F_3$ / $k_B T$', labelpad=0)
        thets = np.linspace(0, 2*np.pi, 200)
        r = sig/2
        plt.plot(mux-r*np.cos(thets), muy-r*np.sin(thets), linewidth=3, color='r')
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.savefig('{}.png'.format(figname))
        plt.savefig('{}.pdf'.format(figname))
        print('Scatter plot saved to {0}.png and {0}.pdf\n'.format(figname))
        plt.show()

    alpreds = {}
    alprederrs = {}

    print('-'*20)
    print('Sampling df2 and df3 to obtain errors in alignment')
    for chem in testDf.index:
        df2 = testDf['df2_pred'].loc[chem]
        df3 = testDf['df3_pred'].loc[chem]
        df2samp = np.random.normal(df2, df2err, alSamps)
        df3samp = np.random.normal(df3, df3err, alSamps)
        alignsamp = gauss([df2samp, df3samp], *fitparms)
        alpreds[chem] = np.mean(alignsamp)
        alprederrs[chem] = np.std(alignsamp)

    testDf['align_predSamp'] = pd.Series(alpreds)
    testDf['align_predSamperr'] = pd.Series(alprederrs)
    testDf['align_pred'] = pd.Series([gauss([testDf['df2_pred'].loc[chem], testDf['df3_pred'].loc[chem]], *fitparms) for chem in testDf.index], index=testDf.index)
    testDf['align_prederr'] = pd.Series([gauss([testDf['df2_pred'].loc[chem], testDf['df3_pred'].loc[chem]], *fitparms)*(np.abs(testDf['df2_pred'].loc[chem]-fitparms[1])*df2err + np.abs(testDf['df3_pred'].loc[chem]-fitparms[2])*df3err)*2*fitparms[3] for chem in testDf.index], index=testDf.index)

    if savepreds:
        testDf[savekeys].loc[predChems].to_csv(saveCsv)
        print('Predictions saved to {}\n'.format(saveCsv))


    topsort = testDf.sort_values('align_predSamp', ascending=False)

    print('-'*20)
    print('Selected chemistries. Predicted df2, df3, alignment directly from df2 and df3, and alignment from sampling df2 and df3 is printed.\n')
    print(testDf.loc[selChems][savekeys])

    print('-'*20)
    print('Top {} chemistries ranked by sampled alignment.\n'.format(topn))
    print(topsort.iloc[:topn][savekeys])

    print('-'*20)
    print('Top {} chemistries ranked by alignment.\n'.format(topn))
    topsort = testDf.sort_values('align_pred', ascending=False)
    print(topsort.iloc[:topn][savekeys])
