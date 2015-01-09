# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 12:18:17 2014
National Data Science Bowl - Plankton Identification
@author: Laurens
"""
#Import libraries for doing image analysis
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier as RF
import glob
import os
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from matplotlib import colors
from pylab import cm
from skimage import segmentation
from skimage.morphology import watershed
from skimage import measure
from skimage import morphology
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import peak_local_max
# make graphics inline
#%pyplot inline

import warnings
warnings.filterwarnings("ignore")
 
# get the classnames from the directory structure. 
directory_names = list(set(glob.glob(os.path.join("C:\Users\Laurens\Documents\Kaggle\National Data Science Bowl","train", "*"))\
 ).difference(set(glob.glob(os.path.join("C:\Users\Laurens\Documents\Kaggle\National Data Science Bowl","train","*.*")))))

# Example image
# This example was chosen for because it has two noncontinguous pieces
# that will make the segmentation example more illustrative
example_file = glob.glob(os.path.join(directory_names[12],"*.jpg"))[12]
#example_file2 = glob.glob(os.path.join(directory_names[12],"*.jpg"))[13]
print example_file
im = imread(example_file, as_grey=True)
#im2 = imread(example_file2, as_grey=True)
plt.imshow(im, cmap=cm.gray)
plt.show()
#plt.imshow(im2, cmap=cm.gray)
#plt.show()

# First we threshold the image by only taking values greater than the mean to reduce noise in the image
# to use later as a mask
f = plt.figure(figsize=(12,3))
imthr = im.copy()
imthr = np.where(im > np.mean(im),0.,1.0)
sub1 = plt.subplot(1,4,1)
plt.imshow(im, cmap=cm.gray)
sub1.set_title("Original Image")

sub2 = plt.subplot(1,4,2)
plt.imshow(imthr, cmap=cm.gray_r)
sub2.set_title("Thresholded Image")

imdilated = morphology.dilation(imthr, np.ones((4,4)))
sub3 = plt.subplot(1, 4, 3)
plt.imshow(imdilated, cmap=cm.gray_r)
sub3.set_title("Dilated Image")

labels = measure.label(imdilated)#returns distinct clusters of connected pixels
labels = imthr*labels
labels = labels.astype(int)
sub4 = plt.subplot(1, 4, 4)
sub4.set_title("Labeled Image")
plt.imshow(labels)    

# calculate common region properties for each region within the segmentation
regions = measure.regionprops(labels)
# find the largest nonzero region
def getLargestRegion(props=regions, labelmap=labels, imagethres=imthr):
    regionmaxprop = None
    for regionprop in props:
        # check to see if the region is at least 50% nonzero
        if sum(imagethres[labelmap == regionprop.label])*1.0/regionprop.area < 0.50:
            continue
        if regionmaxprop is None:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    return regionmaxprop
    
regionmax = getLargestRegion()
plt.imshow(np.where(labels == regionmax.label,1.0,0.0))
plt.show()

print regionmax.minor_axis_length/regionmax.major_axis_length

#Create repeatable function for the steps above
def getMinorMajorRatio(image):
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image),0.,1.0)

    #Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((4,4)))

    # Create the label list
    label_list = measure.label(imdilated, neighbors=8)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)
    
    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)
    
    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
        ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
             
    return ratio
    
#Create repeatable function for the steps above - calculate solidity (convex hull vs. region size)
def getSolidity(image):
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image),0.,1.0)

    #Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((4,4)))

    # Create the label list
    label_list = measure.label(imdilated, neighbors=8)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)
    
    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)
    
    # guard against cases where the segmentation fails by providing zeros
    solidity = 0.0
    if ((not maxregion is None) and  (maxregion.solidity != 0.0)):
        solidity = 0.0 if maxregion is None else  maxregion.solidity
    return solidity
    

#Create repeatable function for the steps above - calculate solidity (convex hull vs. region size)
def getPerimeter(image):
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image),0.,1.0)

    #Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((4,4)))

    # Create the label list
    label_list = measure.label(imdilated, neighbors=8)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)
    
    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)
    
    # guard against cases where the segmentation fails by providing zeros
    perimeter = 0.0
    if ((not maxregion is None) and  (maxregion.perimeter != 0.0)):
        perimeter = 0.0 if maxregion is None else  maxregion.perimeter
    return perimeter
    
#Create repeatable function for the steps above - calculate solidity (convex hull vs. region size)
def getArea(image):
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image),0.,1.0)

    #Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((4,4)))

    # Create the label list
    label_list = measure.label(imdilated, neighbors=8)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)
    
    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)
    
    # guard against cases where the segmentation fails by providing zeros
    filled_area = 0.0
    if ((not maxregion is None) and  (maxregion.filled_area != 0.0)):
        filled_area = 0.0 if maxregion is None else  maxregion.filled_area
    return filled_area
    
 # Rescale the images and create the combined metrics and training labels
print "Counting total number of images"
#get the total training images
numberofImages = 0
for folder in directory_names:
    for fileNameDir in os.walk(folder):   
        for fileName in fileNameDir[2]:
             # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            numberofImages += 1

# We'll rescale the images to be 25x25
maxPixel = 50
imageSize = maxPixel * maxPixel
num_rows = numberofImages # one row for each image in the training dataset
num_features = imageSize + 3 # 1 for our ratio, 1 for solidity, 1 for perimeter/area ratio

print "Creating the initial feature vector"
# X is the feature vector with one row of features per image
# consisting of the pixel values and our metrics
X = np.zeros((num_rows, num_features), dtype=float)
# y is the numeric class label 
y = np.zeros((num_rows))

files = []
# Generate training data
i = 0    
label = 0
# List of string of class names
namesClasses = list()

print "Reading images"
# Navigate through the list of directories
for folder in directory_names:
    # Append the string class name for each class
    currentClass = folder.split(os.pathsep)[-1]
    namesClasses.append(currentClass)
    for fileNameDir in os.walk(folder):   
        for fileName in fileNameDir[2]:
            # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            
            # Read in the images and create the features
            nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)            
            image = imread(nameFileImage, as_grey=True)
            files.append(nameFileImage)
            axisratio = getMinorMajorRatio(image)
            region_solidity = getSolidity(image)
            region_perimeter = getPerimeter(image)
            region_area = getArea(image)
            if region_perimeter == 0:
                region_perimeter = 1
            if region_area == 0.0:
                region_area = 1.0
            perimeter_area_ratio = region_perimeter/region_area
            
            print "resizing " + str(i) + " of 30336"
            image = resize(image, (maxPixel, maxPixel))
            
            # Store the rescaled image pixels and the ratios
            X[i, 0:imageSize] = np.reshape(image, (1, imageSize))
            X[i, imageSize] = axisratio
            X[i, imageSize+1] = region_solidity
            X[i, imageSize+2] = perimeter_area_ratio           
            
            # Store the classlabel
            y[i] = label
            i += 1
            # report progress for each 5% done  
            report = [int((j+1)*num_rows/20.) for j in range(20)]
            if i in report: print np.ceil(i *100.0 / num_rows), "% done"
    label += 1
    
print "Training"
# n_estimators is the number of decision trees
# max_features also known as m_try is set to the default value of the square root of the number of features
print "Creating Classifier"
clf = RF(n_estimators=500, n_jobs=3);
scores = cross_validation.cross_val_score(clf, X, y, cv=5, n_jobs=1);
print "Accuracy of all classes"
print np.mean(scores)

    
def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss
    
# Get the probability predictions for computing the log-loss function
#Difference between K-folds below and above is use of clf.predict.proba()
kf = KFold(y, n_folds=5)
# prediction probabilities number of samples, by number of classes
y_pred = np.zeros((len(y),len(set(y))))
for train, test in kf:
    X_train, X_test, y_train, y_test = X[train,:], X[test,:], y[train], y[test]
    clf = RF(n_estimators=200, n_jobs=3)
    clf.fit(X_train, y_train)
    y_pred[test] = clf.predict_proba(X_test)
#print classification_report(y, y_pred, target_names=namesClasses)

multiclass_log_loss(y, y_pred)    


header = "acantharia_protist_big_center,acantharia_protist_halo,acantharia_protist,amphipods,appendicularian_fritillaridae,appendicularian_s_shape,appendicularian_slight_curve,appendicularian_straight,artifacts_edge,artifacts,chaetognath_non_sagitta,chaetognath_other,chaetognath_sagitta,chordate_type1,copepod_calanoid_eggs,copepod_calanoid_eucalanus,copepod_calanoid_flatheads,copepod_calanoid_frillyAntennae,copepod_calanoid_large_side_antennatucked,copepod_calanoid_large,copepod_calanoid_octomoms,copepod_calanoid_small_longantennae,copepod_calanoid,copepod_cyclopoid_copilia,copepod_cyclopoid_oithona_eggs,copepod_cyclopoid_oithona,copepod_other,crustacean_other,ctenophore_cestid,ctenophore_cydippid_no_tentacles,ctenophore_cydippid_tentacles,ctenophore_lobate,decapods,detritus_blob,detritus_filamentous,detritus_other,diatom_chain_string,diatom_chain_tube,echinoderm_larva_pluteus_brittlestar,echinoderm_larva_pluteus_early,echinoderm_larva_pluteus_typeC,echinoderm_larva_pluteus_urchin,echinoderm_larva_seastar_bipinnaria,echinoderm_larva_seastar_brachiolaria,echinoderm_seacucumber_auricularia_larva,echinopluteus,ephyra,euphausiids_young,euphausiids,fecal_pellet,fish_larvae_deep_body,fish_larvae_leptocephali,fish_larvae_medium_body,fish_larvae_myctophids,fish_larvae_thin_body,fish_larvae_very_thin_body,heteropod,hydromedusae_aglaura,hydromedusae_bell_and_tentacles,hydromedusae_h15,hydromedusae_haliscera_small_sideview,hydromedusae_haliscera,hydromedusae_liriope,hydromedusae_narco_dark,hydromedusae_narco_young,hydromedusae_narcomedusae,hydromedusae_other,hydromedusae_partial_dark,hydromedusae_shapeA_sideview_small,hydromedusae_shapeA,hydromedusae_shapeB,hydromedusae_sideview_big,hydromedusae_solmaris,hydromedusae_solmundella,hydromedusae_typeD_bell_and_tentacles,hydromedusae_typeD,hydromedusae_typeE,hydromedusae_typeF,invertebrate_larvae_other_A,invertebrate_larvae_other_B,jellies_tentacles,polychaete,protist_dark_center,protist_fuzzy_olive,protist_noctiluca,protist_other,protist_star,pteropod_butterfly,pteropod_theco_dev_seq,pteropod_triangle,radiolarian_chain,radiolarian_colony,shrimp_caridean,shrimp_sergestidae,shrimp_zoea,shrimp-like_other,siphonophore_calycophoran_abylidae,siphonophore_calycophoran_rocketship_adult,siphonophore_calycophoran_rocketship_young,siphonophore_calycophoran_sphaeronectes_stem,siphonophore_calycophoran_sphaeronectes_young,siphonophore_calycophoran_sphaeronectes,siphonophore_other_parts,siphonophore_partial,siphonophore_physonect_young,siphonophore_physonect,stomatopod,tornaria_acorn_worm_larvae,trichodesmium_bowtie,trichodesmium_multiple,trichodesmium_puff,trichodesmium_tuft,trochophore_larvae,tunicate_doliolid_nurse,tunicate_doliolid,tunicate_partial,tunicate_salp_chains,tunicate_salp,unknown_blobs_and_smudges,unknown_sticks,unknown_unclassified".split(',')
labels = map(lambda s: s.split('\\')[-1], namesClasses)
#get the total test images
print "Getting all test images"
fnames = glob.glob(os.path.join("C:\Users\Laurens\Documents\Kaggle\National Data Science Bowl", "test", "*.jpg"))
numberofTestImages = len(fnames)
X_test = np.zeros((numberofTestImages, num_features), dtype=float)
images = map(lambda fileName: fileName.split('\\')[-1], fnames)

i = 0
# report progress for each 5% done  
report = [int((j+1)*numberofTestImages/20.) for j in range(20)]
for fileName in fnames:
    # Read in the images and create the features
    print "processing test image " + str(i) + " of a lot"
    image = imread(fileName, as_grey=True)
    axisratio = getMinorMajorRatio(image)
    region_solidity = getSolidity(image)
    region_perimeter = getPerimeter(image)
    region_area = getArea(image)
    if region_perimeter == 0:
        region_perimeter = 1
    if region_area == 0.0:
        region_area = 1.0
    perimeter_area_ratio = region_perimeter/region_area
    image = resize(image, (maxPixel, maxPixel))

    # Store the rescaled image pixels and the axis ratio
    X_test[i, 0:imageSize] = np.reshape(image, (1, imageSize))
    X_test[i, imageSize] = axisratio
    X_test[i, imageSize+1] = region_solidity
    X_test[i, imageSize+2] = perimeter_area_ratio
 
    i += 1
    if i in report: print np.ceil(i *100.0 / numberofTestImages), "% done"

print "Begin prediction of test images"    
y_pred = clf.predict_proba(X_test)
print "Prediction of test images complete"
y_pred.shape
df = pd.DataFrame(y_pred, columns=labels, index=images)
df.index.name = 'image'
df = df[header]
df.to_csv("C:\Users\Laurens\Documents\Kaggle\National Data Science Bowl\submission.csv")
