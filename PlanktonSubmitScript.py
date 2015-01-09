# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 08:21:48 2014

@author: Laurens
"""


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
    image = resize(image, (maxPixel, maxPixel))

    # Store the rescaled image pixels and the axis ratio
    X_test[i, 0:imageSize] = np.reshape(image, (1, imageSize))
    X_test[i, imageSize] = axisratio
    X_test[i, imageSize+1] = region_solidity
 
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
