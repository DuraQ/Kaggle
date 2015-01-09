# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 23:09:34 2014
Starter to code to plot plankton Class Separation
@author: Laurens
"""

#19/12 changed "ratio" to "mean"
#Because mean outputs a large number, the features is divided by zero

# Loop through the classes two at a time and compare their distributions of the Width/Length Ratio

#Create a DataFrame object to make subsetting the data on the class 
df = pd.DataFrame({"class": y[:], "mean": X[:, num_features+1]})

f = plt.figure(figsize=(30, 20))
#we suppress zeros and choose a few large classes to better highlight the distributions.
df = df.loc[df["mean"] > 0]
minimumSize = 20 
counts = df["class"].value_counts()
largeclasses = [int(x) for x in list(counts.loc[counts > minimumSize].index)]
# Loop through 40 of the classes 
for j in range(0,40,2):
    subfig = plt.subplot(4, 5, j/2 +1)
    # Plot the normalized histograms for two classes
    classind1 = largeclasses[j]
    classind2 = largeclasses[j+1]
    n, bins,p = plt.hist(df.loc[df["class"] == classind1]["mean"].values,\
                         alpha=0.5, bins=[x*0.01 for x in range(100)], \
                         label=namesClasses[classind1].split(os.sep)[-1],normed=1)

    n2, bins,p = plt.hist(df.loc[df["class"] == (classind2)]["mean"].values,\
                          alpha=0.5, bins=bins, label=namesClasses[classind2].split(os.sep)[-1],normed=1)
    subfig.set_ylim([0.,10.])
    plt.legend(loc='upper right')
    plt.xlabel("mean")