## Users’ Manual of PseU-ST (Version 1.0.0)

Pseudouridine (Ψ) is one of the most abundant RNA modifications found in a variety RNA types, and it plays a significant role in many biological processes. The key to studying the various biochemical functions and mechanisms of Ψ is to identify the corresponding Ψ sites. However, identifying Ψ sites using experimental methods is expensive and time-consuming. Therefore, it is necessary to develop computational methods which can accurately predict Ψ sites based on RNA sequence information. In this study, we proposed a new model called PseU-ST to identify Ψ sites in Homo sapiens (H. sapiens), Saccharomyces cerevisiae (S. cerevisiae), and Mus musculus (M. musculus). We selected the best six coding schemes and four machine learning algorithms based on a comprehensive test of almost all RNA sequence coding schemes in the iLearnPlus software package, and selected the optimal features for each encoding scheme using chi-square and incremental feature selection (IFS) algorithms. Then, we selected the optimal feature combination and the best base-classifier combination for each species through an extensive performance comparison and employed a stacking strategy to build the predictive model. Empirical performance benchmarking tests demonstrated that PseU-ST achieved better prediction performance compared to other existing models. PseU-ST’s accuracy scores were 93.64%, 87.74%, and 89.64% on H_990, S_628, and M_944, respectively, representing increments of 13.94%, 6.05%, and 0.26% versus the best existing methods on the same benchmark training datasets. The data indicate that PseU-ST is a very competitive prediction model for identifying RNA Ψ sites in H. sapiens, M. musculus, and S. cerevisiae. The code and data used in PseU-ST are available at (https://github.com/jluzhangxinrubio/PseU-ST/). In addition, we found that the Position-specific trinucleotide propensity based on single strand (PSTNPss) and Position-specific of three nucleotides (PS3) features play an important role in Ψ site identification.

### Preamble
*Download and install the anaconda (64 bit) platform: Download from https://www.anaconda.com/download/ .
*Install sklearn 
*Install mxltend

### Feature descriptor extraction
*Download the training and independent testing datasets from http://lin-group.cn/server /iRNAPseu/data, and organize these datasets into special FASTA format.
*Input the formatted sequences into ilearnplus (https://ilearnplus.erc.monash.edu/).
*Select the feature descriptor and configure the descriptor parameters: In “Select feature descriptor” panel, select the various descriptor such as “PSTNPss”and “PS3”; In “Select output format for feature” panel, select the CSV format; In “Cluster methods” panel, select the K-Means; In “Feature normalization methods” panel, select the ZScore ; In “Feature selection methods” panel, select the Chi-Square, and enter the appropriate dimension number in “Number of selected features”panel; In “Dimension reduction methods” panel, select the tsne. Finally, select “RF” as the machine learning algorithm; Select “5-fold cross-validation” as evaluation strategy. 
*Save the results: Download the output results, then merge the “Feature_selection
_testing” and “Feature_selection_training” in the output results obtained by various feature extraction methods, and rename them to “testing_set” and “training_set” respectively.

### Run program 
*Put the code, “testing_set”, and "training_set" in the same folder.
*Open Anaconda platform, run the code using Python, and results can be obtained. 
