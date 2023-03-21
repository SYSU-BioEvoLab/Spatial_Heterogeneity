# Spatial_Heterogeneity

Title: Mutation divergence over space in tumour expansion
weblink: https://www.biorxiv.org/content/10.1101/2022.12.21.521509v1

Mutation accumulation in tumour evolution is one major cause of intra-tumour heterogeneity (ITH), which often leads to drug resistance during treatment. Previous studies with multi-region sequencing have shown that mutation divergence among samples within the patient is common, and the importance of spatial sampling to obtain a complete picture in tumour measurements. However, quantitative comparisons of the relationship between mutation heterogeneity and tumour expansion modes, sampling distances as well as the sampling methods are still few. Here, we investigate how mutations diverge over space by varying the sampling distance and tumour expansion modes using individual based simulations. We measure ITH by the Jaccard index between samples and quantify how ITH increases with sampling distance, the pattern of which holds in various sampling methods and sizes. We also compare the inferred mutation rates based on the distributions of Variant Allele Frequencies (VAF) under different tumour expansion modes and sampling sizes. In exponentially fast expanding tumours, a mutation rate can always be inferred in any sampling size. However, the accuracy compared to the true value decreases when the sampling size decreases, where small sampling sizes result in a high estimate of the mutation rate. In addition, such an inference becomes unreliable when the tumour expansion is slower such as in surface growth.



Author contributions statement
W.H., B.W., G.D. conceived and supervised this study, H.L. W. H., and B. W. developed the methods and modelling ideas. H.Y. L. implemented computational simulations and analysis. F. T. , Y. H., L.D. , Z. Y. ,X.F. and L. W. assisted H. L. in completing parts of the algorithms. H.L., B.W. and W.H wrote the manuscript with the input of all other authors.

[Cancer growth model]
  [2023/3/28]

   2^15 cancer cell    mutation rate =10  with push rate
Plot VAF   cumulative      fitness cumulative

 The code using the Object functions and structs to define the ID
 and mutation type of daughter cell, and the overall code
 is shorter and several times faster, in addition, the test
 code could detect the whole cancer cell VAF.
'''
