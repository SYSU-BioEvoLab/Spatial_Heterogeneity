# Spatial_Heterogeneity

Title: Mutation divergence over space in tumour expansion
https://www.biorxiv.org/content/10.1101/2022.12.21.521509v1

Mutation accumulation in tumour evolution is one major cause of intra-tumour heterogeneity (ITH), which often leads to drug resistance during treatment. Previous studies with multi-region sequencing have shown that mutation divergence among samples within the patient is common, and the importance of spatial sampling to obtain a complete picture in tumour measurements. However, quantitative comparisons of the relationship between mutation heterogeneity and tumour expansion modes, sampling distances as well as the sampling methods are still few. Here, we investigate how mutations diverge over space by varying the sampling distance and tumour expansion modes using individual based simulations. We measure ITH by the Jaccard index between samples and quantify how ITH increases with sampling distance, the pattern of which holds in various sampling methods and sizes. We also compare the inferred mutation rates based on the distributions of Variant Allele Frequencies (VAF) under different tumour expansion modes and sampling sizes. In exponentially fast expanding tumours, a mutation rate can always be inferred in any sampling size. However, the accuracy compared to the true value decreases when the sampling size decreases, where small sampling sizes result in a high estimate of the mutation rate. In addition, such an inference becomes unreliable when the tumour expansion is slower, such as in surface growth.



<img width="1671" alt="image" src="https://user-images.githubusercontent.com/88959742/226506959-497673c9-cd2e-43a2-bb1d-5305a8b21777.png">
Figure1. The spatial growth under different push rates.a) when push rate p=0,it refers to the surface growth, where only cells on the surface have an empty spot in its direct neighbours and can divide. For other values of the push rate, cells not on the surface can also divide by creating an empty spot among its neighbours through pushing. We allocated a unique ID for each new cell. The larger the cell ID, the later the cell was reproduced. b) To reach the same tumour size, surface growth takes more generations of reproductions compared to exponential growth under
p = 1. While mutation accumulation only happens in cell divisions in our model, for tumours of the same size, the push rate will impact on the mutation burden in tumours and also the spatial distribution of those mutations. The shadowed area is 100 simulations under the corresponding push rate, where their averages are shown as solid lines. (The fig.1b is 100 times simulation, cell number is around 215, p = 1 grow to 215 cells only 15 generations. when p = 0 divide 100 generations can grow about 27500 cells)



Growth and mutation distribution in cancer models
![image](https://user-images.githubusercontent.com/88959742/226506803-30c608bf-1480-4feb-8d3e-c5b1e5bf5797.png)
Figure2. The impact of push rates on distributions of early mutations.a) We record the identities of early mutations, including the mutation ID and the time when they were first present in the population. b) One example of the spatial distribution of unique mutations from four cells after the second round of cell division. (the distribution of 4 types of mutation in the second generation, cell number is about 214). c) Frequencies of all early mutations in growing tumours. These mutations arise at different rounds of early cell divisions (300 times simulation, cell number is about 214, cut last 3 points). While the push rates change, thus the spatial distributions of those mutations differ as in b); their final frequencies are 0.5, 0.25, 0.125, 0.0625 and 0.03125 with the push rate. Here, the red lines in colourful bars are the average mean value of frequencies of all mutations from 300 times simulation realisations, and the bars are the variance. (λ = 10, the final tumour size is 214.)

The heterogeneity of tumour cells was calculated by Jaccard index of sampling points
<img width="950" alt="image" src="https://user-images.githubusercontent.com/88959742/226507119-e84ae334-27e4-47cf-bc9a-fe67fd850d81.png">
Figure4. The mutation divergence over space. Pairwise comparison of mutation divergences between samples are measured by Jaccard Index. As the sampling distance increases, fewer mutations are shared between the samples, and the smaller the Jaccard Index is. a), p = 0. b), p = 0.125. c), p = 1. This holds for any push rates and any sampling size. For the surface growth a), the Jaccard index decreases faster to 0 when the sampling distance increases compared to exponential growth c). This effect is stronger when the sampling size is smaller (e.g. orange dots compared to purple dots). (1-time simulation, sampling 500 points for each sample size, cell number is about 214, cut last 3 points)




Author contributions statement:
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
