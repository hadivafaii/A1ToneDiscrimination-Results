Section 2
=======================

In this phase, we explore whether latent variable modeling approach is suitable for this dataset.  We also combine datasets together to identify global features of the system that is shared across subjects and experimental sessions.

## Content:

### Simple AutoEncoder (AE)
This is the most simple AE I can build.  There is nothing fancy in the architecture and it does not respect temporal dependence in data.  This i.i.d assumption at every time point is completely wrong and this approach will have limited success, if at all.  However, it is still interesting to see what we can learn from these experiments.

%### Slightly less simple autoencoder:
%We explore a lower dimensional projection of the data and observe that such structure exists in data.  Different trial types have different %characteristics that can be captured in a lower dimensional space using LDA analysis.  This structrue exists for individual experiments %separately but we do not know if this can be seen globally.  This will be the subject of next section
