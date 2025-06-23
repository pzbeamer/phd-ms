
library('SpatialPCA')
sample_names=c("151507", "151508", "151509", "151510", "151669", "151670", "151671" ,"151672","151673", "151674" ,"151675" ,"151676")
i=9 # Here we take the 9th sample as example, in total there are 12 samples (numbered as 1-12), the user can test on other samples if needed.
clusterNum=c(7,7,7,7,5,5,5,5,7,7,7,7) # each sample has different ground truth cluster number
load(paste0("DLPFC/LIBD_sample9.RData")) 
print(dim(count_sub)) # The count matrix
print(dim(xy_coords)) # The x and y coordinates. We flipped the y axis for visualization.

# location matrix: n x 2, count matrix: g x n.
# here n is spot number, g is gene number.
xy_coords = as.matrix(xy_coords)
rownames(xy_coords) = colnames(count_sub) # the rownames of location should match with the colnames of count matrix
LIBD = CreateSpatialPCAObject(counts=count_sub, location=xy_coords, project = "SpatialPCA",gene.type="spatial",sparkversion="spark",numCores_spark=5,gene.number=3000, customGenelist=NULL,min.loctions = 20, min.features=20)
LIBD = SpatialPCA_buildKernel(LIBD, kerneltype="gaussian", bandwidthtype="SJ",bandwidth.set.by.user=NULL)
LIBD = SpatialPCA_EstimateLoading(LIBD,fast=FALSE,SpatialPCnum=20) 
LIBD = SpatialPCA_SpatialPCs(LIBD, fast=FALSE)

