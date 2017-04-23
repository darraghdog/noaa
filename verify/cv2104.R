rm(list=ls())
library(data.table)
library(Metrics)
# Best yolo input size is 
yolo <- fread("~/Downloads/yolo_seals.csv")
# rfcn <- fread("~/Dropbox/noaa/feat/comp4_30000_det_test_seals.txt")
rfcn <- fread("~/Dropbox/noaa/feat/comp4_30K_det_trainval_seals.txt")
rfcn$V1 = gsub("/home/ubuntu/noaa/darknet/seals/JPEGImagesBlk/", "", rfcn$V1)
# yolo <- fread("~/Downloads/yolo_seals2176.csv")
getwd()

bad_train_ids = c(
  3, 7, 9, 21, 30, 34, 71, 81, 89, 97, 151, 184, 215, 234, 242, 
  268, 290, 311, 331, 344, 380, 384, 406, 421, 469, 475, 490, 499, 
  507, 530, 531, 605, 607, 614, 621, 638, 644, 687, 712, 721, 767, 
  779, 781, 794, 800, 811, 839, 840, 869, 882, 901, 903, 905, 909, 
  913, 927, 946)

# Prune the high proba
rfcn = rfcn[V2>.75]
rfcn = rfcn[(V5-V3)<150]
rfcn = rfcn[(V6-V4)<150]

# rfcn = rfcn[V3>5|V4>5|V2>.85]

rfcn[,ct:=.N, by=V1]

y <- fread("~/Downloads/train-noaa.csv")
y = y[train_id %% 2 == 1]
# View(y)

# y_pred = yolo[,.(.N), by=V1]
y_pred = rfcn[,.(.N), by=V1]
y_pred[,img := as.integer(unlist(lapply(strsplit(V1, "_"), function (x) x[1])))]
y_pred = y_pred[,sum(N), by=img]

# take out bad ids
y = y[!train_id %in% bad_train_ids]
y_pred = y_pred[!img %in% bad_train_ids]


id = setdiff(intersect(y$train_id, y_pred$img), bad_train_ids)
y = y[order(train_id)][train_id %in% id]
y_pred = y_pred[order(img)][img %in% id]
y[,sum:=rowSums(y[,-1,with=F])]

# Check correlation
cor(y$sum, y_pred$V1)
plot(y$sum, y_pred$V1)


# Check up on RMSE
cols = c("adult_males", "subadult_males", "adult_females", "juveniles", "pups")
mean_rmse = function(yact, ypred){
  rmse_vec = c()
  for(var in cols) {
    print(paste0(rmse(yact[[var]], ypred[[var]]), "...", var))
    rmse_vec = c(rmse_vec, rmse(yact[[var]], ypred[[var]]))
  }
  mean(rmse_vec)
}
# Create the avg pred
rep.row<-function(x,n) data.table(matrix(rep(x,each=n),nrow=n))
avg = c(5,	4,	26,	15,	11)
avg = c(5,	4,	40,	20,	19)
y_avg = rep.row(avg, nrow(y_pred))
names(y_avg) = cols

mean_rmse(y, y_avg)

# Lets optimise
y_opt = y_avg
y_opt$adult_males =  0.1 * (y_pred$V1)
y_opt[["adult_males"]][y_opt[["adult_males"]]<0]=0
y_opt$adult_females =  y_pred$V1-50
y_opt[["adult_females"]][y_opt[["adult_females"]]<0]=0
y_opt$juveniles =  0.3 * (y_pred$V1-50)
y_opt[["juveniles"]][y_opt[["juveniles"]]<0]=0
y_opt$pups =  0.6 * (y_pred$V1-50)
y_opt[["pups"]][y_opt[["pups"]]<0]=0
mean_rmse(y, y_opt)

# Plot
plot(y$adult_males, y_pred$V1)
plot(y$adult_females, y_opt$adult_females)

############################
# Make a sub
############################
rm(list=ls())
rfcn <- fread("~/Dropbox/noaa/feat/comp4_30000_det_test_all_seals.txt")
ssub = fread("~/Dropbox/noaa/data/sample_submission.csv")

# Prune the high proba
rfcn = rfcn[V2>.85]
rfcn = rfcn[(V5-V3)<150]
rfcn = rfcn[(V6-V4)<150]
# rfcn = rfcn[V3>5|V4>5|V2>.85]
rfcn[,ct:=.N, by=V1]
y_pred = rfcn[,.(.N), by=V1]

y_pred$V1 = gsub("/home/ubuntu/noaa/darknet/seals/JPEGImagesTest/", "", y_pred$V1)
y_pred[,img := as.integer(unlist(lapply(strsplit(V1, "_"), function (x) x[1])))]
y_pred = y_pred[,sum(N), by=img]
y_pred = y_pred[order(img)]

# Check up on RMSE
cols = c("adult_males", "subadult_males", "adult_females", "juveniles", "pups")
# Create the avg pred
rep.row<-function(x,n) data.table(matrix(rep(x,each=n),nrow=n))
avg = c(5,	4,	26,	15,	11)
y_avg = cbind(y_pred$img, rep.row(avg, nrow(y_pred)))
names(y_avg) = c("test_id", cols)


# Lets optimise
y_opt = y_avg
y_opt$adult_males =  0.1 * (y_pred$V1)
y_opt[["adult_males"]][y_opt[["adult_males"]]<0]=0
y_opt$adult_females =  y_pred$V1-50
y_opt[["adult_females"]][y_opt[["adult_females"]]<0]=0
y_opt$juveniles =  0.3 * (y_pred$V1-50)
y_opt[["juveniles"]][y_opt[["juveniles"]]<0]=0
y_opt$pups =  0.6 * (y_pred$V1-50)
y_opt[["pups"]][y_opt[["pups"]]<0]=0
hist(y_pred$V1, breaks = 300)

# Add the missing ones
y_opt = rbind(y_opt, ssub[!test_id %in% y_opt$test_id])
y_opt = y_opt[order(test_id)]
for (var in cols) y_opt[[var]] = round(y_opt[[var]]) 

# # Now lets do a blend with the average
# y_avg = cbind(y_opt$test_id, rep.row(avg, nrow(y_opt)))
# names(y_avg) = c("test_id", cols)
# for (var in cols) y_opt[[var]] = round( 0.6*  y_opt[[var]] + 0.4 *  y_avg[[var]]  ) 

# Lets try a sub
write.csv(y_opt, paste0("~/Dropbox/noaa/sub/sub-RFCN-avgblend-", Sys.Date(), ".csv"), row.names = F)

# 0.75 cut off was 25.26
# 0.85 cut off was 22.34
# 0.9 cut off was 24.36
# Avg blend was 22.85 - no improvement.