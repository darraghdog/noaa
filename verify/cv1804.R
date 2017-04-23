rm(list=ls())
library(data.table)
library(Metrics)
# Best yolo input size is 
yolo <- fread("~/Downloads/yolo_seals.csv")
rfcn <- fread("~/Dropbox/noaa/feat/comp4_30000_det_test_seals.txt")
# yolo <- fread("~/Downloads/yolo_seals2176.csv")
getwd()

bad_train_ids = c(
  3, 7, 9, 21, 30, 34, 71, 81, 89, 97, 151, 184, 215, 234, 242, 
  268, 290, 311, 331, 344, 380, 384, 406, 421, 469, 475, 490, 499, 
  507, 530, 531, 605, 607, 614, 621, 638, 644, 687, 712, 721, 767, 
  779, 781, 794, 800, 811, 839, 840, 869, 882, 901, 903, 905, 909, 
  913, 927, 946)

# yolo = yolo[V2>.3]
# yolo[,ct:=.N, by=V1]
# yolo[,V1:=NULL]
rfcn = rfcn[V2>.7]
rfcn = rfcn[(V5-V3)<150]
rfcn = rfcn[(V6-V4)<150]
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

par(mfrow=c(2,1))
hist(rowSums(y[!train_id %in% y_pred$img][,-1,with=F]), ylim=c(0,60), xlim=c(0,1200), breaks = 200)
hist(rowSums(y[train_id %in% y_pred$img][,-1,with=F]), ylim=c(0,60), xlim=c(0,1200),  breaks = 200)
hist(rowSums(y[!train_id %in% y_pred$img][,-1,with=F]), breaks = 100)

var = "adult_females"
id = y$train_id %in%  y_pred[V1<300]$img
mean(y[id][train_id %% 2 == 1][!train_id %in% y_pred$img][[var]])
mean(y[id][train_id %% 2 == 1][train_id %in% y_pred$img][[var]])
mean(y[id][train_id %% 2 == 1][[var]])
mean(y[!id][train_id %% 2 == 1][[var]])
id = setdiff(intersect(y$train_id, y_pred$img), bad_train_ids)

y = y[order(train_id)][train_id %in% id]
y_pred = y_pred[order(img)][img %in% id]
y[,sum:=rowSums(y[,-1,with=F])]

# Some plots and correlation
cols
for(var in cols) print(paste(round(cor(y[[var]], y_pred$V1), 3), "...", var))

plot(y$sum, y_pred$V1, xlim=c(0,1300), ylim=c(0,1300))
plot(y$sum, y_pred$V1, xlim=c(0,50), ylim=c(0,50))
par(mfrow=c(3,2))
for(var in cols) hist(y[[var]], main=var, breaks = 100)
par(mfrow=c(3,2))
for(var in cols) plot(y[[var]], y_pred$V1, main=var, xlim = c(0, 400))

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
y_avg = rep.row(avg, nrow(y_pred))
names(y_avg) = cols

mean_rmse(y, y_avg)

# Lets optimise
hist(y_pred$V1, breaks = 100, xlim=c(1,100))
y_opt = y_avg
y_opt$adult_males =  (5* y_pred$V1) / 50
y_opt$adult_females =  (35* y_pred$V1) /50
y_opt$juveniles =  (10* y_pred$V1) / 50 # ... not a big effect
y_opt$pups =  (20* y_pred$V1) / 50
mean_rmse(y, y_opt)




