rm(list=ls())
library(data.table)
yolo <- fread("~/Downloads/yolo_seals.csv")
getwd()

bad_train_ids = c(
  3, 7, 9, 21, 30, 34, 71, 81, 89, 97, 151, 184, 215, 234, 242, 
  268, 290, 311, 331, 344, 380, 384, 406, 421, 469, 475, 490, 499, 
  507, 530, 531, 605, 607, 614, 621, 638, 644, 687, 712, 721, 767, 
  779, 781, 794, 800, 811, 839, 840, 869, 882, 901, 903, 905, 909, 
  913, 927, 946)

yolo = yolo[V2>.3]
yolo[,ct:=.N, by=V1]
yolo[,V1:=NULL]

y <- fread("~/Downloads/train-noaa.csv")
y = y[train_id %% 2 == 1]
# View(y)

y_pred = yolo[,.(.N), by=V1]
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

par(mfrow=c(1,1))
cor(y$sum, y_pred$V1)
plot(y$sum, y_pred$V1, xlim=c(0,1300), ylim=c(0,1300))
# [1] 0.6977999
