rm(list=ls())
gc();gc();gc();gc();gc();gc();gc();gc();gc()
library(data.table)
library(Hmisc)
library(xgboost)
library(Metrics)
setwd("~/Dropbox/noaa/coords")

# Functions
getBreaks = function(df, brk, nm){
  for(i in brk){
    dftmp = df[,(sum(predSeal>i)),by = "bigimg"]
    names(dftmp)[2] = paste0(nm, "above", i)
    if (exists("dffull")){
      dffull = cbind(dffull, dftmp[order(bigimg)][,2,with=F])
    }else{
      dffull = dftmp[order(bigimg)]
    }
  }
  names(dffull)[1] = "img"
  dffull[["img"]] = as.integer(dffull[["img"]])
  dffull[order(img)]
}


# List of bad training files
bad_train_ids = c(
  3, 7, 9, 21, 30, 34, 71, 81, 89, 97, 151, 184, 215, 234, 242, 
  268, 290, 311, 331, 344, 380, 384, 406, 421, 469, 475, 490, 499, 
  507, 530, 531, 605, 607, 614, 621, 638, 644, 687, 712, 721, 767, 
  779, 781, 794, 800, 811, 839, 840, 869, 882, 901, 903, 905, 909, 
  913, 927, 946)

# Load up training and test files
meta_trn <- fread("train_meta1.csv")[,-1,with=F]
meta_tst <- fread("test_meta1.csv")[,-1,with=F]
target_cols = names(meta_trn)[2:6]
resnfile = c(paste0("resnet50CVPreds2604_fold", 1:2, ".csv"))
vggfile = c(paste0("vggCVPreds2604_fold", 1:2, ".csv"))
resn50trn = rbind(fread(resnfile[1]), fread(resnfile[2]))
vggtrn = rbind(fread(vggfile[1]), fread(vggfile[2]))
resn50tst = fread("resnet50TestPreds2604.csv")  
vggtst = fread("vggTestPreds2604.csv")

# Get a function to break up the block predictions
vggtrn[, bigimg := unlist(lapply(strsplit(img, "_"), function(x) x[1]))]
vggtst[, bigimg := unlist(lapply(strsplit(img, "_"), function(x) x[1]))]
vggbigtrn = getBreaks(vggtrn, c(seq(.2,.8,.2), .9, .95), "vgg")
vggbigtst = getBreaks(vggtst, c(seq(.2,.8,.2), .9, .95), "vgg")
resn50trn[, bigimg := unlist(lapply(strsplit(img, "_"), function(x) x[1]))]
resn50tst[, bigimg := unlist(lapply(strsplit(img, "_"), function(x) x[1]))]
resn50bigtrn = getBreaks(resn50trn, c(seq(.2,.8,.2), .9, .95), "resn50")
resn50bigtst = getBreaks(resn50tst, c(seq(.2,.8,.2), .9, .95), "resn50")

ct_trn = merge(resn50bigtrn, vggbigtrn, all = T, by = "img") 
ct_tst = merge(resn50bigtst, vggbigtst, all = T, by = "img") 

# Create training and test sets
names(meta_tst)[1] = names(meta_trn)[1]  = "img"
Xtrn = merge(meta_trn, ct_trn, all = T, by = "img")
Xtrn = Xtrn[!Xtrn$img %in% bad_train_ids]
Xtst = merge(meta_tst, ct_tst, all = T, by = "img")
Xtrn[is.na(Xtrn)] <- 0
Xtst[is.na(Xtst)] <- 0
Ytrn = Xtrn[,target_cols, with=F]
trn_img = Xtrn$img
tst_img = Xtst$img
folds = list(trn_img%%2==0, trn_img%%2==1)
Xtrn = as.matrix(Xtrn[,setdiff(names(Xtrn), c(target_cols, "img")), with=F] )
Xtst = as.matrix(Xtst[,setdiff(names(Xtst), c(target_cols, "img")), with=F] )

# Model Cross validate
result = c()
for( var in target_cols){
  set.seed(100)
  y = Ytrn[[var]]
  result_tmp = rep(NA, length(y))
  for(fold in folds){
    train.xgb <- xgb.DMatrix(Xtrn[fold,], label = y[fold])
    model <- xgb.train(data = train.xgb,
                    eta = 0.1,
                    nrounds = 400,
                    verbose = 0,
                    colsample_bytree = 0.7,
                    max_depth = 4,
                    objective = 'reg:linear',
                    eval_metric = 'rmse')
    result_tmp[!fold] = pmax(0, round(predict(model, Xtrn[!fold,])))
  }
  print(paste0("Results for: ", var))
  #print(xgb.importance(model, feature_names = colnames(Xtrn))[1:8,])
  result = c(result, rmse(y, result_tmp))
  print(result[length(result)])
  print("")
}
print(mean(result))

# [1] "Results for: adult_males"
# [1] 4.055452
# [1] "Results for: subadult_males"
# [1] 6.590218
# [1] "Results for: adult_females"
# [1] 35.25834
# [1] "Results for: juveniles"
# [1] 45.02745
# [1] "Results for: pups"
# [1] 30.59307
# > print(mean(result))
# [1] 24.30491


# Make the sub file
subDt = data.table(data.frame(test_id = tst_img))
for( var in target_cols){
  set.seed(100)
  y = Ytrn[[var]]
  train.xgb <- xgb.DMatrix(Xtrn, label = y)
  model <- xgb.train(data = train.xgb,
                       eta = 0.1,
                       nrounds = 400,
                       verbose = 0,
                       colsample_bytree = 0.7,
                       max_depth = 4,
                       objective = 'reg:linear',
                       eval_metric = 'rmse')
  subDt[[var]] = pmax(0, round(predict(model, Xtst)))
  }

sub_ct = fread("../sub/sub-RFCN-2017-04-27-0.2_tune.csv")
sub_out = sub_ct
for(var in target_cols) print(paste0(round(cor(sub_out[[var]], sub_ct[[var]]), 3), " ... ", var))
par(mfrow=c(2,3))
for(var in target_cols) plot(sub_ct[[var]], subDt[[var]], main = var)
for(var in target_cols) sub_out[[var]] = round((0.5*sub_ct[[var]]) + (0.5*subDt[[var]]))
sub_out
sub_ct
subDt

write.csv(sub_out, paste0("../sub/sub-xgbA-ct-", Sys.Date(), ".csv"), row.names = F)


  