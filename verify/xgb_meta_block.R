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
    dftmp = df[,(sum(predSeal>i)),by = "img"]
    names(dftmp)[2] = paste0(nm, "above", i)
    if (exists("dffull")){
      dffull = cbind(dffull, dftmp[order(img)][,2,with=F])
    }else{
      dffull = dftmp[order(img)]
    }
  }
  return(dffull)
}


# List of bad training files
bad_train_ids = c(
  3, 7, 9, 21, 30, 34, 71, 81, 89, 97, 151, 184, 215, 234, 242, 
  268, 290, 311, 331, 344, 380, 384, 406, 421, 469, 475, 490, 499, 
  507, 530, 531, 605, 607, 614, 621, 638, 644, 687, 712, 721, 767, 
  779, 781, 794, 800, 811, 839, 840, 869, 882, 901, 903, 905, 909, 
  913, 927, 946)

# Load up training and test meta files
meta_trn1 <- fread("train_meta1.csv")[,-1,with=F]
meta_tst1 <- fread("test_meta1.csv")[,-1,with=F]
meta_trn2 <- fread("train_meta2.csv")[,1:3,with=F]
meta_tst2 <- fread("test_meta2.csv")[,1:3,with=F]
names(meta_trn2)[1] = names(meta_tst2)[1] = "id"
meta_trn = merge(meta_trn1, meta_trn2, all=T,by="id")
meta_tst = merge(meta_tst1, meta_tst2, all=T,by="id")
rm(meta_trn1, meta_trn2, meta_tst1, meta_tst2)

# Load up training and test meta block files
meta_block_trn1 <- fread("train_block_meta1.csv")[,-1,with=F][mask_size<1]
meta_block_tst1 <- fread("test_block_meta1.csv")[,-1,with=F][mask_size<1]
meta_block_trn1$id_block = gsub(".jpg", "", meta_block_trn1$id_block)
meta_block_tst1$id_block = gsub(".jpg", "", meta_block_tst1$id_block)
meta_block_trn2 <- fread("train_block_meta2.csv")[,1:3,with=F]
meta_block_tst2 <- fread("test_block_meta2.csv")[,1:3,with=F]

names(meta_block_trn2)[1] = names(meta_block_tst2)[1] = 
  names(meta_block_trn1)[1] = names(meta_block_tst1)[1] = "id_block"
meta_block_trn = merge(meta_block_trn1, meta_block_trn2, all=T,by="id_block")
meta_block_tst = merge(meta_block_tst1, meta_block_tst2, all=T,by="id_block")
rm(meta_block_trn1, meta_block_trn2, meta_block_tst1, meta_block_tst2)

# Get the coverage a proportion of the masksize
maskProp = function(df){
  df[["sealCoverage"]] = df[["sealCoverage"]]/(1-df[["mask_size"]])
  df[["sealOverlap2"]] = df[["sealOverlap2"]]/(1-df[["mask_size"]])
  return(df)
}
meta_trn = maskProp(meta_trn)
meta_tst = maskProp(meta_tst)
meta_block_trn = maskProp(meta_block_trn)
meta_block_tst = maskProp(meta_block_tst)


target_cols = names(meta_trn)[2:6]

resnfile = c(paste0("resnet50CVPreds2604_fold", 1:2, ".csv"))
vggfile = c(paste0("vggCVPreds2604_fold", 1:2, ".csv"))
resn50trn = rbind(fread(resnfile[1]), fread(resnfile[2]))
vggtrn = rbind(fread(vggfile[1]), fread(vggfile[2]))
resn50tst = fread("resnet50TestPreds2604.csv")  
vggtst = fread("vggTestPreds2604.csv")

# Get a function to break up the block predictions
#vggtrn[, bigimg := unlist(lapply(strsplit(img, "_"), function(x) x[1]))]
#vggtst[, bigimg := unlist(lapply(strsplit(img, "_"), function(x) x[1]))]
vggsmalltrn = getBreaks(vggtrn, c(seq(.2,.8,.2), .9, .95), "vgg")
vggsmalltst = getBreaks(vggtst, c(seq(.2,.8,.2), .9, .95), "vgg")
#resn50trn[, bigimg := unlist(lapply(strsplit(img, "_"), function(x) x[1]))]
#resn50tst[, bigimg := unlist(lapply(strsplit(img, "_"), function(x) x[1]))]
resn50smalltrn = getBreaks(resn50trn, c(seq(.2,.8,.2), .9, .95), "resn50")
resn50smalltst = getBreaks(resn50tst, c(seq(.2,.8,.2), .9, .95), "resn50")

ct_trn = merge(resn50smalltrn, vggsmalltrn, all = T, by = "img") 
ct_tst = merge(resn50smalltst, vggsmalltst, all = T, by = "img") 
rm(vggtrn, vggsmalltrn, resn50trn, resn50smalltrn)
rm(vggtst, vggsmalltst, resn50tst, resn50smalltst)
gc();gc();gc();gc();gc();gc()

# Create training and test sets
names(meta_tst)[1] = names(meta_trn)[1]  
  names(meta_block_tst)[1] = names(meta_block_trn)[1]  = "img"
Xtrn = merge(meta_block_trn, ct_trn, all = T, by = "img")
Xtrn = Xtrn[!as.numeric(unlist(lapply(strsplit(Xtrn$img, "_"), function(x) x[1]))) %in% bad_train_ids]
Xtst = merge(meta_block_tst, ct_tst, all = T, by = "img")

Xtrn[is.na(Xtrn)] <- 0
Xtst[is.na(Xtst)] <- 0

trn_img = as.numeric(unlist(lapply(strsplit(Xtrn$img, "_"), function(x) x[1])))
tst_img = as.numeric(unlist(lapply(strsplit(Xtst$img, "_"), function(x) x[1])))

Ytrn = Xtrn[,target_cols, with=F]
Ytrn_agg = cbind(trn_img, Ytrn)
Ytrn_agg = Ytrn_agg[, lapply(.SD, sum, na.rm=TRUE), by="trn_img" ][order(trn_img)]

folds = list(trn_img%%2==0, trn_img%%2==1)
Xtrn = as.matrix(Xtrn[,setdiff(names(Xtrn), c(target_cols, "img")), with=F] )
Xtst = as.matrix(Xtst[,setdiff(names(Xtst), c(target_cols, "img")), with=F] )

# Model Cross validate
result = result_agg = c()
for( var in target_cols){
  set.seed(100)
  y = Ytrn[[var]]
  y_agg = Ytrn_agg[[var]]
  result_tmp = rep(NA, length(y))
  result_agg_tmp = rep(NA, length(y))
  for(fold in folds){
    train.xgb <- xgb.DMatrix(Xtrn[fold,], label = y[fold])
    model <- xgb.train(data = train.xgb,
                    eta = 0.1,
                    nrounds = 400,
                    verbose = 0,
                    colsample_bytree = 0.4,
                    max_depth = 4,
                    objective = 'reg:linear',
                    eval_metric = 'rmse')
    result_tmp[!fold] = pmax(0, round(predict(model, Xtrn[!fold,])))
    result_agg_tmp[!fold] = round(predict(model, Xtrn[!fold,]))
  }
  result_agg_tmp = round(tapply(result_agg_tmp, trn_img, sum))
  print(paste0("Results for: ", var))
  #print(xgb.importance(model, feature_names = colnames(Xtrn))[1:8,])
  result = c(result, rmse(y, result_tmp))
  result_agg = c(result_agg, rmse(y_agg, result_agg_tmp))
  print(result[length(result)])
  print(result_agg[length(result_agg)])
  print("")
}
print(mean(result))
print(mean(result_agg))

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
                     colsample_bytree = 0.4,
                     max_depth = 4,
                     objective = 'reg:linear',
                     eval_metric = 'rmse')
  subDt[[var]] = pmax(0, round(predict(model, Xtst)))
  }

subDt = subDt[, lapply(.SD, sum, na.rm=TRUE), by="test_id" ][order(test_id)]

sub_ct = fread("../sub/sub-RFCN-2017-04-27-0.2_tune.csv")
sub_out = sub_ct
for(var in target_cols) print(paste0(round(cor(sub_out[[var]], sub_ct[[var]]), 3), " ... ", var))
par(mfrow=c(2,3))
for(var in target_cols) plot(sub_ct[[var]], subDt[[var]], main = var, 
                             ylim = c(0, max(c(sub_ct[[var]], subDt[[var]]))),
                             xlim = c(0, max(c(sub_ct[[var]], subDt[[var]]))))
for(var in target_cols) sub_out[[var]] = round((0.5*sub_ct[[var]]) + (0.5*subDt[[var]]))
sub_out
sub_ct
subDt

write.csv(sub_out, paste0("../sub/sub-xgb-at-block-lvl-", Sys.Date(), ".csv"), row.names = F)


  