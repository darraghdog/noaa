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
meta_trn1 <- fread("train_meta1.csv")[,-1,with=F]
meta_tst1 <- fread("test_meta1.csv")[,-1,with=F]
meta_trn2 <- fread("train_meta2.csv")[,1:3,with=F]
meta_tst2 <- fread("test_meta2.csv")[,1:3,with=F]
names(meta_trn2)[1] = names(meta_tst2)[1] = "id"
meta_trn = merge(meta_trn1, meta_trn2, all=T,by="id")
meta_tst = merge(meta_tst1, meta_tst2, all=T,by="id")

meta_trn[["sealCoverage"]] = meta_trn[["sealCoverage"]]/(1-meta_trn[["mask_size"]])
meta_trn[["sealOverlap2"]] = meta_trn[["sealOverlap2"]]/(1-meta_trn[["mask_size"]])
meta_tst[["sealCoverage"]] = meta_tst[["sealCoverage"]]/(1-meta_tst[["mask_size"]])
meta_tst[["sealOverlap2"]] = meta_tst[["sealOverlap2"]]/(1-meta_tst[["mask_size"]])
rm(meta_tst1, meta_tst2, meta_trn1, meta_trn2)

target_cols = names(meta_trn)[2:6]
resnfile = c(paste0("resnet50CVPreds2604_fold", 1:2, ".csv"))
vggfile = c(paste0("vggCVPreds2604_fold", 1:2, ".csv"))
resn50trn = rbind(fread(resnfile[1]), fread(resnfile[2]))
vggtrn = rbind(fread(vggfile[1]), fread(vggfile[2]))
resn50tst = fread("resnet50TestPreds2604.csv")  
vggtst = fread("vggTestPreds2604.csv")
resnClass = c(paste0("resnet50CVclassPreds0405_fold", 1:2, ".csv"))


# Class Predictions
resnclfile = c(paste0("resnet50CVclassPreds0405_fold", 1:2, ".csv"))
resn50cltrn = rbind(fread(resnclfile[1]), fread(resnclfile[2]))[,c(1,3,5,7,9,11),with=F]
resn50cltst = fread("resnet50TestclassPreds2604.csv")[,1:6,with=F]
for(var in c(target_cols)) resn50cltrn[[paste0("cutclass0.2_", var)]] = ifelse(resn50cltrn[[var]]>0.2, 1, 0)
for(var in c(target_cols)) resn50cltrn[[paste0("cutclass0.8_", var)]] = ifelse(resn50cltrn[[var]]>0.8, 1, 0)
for(var in c(target_cols)) resn50cltst[[paste0("cutclass0.2_", var)]] = ifelse(resn50cltst[[var]]>0.2, 1, 0)
for(var in c(target_cols)) resn50cltst[[paste0("cutclass0.8_", var)]] = ifelse(resn50cltst[[var]]>0.8, 1, 0)
resn50cltrn = resn50cltrn[,setdiff(names(resn50cltrn), c("others", target_cols)), with=F]
resn50cltst = resn50cltst[,setdiff(names(resn50cltst), c("others", target_cols)), with=F]
resn50cltrn[, img := unlist(lapply(strsplit(img, "_"), function(x) x[1]))]
resn50cltst[, img := unlist(lapply(strsplit(img, "_"), function(x) x[1]))]
resn50cltrn = resn50cltrn[, lapply(.SD, sum, na.rm=TRUE), by="img" ][order(img)]
resn50cltst = resn50cltst[, lapply(.SD, sum, na.rm=TRUE), by="img" ][order(img)]
resn50cltrn[,img:=as.integer(img)]
resn50cltst[,img:=as.integer(img)]

# Get a function to break up the block predictions
vggtrn[, bigimg := unlist(lapply(strsplit(img, "_"), function(x) x[1]))]
vggtst[, bigimg := unlist(lapply(strsplit(img, "_"), function(x) x[1]))]
vggbigtrn = getBreaks(vggtrn, .8, "vgg")
vggbigtst = getBreaks(vggtst, .8, "vgg")
resn50trn[, bigimg := unlist(lapply(strsplit(img, "_"), function(x) x[1]))]
resn50tst[, bigimg := unlist(lapply(strsplit(img, "_"), function(x) x[1]))]
resn50bigtrn = getBreaks(resn50trn, c(seq(.3,.9,.3), .95), "resn50")
resn50bigtst = getBreaks(resn50tst, c(seq(.3,.9,.3), .95), "resn50")

ct_trn = merge(resn50bigtrn, vggbigtrn, all = T, by = "img") 
ct_trn = merge(ct_trn, resn50cltrn, all = T, by = "img") 
ct_tst = merge(resn50bigtst, vggbigtst, all = T, by = "img") 
ct_tst = merge(ct_tst, resn50cltst, all = T, by = "img") 

rm(vggtrn, vggbigtrn, resn50trn, resn50bigtrn)
rm(vggtst, vggbigtst, resn50tst, resn50bigtst)
gc();gc();gc();gc();gc();gc()
gc();gc();gc();gc();gc();gc()

# Create training and test sets
names(meta_tst)[1] = names(meta_trn)[1]  = "img"
Xtrn = merge(meta_trn, ct_trn, all = T, by = "img")
Xtrn = Xtrn[!Xtrn$img %in% bad_train_ids]
Xtst = merge(meta_tst, ct_tst, all = T, by = "img")


# Xgb set up
Xtrn[is.na(Xtrn)] <- 0
Xtst[is.na(Xtst)] <- 0
Ytrn = Xtrn[,target_cols, with=F]
trn_img = Xtrn$img
tst_img = Xtst$img
folds = list(trn_img%%2==0, trn_img%%2==1)
Xtrn = as.matrix(Xtrn[,setdiff(names(Xtrn), c(target_cols, "img")), with=F] )
Xtst = as.matrix(Xtst[,setdiff(names(Xtst), c(target_cols, "img")), with=F] )

puppy = resn50cltrn[as.integer(img) %in% trn_img][order(as.integer(img))]$cutclass0.2_pups
plot(Ytrn[trn_img %in% as.integer(resn50cltrn$img)]$pups, puppy, pch=19, col="blue")
points(result_tmp[trn_img %in% as.integer(resn50cltrn$img)], puppy, col="red", pch=19)

rmse(Ytrn[trn_img %in% as.integer(resn50cltrn$img)]$pups, puppy)
rmse(Ytrn[trn_img %in% as.integer(resn50cltrn$img)]$pups, result_tmp[trn_img %in% as.integer(resn50cltrn$img)])


puppy = resn50cltst[as.integer(img) %in% tst_img][order(as.integer(img))]$cutclass0.2_pups
sub_best = fread("../sub/sub-xgb-fix-coverage-bug-2017-05-06.csv")
plot(sub_best[test_id %in% as.integer(resn50cltst$img)]$pups, puppy, pch=19, col="blue")
sub_best[test_id %in% as.integer(resn50cltst$img)][puppy <10 & pups > 100  ]

juv = resn50cltst[as.integer(img) %in% tst_img][order(as.integer(img))]$cutclass0.2_juveniles
plot(sub_best[test_id %in% as.integer(resn50cltst$img)]$juveniles, juv, pch=19, col="blue")
sub_best[test_id %in% as.integer(resn50cltst$img)][juv <100 & juveniles > 200  ]

table(rowSums(sub_best[,-1,with=F])<400)
table(rowSums(Ytrn[,-1,with=F])<400)

# Model Cross validate
result = c()
result_ls = list()
for( var in target_cols){
  set.seed(100)
  y = Ytrn[[var]]
  result_tmp = rep(NA, length(y))
  exclude = apply(expand.grid(setdiff(target_cols, var), c("cutclass0.2_", "cutclass0.8_")), 1, function(x) paste(x[2], x[1], sep="")) 
  cols = setdiff(colnames(Xtrn), exclude)
  for(fold in folds){
    train.xgb <- xgb.DMatrix(Xtrn[fold,cols], label = y[fold])
    model <- xgb.train(data = train.xgb,
                    eta = 0.1,
                    nrounds = 400,
                    verbose = 0,
                    colsample_bytree = 0.4,
                    max_depth = 4,
                    objective = 'reg:linear',
                    eval_metric = 'rmse')
    result_tmp[!fold] = pmax(0, round(predict(model, Xtrn[!fold,cols])))
  }
  result_ls[[length(result_ls)+1]] = result_tmp
  print(paste0("Results for: ", var))
  #print(xgb.importance(model, feature_names = colnames(Xtrn[,cols]))[1:8,])
  result = c(result, rmse(y, result_tmp))
  print(result[length(result)])
}
print(mean(result))

# [1] "Results for: adult_males"
# [1] 3.887157
# [1] "Results for: subadult_males"
# [1] 6.333422
# [1] "Results for: adult_females"
# [1] 35.51293
# [1] "Results for: juveniles"
# [1] 39.27167
# [1] "Results for: pups"
# [1] 28.35539
# > print(mean(result))
# [1] 22.67211


# Make the sub file
subDt = data.table(data.frame(test_id = tst_img))
for( var in target_cols){
  set.seed(100)
  y = Ytrn[[var]]
  exclude = apply(expand.grid(setdiff(target_cols, var), c("cutclass0.2_", "cutclass0.8_")), 1, function(x) paste(x[2], x[1], sep="")) 
  cols = setdiff(colnames(Xtrn), exclude)
  train.xgb <- xgb.DMatrix(Xtrn[,cols], label = y)
  model <- xgb.train(data = train.xgb,
                     eta = 0.1,
                     nrounds = 400,
                     verbose = 0,
                     colsample_bytree = 0.4,
                     max_depth = 4,
                     objective = 'reg:linear',
                     eval_metric = 'rmse')
  subDt[[var]] = pmax(0, round(predict(model, Xtst[,cols])))
  }

sub_ct = fread("../sub/sub-RFCN-2017-04-27-0.2_tune.csv")
sub_best = fread("../sub/sub-xgb-fix-coverage-bug-2017-05-06.csv")
sub_out = sub_ct
for(var in target_cols) print(paste0(round(cor(sub_out[[var]], sub_ct[[var]]), 3), " ... ", var))
par(mfrow=c(2,3))
for(var in target_cols) plot(sub_ct[[var]], subDt[[var]], main = var)
for(var in target_cols) plot(sub_best[[var]], subDt[[var]], main = var)
for(var in target_cols) sub_out[[var]] = round((0.33*sub_ct[[var]]) + (0.34*sub_best[[var]]) + (0.33*subDt[[var]]))
sub_out
sub_best
sub_ct
subDt

write.csv(sub_out, paste0("../sub/sub-xgb-classesB-", Sys.Date(), ".csv"), row.names = F)


no_pup = resn50cltst[order(as.numeric(img))]$cutclass0.2_pups==0
sqrt(mean((sub_best$pups[no_pup])^2))
