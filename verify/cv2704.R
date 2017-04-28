
############################
# Make a sub
############################
rm(list=ls())
gc();gc();gc();gc();gc();gc();gc();gc();gc()
gc();gc();gc();gc();gc();gc();gc();gc();gc()
# rfcn <- fread("~/Dropbox/noaa/feat/comp4_30000_det_test_all_seals.txt")
rfcn = fread("~/Dropbox/noaa/coords/vggTestPreds2604.csv")
#rfcn = rfcn[V2>.5]
#write.csv(rfcn, "~/Dropbox/noaa/feat/comp4_30000_det_test_all_seals_subset.txt", row.names = F)
hist(rfcn$predSeal)
ssub = fread("~/Dropbox/noaa/data/sample_submission.csv")

# Prune the high proba
rfcn = rfcn[predSeal>.2]
rfcn[,ct:=.N, by=img]
y_pred = rfcn[,.(.N), by=img]

y_pred[,img := as.integer(unlist(lapply(strsplit(img, "_"), function (x) x[1])))]
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
y_opt$adult_males =  0.12 * (y_pred$V1)
y_opt$subadult_males =  0.1 * (y_pred$V1)
y_opt$adult_females =  0.6 * y_pred$V1
y_opt$juveniles =  0.3 * (y_pred$V1)
y_opt$pups =  0.2 * (y_pred$V1)
hist(y_pred$V1, breaks = 300)

# Add the missing ones
y_opt = rbind(y_opt, ssub[!test_id %in% y_opt$test_id])
y_opt = y_opt[order(test_id)]
for (var in cols) y_opt[[var]] = round(y_opt[[var]]) 

# # Lets try a sub
write.csv(y_opt, paste0("~/Dropbox/noaa/sub/sub-RFCN-", Sys.Date(), "-0.2_tunemore.csv"), row.names = F)
