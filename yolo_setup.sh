# Copy in config files specific for our case
cp voc.seals.names darknet/data/voc.seals.names
cp voc.seals.data darknet/cfg/voc.seals.data
cp coco.c darknet/src/coco.c
cp yolo-voc-seals544.cfg darknet/cfg/yolo-voc-seals544.cfg
cp yolo-voc-seals544small.cfg darknet/cfg/yolo-voc-seals544small.cfg
cp yolo-voc-seals544small-blk.cfg darknet/cfg/yolo-voc-seals544small-blk.cfg
#mkdir darknet/backupseals
#mkdir darknet/backup
# then run ....
# nohup ./darknet detector train cfg/voc.seals.data cfg/yolo-voc-seals544.cfg darknet19_448.conv.23  > nohup544.out 2>&1&
# ./darknet detector valid cfg/voc.seals.data   cfg/yolo-voc-seals544.cfg   backup/yolo-voc-seals544_12000.weights


# nohup ./darknet detector train cfg/voc.seals.data cfg/yolo-voc-seals544small.cfg darknet19_448.conv.23  > nohup544.out 2>&1&
# ./darknet detector valid cfg/voc.seals.data   cfg/yolo-voc-seals544small.cfg   backup/yolo-voc-seals544_12000.weights
# ./darknet detector test cfg/voc.seals.data cfg/yolo-voc-seals544small.cfg backup/yolo-voc-seals544small_6000.weights seals/JPEGImages/591_20.jpg

# nohup ./darknet detector train cfg/voc.seals.data cfg/yolo-voc-seals544small-blk.cfg darknet19_448.conv.23  > nohup544small-blk.out 2>&1&
# nohup ./darknet detector valid cfg/voc.seals.data cfg/yolo-voc-seals544small-blk.cfg whichever_weights  > nohup544small-blkpred.out 2>&1&
