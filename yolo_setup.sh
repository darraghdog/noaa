# Copy in config files specific for our case
cp voc.seals.names darknet/data/voc.seals.names
cp voc.seals.data darknet/cfg/voc.seals.data
cp coco.c darknet/src/coco.c
cp yolo-voc-seals544.cfg darknet/cfg/yolo-voc-seals544.cfg
mkdir darknet/backupseals
mkdir darknet/backup
