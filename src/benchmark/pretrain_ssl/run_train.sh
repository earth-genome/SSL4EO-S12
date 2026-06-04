cd /home/snau/code/SSL4EO-S12/src/benchmark/pretrain_ssl
export VENV_DIR=/data/venv
export DATA_DIR=/data
export DATASET_SIZE=172348
VENV_DIR=/data/venv DATA_DIR=/data DATASET_SIZE=172348 nohup bash ssl4eo_dinov3/scripts/train_gcp.sh /data/ssl4eo_s2c_uint8.lmdb /data/out/vits16 &
