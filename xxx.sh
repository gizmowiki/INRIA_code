source ~/.profile
module load opencv2.4.13
module load cuda/8.0
module load cudnn/5.1-cuda-8.0

echo "Now copying data to RAM"
mkdir /dev/shm/people_detect/
cp -rv /data/stars/user/rpandey/head_dataset/crop/* /dev/shm/head_detect/
cd /dev/shm/head_detect/positive/
find `pwd` -maxdepth 2 -type f -name "*.jpg"| head -120 > ../files_positive.txt
cd /dev/shm/head_detect/negative/
find `pwd` -maxdepth 2 -type f -name "*.jpg"| head -200 > ../files_negative.txt

python /home/rpandey/head_detect/head_detect_upd.py

echo "Now removing all files from RAM"
rm -rf /dev/shm/head_detect/
