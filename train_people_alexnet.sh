source ~/.profile
module load opencv2.4.13
module load cuda/8.0
module load cudnn/5.1-cuda-8.0

echo "Now copying data to RAM"
# mkdir /dev/shm/people_detect/
mkdir /dev/shm/people_detect/




time cp -rv /local/people_detect/* /dev/shm/people_detect/


cd /dev/shm/people_detect/positives/
find `pwd` -maxdepth 2 -type f -name "*.jpg" > ../files_positive.txt
cd /dev/shm/people_detect/negatives/
find `pwd` -maxdepth 2 -type f -name "*.jpg" > ../files_negative.txt


/usr/bin/python /home/rpandey/people_detect/alexnet_people_depth.py

echo "Now removing all files from RAM"
rm -rf /dev/shm/people_detect/
# rm -rf /local/people_detect/
