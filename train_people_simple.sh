source ~/.profile
module load opencv2.4.13
module load cuda/8.0
module load cudnn/5.1-cuda-8.0

mkdir /dev/shm/people_depth/
time cp -iprv /local/people_depth/positives /dev/shm/people_depth/

cd /dev/shm/people_depth/positives/
find `pwd` -maxdepth 2 -type f -name "*.png" > ../files_positive.txt
# cd /local/people_depth/negatives_upd/

# find `pwd` -maxdepth 2 -type f -name "*.png"| head -40000> ../files_negative.txt

# find `pwd` -maxdepth 2 -type f -name "*.png"| head -985076> ../files_negative.txt
# find `pwd` -maxdepth 2 -type f -name "*.jpg"| head -40000> ../files_negative.txt

/usr/bin/python /home/rpandey/people_detect/scaled_people_depth.py

echo "Now removing all files from RAM"
# rm -rf /dev/shm/people_detect/
# rm -rf /local/people_detect/
