

# cd /data/stars/share/people_depth/people-depth/fulldata/positives
# cd /local/people_detect/positives/
# find `pwd` -type f -exec cp -v {} /local/people_detect/positives/ \;
# find `pwd` -maxdepth 2 -type f -name "*.jpg" > ../files_positive.txt
rsync -rv --ignore-existing /data/stars/share/people_depth/people-depth/fulldata_updated/negatives /local/people_orig_depth/
