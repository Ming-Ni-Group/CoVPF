## grep Omicron variants
grep 'VOI GRA' ./metadata.csv > ./metadata_merge1.csv
grep 'VUM GRA' ./metadata.csv > ./metadata_merge2.csv
grep 'VOC Omicron' ./metadata.csv > ./metadata_merge3.csv
cat merge1 >> merge2 >> merge3

## Delete tabs and run process_metadata.py - extract countries or regions and  Spike/RBD information
sed -i 's/ //g' ./metadata_Omicron.csv

## Process the extracted mutation data: "['aaa', 'bbb', 'ccc']": remove '; remove spaces; remove [; remove]. Get "aaa,bbb,ccc".
sed -i "s/'//g" ./Omicron_RBD_experimental_data.csv
sed -i "s/ //g" ./Omicron_RBD_experimental_data.csv
sed -i "s/\[//g" ./Omicron_RBD_experimental_data.csv
sed -i "s/\]//g" ./Omicron_RBD_experimental_data.csv

## Screening of time to remove irregularities
awk -F "-" '{print NF - 1}' ./Omicron_RBD_experimental_data.csv > ./Time_completed_index.csv
paste ./Time_completed_index.csv ./Omicron_RBD_experimental_data.csv > ./Omicron_time.csv
grep -E ^"2" ./Omicron_time.csv > ./Omicron_time_2.csv
sed 's/..//' ./Omicron_time_2.csv > ./Omicron_time_delete.csv
head -1 ./Omicron_RBD_experimental_data.csv > ./Omicron_experimental_data_nosort.csv
cat ./Omicron_time_delete.csv >> ./Omicron_experimental_data_nosort.csv
sort -t, -k1n ./Omicron_experimental_data_nosort.csv > ./Omicron_experimental_data.csv

## Process the data:
# Delete lines with NAN
# Delete error data before 2021-11-01: 
sed -i 'm,nd' ./Omicron_experimental_data.csv

## After counting, remove erroneous noise: e.g. A, AY, etc. sublineages