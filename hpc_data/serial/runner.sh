rm -rf output101 output1011 output1012 output1013 output1014 output1015

mkdir output101
./main 100 200 0.01
mv output101 output1011

mkdir output101
./main 100 200 0.1
mv output101 output1012

mkdir output101
./main 100 500 0.01
mv output101 output1013

mkdir output101
./main 100 500 0.05
mv output101 output1014

mkdir output101
./main 100 500 0.1
mv output101 output1015
