#!/bin/bash
./clean.sh

echo -n "Running example.py ... "
python3 example.py > example.out 2> /dev/null
grep -A40 "> Training data" example_orig.out > report.old
grep -A40 "> Training data" example.out > report.new
check=`paste report.old report.new | awk '($0!~/^>/ && $0!~/^\s+$/){diff=sqrt(($4-$2)**2); if(diff>0.0){loc=1; print 1; exit} }END{if(loc==0) print 0}'`
if [ "$check" -eq 0 ]
then
    echo "PASSED"
else
    echo "FAILED"
    paste -d "~" report.old report.new | sed 's/~/         |/g'
    exit -1
fi
