#!/bin/bash

function runDir()
{
    example_name=$1

    echo -n "Running $example_name ... "
    python3 ${example_name}.py > ${example_name}.out 2> /dev/null
    grep -A40 "> Training data" ${example_name}_orig.out > report.old
    grep -A40 "> Training data" ${example_name}.out > report.new
    check=`paste report.old report.new | awk '($0!~/^>/ && $0!~/^\s+$/){diff=sqrt(($NF-$(NF/2))**2); if(diff>0.0){loc=1; print 1; exit} }END{if(loc==0) print 0}'`
    if [ "$check" -eq 0 ]
    then
        echo "PASSED"
        rm report.old report.new
        rm ${example_name}.out
        [ -x "clean.sh" ] && ./clean.sh
    else
        echo "FAILED"
        paste -d "~" report.old report.new | sed 's/~/         |/g'
        exit -1
    fi
}

function main()
{
    for dir in `ls`
    do
        [ $dir = "test.sh" ] && continue

        cd $dir
        runDir $dir
        cd ..
    done
}

main $*
