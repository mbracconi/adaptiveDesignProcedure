#!/bin/bash

[ -z "$PYTHON_EXEC" ] && export PYTHON_EXEC=python3
[ -z "$MPLBACKEND" ] && export MPLBACKEND=Agg

function runDir()
{
    example_name=$1

    echo -n "Running $example_name ... "
    $PYTHON_EXEC ${example_name}.py > ${example_name}.out 2> ${example_name}.err
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
        echo "ERROR messages:"
        cat ${example_name}.err
        exit -1
    fi
}

function main()
{
    echo "Using `$PYTHON_EXEC --version`"
    echo "Using Scikit-learn `$PYTHON_EXEC -c "import sklearn; print(sklearn.__version__)"`"
    echo -n "Checking adaptiveDesignProcedure ... "
    $PYTHON_EXEC -c "import adaptiveDesignProcedure" && echo "OK"

    for dir in `ls`
    do
        [ $dir = "test.sh" ] && continue

        cd $dir
        runDir $dir
        cd ..
    done
}

main $*
