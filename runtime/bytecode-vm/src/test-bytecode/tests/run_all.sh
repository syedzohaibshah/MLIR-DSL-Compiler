#!/bin/bash
# Runs every test file through the test runner and checks the result against the
# name convention: err_* must FAIL, ok_* must pass. Run from this folder.
# The runner binary path can be overridden: RUNNER=../../../test_runner ./run_all.sh

RUNNER="${RUNNER:-../../../test_runner}"

if [ ! -x "$RUNNER" ]; then
    echo "runner not found at $RUNNER -- build it first (see README.md)"
    exit 2
fi

pass=0
fail=0
for f in err_*.txt ok_*.txt; do
    out=$("$RUNNER" "$f" 2>&1)
    code=$?
    case "$f" in
        err_*) want="fail" ;;
        ok_*)  want="ok"   ;;
    esac

    if [ "$want" = "fail" ] && [ "$code" -ne 0 ]; then
        echo "PASS  $f  ->  $out"; pass=$((pass+1))
    elif [ "$want" = "ok" ] && [ "$code" -eq 0 ]; then
        echo "PASS  $f  ->  $out"; pass=$((pass+1))
    else
        echo "WRONG $f  (wanted $want)  ->  $out"; fail=$((fail+1))
    fi
done

echo "----"
echo "$pass passed, $fail wrong"
