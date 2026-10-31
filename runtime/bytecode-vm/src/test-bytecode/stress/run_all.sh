#!/bin/bash
# Runs every test file through the runner and prints a clean PASS/FAIL table.
# The runner decides PASS/FAIL from the file name (err_* must fail, ok_* must pass).
#
# Device log lines ("[emunpu] ...", "Launching kernel ...") go to a detail file,
# so the summary stays readable. Read detail.log to see what each run printed.
#
# Run from this folder:
#   ./run_all.sh
# Override the runner path if needed:
#   RUNNER=../../../test_runner ./run_all.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
RUNNER="${RUNNER:-$HERE/../../../build/test_runner}"
DETAIL="detail.log"

if [ ! -x "$RUNNER" ]; then
    echo "runner not found or not executable at: $RUNNER"
    echo "build it first (see README.md), or set RUNNER=/path/to/test_runner"
    exit 2
fi

: > "$DETAIL"   # truncate the detail log

pass=0
fail=0
failed_names=""

printf '%-6s %-38s %s\n' "RESULT" "TEST" "DETAIL"
printf '%-6s %-38s %s\n' "------" "----" "------"

for f in err_*.txt ok_*.txt; do
    [ -e "$f" ] || continue

    # Device logs -> detail file. The last line of stdout is the verdict.
    full=$("$RUNNER" "$f" 2>>"$DETAIL")
    code=$?

    {
        echo "===== $f (exit $code) ====="
        echo "$full"
        echo
    } >> "$DETAIL"

    verdict=$(echo "$full" | tail -n 1)
    status=$(echo "$verdict" | awk '{print $1}')
    detail=$(echo "$verdict" | sed 's/^[^ ]*  *[^ ]*  *->  *//')

    if [ "$code" -eq 0 ]; then
        printf '%-6s %-38s %s\n' "PASS" "$f" "$detail"
        pass=$((pass+1))
    else
        printf '%-6s %-38s %s\n' "FAIL" "$f" "$detail"
        fail=$((fail+1))
        failed_names="$failed_names $f"
    fi
done

echo "-------------------------------------------------------------"
echo "TOTAL: $pass passed, $fail failed   (device logs in $DETAIL)"
[ -n "$failed_names" ] && echo "FAILED:$failed_names"
[ "$fail" -eq 0 ]
