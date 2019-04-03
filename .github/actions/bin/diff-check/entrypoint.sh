#!/bin/sh

for var in "$@"; do
    echo "Running git diff for $var..."
    if sh -c "git diff origin/develop --exit-code $var"; then
        echo "$var wasn't changed, failing!"
        exit 255 
    else
        exit_code=$?
        echo "$var was changed, passing!"
        exit 0
    fi
done
