#!/bin/bash
echo -ne "Not comments (code + blank):\n"
grep --color -cv "^\s*#" *.py

echo -ne "Comments:\n"
grep --color -c "^\s*#" *.py

echo -ne "Blank lines:\n"
grep --color -c "^\s*$" *.py

echo -ne "Total:\n"
grep --color -c "^.*" *.py
