#!/bin/bash
echo "Total Games: $(wc -l $1)"
echo "Unique Games: $(sort $1 | uniq | wc -l)"

