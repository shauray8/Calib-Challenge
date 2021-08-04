#!/bin/bash
python utils.py --data $1
echo will this thing ever work > sucker.txt
rm sucker.txt
echo done bitches
