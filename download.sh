#!/bin/bash

while read -r line;
	do
		if [[ $line == *"LST_"* ]]; then
			wget -P "./LST_DATA" $line;
		fi
	done
