#!/bin/bash

cd ./money_person_xml

for file in ./*
do
    echo $file
    sed -i '/<?xml version*/d' $file
    sed -i '/<annotation*/d' $file
done

echo '----------add cleaned annotations----------'
sed -i '1 i<annotation verified="yes">' *.xml