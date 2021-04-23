for FILE in *; do gdal_translate -of XYZ $FILE $FILE.xyz; done
