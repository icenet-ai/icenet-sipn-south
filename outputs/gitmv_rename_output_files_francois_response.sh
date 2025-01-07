# This script renames the output diagnostic files for diagnostic 1 and 3 which were
# missing a hyphen or the start/end dates in the filename.
# Not needed in up to date code.

find . -type f -name '*totalarea.txt' -exec sh -c '
  for file; do
    new_name=$(echo "$file" | sed "s/totalarea.txt/total-area.txt/")
    git mv "$file" "$new_name"
  done
' sh {} +

# The netCDF files are not being version controlled, so, using `mv` instead of `git mv`
find . -type f -name '*concentration.nc' -exec sh -c '
  for file; do
    new_name=$(echo "$file" | sed "s/_concentration.nc/_20241201-20250228_concentration.nc/")
    mv "$file" "$new_name"
  done
' sh {} +

