#!/usr/bin/env bash

# This script replaces dummy directories with symbolic links to your
# own directories ~/ndlinks/[dataset|dissection|sourcedata|zoo], which
# of course may be further symbolic links to wherever you choose
# to keep your datasets, models, and dissections.

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

# Remove dummy directories from git using sparse-checkout
if [ -e .git/info ]
then
git config core.sparsecheckout true
cat << EOF >> .git/info/sparse-checkout
!dataset/*
!dissection/*
!sourcedata/*
!zoo/*
/*
EOF
fi

# Remove dummy directories
for DUMMY in dataset dissection sourcedata zoo
do
  if [ -L ${dummy} ]
  then
    rm ${DUMMY}
    echo "Removed link ${DUMMY}"
  fi
  if [ -d ${DUMMY} ]
  then
    rm -f ${DUMMY}/README.md
    rmdir ${DUMMY}
    echo "Removed dummy directory ${DUMMY}"
  fi
  ln -s --target-directory=. ~/ndlinks/${DUMMY}
  echo "Created link ${DUMMY}"
done
