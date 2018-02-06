#!  /bin/bash
# Author:  Lam Nguyen

# This script is designed to be run from the docker directory

set -e  # Fail on any command with non-zero exit
set -u  # No unset variables

ngraph_mx_dir="$(realpath ../..)"

# The docker image ID is currently just the git SHA of this cloned repo.
# We need this ID to know which docker image to run with.
# Note that the docker image must have been previously built using the
# make-docker-mx-ngraph-base.sh script (in the same directory as this script).
IMAGE_ID="$(git rev-parse HEAD)"

docker run --rm \
       --env RUN_UID="$(id -u)" \
       --env RUN_CMD='/home/dockuser/ng-mx/docker/scripts/run-mx-ngraph-build-and-test.sh' \
       --env http_proxy=http://proxy-us.intel.com:911 \
       --env https_proxy=https://proxy-us.intel.com:911 \
       -v "${ngraph_mx_dir}:/home/dockuser/ng-mx" \
       "mxnet:${IMAGE_ID}" /home/run-as-user.sh