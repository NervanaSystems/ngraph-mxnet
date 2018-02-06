#!  /bin/sh
# Author:  Lam Nguyen

# This script is designed to be run from the docker/Dockerfiles directory

set -e  # Fail on any command with non-zero exit
set -u  # No unset variables

DOCKER_FILE='Dockerfile.ci.mxnet'

# The docker image ID is currently just the git SHA of this cloned repo
IMAGE_ID="$(git rev-parse HEAD)"

# Context is the docker directory, to avoid including all of
# ngraph-mxnet in the context.
docker build  --rm=true  --build-arg http_proxy=http://proxy-us.intel.com:911  --build-arg https_proxy=https://proxy-us.intel.com:911  -f="${DOCKER_FILE}"  -t="mxnet:${IMAGE_ID}"   ..