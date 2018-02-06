#!/bin/bash
#Author:  Lam Nguyen
set -u
set -e

declare SCRIPT_NAME="$(basename "${0}")"
declare THIS_SCRIPT_DIR="$( cd $(dirname "${BASH_SOURCE[0]}") && pwd )"
declare NGRAPH_INSTALL_DIR="${HOME}/ngraph_dist"
declare MX_DIR="$(cd "${THIS_SCRIPT_DIR}/../.." && pwd)"

echo "****************************************************************************************"
echo "Build and install MXnet..."
echo "****************************************************************************************"

if [[ -z "${NGRAPH_INSTALL_DIR:-}" ]]; then
    export LD_LIBRARY_PATH="${HOME}/ng_mx/ngraph_dist/lib/"
fi

cd "${MX_DIR}"
git submodule update --init
make -j $(nproc)

if [ ! -f "./lib/libmxnet.so" ] ; then
  ( >&2 echo "FATAL ERROR: Can not found libmxnet.so. Exiting ...." )
  exit 1
else
   echo "Success to install ngraph-mxnet."
fi