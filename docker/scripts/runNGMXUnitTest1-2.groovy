// *******************************************************************************
// Copyright 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ********************************************************************************

// Install Ngraph_Mxnet
// Author:  Lam Nguyen       

//Temporary to remove the dockerID
def call(String dockerID, String pythonVersion){

    try {

        // Build NGMXWheel()
        echo 'INFO: Building the ngraph-mxnet wheel from source'
        runNGMXUnitTest(dockerID, pythonVersion)

    } catch(e) {

        echo "===================="
        echo "ERROR: Exception caught in module which gets ngraph-mxnet wheel - runNGMXUnitTest()"
        echo "ERROR: ${e}"
        echo "===================="

        echo ' '
        echo "Build marked as FAILURE"
        currentBuild.result = 'FAILURE'

    }  // catch

}  //  def call(...)

return this;

//********* installNGMX() ***********
//
// Build the MX Wheel from a cloned ngraph-mxnet repository in the top
// level of the workspace.
//
// Returns: nothing
//
// NOTE: Sets global currentBuild.result to FAILURE if build fails.
//
// External dependencies: highly dependent on cloned repo of ngraph-mxnet
// (cloned here) and ngraph_dist_gcc.tgz (currently in ngraph repo).
def runNGMXUnitTest(String dockerID, String pythonVersion){

    // Early check to see if ngraph_dist_gcc.tgz is available.  If not (due to
    // ngraph build failure, or Artifactory not finding the proper
    // version), then exit early.  This makes finding the first error in the
    // log easier, and failures terminate more quickly since the NG-MX Docker
    // image (which includes bazel) does not need to be built.
    sh '''
        if [ ! -f ngraph_dist_gcc.tgz ] ; then
            echo 'ERROR: Could not find an ngraph_dist_gcc.tgz file to build ngraph-mxnet with, in buildNGMXWheel()'
            exit 1
        fi
    '''
    try {
        withEnv(["DOCKER_ID=${dockerID}","PYTHON_VERSION_NUMBER=${pythonVersion}"]){
            // Build ngraph-mxnet reference-OS Docker image
            sh '''
                set -e
                echo "ngraph-mxnet docker build started at `date`"
                cd ngraph-mxnet/docker/Dockerfiles/
                ./docker-mx-ng-set-up-unit-test.sh ${DOCKER_ID}-base-mxnet ${PYTHON_VERSION_NUMBER}
                cd "$WORKSPACE"
                echo "ngraph-mxnet docker build completed at `date`"
            '''
        }//withEnv

    } catch(e) {

        echo "===================="
        echo "ERROR: Exception caught in module which builds the ngraph-mxnet repo - runNGMXUnitTest()"
        echo "ERROR: ${e}"
        echo "===================="

        echo ' '
        echo "Build marked as FAILURE"
        currentBuild.result = 'FAILURE'

    }  // catch

}  // def runNGMXUnitTest()