# *******************************************************************************
# * Copyright 2018 Intel Corporation
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# ********************************************************************************
# Author:  Lam Nguyen

# Test intended to be run using pytest
#
# If pytest is not installed on your server, you can install it in a virtual
# environment:
#
# This test has no command-line parameters, as it is run via pytest.
# TEST_OMP_NUM_THREADS        Number of OpenMP threads; default=56 
# TEST_RUN_BENCHMARK_LOG_DIR  Optional: directory to write log files to 

# JUnit XML files can be generated using pytest's command-line options.
# For example:
#
#     $ pytest -s ./test_benchmark_score_validation.py --junit-xml=../validation_benchmark_score_validation.xml --junit-prefix=benchmark_score_validation
#

import sys
import os
import re

import lib_validation_testing as VT

# export OMP_NUM_THREADS=56;export KMP_AFFINITY=granularity=fine,compact,1,0; python example/image-classification/benchmark_score.py 

# Get variables
if (os.environ.get('TEST_OMP_NUM_THREADS') != ''):
    ompNumThreads = os.environ.get('TEST_OMP_NUM_THREADS')
else:
    ompNumThreads = 56

if (os.environ.get('TEST_KMP_AFFINITY') != ''):
    kmpAff = os.environ.get('TEST_KMP_AFFINITY')
else:
    kmpAff = "granularity=fine,compact,1,0"

# Relative path (from top of repo) to mnist_softmax_xla.py script
benchmarkScoreScriptPath = 'example/image-classification/benchmark_score.py'

# Python program to run script.  This should just be "python" (or "python2"),
# as the virtualenv relies on PATH resolution to find the python executable
# in the virtual environment's bin directory.
pythonProg = 'python'

def test_benchmark_score_validation():
    print("TEST_RUN_BENCHMARK_LOG_DIR = {}".format(os.environ.get('TEST_RUN_BENCHMARK_LOG_DIR')))
    script = os.path.join(os.environ.get('TEST_RUN_BENCHMARK_LOG_DIR'), benchmarkScoreScriptPath)
    VT.checkScript(script)

    # Run with NGraph CPU backend, saving timing and accuracy
    ngraphLog = VT.runBenchmarkScoreScript(logID=' nGraph',
                                  script=script,
                                  python=pythonProg,
                                  ompNumThreads=ompNumThreads,
                                  kmpAff=kmpAff)
    ngraphResults = processOutput(ngraphLog)
    
    lDir = None

    ## Need to update the refAccPercent from the paper.
    #tmp_refAccPercent = 3.4

    if os.environ.has_key('TEST_RUN_BENCHMARK_LOG_DIR'):
        lDir = os.path.abspath(os.environ['TEST_RUN_BENCHMARK_LOG_DIR'])
        VT.writeLogToFile(ngraphLog,
                          os.path.join(lDir, 'test_benchmark_score_cpu_ngraph.log'))
        # Write Jenkins description, for quick perusal of results
        writeJenkinsDescription(ngraphResults, os.path.join(lDir,'test_benchmark_score_cpu_jenkins_oneline.log'))

    print
    print '-------------------------------- Benchmark Score Testing Summary ----------------------------------------'


    summaryLog = None
    if lDir != None:
        summaryLog = os.path.join(lDir, 'test_benchmark_score_cpu_summary.log')

    logOut = VT.LogAndOutput(logFile=summaryLog)

    # Report commands
    logOut.line()
    #logOut.line('Run with default CPU: %s' % referenceResults['command'])
    logOut.line('Run with NGraph CPU: %s' % ngraphResults['command'])

    # Report parameters -- NEED TO DO
    logOut.line()
    logOut.line('useNGraph: true')
    logOut.line('OMP Num Threads :       %d (fixed)' % ompNumThreads)
    logOut.line('KMP_AFFINITY=granularity=fine,compact,1,0')
    
# End: test_benchmark_score_validation()

# Returns dictionary with results extracted from the run:
#     'command':    Command that was run
def processOutput(log):

    command = None

    # Dummy processing for proof-of-concept
    lineCount = 0
    for line in log:

        if re.match('Command is:', line):
            if command == None:
                lArray = line.split('"')
                command = str(lArray[1].strip('"'))
                print 'Found command = [%s]' % command
            else:
                raise Exception('Multiple command-is lines found')
        break
        lineCount += 1

    return {'command': command}

# End: processOutput

def writeJenkinsDescription(ngResults, fileName, trainEpochs):
#def writeJenkinsDescription(refResults, ngResults, fileName):

    print 'Jenkins description written to %s' % fileName

    try: 

        fOut = open( fileName, 'w')
        fOut.write( 'benchmark_score - for 5 models: alexnet, vgg-16, inception-bn, inception-v3, resnet-50, resnet-152 with command : %s'
                    % (ngResults['command']))

        fOut.close()

    except Exception as e:
        print 'Unable to write Jenkins description file - %s' % e

# End: writeJenkinsDescription()