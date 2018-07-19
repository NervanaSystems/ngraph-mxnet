# Test intended to be run using pytest
#
# If pytest is not installed on your server, you can install it in a virtual
# environment:
#
# Set up a virtual environment
#   Python 2: virtualenv -p python2.7 .venv && . .venv/bin/activate
#   Python 3: python3 -m venv .venv && . .venv/bin/activate
#   $ pip -U pytest
#   $ pytest test_deepmark_inception_v4.py
#   $ deactivte
#
# This test has no command-line parameters, as it is run via pytest.
# This test does have environment variables that can alter how the run happens:
#
#     Parameter              Purpose & Default (if any)
#
#     TEST_OMP_NUM_THREADS       Number of OMP THREADS
#                                default=28
#     TEST_KMP_BLOCKTIME         Sets the time, in milliseconds, that a thread should wait
#                                default=200
#     TEST_DEEPMARK_LOG_DIR  directory to write log files to
#     TEST_BATCH_SIZE            BatchSize
#     TEST_KMP_AFFINITY          KMP_AFFINITY
#
# JUnit XML files can be generated using pytest's command-line options.
# For example:
#
#     $ pytest -s ./test_deepmark_inception-v4.py --junit-xml=../validation_deepmark_inception_v4_cpu.xml --junit-prefix=daily_validation_inception_v4_cpu

import sys
import os
import re

import lib_validation_testing as VT

# TEST_OMP_NUM_THREADS
if (os.environ.get('TEST_OMP_NUM_THREADS') != ''):
    ompNumThreads= os.environ.get('TEST_OMP_NUM_THREADS')
else:
    ompNumThreads = 28

# TEST_DEEPMARK_LOG_DIR
if (os.environ.get('TEST_DEEPMARK_LOG_DIR') != ''):
    sourceDir= os.environ.get('TEST_DEEPMARK_LOG_DIR')

# TEST_KMP_BLOCKTIME
if (os.environ.get('TEST_KMP_BLOCKTIME') != ''):
    kmpBlocktime= os.environ.get('TEST_KMP_BLOCKTIME')
else:
    kmpBlocktime = 1

# TEST_BATCH_SIZE
if (os.environ.get('TEST_BATCH_SIZE') != ''):
    batchsize= os.environ.get('TEST_BATCH_SIZE')
else:
    batchsize = 128


# TEST_KMP_AFFINITY
if (os.environ.get('TEST_KMP_AFFINITY') != ''):
    kmpAff= os.environ.get('TEST_KMP_AFFINITY')
else:
    kmpAff = "granularity=fine,compact,1,0"

benchmarkScriptPath = "benchmark.py"
# Python program to run script.  This should just be "python" (or "python2"),
# as the virtualenv relies on PATH resolution to find the python executable
# in the virtual environment's bin directory.
pythonProg = 'python'


def test_deepmark_inception_v4_cpu_backend():
    
    script = os.path.join(os.environ.get('TEST_DEEPMARK_LOG_DIR'), benchmarkScriptPath)
    VT.checkScript(script)
    # Run with NGraph CPU backend, saving timing and accuracy
    ngraphLog = VT.runInceptionV4Script(sourceDir=sourceDir,
                                logID=' nGraph',
                                script=script,
                                ompNumThreads=ompNumThreads,
                                kmpAff=kmpAff,
                                kmpBlocktime=kmpBlocktime,
                                batchsize=batchsize)
    ngraphResults = processOutput(ngraphLog)
    
    lDir = None

    lDir = os.path.abspath(os.environ['TEST_DEEPMARK_LOG_DIR'])

    VT.writeLogToFile(ngraphLog, os.path.join(lDir, 'test_deepmark_inception_v4_cpu_ngraph.log'))
    VT.checkScript(os.path.join(lDir, 'test_deepmark_inception_v4_cpu_ngraph.log'))

    writeJenkinsDescription(ngraphLog, os.path.join(lDir, 'test_deepmark_inception_v4_jenkins_oneline.log'))


    print("----- deepmark INCEPTION V4 Testing Summary ----------------------------------------")

    summaryLog = None
    if lDir != None:
        summaryLog = os.path.join(lDir, 'test_deepmark_inception_v4_cpu_summary.log')

    logOut = VT.LogAndOutput(logFile=summaryLog)

    # Report commands
    logOut.line()
    logOut.line("Run with NGraph CPU: {}".format(ngraphResults['command']))

    # Report parameters
    logOut.line()
    logOut.line("Batch size:       {} (fixed)".format(batchsize))
    logOut.line("OMP_NUM_THREADS:       {} (fixed)".format(ompNumThreads))
    logOut.line("KMP_AFFINITY:       {} (fixed)".format(kmpAff))

# End: test_deepmark_inception_v4_cpu_backend()


# Returns array of captured stdout/stderr lines, for post-processing


# Returns dictionary with results extracted from the run:
#     'command':    Command that was run

def processOutput(log):

    command = None
    network = None
    one_line = None
    #type_inference = None
    #batch_size = None
    #omp = None
    #throughput = None
    #latency = None

    # Dummy processing for proof-of-concept
    lineCount = 0
    for line in log:
        if re.match("Command is:", line):
            if command == None:
                lArray = line.split('"')
                command = str(lArray[1].strip('"'))
                print("Found command = [{}]".format(command))
            else:
                raise Exception("Multiple command-is lines found")

        if re.match("network:", line):
            if one_line == None:
                one_line = line
                print("Found one_line = {}".format(one_line))
            else:
                raise Exception("Multiple network lines found")
        
        lineCount += 1
    return {'command': command,
            'one_line': one_line}
# End: processOutput

def writeJenkinsDescription(ngResults, fileName):

    print("Jenkins description written to {}".format(fileName))

    try: 

        fOut = open( fileName, 'w')

        fOut.write("Inception-v4 type: {}\n\t{}".format(ngResults['command'],ngResults['one_line']))

        fOut.close()

    except Exception as e:
        print("Unable to write Jenkins description file - {}".format(e))

# End: writeJenkinsDescription()
