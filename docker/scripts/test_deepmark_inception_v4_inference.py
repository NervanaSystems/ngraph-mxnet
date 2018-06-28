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
#

import sys
import os
import re

import lib_validation_testing as VT

# TEST_OMP_NUM_THREADS
if (os.environ.get('TEST_OMP_NUM_THREADS') != ''):
    ompNumThreads= os.environ.get('TEST_OMP_NUM_THREADS')
else:
    ompNumThreads = 28

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

# Relative path (from top of repo) to mnist_softmax_xla.py script
benchmarkScriptPath = 'bmark.sh'

# Python program to run script.  This should just be "python" (or "python2"),
# as the virtualenv relies on PATH resolution to find the python executable
# in the virtual environment's bin directory.
pythonProg = 'python'


def test_deepmark_inception_v4_cpu_backend():
    
    script = os.path.join(os.environ.get('TEST_DEEPMARK_LOG_DIR'), benchmarkScriptPath)
    print("script = {}".format(script))
    VT.checkScript(script)

    # Run with NGraph CPU backend, saving timing and accuracy
    ngraphLog = VT.runInceptionV4Script(logID=' nGraph',
                                script=script,
                                ompNumThreads=ompNumThreads,
                                kmpAff=kmpAff,
                                kmpBlocktime=kmpBlocktime,
                                batchsize=batchsize)
    ngraphResults = processOutput(ngraphLog)
    
    lDir = None
    if os.environ.has_key('TEST_DEEPMARK_LOG_DIR'):
        lDir = os.path.abspath(os.environ['TEST_DEEPMARK_LOG_DIR'])
        VT.writeLogToFile(ngraphLog,
                          os.path.join(lDir, 'test_deepmark_inception_v4_cpu_ngraph.log'))
        # Write Jenkins description, for quick perusal of results
        #writeJenkinsDescription(ngraphResults,
        #                        os.path.join(lDir,
        #                            'test_mlp_mnist_cpu_jenkins_oneline.log'),'test_mlp_mnist_cpu_jenkins_oneline.log')

        writeJenkinsDescription(ngraphResults,  os.path.join(lDir,
                                    'test_deepmark_inception_v4_jenkins_oneline.log'))

    print("----- deepmark INCEPTION V4 Testing Summary ----------------------------------------")

    summaryLog = None
    if lDir != None:
        summaryLog = os.path.join(lDir, 'test_deepmark_inception_v4_cpu_summary.log')

    logOut = VT.LogAndOutput(logFile=summaryLog)

    # Report commands
    logOut.line()
    #logOut.line('Run with default CPU: %s' % referenceResults['command'])
    logOut.line('Run with NGraph CPU: %s' % ngraphResults['command'])

    # Report parameters
    logOut.line()
    logOut.line('Batch size:       %d (fixed)' % trainBatchSize)
    logOut.line('nGraph priority:  %d (fixed)' % 70)
    logOut.line('nGraph back-end:  %s (fixed)' % 'CPU')

# End: test_deepmark_inception_v4_cpu_backend()


# Returns array of captured stdout/stderr lines, for post-processing


# Returns dictionary with results extracted from the run:
#     'command':    Command that was run

def processOutput(log):

    command = None
    network = None
    type_inference = None
    batch_size = None
    omp = None
    throughput = None
    latency = None

    # Dummy processing for proof-of-concept
    lineCount = 0
    for line in log:

        if re.match('Command is:', line):
            if command == None:
                lArray = line.split('"')
                command = str(lArray[1].strip('"'))
                print("Found command = [{}]".format(command))
            else:
                raise Exception("Multiple command-is lines found")

        if re.match('network:', line):
            if throughput == None:
                lArray = line.split()
                network = float(lArray[1].strip())
                print("Found network = {}".format(network))
            else:
                raise Exception("Multiple network lines found")
        
        if re.match('type:', line):
            if type_inference == None:
                lArray = line.split()
                type_inference = float(lArray[2].strip())
                print("Found type = {}".format(type_inference))
            else:
                raise Exception("Multiple type lines found")

        if re.match('batch_size:', line):
            if batch_size == None:
                lArray = line.split()
                batch_size = float(lArray[3].strip())
                print("Found batch_size = {}".format(batch_size))
            else:
                raise Exception("Multiple batch_size lines found")

        if re.match('OMP:', line):
            if omp == None:
                lArray = line.split()
                omp = float(lArray[4].strip())
                print("Found omp = {}".format(omp))
            else:
                raise Exception("Multiple omp lines found")

        if re.match('throughput:', line):
            if throughput == None:
                lArray = line.split()
                throughput = float(lArray[5].strip())
                print("Found throughput = {}".format(throughput))
            else:
                raise Exception("Multiple throughput lines found")
                
        if re.match('latency:', line):
            if latency == None:
                lArray = line.split()
                latency = float(lArray[6].strip())
                print("Found latency = {}".format(latency))
            else:
                raise Exception("Multiple time-elapsed lines found")

        lineCount += 1

    return {'command': command,
            'network':network,
            'type':type_inference,
            'batch_size':batch_size,
            'OMP': omp,     
            'throughput': throughput,
            'latency': latency}

# End: processOutput


#def writeJenkinsDescription(refResults, ngResults, fileName):
def writeJenkinsDescription(ngResults, fileName):

    print("Jenkins description written to {}".format(fileName))

    try: 

        fOut = open( fileName, 'w')

        fOut.write( 'Inception-v4 type: %s;  batch_size %s; omp %s; throughput : %5.2f%%; latency %4.2f'
                    % (ngResults['batch_size'], ngResults['omp'], ngResults['throughput'], ngResults['latency']))

        fOut.close()

    except Exception as e:
        print("Unable to write Jenkins description file - {}".format(e))

# End: writeJenkinsDescription()
