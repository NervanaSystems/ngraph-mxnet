# Test intended to be run using pytest
#
# If pytest is not installed on your server, you can install it in a virtual
# environment:
#
# Set up a virtual environment
#   Python 2: virtualenv -p python2.7 .venv && . .venv/bin/activate
#   Python 3: python3 -m venv .venv && . .venv/bin/activate
#   $ pip -U pytest
#   $ pytest test_mnist_cpu_daily_validation.py
#   $ deactivte
#
# This test has no command-line parameters, as it is run via pytest.
# This test does have environment variables that can alter how the run happens:
#
#     Parameter              Purpose & Default (if any)
#
#     TEST_MLP_MNIST_ITERATIONS  Number of iterations (steps) to run;
#                                default=100000
#     TEST_MLP_MNIST_DATA_DIR    Directory where MNIST datafiles are located
#     TEST_MLP_MNIST_LOG_DIR     Optional: directory to write log files to
#
# JUnit XML files can be generated using pytest's command-line options.
# For example:
#
#     $ pytest -s ./test_mnist_cpu_daily_validation.py --junit-xml=../validation_tests_mnist_mlp_cpu.xml --junit-prefix=daily_validation_mnist_mlp_cpu
#

import sys
import os
import re

import lib_validation_testing as VT


# Constants

## MLP_MNIST : Parameters and its default values.
#gpus = None
#batch_size = 64
#disp_batches = 100
#num_epochs = 20
#lr = .05
#lr_step_epochs = 10
#--num-examples type=int, default=60000
#--num-classes type=int default=10
#--with-nnp : False
#--add_stn : False


# Acceptable accuracy
acceptableAccuracy = 2.0  # 1.0%, delta must be calculated from percentages

# As per Yann Lecun's description of the MNIST data-files at URL:
#   http://yann.lecun.com/exdb/mnist/
trainEpochs = 20 

trainBatchSize = 64

# Relative path (from top of repo) to mnist_softmax_xla.py script
mnistScriptPath = 'example/image-classification/train_mnist.py'

# Python program to run script.  This should just be "python" (or "python2"),
# as the virtualenv relies on PATH resolution to find the python executable
# in the virtual environment's bin directory.
pythonProg = 'python'


def test_mlp_mnist_cpu_backend():
    
    script = os.path.join(os.environ.get('TEST_MLP_MNIST_LOG_DIR'), mnistScriptPath)
    VT.checkScript(script)
    TEST_MLP_MNIST_DATA_DIR=os.environ.get('TEST_MLP_MNIST_LOG_DIR', None)

    print("TEST_MLP_MNIST_DATA_DIR = {}".format(TEST_MLP_MNIST_DATA_DIR))

    dataDir = os.environ.get('TEST_MLP_MNIST_DATA_DIR', None)

    # Run with Google CPU defaults, saving timing and accurac

    # Run with NGraph CPU backend, saving timing and accuracy
    ngraphLog = VT.runMnistScript(logID=' nGraph',
                                  script=script,
                                  python=pythonProg,
                                  dataDirectory=dataDir)
    ngraphResults = processOutput(ngraphLog)
    
    lDir = None
    if os.environ.has_key('TEST_MLP_MNIST_LOG_DIR'):
        lDir = os.path.abspath(os.environ['TEST_MLP_MNIST_LOG_DIR'])
        VT.writeLogToFile(ngraphLog,
                          os.path.join(lDir, 'test_mlp_mnist_cpu_ngraph.log'))
        # Write Jenkins description, for quick perusal of results
        #writeJenkinsDescription(ngraphResults,
        #                        os.path.join(lDir,
        #                            'test_mlp_mnist_cpu_jenkins_oneline.log'),'test_mlp_mnist_cpu_jenkins_oneline.log')

        writeJenkinsDescription(ngraphResults,  os.path.join(lDir,
                                    'test_mlp_mnist_cpu_jenkins_oneline.log'))

    print
    print '----- MNIST-MLP Testing Summary ----------------------------------------'


    summaryLog = None
    if lDir != None:
        summaryLog = os.path.join(lDir, 'test_mlp_mnist_cpu_summary.log')

    logOut = VT.LogAndOutput(logFile=summaryLog)

    # Report commands
    logOut.line()
    #logOut.line('Run with default CPU: %s' % referenceResults['command'])
    logOut.line('Run with NGraph CPU: %s' % ngraphResults['command'])

    # Report parameters
    logOut.line()
    logOut.line('Batch size:       %d (fixed)' % trainBatchSize)
    logOut.line('Epoch  :       %d (fixed)' % trainEpochs)
    logOut.line('nGraph priority:  %d (fixed)' % 70)
    logOut.line('nGraph back-end:  %s (fixed)' % 'CPU')
    logOut.line('Data directory:   %s' % dataDir)

    ## Need to update the refAccPercent from the paper.
    #refAccPercent = float(referenceResults['accuracy']) * 100.0
    #tmp_refAccPercent = 3.4
    #kAcceptableAccuracy = 400

    #refAccPercent = tmp_refAccPercent * 100.0
    ngAccPercent = float(ngraphResults['accuracy']) * 100.0

    # Report accuracy
    #deltaAccuracy = abs(float(refAccPercent) - ngAccPercent)
    logOut.line()
    #logOut.line('Run with default CPU accuracy: %7.4f%%' % float(refAccPercent))
    logOut.line('Run with NGraph CPU accuracy: %7.4f%%' % ngAccPercent)
    #logOut.line('Accuracy delta: %6.4f%%' % deltaAccuracy)
    #logOut.line('Acceptable accuracy delta is <= %6.4f%%'
    #            % float(kAcceptableAccuracy))
    # Assert for out-of-bounds accuracy
    #assert deltaAccuracy <= kAcceptableAccuracy
        
    # Report on times
    logOut.line()
    #logOut.line('Run with default CPU took:    %f seconds'
    #            % referenceResults['wallclock'])
    logOut.line('Run with NGraph CPU took: %f seconds'
                % ngraphResults['wallclock'])
    #logOut.line('NGraph was %f times longer than default (wall-clock measurement)'
    #            % (ngraphResults['wallclock'] / referenceResults['wallclock']))

# End: test_mlp_mnist_cpu_backend()


# Returns array of captured stdout/stderr lines, for post-processing


# Returns dictionary with results extracted from the run:
#     'command':    Command that was run
#     'accuracy':   Accuracy reported for the run
#     'wallclock':  How many seconds the job took to run
def processOutput(log):

    command = None
    accuracy = None
    wallclock = None

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

        if re.match('Accuracy:', line):
            if accuracy == None:
                lArray = line.split()
                accuracy = float(lArray[1].strip())
                print 'Found accuracy = %f' % accuracy
            else:
                raise Exception('Multiple accuracy lines found')
                
        if re.match('Run length:', line):
            if wallclock == None:
                lArray = line.split()
                wallclock = float(lArray[2].strip())
                print 'Found wallclock = %f' % wallclock
            else:
                raise Exception('Multiple time-elapsed lines found')

        lineCount += 1

    # Make exact zero instead be a very tiny number, to avoid divide-by-zero
    # calculations
    if accuracy == 0.0 or accuracy == None:   accuracy = 0.000000001
    if wallclock == 0.0 or wallclock == None:  wallclock = 0.000000001

    return {'command': command,
            'accuracy': accuracy,
            'wallclock': wallclock}

# End: processOutput


#def writeJenkinsDescription(refResults, ngResults, fileName):
def writeJenkinsDescription(ngResults, fileName):

    print 'Jenkins description written to %s' % fileName

    try: 

        fOut = open( fileName, 'w')

        #refAccPercent = float(refResults['accuracy']) * 100.0
        #ngAccPercent = float(ngResults['accuracy']) * 100.0

        #fOut.write( 'MNIST-MLP accuracy - ref: %5.2f%%, ngraph: %5.2f%%, delta %4.2f; ngraph %4.2fx slower; %d steps'
        #            % (refAccPercent, ngAccPercent,
        #               abs(refAccPercent - ngAccPercent),
        #               (ngResults['wallclock']/refResults['wallclock'])))
        fOut.write( 'MNIST-MLP accuracy - ngraph: %5.2f%%; ngraph speed %4.2f'
                    % (ngResults['accuracy'], ngResults['wallclock']))

        fOut.close()

    except Exception as e:
        print 'Unable to write Jenkins description file - %s' % e

# End: writeJenkinsDescription()
