# Test intended to be run using pytest
#
# If pytest is not installed on your server, you can install it in a virtual
# environment:
#
# Set up a virtual environment
#   Python 2: virtualenv -p python2.7 .venv && . .venv/bin/activate
#   Python 3: python3 -m venv .venv && . .venv/bin/activate
#   $ pip -U pytest
#   $ pytest test_resnet_i1k_daily_validation.py 
#   $ deactivte
#
# This test has no command-line parameters, as it is run via pytest.
# This test does have environment variables that can alter how the run happens:
#
#     Parameter              Purpose & Default (if any)
#
#     TEST_MX_NG_RESNET_NUM_LAYERS    Number of layers for resnet network; default=50                            
#     TEST_MX_RESNET_NUM_CLASSES   Number of classes for resnet network; default=1000
#     TEST_MX_NG_RESNET_NUM_EXAMPLES  Number of examples for resnet network; default=1281167
#     TEST_MX_NG_RESNET_IMAGE_SHAPE   Image shape; default= '3,224,224'
#     TEST_MX_NG_RESNET_PAD_SIZE      Pad size; default= 4
#     TEST_MX_NG_RESNET_BATCH_SIZE    Batch size; default= 128
#     TEST_RESNET_I1K_EPOCHS          Number of epochs; default = 80
#     TEST_MX_NG_RESNET_LR            Learning rate; default = .05
#     TEST_MX_NG_RESNET_LR_STEP_EPOCHS Learning Step Epochs; default '30,60''
#     TEST_MX_NG_RESNET_WITH_NNP       Using NNP transformer; default = False
#     TEST_RESNET_I1K_LOG_DIR     Optional: directory to write log files to
#     TEST_RESNET_I1K_DATA_DIR    Directory where CIRAF10 datafiles are located

# JUnit XML files can be generated using pytest's command-line options.
# For example:
#
#     $ pytest -s ./test_resnet_i1k_daily_validation.py --junit-xml=../validation_tests_resnet_i1k.xml --junit-prefix=daily_validation_resnet_i1k
#

import sys
import os
import re
import subprocess
import lib_validation_testing as VT

# Acceptable accuracy
if (os.environ.get('TEST_MX_NG_RESNET_ACCEPTABLE_ACCURACY') != ''):
    acceptableAccuracy = os.environ.get('TEST_MX_NG_RESNET_ACCEPTABLE_ACCURACY')  # 1.0%, delta must be calculated from percentages
else:
    acceptableAccuracy = 1

# Num Layers
if (os.environ.get('TEST_MX_NG_RESNET_NUM_LAYERS') != ''):
    trainNumLayers = int(os.environ.get('TEST_MX_NG_RESNET_NUM_LAYERS'))
else:
    trainNumLayers = 50

# Num Classes
if (os.environ.get('TEST_MX_RESNET_NUM_CLASSES') != ''):
    trainNumClasses = int(os.environ.get('TEST_MX_RESNET_NUM_CLASSES'))
else:
    trainNumClasses = 1000

# Num Examples
if (os.environ.get('TEST_MX_NG_RESNET_NUM_EXAMPLES') != ''):
    trainNumExamples = int(os.environ.get('TEST_MX_NG_RESNET_NUM_EXAMPLES'))
else:
    trainNumExamples = 1281167

# IMAGE_SHAPE 
if (os.environ.get('TEST_MX_NG_RESNET_IMAGE_SHAPE') != ''):
    trainImageShape = os.environ.get('TEST_MX_NG_RESNET_IMAGE_SHAPE')
else:
    trainImageShape = '3,224,224'

# Pad size
if (os.environ.get('TEST_MX_NG_RESNET_PAD_SIZE') != ''): 
    trainPadSize = int(os.environ.get('TEST_MX_NG_RESNET_PAD_SIZE').strip())
else:
    trainPadSize = 4

# Pad size
if (os.environ.get('TEST_MX_NG_RESNET_BATCH_SIZE') != ''): 
    trainBatchSize = int(os.environ.get('TEST_MX_NG_RESNET_BATCH_SIZE').strip())
else:
    trainBatchSize = 128

# Epochs
if (os.environ.get('TEST_RESNET_I1K_EPOCHS') != ''):
    trainEpochs = int(os.environ.get('TEST_RESNET_I1K_EPOCHS').strip())
    print ("===== TEST_RESNET_I1K_EPOCHS : {}".format(trainEpochs))
else:
    trainEpochs = 1

# LR
if (os.environ.get('TEST_MX_NG_RESNET_LR') != ''): 
    trainLr = float(os.environ.get('TEST_MX_NG_RESNET_LR').strip())
else:
    trainLr = 0.1

# Learning Step Epochs
if (os.environ.get('TEST_MX_NG_RESNET_LR_STEP_EPOCHS') != ''): 
    trainLrStepEpochs = os.environ.get('TEST_MX_NG_RESNET_LR_STEP_EPOCHS').strip()
else:
    trainLrStepEpochs = '30,60'

# With NNP 
trainWithNPP = os.environ.get('TEST_MX_NG_RESNET_WITH_NNP')

resnetI1KScriptPath = 'example/image-classification/train_imagenet.py'

# Python program to run script.  This should just be "python" (or "python2"),
# as the virtualenv relies on PATH resolution to find the python executable
# in the virtual environment's bin directory.
pythonProg = 'python'

def test_resnet_i1k_daily_validation():
    print("TEST_RESNET_I1K_LOG_DIR = {}".format(os.environ.get('TEST_RESNET_I1K_LOG_DIR')))
    script = os.path.join(os.environ.get('TEST_RESNET_I1K_LOG_DIR'), resnetI1KScriptPath)
    VT.checkScript(script)
    # Check if the data exist 
    TEST_I1K_DATA_DIR=os.environ.get('TEST_RESNET_I1K_LOG_DIR', None)
    
    dataDir = os.environ.get('TEST_RESNET_I1K_DATA_DIR', None)
    VT.checkScript("/dataset/mxnet_imagenet/train.rec")
    # Run with NGraph CPU backend, saving timing and accuracy
    ngraphLog = VT.runResnetI1KScript(logID=' nGraph',
                                  script=script,
                                  python=pythonProg,
                                  dataDirectory=dataDir,
                                  trainNumLayers = trainNumLayers,
                                  trainNumClasses = trainNumClasses,
                                  trainNumExamples = trainNumExamples,
                                  trainImageShape = trainImageShape,
                                  trainPadSize = trainPadSize,
                                  trainBatchSize = trainBatchSize,
                                  trainEpochs = trainEpochs,
                                  trainLr = trainLr,
                                  trainLrStepEpochs = trainLrStepEpochs,
                                  trainWithNPP = trainWithNPP)
    ngraphResults = processOutput(ngraphLog)
    
    lDir = None

    ## Need to update the refAccPercent from the paper.
    #tmp_refAccPercent = 3.4

    if os.environ.has_key('TEST_RESNET_I1K_LOG_DIR'):
        lDir = os.path.abspath(os.environ['TEST_RESNET_I1K_LOG_DIR'])
        VT.writeLogToFile(ngraphLog,
                          os.path.join(lDir, 'test_resnet_i1k_cpu_ngraph.log'))
        # Write Jenkins description, for quick perusal of results
        writeJenkinsDescription(ngraphResults, os.path.join(lDir,'test_resnet_i1k_cpu_jenkins_oneline.log'), trainEpochs)

    print("----- RESNET_I1K Testing Summary ----------------------------------------")

    summaryLog = None
    if lDir != None:
        summaryLog = os.path.join(lDir, 'test_resnet_i1k_cpu_summary.log')

    logOut = VT.LogAndOutput(logFile=summaryLog)

    # Report commands
    logOut.line()
    logOut.line('Run with NGraph CPU: %s' % ngraphResults['command'])

    # Report parameters -- NEED TO DO
    logOut.line()
    logOut.line("Batch size:       {} (fixed)".format(trainBatchSize))
    logOut.line("Epoch  :       {} (fixed)".format(trainEpochs))
    logOut.line("Data directory:   {}".format(dataDir))
    logOut.line("useNGraph: true")
    logOut.line("Num Layers :       {} (fixed)".format(trainNumLayers))
    logOut.line("NumClasses :       {} (fixed)".format(trainNumClasses))
    logOut.line("Num Examples :      {} (fixed)".format(trainNumExamples))
    logOut.line("Image Shape :       {} (fixed)".format(str(trainImageShape)))
    logOut.line("Pad Size :       {} (fixed)".format(trainPadSize))
    logOut.line("Lr :       {} (fixed)".format(trainLr))
    logOut.line("Step Epochs:       {} (fixed)".format(str(trainLrStepEpochs)))
    logOut.line("with NNP:       {} (fixed)".format(str(trainWithNPP)))

    #refAccPercent = tmp_refAccPercent * 100.0
    ngAccPercent = float(ngraphResults['accuracy']) * 100.0

    print("==========================")
    print("ngAccPercent = {}".format(ngAccPercent))
    print("==========================")

    logOut.line()
    logOut.line("Run with NGraph CPU accuracy: {}".format(float(ngAccPercent)))
        
    # Report on times
    logOut.line()
    logOut.line("Run with NGraph CPU took: {} seconds".format(ngraphResults['wallclock']))
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
                print("Found command = [{}]".format(command))
            else:
                raise Exception("Multiple command-is lines found")

        if re.match('Accuracy:', line):
            if accuracy == None:
                lArray = line.split()
                accuracy = float(lArray[1].strip())
                print("Found accuracy = {}".format(float(accuracy)))
            else:
                raise Exception('Multiple accuracy lines found')
                
        if re.match('Run length:', line):
            if wallclock == None:
                lArray = line.split()
                wallclock = float(lArray[2].strip())
                print("Found wallclock = {}".format(float(wallclock)))
            else:
                raise Exception("Multiple time-elapsed lines found")

        lineCount += 1

    # Make exact zero instead be a very tiny number, to avoid divide-by-zero
    # calculations
    if accuracy == 0.0 or accuracy == None:   accuracy = 0.000000001
    if wallclock == 0.0 or wallclock == None:  wallclock = 0.000000001

    return {'command': command,
            'accuracy': accuracy,
            'wallclock': wallclock}

# End: processOutput

def writeJenkinsDescription(ngResults, fileName, trainEpochs):
#def writeJenkinsDescription(refResults, ngResults, fileName):

    print("Jenkins description written to {}".format(fileName))

    try: 

        fOut = open( fileName, 'w')
        fOut.write("RESNET-I1K accuracy - ngraph: {}; ngraph speed {}; {} steps".format(ngResults['accuracy'], ngResults['wallclock'], trainEpochs))

        fOut.close()

    except Exception as e:
        print("Unable to write Jenkins description file - {}".format(e))

# End: writeJenkinsDescription()
