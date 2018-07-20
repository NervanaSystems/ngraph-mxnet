# Test intended to be run using pytest
#
# If pytest is not installed on your server, you can install it in a virtual
# environment:
#
# Set up a virtual environment
#   Python 2: virtualenv -p python2.7 .venv && . .venv/bin/activate
#   Python 3: python3 -m venv .venv && . .venv/bin/activate
#   $ pip -U pytest
#   $ pytest test_resnet_cifar10_daily_validation.py
#   $ deactivte
#
# This test has no command-line parameters, as it is run via pytest.
# This test does have environment variables that can alter how the run happens:
#
#     Parameter              Purpose & Default (if any)
#
#     TEST_MX_NG_RESNET_NUM_LAYERS    Number of layers for resnet network; default=101                             
#     TEST_MX_RESNET_NUM_CLASSES   Number of classes for resnet network; default=10
#     TEST_MX_NG_RESNET_NUM_EXAMPLES  Number of examples for resnet network; default=50000
#     TEST_MX_NG_RESNET_IMAGE_SHAPE   Image shape; default= '3,28,28'
#     TEST_MX_NG_RESNET_PAD_SIZE      Pad size; default= 4
#     TEST_MX_NG_RESNET_BATCH_SIZE    Batch size; default= 128
#     TEST_RESNET110_CIFAR10_EPOCHS    Number of epochs; default = ?
#     TEST_MX_NG_RESNET_LR            Learning rate; default = .05
#     TEST_MX_NG_RESNET_LR_STEP_EPOCHS Learning Step Epochs; default '200,250'
#     TEST_MX_NG_RESNET_WITH_NNP       Using NNP transformer; default = False
#     TEST_RESNET_CIFAR10_LOG_DIR     Optional: directory to write log files to
#     TEST_RESNET_CIFAR10_DATA_DIR    Directory where CIRAF10 datafiles are located

# JUnit XML files can be generated using pytest's command-line options.
# For example:
#
#     $ pytest -s ./test_resnet_cifar10_daily_validation.py --junit-xml=../validation_tests_resnet_cifar10.xml --junit-prefix=daily_validation_resnet_cifar10
#

import sys
import os
import re

import lib_validation_testing as VT

## RESNET : Parameters and its default values
#network = 'resnet'
#num_layers = 110
#num_classes = 10
#num_examples = 50000
#image_shape = '3,28,28'
#pad_size = 4
#batch_size = 128
#num_epochs = 1
#lr = .05
#lr_step_epochs = '200,250'
#--with-nnp : False

# Acceptable accuracy
if (os.environ.get('TEST_MX_NG_RESNET_ACCEPTABLE_ACCURACY') != ''):
    acceptableAccuracy = os.environ.get('TEST_MX_NG_RESNET_ACCEPTABLE_ACCURACY')  # 1.0%, delta must be calculated from percentages
else:
    acceptableAccuracy = 1

# Num Layers
if (os.environ.get('TEST_MX_NG_RESNET_NUM_LAYERS') != ''):
    trainNumLayers = int(os.environ.get('TEST_MX_NG_RESNET_NUM_LAYERS'))
else:
    trainNumLayers = 110

# Num Classes
if (os.environ.get('TEST_MX_RESNET_NUM_CLASSES') != ''):
    trainNumClasses = int(os.environ.get('TEST_MX_RESNET_NUM_CLASSES'))
else:
    trainNumClasses = 10

# Num Examples
if (os.environ.get('TEST_MX_NG_RESNET_NUM_EXAMPLES') != ''):
    trainNumExamples = int(os.environ.get('TEST_MX_NG_RESNET_NUM_EXAMPLES'))
else:
    trainNumExamples = 50000

# IMAGE_SHAPE 
if (os.environ.get('TEST_MX_NG_RESNET_IMAGE_SHAPE') != ''):
    trainImageShape = os.environ.get('TEST_MX_NG_RESNET_IMAGE_SHAPE')
else:
    trainImageShape = '3,28,28'

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
if (os.environ.get('TEST_RESNET110_CIFAR10_EPOCHS') != ''):
    trainEpochs = int(os.environ.get('TEST_RESNET110_CIFAR10_EPOCHS').strip())
    print ("===== TEST_RESNET110_CIFAR10_EPOCHS : {}".format(trainEpochs))
else:
    trainEpochs = 1

# LR
if (os.environ.get('TEST_MX_NG_RESNET_LR') != ''): 
    trainLr = float(os.environ.get('TEST_MX_NG_RESNET_LR').strip())
else:
    trainLr = 0.05

# Learning Step Epochs
if (os.environ.get('TEST_MX_NG_RESNET_LR_STEP_EPOCHS') != ''): 
    trainLrStepEpochs = os.environ.get('TEST_MX_NG_RESNET_LR_STEP_EPOCHS').strip()
else:
    trainLrStepEpochs = '200,250'

# With NNP 
trainWithNPP = os.environ.get('TEST_MX_NG_RESNET_WITH_NNP')

# Relative path (from top of repo) to mnist_softmax_xla.py script
resnetCifar10ScriptPath = 'example/image-classification/train_cifar10.py'

# Python program to run script.  This should just be "python" (or "python2"),
# as the virtualenv relies on PATH resolution to find the python executable
# in the virtual environment's bin directory.
pythonProg = 'python'

def test_resnet_cifar10_daily_validation():
    print("TEST_RESNET_CIFAR10_LOG_DIR = {}".format(os.environ.get('TEST_RESNET_CIFAR10_LOG_DIR')))
    script = os.path.join(os.environ.get('TEST_RESNET_CIFAR10_LOG_DIR'), resnetCifar10ScriptPath)
    VT.checkScript(script)
    TEST_CIFAR10_DATA_DIR=os.environ.get('TEST_RESNET_CIFAR10_LOG_DIR', None)

    dataDir = os.environ.get('TEST_CIFAR10_DATA_DIR', None)

    # Run with NGraph CPU backend, saving timing and accuracy
    ngraphLog = VT.runResnetScript(logID=' nGraph',
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

    if os.environ.has_key('TEST_RESNET_CIFAR10_LOG_DIR'):
        lDir = os.path.abspath(os.environ['TEST_RESNET_CIFAR10_LOG_DIR'])
        VT.writeLogToFile(ngraphLog,
                          os.path.join(lDir, 'test_resnet_cifar10_cpu_ngraph.log'))
        # Write Jenkins description, for quick perusal of results
        #writeJenkinsDescription(tmp_refAccPercent, ngraphResults,
        #                        os.path.join(lDir,
        #                           'test_resnet_cifar_jenkins.log'))
        writeJenkinsDescription(ngraphResults, os.path.join(lDir,'test_resnet_cifar10_cpu_jenkins_oneline.log'), trainEpochs)

    print("----- RESNET_CIFAR10 Testing Summary ----------------------------------------")


    summaryLog = None
    if lDir != None:
        summaryLog = os.path.join(lDir, 'test_resnet_cifar10_cpu_summary.log')

    logOut = VT.LogAndOutput(logFile=summaryLog)

    # Report commands
    logOut.line()
    logOut.line("Run with NGraph CPU: {}".format(ngraphResults['command']))

    # Report parameters -- NEED TO DO
    logOut.line()
    logOut.line("Batch size:       {} (fixed)".format(trainBatchSize))
    logOut.line("Epoch  :       {} (fixed)".format(trainEpochs))
    logOut.line("Data directory:   {}".format(dataDir))
    logOut.line("useNGraph: true")
    logOut.line("Num Layers :       {} (fixed)".format(trainNumLayers))
    logOut.line("NumClasses :       {} (fixed)".format(trainNumClasses))
    logOut.line("Num Examples :       {} (fixed)".format(trainNumExamples))
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

    # Report accuracy
    #deltaAccuracy = abs(float(refAccPercent) - ngAccPercent)
    logOut.line()
    #logOut.line('Run with default CPU accuracy: %7.4f%%' % float(refAccPercent))
    logOut.line("Run with NGraph CPU accuracy: {}".format(ngAccPercent))
    #logOut.line('Accuracy delta: %6.4f%%' % deltaAccuracy)
    #logOut.line('Acceptable accuracy delta is <= %6.4f%%'
    #            % float(acceptableAccuracy))

    #print("==========================")
    #print("deltaAccuracy = {}".format(deltaAccuracy))
    #print("==========================")

    #print("==========================")
    #print("acceptableAccuracy = {}".format(acceptableAccuracy))
    #print("==========================")
    # Assert for out-of-bounds accuracy
    #assert deltaAccuracy <= acceptableAccuracy
        
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
                print("Found accuracy = {}".format(accuracy))
            else:
                raise Exception("Multiple accuracy lines found")
                
        if re.match('Run length:', line):
            if wallclock == None:
                lArray = line.split()
                wallclock = float(lArray[2].strip())
                print("Found wallclock = {}".format(float(wallclock)))
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

def writeJenkinsDescription(ngResults, fileName, trainEpochs):
#def writeJenkinsDescription(refResults, ngResults, fileName):

    print("Jenkins description written to {}".format(fileName))

    try: 

        fOut = open( fileName, 'w')

        #refAccPercent = float(refResults['accuracy']) * 100.0
        #ngAccPercent = float(ngResults['accuracy']) * 100.0

        #fOut.write( 'RESNET_CIFAR10 accuracy - ref: %5.2f%%, ngraph: %5.2f%%, delta %4.2f; ngraph %4.2fx slower; %d steps'
        #            % (refAccPercent, ngAccPercent,
        #               abs(refAccPercent - ngAccPercent),
        #               (ngResults['wallclock']/refResults['wallclock'])))
        fOut.write("RESNET-CIFAR10 accuracy - ngraph: {}; ngraph speed ={}; {} steps".format(ngResults['accuracy'], ngResults['wallclock'], trainEpochs))

        fOut.close()

    except Exception as e:
        print("Unable to write Jenkins description file - {}".format(e))

# End: writeJenkinsDescription()
