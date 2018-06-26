# Library of common classes and functions for validation tests
# Author : Chris L / Lam Nguyen

import sys
import os
import re
import subprocess
import time
import datetime as DT
from random import randint

# ===== checkScript(mnistScript) ========== 
# Function to check if the example/image-classification/train_mnist.py exists.
# Note: download_cifar10() will automatic download data from http://data.mxnet.io/data/cifar10
# Return: Nothing. 
# =========================================
def checkScript(pythonScript):

    if not os.path.isfile(pythonScript):
        raise Exception('WARNING: Script path is not a file: %s'
                        % str(pythonScript))

#End: def checkScript()

# ===== checkMnistData(dataDir)========== 
# Function to check if the Mnist data exists.
# Note: download_mnist() will automatic download data from http://yann.lecun.com/exdb/mnist/
# Return: Nothing. 
# =========================================
def  checkMnistData(dataDir):

    dataFiles = ['t10k-images-idx3-ubyte.gz',
                 't10k-labels-idx1-ubyte.gz',
                 'train-images-idx3-ubyte.gz',
                 'train-labels-idx1-ubyte.gz']

    if dataDir == None:
        raise Exception('Data directory was not specified in TEST_MLP_MNIST_DATA_DIR')

    if not os.path.isdir(dataDir):
        raise Exception('Data directory %s is not actually a directory'
                        % str(dataDir))

    for f in dataFiles:
        if not os.path.isfile(os.path.join(dataDir, f)):
            raise Exception('Data file %s not found in %s'
                            % (f, dataDir))

# End: checkMnistData()             

# ===== checkDataFiles(dataDir)========== 
# Function to check if the data exists.
# Note: download_cifar10() will automatic download data http://data.mxnet.io/data/cifar10/cifar10_val.rec
# Return: Nothing. 
# =========================================
def  checkCifar10DataFiles(dataDir):

    dataFiles = ['cifar10_val.rec',
                'cifar10_train.rec']

    if dataDir == None:
        raise Exception('Data directory was not specified in TEST_RESNET_CIFAR10_DATA_DIR')

    if not os.path.isdir(dataDir):
        raise Exception('Data directory %s is not actually a directory'
                        % str(dataDir))

    for f in dataFiles:
        if not os.path.isfile(os.path.join(dataDir, f)):
            raise Exception('Data file %s not found in %s'
                            % (f, dataDir))

# End: checkMnistData()                          


# ===== writeLogToFile(logArray, fileName)========== 
# Function to write into a log file
# Return: Nothing. 
# =========================================
def writeLogToFile(logArray, fileName):

    print 'Log written to %s' % fileName

    fOut = open(fileName, 'w')
    for line in logArray:  fOut.write('%s\n' % str(line))
    fOut.close()

# End: writeLogToFile()

# ===== writeJsonToFile(jsonString, fileName) ========== 
# Function to write into a log file from a json file.
# Return: Nothing. 
# =========================================
def writeJsonToFile(jsonString, fileName):

    print 'JSON results written to %s' % fileName

    fOut = open(fileName, 'w')
    fOut.write('%s\n' % str(jsonString))
    fOut.close()

# ===== LogAndOutput ========== 
#  Define a LogAndOutput class.
# Return: Nothing. 
# =========================================

class LogAndOutput(object) :

    def __init__(self, logFile=None):

        if logFile == None :
            self.out = None

        else:
            try:
                self.out = open(logFile, 'w')
            except Exception as e:
                raise Exception('Unable to open log-file %s in LogAndOutput() due to exception: %s'
                                % (logFile, e))
        # End: else

    # End: def __init__()


    def  line(self, message=''):

        print('%s' % str(message))
        if self.out != None: self.out.write('%s\n' % str(message))

    # End: def print()

    def  flush( self ):

        sys.stdout.flush()
        if self.out != None: self.out.flush()

# End: LogAndOutput()


def runMnistScript(script=None,          # Script to run
                   dataDirectory=None,   # --data_dir, where MNIST data is
                   python=None,          # Which python to use
                   logID=''):            # Log line prefix

    print
    print 'MNIST script being run with:'
    print '    script:         %s' % str(script)
    print '    dataDirectory:  %s' % str(dataDirectory)
    print '    python:         %s' % str(python)
    print '    logID:          %s' % str(logID)

    if dataDirectory != None:
        optDataDir = '--data_dir %s' % dataDirectory
        print 'Using data directory %s' % dataDirectory
    else: optDataDir = ''

    print 'Setting up run in nGraph environment'
    print("the Python version is: {}".format(os.environ['PYTHON_VERSION_NUMBER']))
    cmd = "python{} {}".format(os.environ['PYTHON_VERSION_NUMBER'],script )
    print("The command for Mnist Script is: {}".format(cmd))

    # Hook for testing results detection without having to run multi-hour
    # FW+Dataset tests
    if (os.environ.has_key('MX_NG_DO_NOT_RUN')
        and len(os.environ['MX_NG_DO_NOT_RUN']) > 0):
        runLog = runFakeCommand(command=cmd, logID=logID)
    else:
        runLog = runCommand(command=cmd, logID=logID)

    return runLog

# End: def runMnistScript()


def runResnetScript(script=None,          # Script to run
                    useNGraph=True,       # False->reference, True->nGraph++
                    dataDirectory=None,   # --data_dir, where MNIST data is
                    trainEpochs=None,          # Epochs to run
                    trainBatchSize=None,        # Batch size for training
                    python=None,          # Which python to use
                    verbose=False,        # If True, enable log_device_placement
                    logID='',
                    trainNumLayers = None,
                    trainNumClasses = None,
                    trainNumExamples=None,
                    trainImageShape = None,
                    trainPadSize = None,
                    trainLr = None,
                    trainLrStepEpochs = None,
                    trainWithNPP = None):            # Log line prefix

    print
    print 'Resnet script being run with:'
    print '    script:         %s' % str(script)
    print '    useNGraph:      %s' % str(useNGraph)
    print '    dataDirectory:  %s' % str(dataDirectory)
    print '    trainEpochs:         %s' % str(trainEpochs)
    print '    trainBatchSize:  %s' % str(trainBatchSize)
    print '    python:         %s' % str(python)
    print '    logID:          %s' % str(logID)
    print '    trainNumLayers:         %s' % str(trainNumLayers)
    print '    trainNumClasses:          %s' % str(trainNumClasses)
    print '    trainNumExamples:          %s' % str(trainNumExamples)   
    print '    trainImageShape:         %s' % str(trainImageShape)
    print '    trainPadSize:          %s' % str(trainPadSize)
    print '    trainLr:         %s' % str(trainLr)
    print '    trainLrStepEpochs:          %s' % str(trainLrStepEpochs)
    print '    trainWithNPP:          %s' % str(trainWithNPP)

    if trainEpochs is None or int(trainEpochs) == 0:
        raise Exception('runResnetScript() called without parameter num_epochs')

    if trainBatchSize is None:
        raise Exception('runResnetScript() called without parameter batch_size')

    #which python
    process = subprocess.Popen(['which','python'], stdout=subprocess.PIPE)
    python_lib, err = process.communicate()
    if (python_lib == ''):
        python_lib = "python"

    # -u puts python in unbuffered mode
    if (trainWithNPP):
        cmd = ("{} {} --network {} --batch-size {} --num-layers {} --num-epochs {} --num-classes {} --num-examples {} --image-shape {} --pad-size {} --lr {} --lr-step-epochs {} --with-nnp".format(python_lib.strip(), script, "resnet", trainBatchSize, trainNumLayers, 
        trainEpochs, trainNumClasses, trainNumExamples, str(trainImageShape).strip(), trainPadSize, trainLr, str(trainLrStepEpochs).strip()))
        print("The Command for Resnet is: {}".format(cmd))
    else:
        cmd = ("{} {} --network {} --batch-size {} --num-layers {} --num-epochs {} --num-classes {} --num-examples {} --image-shape {} --pad-size {} --lr {} --lr-step-epochs {}".format(python_lib.strip(), script, "resnet", trainBatchSize, trainNumLayers,
        trainEpochs, trainNumClasses, trainNumExamples, str(trainImageShape).strip(), trainPadSize, trainLr, str(trainLrStepEpochs).strip()))
        print("The Command for Resnet is: {}".format(cmd))

    # Hook for testing results detection without having to run multi-hour
    # Framework+Dataset tests
    if (os.environ.has_key('MX_NG_DO_NOT_RUN')
        and len(os.environ['MX_NG_DO_NOT_RUN']) > 0):
        runLog = runFakeCommand(command=cmd, logID=logID)
    else:
        runLog = runCommand(command=cmd, logID=logID)

    return runLog

# End: def runResnetScript()


def runBenchmarkScoreScript(script=None,          # Script to run
                   python=None,          # Which python to use
                   logID='',
                   ompNumThreads=None,
                   kmpAff=None):            # Log line prefix

    print
    print 'Benchmark script being run with:'
    print '    script:         %s' % str(script)
    print '    python:         %s' % str(python)
    print '    logID:          %s' % str(logID)

    print 'Setting up run in nGraph environment'
    print("the Python version is: {}".format(os.environ['PYTHON_VERSION_NUMBER']))
    cmd = "export OMP_NUM_THREADS={}; export KMP_AFFINITY={};{} {}".format(ompNumThreads, kmpAff, python_lib.strip(),script )
    print("The command for benchmark_score script is: {}".format(cmd))
    runLog = runCommand(command=cmd, logID=logID)

    return runLog

# End: def runBenchmarkScoreScript()


def runInceptionV4Script(script=None,          # Script to run
                   logID='',
                   ompNumThreads=None,
                   kmpAff=None,
                   kmpBlocktime=None,
                   batchsize=None):            # Log line prefix

    print
    print 'DeepMark script being run with:'
    print '    script:         %s' % str(script)
    print '    logID:          %s' % str(logID)

    print 'Setting up run in nGraph environment'
    print("The Python version is: {}".format(os.environ['PYTHON_VERSION_NUMBER']))
    #cmd = "export OMP_NUM_THREADS={}; export KMP_AFFINITY={};export KMP_BLOCKTIME= {}; numactl -l .{} --network inception-v4 --batch-size {}".format(ompNumThreads, kmpAff, kmpBlocktime,script, batchsize.strip())
    cmd = "OMP_NUM_THREADS={} KMP_AFFINITY={} KMP_BLOCKTIME={} .{} --network inception-v4 --batch-size {}".format(ompNumThreads, kmpAff, kmpBlocktime,script, batchsize.strip())
    print("The command for deepmark script is: {}".format(cmd))
    runLog = runCommand(command=cmd, logID=logID)

    return runLog

# End: def runInceptionV4Script()


def runCommand(command=None,  # Script to run
               logID=''):     # Log line prefix

    print
    print 'Command being run with:'
    print '    command: %s' % str(command)
    print '    logID:   %s' % str(logID)

    quiet = False   # Used for filtering messages, not currently activated
    patterns = [ ]

    log = []

    if command == None:
        raise Exception('runCommand() called with empty command parameter')

    cmd = command

    cmdMsg = 'Command is: "%s"' % str(cmd)
    print cmdMsg
    log.append(cmdMsg)

    sTime = DT.datetime.today()
    subP = subprocess.Popen(cmd,
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    print 'Subprocess started at %s' % str(sTime)
    print 'Subprocess started with PID %d' % subP.pid

    retCode = None
    while True:
        retCode = subP.poll()
        if not retCode == None: break
        line = subP.stdout.readline()
        if line != None:
            log.append(line.strip())
            timeStr = str(DT.datetime.time(DT.datetime.now()))
            if not quiet:
                sys.stdout.write('%s%s: %s\n' % (timeStr, logID, line.strip()))
            else:
                if any(re.match(regex, line) for regex in patterns):
                    sys.stdout.write('%s%s: %s\n'
                                     % (timeStr, logID, line.strip()))
            sys.stdout.flush()

    eTime = DT.datetime.now()
    print 'Subprocess completed at %s' % str(eTime)
    elapsed = timeElapsedString(sTime,eTime)
    log.append(elapsed)
    print(elapsed)

    subP = None  # Release the subprocess Popen object

    if retCode != 0:
        print('ERROR: Subprocess (%s) returned non-zero exit code %d'
              % (cmd, retCode))
    else:
        print('Subprocess returned exit code %d' % retCode)

    assert retCode == 0  # Trigger a formal assertion

    return log

# End: def runCommand()

## Need to look for the documents.
def runFakeCommand(command=None, logID=''):

    print('')
    print('Fake command being run, to test testing infrastructure:')
    print('    logID: %s' % str(logID))
    print('')
    print('Would have run:')
    print('    command: %s' % str(command))

    sTime = DT.datetime.today()
    print('Fake command started at %s' % str(sTime))

    # Sleep a random amount, so we can test 
    time.sleep(randint(5,15))

    eTime = DT.datetime.now()
    print('Fake command completed at %s' % str(eTime))
    elapsed = timeElapsedString(sTime, eTime)
    print(elapsed)

    return(['Fake log',
            'Nothing run',
            "{'loss': 15.762859, 'global_step': 391, 'accuracy': 0.1045}",
            elapsed])


def timeElapsedString(startTime, endTime):

    timeElapsed = endTime - startTime

    return('Run length: %s seconds (%s)'
           % (timeElapsed.total_seconds(), str(timeElapsed)))
