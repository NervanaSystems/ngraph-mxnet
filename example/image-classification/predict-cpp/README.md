# Image Classification Example Using the C Predict API
This is a simple predictor which shows how to use the MXNet C Predict API for image classification with a pre-trained ImageNet model in a single thread and multiple threads.

## Prerequisites

* OpenCV for image processing: `USE_OPENCV` is set to true by default when [building from source](https://mxnet.incubator.apache.org/install/build_from_source.html)

## How to Use this Example

### Download the Model Artifacts
1. You will need the model artifacts for the Inception ImageNet model. You can download these from http://data.mxnet.io/mxnet/models/imagenet/inception-bn/
2. Place them into a `model/Inception/` subfolder, or if not, you will need to edit the source file and update the paths in the Build step.

* [model/Inception/Inception-BN-symbol.json](http://data.mxnet.io/mxnet/models/imagenet/inception-bn/Inception-BN-symbol.json)
* [model/Inception/Inception-BN-0126.params](http://data.mxnet.io/mxnet/models/imagenet/inception-bn/Inception-BN-0126.params)
* [model/Inception/synset.txt](http://data.mxnet.io/mxnet/models/imagenet/synset.txt)

### Build
1. If using a different location for the model artifacts, edit `image-classification-predict.cc` file, and change the following lines to your artifacts' paths:
  ```c
    // Models path for your model, you have to modify it
    std::string json_file = "model/Inception/Inception-BN-symbol.json";
    std::string param_file = "model/Inception/Inception-BN-0126.params";
    std::string synset_file = "model/Inception/synset.txt";
    std::string nd_file = "model/Inception/mean_224.nd";
  ```

2. You may also want to change the image size and channels:
  ```c
    // Image size and channels
    int width = 224;
    int height = 224;
    int channels = 3;
  ```

3. Simply just use our Makefile to build:
  ```bash
  make
  ```

### Run
Run the example by passing it an image that you want to classify. If you don't have one handy, run the following to get one:

  ```bash
  wget https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/Honeycrisp.jpg/1920px-Honeycrisp.jpg
  ```

Then run the `image-classification-predict` program, passing the image as the first argument and the number of threads as the second parameter.

  ```bash
  ./image-classification-predict 1920px-Honeycrisp.jpg 1
  ```

## Tips

* If you don't run it in the MXNet root path, you may need to copy the `lib` folder here.

## Author
* **Xiao Liu**

* E-mail: liuxiao@foxmail.com

* Homepage: [www.liuxiao.org](http://www.liuxiao.org/)

## Thanks
* pertusa (for Makefile and image reading check)

* caprice-j (for reading function)

* sofiawu (for sample model)

* piiswrong and tqchen (for useful coding suggestions)
