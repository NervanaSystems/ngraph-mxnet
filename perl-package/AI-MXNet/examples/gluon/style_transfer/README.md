<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

This directory provides AI::MXNet Implementation of MSG-Net real time style transfer, https://arxiv.org/abs/1703.06953

### Stylize Images Using Pre-trained MSG-Net
Download the pre-trained model:

        ./get_data.sh

Test the model:

        ./style_transfer.pl --content-image <path or url> --style-image < path or url> --content-size 512

More options:

        * --content-image: path or url to content image you want to stylize.
        * --style-image:   path or url to style image.
        * --model:         path to the pre-trained model to be used for stylizing the image if you use your custom model
        * --output-image:  path for saving the output image, default is 'out.jpg'
        * --content-size:  the output image size, default is 512 pixels for the shorter side,
                             decrease the size if your computer is low on RAM and the script fails.

<img title="Pembroke Welsh Corgi Kyuubi is enjoying Total Solar Eclipse of Aug 2017 in Salem, OR"
    alt="Pembroke Welsh Corgi Kyuubi is enjoying Total Solar Eclipse of Aug 2017 in Salem, OR"
    src ="http://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/kyuubi.jpg" width="512px" />
<img title="Style image: Kazimir Malevich, Black Square"
    alt="Style image: Kazimir Malevich, Black Square"
    src ="http://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/kyuubi_blacksquare.jpg" width="512px" />
<img title="Style image: random ornate stone wall image"
    alt="Style image: random ornate stone wall image"
    src ="http://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/kyuubi_mural.jpg" width="512px" />
<img title="Style image: Salvador Dali, The Enigma of Desire"
    alt="Style image: Salvador Dali, The Enigma of Desire"
    src ="http://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/kyuubi_dali.jpg" width="512px" />
<img title="Style image: Vincent van Gogh, The Starry Night"
    alt="Style image: Vincent van Gogh, The Starry Night"
    src ="http://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/kyuubi_starry.jpg" width="512px" />
