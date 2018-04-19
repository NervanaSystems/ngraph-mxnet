#!/bin/bash

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# (Yizhi) This is mainly inspired by the script in apache/spark.
# I did some modificaiton to get it with our project.
#

set -e

if [[ ($# -ne 2) || ( $1 == "--help") ||  $1 == "-h" ]]; then
  echo "Usage: $(basename $0) [-h|--help] <from_artifactId> <to_artifactId>" 1>&2
  exit 1
fi

FROM_ARTIFACT_ID=$1
TO_ARTIFACT_ID=$2

sed_i() {
  perl -p -000 -e "$1" "$2" > "$2.tmp" && mv "$2.tmp" "$2"
}
   
export -f sed_i
 
BASEDIR=$(dirname $0)/..

find "$BASEDIR" -name 'pom.xml' -not -path '*target*' -print \
  -exec bash -c \
  "sed_i 's/(<artifactId)>'$FROM_ARTIFACT_ID'(<\/artifactId>)/\1>'$TO_ARTIFACT_ID'\2/g' {}" \;

# Change assembly including settings
# <includes>
# 	<include>org.apache.mxnet:libmxnet-scala-linux-x86_64-cpu:so</include>
# </includes>
find "$BASEDIR" -name 'assembly.xml' -not -path '*target*' -print \
  -exec bash -c \
  "sed_i 's/(<include>.*mxnet):'$FROM_ARTIFACT_ID'(:.*<\/include>)/\1:'$TO_ARTIFACT_ID'\2/g' {}" \;
