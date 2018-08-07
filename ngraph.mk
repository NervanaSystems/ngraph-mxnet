# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

####################################################################################################
# USAGE / OVERVIEW:
####################################################################################################
# Including this .mk file has the following effects:
#
# - If any problem is noticed, calls $(error ...).
#
# - Assuming no error is noticed:
#   - Sets the following variables (perhaps to the empty string):
#     NGRAPH_CFLAGS  - Compiler args needed to build the MXnet-nGraph bridge code.
#     NGRAPH_LDFLAGS - Linker args needed to build the MXnet-nGraph bridge code.
#
#   - In whatever way is appropriate, defines new goals and adds them as
#     dependencies of the 'all' and/or 'clean' targets.
#
#   - Defines a target named 'ngraph' that represents *required* steps for the local build and
#     local installation of nGraph have been completed.  If USE_NGRAPH != 1, the 'ngraph'
#     target does nothing and is trivially satisfied.
#
# Input variables:
#   USE_NGRAPH
#   USE_NGRAPH_DISTRIBUTED
#   USE_NGRAPH_IE
#   NGRAPH_EXTRA_CMAKE_FLAGS
#   NGRAPH_EXTRA_MAKE_FLAGS

#===================================================================================================

# Check for some configuration problems...
ifeq ($(USE_NGRAPH), 1)
  ifeq ($(USE_MKLDNN), 1)
    $(error "Cannot have both USE_NGRAPH=1 and USE_MKLDNN=1: they require different MKLDNN versions.")
  endif
endif

ifneq ($(NGRAPH_DIR),)
  $(warning "WARNING: MXnet's build system ignores the value of NGRAPH_DIR.")
endif

#===================================================================================================

NGRAPH_SRC_DIR := $(ROOTDIR)/3rdparty/ngraph
NGRAPH_BUILD_DIR := $(ROOTDIR)/3rdparty/ngraph/build
NGRAPH_INSTALL_DIR := $(ROOTDIR)/3rdparty/ngraph/install
MXNET_LIB_DIR := $(ROOTDIR)/lib

# The 'clean' target should remove nGraph-related generated files, regardless of whether or not
# the current Make invocation has USE_NGRAPH=1 ...
.PHONY: ngraph_clean
clean: ngraph_clean
ngraph_clean:
	$(RM) -r "$(NGRAPH_BUILD_DIR)"
	$(RM) -r "$(NGRAPH_INSTALL_DIR)"

ifeq ($(USE_NGRAPH), 1)

.PHONY: ngraph
all: ngraph
ngraph:
	mkdir -p "$(NGRAPH_BUILD_DIR)"
	@echo
	@echo ABOUT TO CONFIGURE AND BUILD 3rdparty/ngraph...
	@echo

	@# All of the following commands must run in the same subshell, because we want the 'cd'
	@# directory to be in effect for all of them.  Thus our use of ';' ...
	@if echo "$(NGRAPH_EXTRA_CMAKE_FLAGS)" | grep -q -e '-DCMAKE_INSTALL_PREFIX='; then \
	  echo; \
	  echo "It looks like NGRAPH_EXTRA_CMAKE_FLAGS is specifying a CMAKE_INSTALL_PREFIX value. This is not supported." >&2; \
	  echo; \
	  exit 1; \
	  fi
	cd "$(NGRAPH_BUILD_DIR)"; \
	cmake "$(NGRAPH_SRC_DIR)" \
	  -DCMAKE_INSTALL_PREFIX="$(NGRAPH_INSTALL_DIR)" \
	  $(NGRAPH_EXTRA_CMAKE_FLAGS); \
	$(MAKE) all install $(NGRAPH_EXTRA_MAKE_FLAGS)

	# Copy contents of nGraph's 'lib' directory into MXnet's lib directory, taking care to
	# preserve the relative symlinks used to support Linux shared-object versioning.
	@echo
	@echo COPYING 3rdparty/ngraph -SUPPLIED .SO FILES INTO MXNET LIBS DIRECTORY...
	@echo
	mkdir -p "$(MXNET_LIB_DIR)"
	cd "$(NGRAPH_INSTALL_DIR)/lib"; tar c --to-stdout . | tar x --dereference --directory "$(MXNET_LIB_DIR)"

	@echo
	@echo 3rdparty/ngraph INTERNAL BUILD/INSTALLATION SUCCESSFUL
	@echo


  # Set NGRAPH_CFLAGS ...
  NGRAPH_CFLAGS = \
    "-I$(NGRAPH_INSTALL_DIR)/include" \
    "-I$(ROOTDIR)/src/ngraph" \
    -DMXNET_USE_NGRAPH=1

  ifeq ($(USE_NGRAPH_IE),1)
    NGRAPH_CFLAGS += -DMXNET_USE_NGRAPH_IE=1
  endif

  ifeq ($(USE_NGRAPH_DISTRIBUTED), 1)
    MPI_COMPILE_FLAGS = $(shell mpicxx --showme:compile)
    NGRAPH_CFLAGS += \
      $(MPI_COMPILE_FLAGS) \
      -DMXNET_USE_NGRAPH_DISTRIBUTED=1
  endif

  # These following libraries are part of ngraph, and provide symbols required by nGraph's
  # public-interface header files, BUT are not reported as dependencies in libngraph.so's
  # ELF header.  We enumerate them here so we can make sure they get linked in appropriately.
  NGRAPH_HELPER_LIBS_LDFLAGS_ = \
    -L$(MXNET_LIB_DIR) \
    -lcpu_backend

  # nGraph provides some libraries that may compete with other libraries already installed
  # on the system. This provides the link-flags that, if provided early enough on the static-linker
  # command line, should ensure that the nGraph-supplied version is preferred.
  NGRAPH_COMMON_LIBRARY_LDFLAGS_ = \
    -Wl,-rpath-link=$(MXNET_LIB_DIR_) \
    -L$(MXNET_LIB_DIR) \
    -ltbb \
    -liomp5 \
    -lmklml_intel \
    -lmkldnn

  NGRAPH_LDFLAGS_ = \
    $(NGRAPH_COMMON_LIBRARY_LDFLAGS_) \
    -Wl,-rpath-link=$(MXNET_LIB_DIR) \
    -L$(MXNET_LIB_DIR) \
    -lngraph \
    $(NGRAPH_HELPER_LIBS_LDFLAGS_)

  ifeq ($(USE_NGRAPH_DISTRIBUTED), 1)
    NGRAPH_LDFLAGS_ += $(shell mpicxx --showme:link)
  endif

  # The static-link-time flags for creating .so files that will dynamically link to nGraph's libs
  # at runtime.
  NGRAPH_LDFLAGS_FOR_SHARED_LIBS := \
    $(NGRAPH_LDFLAGS_) \
    -Wl,-rpath='$${ORIGIN}'

  # The static-link-time flags for creating .so files that will dynamically link to nGraph's libs
  # at runtime.
  # We don't specify '-Wl,-rpath' for this usage scenario because MXnet does not places its
  # shared objects in the same directory as its executable programs.
  NGRAPH_LDFLAGS_FOR_PROGS := \
    $(NGRAPH_LDFLAGS_)

else
  #-------------------------------------------------------------------------------------------------
  # USE_NGRAPH != 1
  #-------------------------------------------------------------------------------------------------
  NGRAPH_CFLAGS :=
  NGRAPH_LDFLAGS_FOR_SHARED_LIBS :=
  NGRAPH_LDFLAGS_FOR_PROGS :=

  # The 'ngraph' target is a no-op.
ngraph:

endif

