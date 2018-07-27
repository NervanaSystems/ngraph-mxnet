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

  # Set NGRAPH_LDFLAGS ...
  NGRAPH_LDFLAGS = \
    "-L$(MXNET_LIB_DIR)" \
    -Wl,-rpath='$${ORIGIN}'

  # This is a workaround for a bug in Gnu's 'ld' linker, in which it fails to properly interpret
  # RPATH=$ORIGIN. The bug is now fixed, but present in Ubuntu 16.04's version of 'ld'.
  NGRAPH_LDFLAGS += \
    "-Wl,-rpath-link=$(MXNET_LIB_DIR)"

  ifeq ($(USE_NGRAPH_IE),1)
    NGRAPH_LDFLAGS += \
      -lngraph \
      -Wl,--as-needed
  else
    NGRAPH_LDFLAGS += \
      -liomp5 \
      -lcpu_backend \
      -lngraph \
      -lmklml_intel \
      -Wl,--as-needed
  endif

  ifeq ($(USE_NGRAPH_DISTRIBUTED), 1)
    MPI_LINK_FLAGS = $(shell mpicxx --showme:link)
    NGRAPH_LDFLAGS += $(MPI_LINK_FLAGS)
  endif
else
  #-------------------------------------------------------------------------------------------------
  # USE_NGRAPH != 1
  #-------------------------------------------------------------------------------------------------
  NGRAPH_CFLAGS=
  NGRAPH_LDFLAGS=

  # The 'ngraph' target is a no-op.
ngraph:

endif

