/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file optimizer_op.cc
 * \brief Optimizer operators
 * \author Junyuan Xie
 */
#include "./optimizer_op-inl.h"
#include "./elemwise_op_common.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(SGDParam);
DMLC_REGISTER_PARAMETER(SGDMomParam);
DMLC_REGISTER_PARAMETER(AdamParam);
DMLC_REGISTER_PARAMETER(RMSPropParam);
DMLC_REGISTER_PARAMETER(RMSPropAlexParam);
DMLC_REGISTER_PARAMETER(FtrlParam);

NNVM_REGISTER_OP(sgd_update)
.describe(R"code(Update function for Stochastic Gradient Descent (SDG) optimizer.

It updates the weights using::

 weight = weight - learning_rate * gradient

If weight is of ``row_sparse`` storage type,
only the row slices whose indices appear in grad.indices are updated::

 for row in gradient.indices:
     weight[row] = weight[row] - learning_rate * gradient[row]

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SGDParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<2, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<2, 1, false, true, false>)
.set_attr<FCompute>("FCompute<cpu>", SGDUpdate<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", SGDUpdateEx<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_arguments(SGDParam::__FIELDS__());

NNVM_REGISTER_OP(sgd_mom_update)
.describe(R"code(Momentum update function for Stochastic Gradient Descent (SDG) optimizer.

Momentum update has better convergence rates on neural networks. Mathematically it looks
like below:

.. math::

  v_1 = \alpha * \nabla J(W_0)\\
  v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
  W_t = W_{t-1} + v_t

It updates the weights using::

  v = momentum * v - learning_rate * gradient
  weight += v

Where the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.

If weight and momentum are both of ``row_sparse`` storage type,
only the row slices whose indices appear in grad.indices are updated (for both weight and momentum)::

  for row in gradient.indices:
      v[row] = momentum[row] * v[row] - learning_rate * gradient[row]
      weight[row] += v[row]

)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SGDMomParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<3, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<3, 1>)
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<3, 1, false, true, false>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2};
  })
.set_attr<FCompute>("FCompute<cpu>", SGDMomUpdate<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", SGDMomUpdateEx<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("mom", "NDArray-or-Symbol", "Momentum")
.add_arguments(SGDMomParam::__FIELDS__());

NNVM_REGISTER_OP(mp_sgd_update)
.describe("Updater function for multi-precision sgd optimizer")
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SGDParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<3, 1>)
.set_attr<nnvm::FInferType>("FInferType", MP_SGD_InferType<2, 1, 3>)
.set_attr<FCompute>("FCompute<cpu>", MP_SGDUpdate<cpu>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2};
  })
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "gradient")
.add_argument("weight32", "NDArray-or-Symbol", "Weight32")
.add_arguments(SGDParam::__FIELDS__());

NNVM_REGISTER_OP(mp_sgd_mom_update)
.describe("Updater function for multi-precision sgd optimizer")
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SGDMomParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<4, 1>)
.set_attr<nnvm::FInferType>("FInferType", MP_SGD_InferType<2, 1, 4>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3};
  })
.set_attr<FCompute>("FCompute<cpu>", MP_SGDMomUpdate<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("mom", "NDArray-or-Symbol", "Momentum")
.add_argument("weight32", "NDArray-or-Symbol", "Weight32")
.add_arguments(SGDMomParam::__FIELDS__());

NNVM_REGISTER_OP(adam_update)
.describe(R"code(Update function for Adam optimizer. Adam is seen as a generalization
of AdaGrad.

Adam update consists of the following steps, where g represents gradient and m, v
are 1st and 2nd order moment estimates (mean and variance).

.. math::

 g_t = \nabla J(W_{t-1})\\
 m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
 v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
 W_t = W_{t-1} - \alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon }

It updates the weights using::

 m = beta1*m + (1-beta1)*grad
 v = beta2*v + (1-beta2)*(grad**2)
 w += - learning_rate * m / (sqrt(v) + epsilon)

If w, m and v are all of ``row_sparse`` storage type,
only the row slices whose indices appear in grad.indices are updated (for w, m and v)::

 for row in grad.indices:
     m[row] = beta1*m[row] + (1-beta1)*grad[row]
     v[row] = beta2*v[row] + (1-beta2)*(grad[row]**2)
     w[row] += - learning_rate * m[row] / (sqrt(v[row]) + epsilon)

)code" ADD_FILELINE)
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr_parser(ParamParser<AdamParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<4, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<4, 1>)
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<4, 1, false, true, false>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3};
  })
.set_attr<FCompute>("FCompute<cpu>", AdamUpdate<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", AdamUpdateEx<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("mean", "NDArray-or-Symbol", "Moving mean")
.add_argument("var", "NDArray-or-Symbol", "Moving variance")
.add_arguments(AdamParam::__FIELDS__());


NNVM_REGISTER_OP(rmsprop_update)
.describe(R"code(Update function for `RMSProp` optimizer.

`RMSprop` is a variant of stochastic gradient descent where the gradients are
divided by a cache which grows with the sum of squares of recent gradients?

`RMSProp` is similar to `AdaGrad`, a popular variant of `SGD` which adaptively
tunes the learning rate of each parameter. `AdaGrad` lowers the learning rate for
each parameter monotonically over the course of training.
While this is analytically motivated for convex optimizations, it may not be ideal
for non-convex problems. `RMSProp` deals with this heuristically by allowing the
learning rates to rebound as the denominator decays over time.

Define the Root Mean Square (RMS) error criterion of the gradient as
:math:`RMS[g]_t = \sqrt{E[g^2]_t + \epsilon}`, where :math:`g` represents
gradient and :math:`E[g^2]_t` is the decaying average over past squared gradient.

The :math:`E[g^2]_t` is given by:

.. math::
  E[g^2]_t = \gamma * E[g^2]_{t-1} + (1-\gamma) * g_t^2

The update step is

.. math::
  \theta_{t+1} = \theta_t - \frac{\eta}{RMS[g]_t} g_t

The RMSProp code follows the version in
http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
Tieleman & Hinton, 2012.

Hinton suggests the momentum term :math:`\gamma` to be 0.9 and the learning rate
:math:`\eta` to be 0.001.

)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<RMSPropParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<3, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<3, 1>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs &attrs) {
    return std::vector<uint32_t>{2};
  })
.set_attr<FCompute>("FCompute<cpu>", RMSPropUpdate<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("n", "NDArray-or-Symbol", "n")
.add_arguments(RMSPropParam::__FIELDS__());

NNVM_REGISTER_OP(rmspropalex_update)
.describe(R"code(Update function for RMSPropAlex optimizer.

`RMSPropAlex` is non-centered version of `RMSProp`.

Define :math:`E[g^2]_t` is the decaying average over past squared gradient and
:math:`E[g]_t` is the decaying average over past gradient.

.. math::
  E[g^2]_t = \gamma_1 * E[g^2]_{t-1} + (1 - \gamma_1) * g_t^2\\
  E[g]_t = \gamma_1 * E[g]_{t-1} + (1 - \gamma_1) * g_t\\
  \Delta_t = \gamma_2 * \Delta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t - E[g]_t^2 + \epsilon}} g_t\\

The update step is

.. math::
  \theta_{t+1} = \theta_t + \Delta_t

The RMSPropAlex code follows the version in
http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.

Graves suggests the momentum term :math:`\gamma_1` to be 0.95, :math:`\gamma_2`
to be 0.9 and the learning rate :math:`\eta` to be 0.0001.
)code" ADD_FILELINE)
.set_num_inputs(5)
.set_num_outputs(1)
.set_attr_parser(ParamParser<RMSPropAlexParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<5, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<5, 1>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3, 4};
  })
.set_attr<FCompute>("FCompute<cpu>", RMSPropAlexUpdate<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("n", "NDArray-or-Symbol", "n")
.add_argument("g", "NDArray-or-Symbol", "g")
.add_argument("delta", "NDArray-or-Symbol", "delta")
.add_arguments(RMSPropAlexParam::__FIELDS__());

NNVM_REGISTER_OP(ftrl_update)
.describe(R"code(Update function for Ftrl optimizer.
Referenced from *Ad Click Prediction: a View from the Trenches*, available at
http://dl.acm.org/citation.cfm?id=2488200.

It updates the weights using::

 rescaled_grad = clip(grad * rescale_grad, clip_gradient)
 z += rescaled_grad - (sqrt(n + rescaled_grad**2) - sqrt(n)) * weight / learning_rate
 n += rescaled_grad**2
 w = (sign(z) * lamda1 - z) / ((beta + sqrt(n)) / learning_rate + wd) * (abs(z) > lamda1)

If w, z and n are all of ``row_sparse`` storage type,
only the row slices whose indices appear in grad.indices are updated (for w, z and n)::

 for row in grad.indices:
     rescaled_grad[row] = clip(grad[row] * rescale_grad, clip_gradient)
     z[row] += rescaled_grad[row] - (sqrt(n[row] + rescaled_grad[row]**2) - sqrt(n[row])) * weight[row] / learning_rate
     n[row] += rescaled_grad[row]**2
     w[row] = (sign(z[row]) * lamda1 - z[row]) / ((beta + sqrt(n[row])) / learning_rate + wd) * (abs(z[row]) > lamda1)

)code" ADD_FILELINE)
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr_parser(ParamParser<FtrlParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<4, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<4, 1>)
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<4, 1, false, true, false>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3};
  })
.set_attr<FCompute>("FCompute<cpu>", FtrlUpdate<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", FtrlUpdateEx<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("z", "NDArray-or-Symbol", "z")
.add_argument("n", "NDArray-or-Symbol", "Square of grad")
.add_arguments(FtrlParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
