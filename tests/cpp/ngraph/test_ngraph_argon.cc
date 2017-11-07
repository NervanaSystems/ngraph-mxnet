// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <algorithm>
#include <cmath>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

namespace ngraph
{
    // Simple test for compilation and linking of ngraph-cpp
    TEST(execute, abc)
    {
        auto shape = Shape{2, 2};
        auto A = std::make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto B = std::make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto C = std::make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto rt = std::make_shared<TensorViewType>(element::Float32::element_type(), shape);
        auto f = std::make_shared<Function>((A + B) * C, rt, op::Parameters{A, B, C});

        auto manager = runtime::Manager::get("NGVM");
        auto external = manager->compile(f);
        auto backend = manager->allocate_backend();
        auto cf = backend->make_call_frame(external);

        // Create some tensors for input/output
        auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
        *a = std::vector<float>{1, 2, 3, 4};
        auto b = backend->make_parameterized_tensor_view<element::Float32>(shape);
        *b = std::vector<float>{5, 6, 7, 8};
        auto c = backend->make_parameterized_tensor_view<element::Float32>(shape);
        *c = std::vector<float>{9, 10, 11, 12};
        auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

        (*cf)({a, b, c}, {result});
        ASSERT_EQ((std::vector<float>{54, 80, 110, 144}), result->get_vector());

        (*cf)({b, a, c}, {result});
        ASSERT_EQ((std::vector<float>{54, 80, 110, 144}), result->get_vector());

        (*cf)({a, c, b}, {result});
        ASSERT_EQ((std::vector<float>{50, 72, 98, 128}), result->get_vector());
    }

} // namespace ngraph