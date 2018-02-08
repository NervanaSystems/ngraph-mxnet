Compilation instructions for ngraph and mxnet with ngraph can be found in the ngraph documentation.

Release Notes:
This release enables examples/image_classification/train_mnist.py and examples/rnn/lstm_bucketing.py on CPU. Other models are under development but not fully supported at this time.

Integration testing to date (2/8/2018) has focused on "tests/cpp/*" and "tests/python/unittest/*". Of these tests, we see the following failures.

Ngraph changes the number of nodes in the graph, so the assumptions in this test are no longer valid.
tests/python/unittest/test_module.py::test_monitor

Profiler integration is ongoing but incomplete, so profiler fails
tests/python/unittest/test_profiler.py::test_profiler

The current integration only returns dense arrays, so these tests fail when checking the ouput for sparse tensors.
tests/python/unittest/test_sparse_operator.py::test_elemwise_binary_ops
tests/python/unittest/test_sparse_operator.py::test_sparse_mathematical_core
tests/python/unittest/test_sparse_operator.py::test_sparse_unary_with_numerics

We haven't yet integrated ngraph into the debug string, so memory allocation isn't properly supported and test_zero_prop fails
tests/python/unittest/test_symbol.py::test_zero_prop

Integration testing on other python tests is forthcoming
