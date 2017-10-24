

inline NGraphNodePtr transpose(NgraphNodePtr node, ngraph::Shape in_shape){
    if (in_shape.size() != 2) throw;
    auto out_shape = ngraph::Shape({in_shape[1], in_shape[0]});
    return std::make_shared<ngraph::op::Reshape>(
        node, ngraph::AxisVector{1, 0}, out_shape);
}