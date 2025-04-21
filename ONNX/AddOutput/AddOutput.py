import onnx
import onnx.helper as helper

def add_new_output(onnx_path):
    model = onnx.load(onnx_path)

    # 找到要获取其输出的节点
    target_node = None
    for node in model.graph.node:
        if node.name == '/linear3/Gemm':
            target_node = node
            break
    print(target_node.output)

    new_output = helper.make_tensor_value_info('no_sigmoid_output', onnx.TensorProto.FLOAT, [-1, 10])
    dim1 = new_output.type.tensor_type.shape.dim[0]
    dim1.dim_param = "batch_size"
    model.graph.output.extend([new_output]) # 尾插输出,使用时注意输出顺序

    target_node_output = target_node.output[0]
    new_output_node = onnx.helper.make_node(
        op_type='Identity',  # 使用Identity操作作为示例
        inputs=[target_node_output],
        outputs=['no_sigmoid_output'],
        name='no_sigmoid'
    )
    model.graph.node.append(new_output_node)
    onnx.checker.check_model(model)
    
    print("Inputs:")
    for input in model.graph.input:
        print(f"Name: {input.name}, Type: {input.type}")
    print("\nOutputs:")
    for output in model.graph.output:
        print(f"Name: {output.name}, Type: {output.type}")
        
    onnx.save(model, "model_new.onnx")


if __name__ == "__main__":
    onnx_path = "./model.onnx"
    add_new_output(onnx_path)
