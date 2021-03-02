import grpc
import tritongrpcclient
import cv2
import numpy as np
from argparse import ArgumentParser
from functools import partial
from tritongrpcclient import grpc_service_pb2_grpc
from .InferenceUtils import single_image_normalize


class CustomInferenceServerClient(tritongrpcclient.InferenceServerClient):
    def __init__(self, url, verbose=False):
        super(CustomInferenceServerClient, self).__init__(url, verbose=False)
        channel_opt = [('grpc.max_send_message_length', 10 * 3 * 1024 * 1024),
                       ('grpc.max_receive_message_length', 5 * 1024 * 1024)]
        self._channel = grpc.insecure_channel(url, options=channel_opt)
        self._client_stub = grpc_service_pb2_grpc.GRPCInferenceServiceStub(
            self._channel)
        self._verbose = verbose
        self._stream = None


class TritonHelper:
    def __init__(self, _server_url, _server_port,
                 _model_name, _model_version,
                 _num_inputs, _num_outputs,
                 _is_debug=True):
        self.target_url = '%s:%s' % (_server_url, _server_port)
        try:
            self.triton_client = CustomInferenceServerClient(url=self.target_url,
                                                             verbose=_is_debug)
        except Exception as e:
            raise Exception(f'triton server {self.target_url} {_model_name}connect fail')
        self.model_name = _model_name
        self.model_version = str(_model_version)
        self.numpy_data_type_mapper = {
            np.half.__name__: "FP16",
            np.float32.__name__: "FP32",
            np.float64.__name__: "FP64",
            np.bool.__name__: "BOOL",
            np.uint8.__name__: "UINT8",
            np.int8.__name__: "INT8",
            np.short.__name__: "INT16",
            np.int.__name__: "INT32",
        }
        self.num_inputs = _num_inputs
        self.num_outputs = _num_outputs

    def infer_request(self, *_img_tensor):
        inputs = []
        assert len(_img_tensor) == self.num_inputs, 'infer parameter number error'
        for m_index, m_img_tensor in enumerate(_img_tensor):
            if isinstance(m_img_tensor, np.ndarray):
                assert m_img_tensor.dtype.name in self.numpy_data_type_mapper
                m_infer_input = tritongrpcclient.InferInput(f'INPUT__{m_index}',
                                                            m_img_tensor.shape,
                                                            self.numpy_data_type_mapper[m_img_tensor.dtype.name]
                                                            )
                m_infer_input.set_data_from_numpy(m_img_tensor)
            else:
                raise Exception('Not Implement')
            inputs.append(m_infer_input)
        results = self.triton_client.infer(model_name=self.model_name,
                                           model_version=self.model_version,
                                           inputs=inputs)
        to_return_result = []
        for i in range(self.num_outputs):
            to_return_result.append(results.as_numpy(f'OUTPUT__{i}'))
        return to_return_result


if __name__ == '__main__':
    ag = ArgumentParser()
    ag.add_argument('--url', type=str, required=True, help='triton服务器地址')
    ag.add_argument('--port', type=int, required=True, help='triton服务器grpc端口')
    ag.add_argument('--img_path', type=str, required=True, help='用于测试服务的图片')
    args = ag.parse_args()
    triton_helper = TritonHelper(args.url, args.port, 'DB', 1, 1, 1, False)
    img = cv2.imread(args.img_path)
    origin_h, origin_w = img.shape[:2]
    resized_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    img_transformer = partial(single_image_normalize, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    face_parsing_result = triton_helper.infer_request(img_transformer(resized_img).unsqueeze(0))[0].squeeze(0)
    person_mask = (face_parsing_result != 0).astype(np.uint8) * 255
    resized_mask = cv2.resize(person_mask, (origin_w, origin_h))
    masked_img = cv2.bitwise_and(img, img, mask=resized_mask)
    cv2.imshow('masked_img', masked_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
