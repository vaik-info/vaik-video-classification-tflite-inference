from typing import List, Dict, Tuple
import multiprocessing
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite


class TfliteModel:
    def __init__(self, input_saved_model_path: str = None, classes: Tuple = None, num_thread: int = None):
        self.classes = classes
        num_thread = multiprocessing.cpu_count() if num_thread is None else num_thread
        self.__load(input_saved_model_path, num_thread)

    def inference(self, input_image_list: List[np.ndarray]) -> Tuple[List[Dict], np.ndarray]:
        resized_image_array = self.__preprocess_image_list(input_image_list, self.model_input_shape[1:])
        raw_pred = self.__inference(resized_image_array)
        output = self.__output_parse(raw_pred, len(input_image_list)-1)
        return output, raw_pred

    def __load(self, input_saved_model_path: str, num_thread: int):
        self.interpreter = tflite.Interpreter(model_path=input_saved_model_path, num_threads=num_thread)
        self.interpreter.allocate_tensors()
        self.model_input_shape = self.interpreter.get_input_details()[0]['shape']

    def __preprocess_image_list(self, input_image_list: List[np.ndarray],
                                resize_input_shape: Tuple[int, int, int, int]) -> np.ndarray:
        resized_image_list = []
        for input_image in input_image_list:
            resized_image = self.__preprocess_image(input_image, resize_input_shape[1:3])
            resized_image_list.append(resized_image)
        for _ in range(resize_input_shape[0] - (len(input_image_list) % resize_input_shape[0])):
            resized_image_list.append(np.zeros(resize_input_shape[1:], dtype=np.uint8))
        resized_image_array = np.split(np.stack(resized_image_list), len(resized_image_list)//resize_input_shape[0])
        return np.stack(resized_image_array)
    def __preprocess_image(self, input_image: np.ndarray, resize_input_shape: Tuple[int, int]) -> np.ndarray:
        if len(input_image.shape) != 3:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(input_image.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {input_image.dtype}')

        output_image = np.zeros((*resize_input_shape, input_image.shape[2]),
                                dtype=input_image.dtype)
        pil_image = Image.fromarray(input_image)
        x_ratio, y_ratio = resize_input_shape[1] / pil_image.width, resize_input_shape[0] / pil_image.height
        if x_ratio < y_ratio:
            resize_size = (resize_input_shape[1], round(pil_image.height * x_ratio))
        else:
            resize_size = (round(pil_image.width * y_ratio), resize_input_shape[0])
        resize_pil_image = pil_image.resize(resize_size)
        resize_image = np.array(resize_pil_image)
        output_image[:resize_image.shape[0], :resize_image.shape[1], :] = resize_image
        return output_image

    def __inference(self, resized_frames: np.ndarray) -> np.ndarray:
        if len(resized_frames.shape) != 5:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(resized_frames.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {resized_frames.dtype}')
        raw_pred_list = []
        for resized_images in resized_frames:
            self.__set_input_tensor(resized_images)
            self.interpreter.invoke()
            raw_pred = self.__get_output_tensor()[0]
            raw_pred_list.append(raw_pred)
        return np.concatenate(raw_pred_list, axis=0)

    def __output_parse(self, pred: np.ndarray, max_frame_index: int) -> List[Dict]:
        output_dict_list = []
        pred_index = np.argsort(-pred, axis=-1)
        for index in range(pred.shape[0]):
            output_dict = {'score': pred[index][pred_index[index]].tolist(),
                           'label': [self.classes[class_index] for class_index in pred_index[index]],
                           'start_frame': int(index*self.model_input_shape[1]),
                           'end_frame': int(min(index*self.model_input_shape[1]+self.model_input_shape[1], max_frame_index))}
            output_dict_list.append(output_dict)
        return output_dict_list

    def __set_input_tensor(self, images: np.ndarray):
        input_tensor = self.interpreter.tensor(self.interpreter.get_input_details()[0]['index'])()
        input_tensor.fill(0)
        input_images = images.astype(self.interpreter.get_input_details()[0]['dtype'])
        input_tensor[0, :input_images.shape[0], :input_images.shape[1], :input_images.shape[2]] = input_images

    def __get_output_tensor(self) -> List[np.ndarray]:
        output_details = self.interpreter.get_output_details()
        output_tensor = []
        for index in range(len(output_details)):
            output = self.interpreter.get_tensor(output_details[index]['index'])
            scale, zero_point = output_details[index]['quantization']
            if scale > 1e-4:
                output = scale * (output - zero_point)
            output_tensor.append(output)
        return output_tensor