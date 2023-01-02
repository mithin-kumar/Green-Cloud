import base64
import io
import os
import time
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose
from ts.torch_handler.base_handler import BaseHandler
from extrafiles import Dataset_loader
from model import UNet
# from preprocessing import  split_image


class UNet(BaseHandler):

    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None

    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """

        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)

        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model = UNet(3, 2)
        self.model.model.load_state_dict(
            torch.load(model_dir + '0_checkpoint.pt'))

        self.initialized = True

    def split_image(img, TARGET_SIZE):
        img_tiles = []

        for y in range(0, img.shape[0], TARGET_SIZE):
            for x in range(0, img.shape[1], TARGET_SIZE):
                img_tile = img[y:y + TARGET_SIZE, x:x + TARGET_SIZE]
                if img_tile.shape[0] == TARGET_SIZE and img_tile.shape[1] == TARGET_SIZE:
                    img_tiles.append(img_tile)

        return img_tiles

    def extract_output(model, data_path, save_path, device="cuda"):
        import torch.nn.functional as F
        model.eval()
        img_append = []
        for X in data_path:
            X = X.to(device)
            with torch.no_grad():
                y_pred = model(X)
                logits = F.softmax(y_pred, dim=1)
                aggr = torch.max(logits, dim=1)
                preds = aggr[1].cpu().numpy()  # .flatten()
                # print(preds.shape)
                img_batch = preds.shape[0]
                # print(preds[1][:][:].shape)
                for arr in range(0, img_batch):
                    img_tile = preds[arr][:][:]

                    # appending sub-image
                    if len(img_append) == 0:
                        img_append = img_tile
                    else:
                        img_append = np.append(img_append, img_tile, axis=1)

        return img_append

    def preprocess(self, image):
        img_tiles = self.split_image(image, 512)
        # extraction_set = Dataset_loader(img_tiles)
        # extraction_loader = DataLoader(
        # extraction_set, batch_size=16, shuffle=False, num_workers=2)

        return img_tiles

    def inference(self, model_input):

        appended_image = self.extract_output(
            self.model, model_input, save_path=None, device=self.device)

        return appended_image

    def repatch_image(appended_img, split_count):
        img_h_split = np.hsplit(appended_img, split_count)
        img_repatch = np.vstack(img_h_split)

        return img_repatch

    def postprocess(self, image, output, TARGET_SIZE):
        img_height = image.shape[0]
        img_width = image.shape[1]

        repatched_image = self.repatch_image(output, img_height // TARGET_SIZE)
        return repatched_image

    def handle(self, data, context):

        model_input = self.preprocess(data)

        model_output = self.inference(model_input)

        model_output = self.postprocess(data, model_output, 512)

        cv2.imwrite('reult.png', model_output)
        return model_output
