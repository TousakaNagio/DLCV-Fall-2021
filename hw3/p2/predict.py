import torch

from transformers import BertTokenizer
from PIL import Image
import argparse

from models import caption
from datasets import coco, utils
from configuration import Config
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template

def show_mask_on_image(img, mask):
    mask = skimage.transform.pyramid_expand(mask, upscale=min(img.size[0] // mask.shape[0], img.size[1] // mask.shape[1]), sigma=4)
    mask = skimage.transform.resize(mask, (img.size[1], img.size[0]))
    mask = mask / np.amax(mask)
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * (1 - mask)), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

@torch.no_grad()
def evaluate(image, caption, cap_mask):
    model.eval()
    attn_maps = []
    for i in range(config.max_position_embeddings - 1):
        predictions, attn_map = model(image, caption, cap_mask)
        # print(type(attn_map))
        attn_maps.append(attn_map)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 102:
            return caption, attn_maps

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return (caption, attn_maps)


def main(path, pic, config, dir_path):

    image = Image.open(path)
    image = coco.val_transform(image)
    image = image.unsqueeze(0)

    caption, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings)

    output, attn_maps = evaluate(image, caption, cap_mask)
    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)

    text = ['<start>'] + result.capitalize().split() + ['<end>']
    print(text)
    plt.figure(figsize = (50, 50))
    img = Image.open(path).convert('RGB')

    size = 19
    if pic.split('.')[0] == 'bike' or pic.split('.')[0] == 'sheep':
      size = 19
    elif pic.split('.')[0] == 'girl' or pic.split('.')[0] == 'umbrella':
      size = 13
    else:
      size = 14

    for i in range(len(text)):
        plt.subplot(len(text) // 5 + 1, 5, i + 1)
        plt.text(0, 1, '%s' % (text[i]), color='black', backgroundcolor='white', fontsize=45)
        if i == 0:
            plt.imshow(img)
            continue
        mask = np.mean(np.array([sa.cpu().numpy()[:,i - 1,:].reshape((size, 19)) for sa in attn_maps]), axis = 0)
        plt.imshow(show_mask_on_image(img, mask))
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, path.split('/')[-1].split('.')[0]+'.png'), bbox_inches = 'tight', pad_inches = 0)

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Image Captioning')
  parser.add_argument('--path', type=str, help='path to image', required=True)
  parser.add_argument('--output_path', type=str, help='path to image', required=True)
  args = parser.parse_args()
  image_path = args.path
  os.makedirs(args.output_path, exist_ok = True)

  config = Config()


  model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
  end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

  for pic in os.listdir(args.path):
      main(os.path.join(args.path, pic), pic, config, args.output_path)

# thanks to my collaborator b08902134 曾揚哲