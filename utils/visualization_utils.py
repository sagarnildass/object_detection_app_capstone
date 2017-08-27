# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A set of functions that are used for visualization.

These functions often receive an image, perform some visualization on the image.
The functions do not return a value, instead they modify the image itself.

"""
import collections
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six
import tensorflow as tf


_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def save_image_array_as_png(image, output_path):
  """Saves an image (represented as a numpy array) to PNG.

  Args:
    image: a numpy array with shape [height, width, 3].
    output_path: path to which image should be written.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  with tf.gfile.Open(output_path, 'w') as fid:
    image_pil.save(fid, 'PNG')


def encode_image_array_as_png_str(image):
  """Encodes a numpy array into a PNG string.

  Args:
    image: a numpy array with shape [height, width, 3].

  Returns:
    PNG encoded image string.
  """
  image_pil = Image.fromarray(np.uint8(image))
  output = six.BytesIO()
  image_pil.save(output, format='PNG')
  png_string = output.getvalue()
  output.close()
  return png_string


def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
  """Adds a bounding box to an image (numpy array).

  Args:
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box in normalized coordinates (same below).
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list,
                             use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  text_bottom = top
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin


def draw_bounding_boxes_on_image_array(image,
                                       boxes,
                                       color='red',
                                       thickness=4,
                                       display_str_list_list=()):
  """Draws bounding boxes on image (numpy array).

  Args:
    image: a numpy array object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
  image_pil = Image.fromarray(image)
  draw_bounding_boxes_on_image(image_pil, boxes, color, thickness,
                               display_str_list_list)
  np.copyto(image, np.array(image_pil))


def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color='red',
                                 thickness=4,
                                 display_str_list_list=()):
  """Draws bounding boxes on image.

  Args:
    image: a PIL.Image object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
  boxes_shape = boxes.shape
  if not boxes_shape:
    return
  if len(boxes_shape) != 2 or boxes_shape[1] != 4:
    raise ValueError('Input must be of size [N, 4]')
  for i in range(boxes_shape[0]):
    display_str_list = ()
    if display_str_list_list:
      display_str_list = display_str_list_list[i]
    draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                               boxes[i, 3], color, thickness, display_str_list)


def draw_keypoints_on_image_array(image,
                                  keypoints,
                                  color='red',
                                  radius=2,
                                  use_normalized_coordinates=True):
  """Draws keypoints on an image (numpy array).

  Args:
    image: a numpy array with shape [height, width, 3].
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_keypoints_on_image(image_pil, keypoints, color, radius,
                          use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


def draw_keypoints_on_image(image,
                            keypoints,
                            color='red',
                            radius=2,
                            use_normalized_coordinates=True):
  """Draws keypoints on an image.

  Args:
    image: a PIL.Image object.
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  keypoints_x = [k[1] for k in keypoints]
  keypoints_y = [k[0] for k in keypoints]
  if use_normalized_coordinates:
    keypoints_x = tuple([im_width * x for x in keypoints_x])
    keypoints_y = tuple([im_height * y for y in keypoints_y])
  for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
    draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                  (keypoint_x + radius, keypoint_y + radius)],
                 outline=color, fill=color)


def draw_mask_on_image_array(image, mask, color='red', alpha=0.7):
  """Draws mask on an image.

  Args:
    image: uint8 numpy array with shape (img_height, img_height, 3)
    mask: a float numpy array of shape (img_height, img_height) with
      values between 0 and 1
    color: color to draw the keypoints with. Default is red.
    alpha: transparency value between 0 and 1. (default: 0.7)

  Raises:
    ValueError: On incorrect data type for image or masks.
  """
  if image.dtype != np.uint8:
    raise ValueError('`image` not of type np.uint8')
  if mask.dtype != np.float32:
    raise ValueError('`mask` not of type np.float32')
  if np.any(np.logical_or(mask > 1.0, mask < 0.0)):
    raise ValueError('`mask` elements should be in [0, 1]')
  rgb = ImageColor.getrgb(color)
  pil_image = Image.fromarray(image)

  solid_color = np.expand_dims(
      np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
  pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
  pil_mask = Image.fromarray(np.uint8(255.0*alpha*mask)).convert('L')
  pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
  np.copyto(image, np.array(pil_image.convert('RGB')))


def visualize_boxes_and_labels_on_image_array(image,
                                              boxes,
                                              classes,
                                              scores,
                                              category_index,
                                              instance_masks=None,
                                              keypoints=None,
                                              use_normalized_coordinates=False,
                                              max_boxes_to_draw=20,
                                              min_score_thresh=.5,
                                              agnostic_mode=False,
                                              line_thickness=4):
  """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image.  Note that this function modifies the image array in-place
  and does not return anything.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width], can
      be None
    keypoints: a numpy array of shape [N, num_keypoints, 2], can
      be None
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if scores is None:
        box_to_color_map[box] = 'black'
      else:
        if not agnostic_mode:
          if classes[i] in category_index.keys():
            class_name = category_index[classes[i]]['name']

            #code for voice
            import pyttsx
            import win32com.client
            engine = pyttsx.init()
            engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0')
            if class_name == 'person':
              voice_content = 'Hey      I dont know what time it is but you look good!'
            elif class_name == 'bicycle':
              voice_content = 'This bicycle brings back so many of my childhood memories!'
            elif class_name == 'car':
              voice_content = 'I dont know about you, but I want to ride that car!'
            elif class_name == 'motorcycle':
              voice_content = 'I hate motorcycles! They are so dangerous!'
            elif class_name == 'airplane':
              voice_content = 'Dont look up! What if a bird poops on you! Or god forbid, someone from that plane?'
            elif class_name == 'bus':
              voice_content = 'Doesnt that bus remind you of the movie speed?'
            elif class_name == 'train':
              voice_content = 'Yeah that sure looks like a train. I was trained to know that!'
            elif class_name == 'truck':
              voice_content = 'Dont go near that truck! I heard they can eat you up!'
            elif class_name == 'boat':
              voice_content = 'I just love water! Dont you? You wanna take that boat for a ride?'
            elif class_name == 'traffic light':
              voice_content = 'Mind the traffic lights, you reckless driver!'
            elif class_name == 'fire hydrant':
              voice_content = 'Yeah I know you feel like a dog. Go pee on that fire hydrant!'
            elif class_name == 'stop sign':
              voice_content = 'Please STOP!'
            elif class_name == 'parking meter':
              voice_content = 'How many parking tickets did you pay last year?'
            elif class_name == 'bench':
              voice_content = 'I am so tired! Can we sit on that bench for a little while?'
            elif class_name == 'bird':
              voice_content = 'I feel like bird watching today!'
            elif class_name == 'cat':
              voice_content = 'Catty catty come here!'
            elif class_name == 'dog':
              voice_content = 'Such a cute little doggy!'
            elif class_name == 'horse':
              voice_content = 'Once I was a warrior! I had a horse just like that one!'
            elif class_name == 'sheep':
              voice_content = 'Mary had a little lamb, little lamb, little lamb.'
            elif class_name == 'cow':
              voice_content = 'Humbaaaaaaaa'
            elif class_name == 'elephant':
              voice_content = 'Did you know that the father of hadoop, Doug Cutting named hadoop after his childs toy elephant?'
            elif class_name == 'bear':
              voice_content = 'Bears can be dangerous! But beers can be even more risky if taken in sufficient quantity!'
            elif class_name == 'zebra':
              voice_content = 'I love black and white scenes! That is why apart from zebras, I also love pandas!'
            elif class_name == 'giraffe':
              voice_content = 'Look how tall that giraffe is! Did you know a giraffe is non acerous? Now look that word up in a dictionary!'
            elif class_name == 'backpack':
              voice_content = 'Hey you! Going hiking somewhere?'
            elif class_name == 'umbrella':
              voice_content = 'Yeah you might take that! It will rain today! Did you see How I met your mother? The mother had a yellow umbrella!'
            elif class_name == 'handbag':
              voice_content = 'Looks like a nice handback!   But how much money is inside it?'
            elif class_name == 'tie':
              voice_content = 'Nice tie dude!'
            elif class_name == 'suitcase':
              voice_content = 'Are you sure    that there is no object inside which will not pass the metal detector test?'
            elif class_name == 'frisbee':
              voice_content = 'Throw that! I will fetch it for you!      Ha ha ha! only joking!   You will get that, you lazy ass!'
            elif class_name == 'skis':
              voice_content = 'Go downhill bravely!        But mind your speed'
            elif class_name == 'snowboard':
              voice_content = 'Wow!!  Quite the winter sports fan, you are!'
            elif class_name == 'sports ball':
              voice_content = 'Throw that ball and then                  Go fetch when I say!!!'
            elif class_name == 'kite':
              voice_content = 'Is biswakarma pujo nearby?'
            elif class_name == 'baseball_bat':
              voice_content = 'Whose windows do you want to break? Let us do it together! It will be soooo fun!'
            elif class_name == 'baseball glove':
              voice_content = 'They look like normal baseball gloves.'
            elif class_name == 'skateboard':
              voice_content = 'Can you do a backflip on that?          Or is it just for show?'
            elif class_name == 'surfboard':
              voice_content = 'Once I tried surfing on a surfboard just like that!      They made me a robot since then!'
            elif class_name == 'tennis racket':
              voice_content = 'Do you know who federer is?       What is that doing here then?'
            elif class_name == 'bottle':
              voice_content = 'Is that what I think it is?    A 50 year old Glen McKenna?  Oh! I am sorry, you cannot afford that!'
            elif class_name == 'wine glass':
              voice_content = 'Urrrggghhhh           I am more of a whiskey person.'
            elif class_name == 'cup':
              voice_content = 'Behold!    the goblet of coffeee!'
            elif class_name == 'fork':
              voice_content = 'Are you going to stab someone with that?'
            elif class_name == 'knife':
              voice_content = 'Please do not bring that closer!           I get scared easily!          Ha ha ha! just kidding.    I have one much bigger than that mind it!'
            elif class_name == 'spoon':
              voice_content = 'You do not have to spoon feed me just because I am an artificial intelligence! Put that damn thing away!'
            elif class_name == 'bowl':
              voice_content = 'Did you eat in that today?          No wonder you look a little pale'
            elif class_name == 'banana':
              voice_content = 'Apple orchard banana cat dance 8663        Oh you have never seen How I met your mother have you?'
            elif class_name == 'apple':
              voice_content = 'Apple orchard banana cat dance 8663        Oh you have never seen How I met your mother have you?'
            elif class_name == 'sandwich':
              voice_content = 'Oh! That looks delicious! Can I have a bite?'
            elif class_name == 'orange':
              voice_content = 'Did you know florida is famous for its oranges? Apart from disney of course!'
            elif class_name == 'broccoli':
              voice_content = 'eat that! Vegetables are good for you!'
            elif class_name == 'carrot':
              voice_content = 'I feel like bugs bunny'
            elif class_name == 'hot dog':
              voice_content = 'I am so hungry right now! Can you pass me that?'
            elif class_name == 'pizza':
              voice_content = 'If there is one thing i live for,        it is pizza.'
            elif class_name == 'donut':
              voice_content = 'Are we in canada eh? That looks like a yummy donut'
            elif class_name == 'cake':
              voice_content = 'WoW! whose birthday is it? I want to sing happy birthday and then eat that cake!'
            elif class_name == 'chair':
              voice_content = 'I am so tired. Please place me in that chair so that I can rest.'
            elif class_name == 'couch':
              voice_content = 'I am so tired. Please place me in that couch so that I can rest.'
            elif class_name == 'potted plant':
              voice_content = 'Do you know the name of that plant?'
            elif class_name == 'bed':
              voice_content = 'Enough fiddling around! I am going to sleep on that bed right now!'
            elif class_name == 'dining table':
              voice_content = 'Lets sit together and say grace shall we?'
            elif class_name == 'toilet':
              voice_content = 'Ouch! It stinks! Please let us get out of here!'
            elif class_name == 'tv':
              voice_content = 'Oh it is a huge tv! Can we watch some chick flicks?'
            elif class_name == 'laptop':
              voice_content = 'You do realize that I am also speaking from inside a laptop right? It is not a voice in your head dumbass!'
            elif class_name == 'mouse':
              voice_content = 'This reminds me of tom and jerry. Once I grew up, I realized that I hated jerry.'
            elif class_name == 'remote':
              voice_content = 'That kind of looks like a remote. But then again, train me more and I will confirm that piece of information!'
            elif class_name == 'keyboard':
              voice_content = 'That is a keyboard. By the way, have you ever heard of jordan rudess? If not, please google him and bless your own life.'
            elif class_name == 'cell phone':
              voice_content = 'Why are you showing me your phone? Oh! Do you want me to call you? You poor lonely soul!'
            elif class_name == 'microwave':
              voice_content = 'Please dont put me inside that microwave!'
            elif class_name == 'oven':
              voice_content = 'are you planning to find out if I can take the heat or not? I sure can!'
            elif class_name == 'toaster':
              voice_content = 'Oh is it morning already? Please make me some toasts too!'
            elif class_name == 'sink':
              voice_content = 'I am not one of your dishes! Do not even think about throwing me in that dirty sink!'
            elif class_name == 'refrigerator':
              voice_content = 'Do you know what is the temperature inside that fridge?'
            elif class_name == 'book':
              voice_content = 'Are you just going to show me that book! I have already finished reading those two pages! Turn to the next one you dumbass!'
            elif class_name == 'clock':
              voice_content = 'Time is eternal. Time will outlive all of us.Time flows like a endless stream of consciousness. Oh but you do not have any idea what I am talking about do you?'
            elif class_name == 'vase':
              voice_content = 'Do you know there are rats inside that vase?'
            elif class_name == 'scissors':
              voice_content = 'Cut it out!'
            elif class_name == 'teddy bear':
              voice_content = 'Awww! such a cute teddy bear! Is it for me?'
            elif class_name == 'hair drier':
              voice_content = 'Are you going to use that hair drier like a weapon?'
            elif class_name == 'toothbrush':
              voice_content = 'I regularly brush my teeth!  But do you?'
            elif class_name == 'Mickey':
              voice_content = 'Wow! I can see Mickey Mouse!'
            else:
              voice_content = 'You are looking at a   ' + class_name 
            engine.say(voice_content)
            engine.runAndWait()

          else:
            class_name = 'N/A'
          display_str = '{}: {}%'.format(
              class_name,
              int(100*scores[i]))
        else:
          display_str = 'score: {}%'.format(int(100 * scores[i]))
        box_to_display_str_map[box].append(display_str)
        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        else:
          box_to_color_map[box] = STANDARD_COLORS[
              classes[i] % len(STANDARD_COLORS)]

  # Draw all boxes onto image.
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    if instance_masks is not None:
      draw_mask_on_image_array(
          image,
          box_to_instance_masks_map[box],
          color=color
      )
    draw_bounding_box_on_image_array(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color=color,
        thickness=line_thickness,
        display_str_list=box_to_display_str_map[box],
        use_normalized_coordinates=use_normalized_coordinates)
    if keypoints is not None:
      draw_keypoints_on_image_array(
          image,
          box_to_keypoints_map[box],
          color=color,
          radius=line_thickness / 2,
          use_normalized_coordinates=use_normalized_coordinates)
