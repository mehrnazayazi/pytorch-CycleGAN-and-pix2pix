import cv2
import numpy as np
import random
from PIL import Image, ImageDraw
import argparse


def cut_random_pattern_portion(pattern_path):
    # Open the pattern image
    pattern = Image.open(pattern_path)

    # Convert JPEG to RGBA if it's in a different mode
    pattern = pattern.convert("RGBA")

    # Get the size of the pattern image
    pattern_width, pattern_height = pattern.size

    # Randomly select the top-left corner for the 25x256 portion
    x_offset = random.randint(0, pattern_width - 256)
    y_offset = random.randint(0, pattern_height - 256)

    # Cut out the 25x256 portion
    pattern_portion = pattern.crop((x_offset, y_offset, x_offset + 256, y_offset + 256))

    return pattern_portion

def create_random_fading_patterned_circle_image(size):
    x = random.randint(0, size[0])
    y = random.randint(0, size[1])
    radius = random.randint(20, 80)
    transition_length = random.randint(5, 20)*2+1

    pattern_path = "pattern.jpeg"
    cut_pattern = cut_random_pattern_portion(pattern_path).save("cut_pattern.png")
    src = cv2.imread('cut_pattern.png')
    mask = np.zeros_like(src)
    print(mask.shape)
    print(mask.dtype)
    cv2.circle(mask, (x, y), radius, (255, 255, 255), thickness=-1)
    mask_blur = cv2.GaussianBlur(mask, (transition_length, transition_length), 0)
    dst = src * (mask_blur / 255)

    return dst

def create_random_fading_white_circle_image(size):
    x = random.randint(0, size[0])
    y = random.randint(0, size[1])
    radius = random.randint(20, 80)
    transition_length = random.randint(5, 20)*2+1

    pattern_path = "pattern.jpeg"
    cut_pattern = cut_random_pattern_portion(pattern_path).save("cut_pattern.png")
    src = cv2.imread('cut_pattern.png')
    mask = np.zeros_like(src)
    # print(mask.shape)
    # print(mask.dtype)
    cv2.circle(mask, (x, y), radius, (255, 255, 255), thickness=-1)
    mask_blur = cv2.GaussianBlur(mask, (transition_length, transition_length), 0)

    return mask_blur



def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='create data points for cycleGAN training with patterns')

    # Add options to the parser
    parser.add_argument('--train', action='store_true' , help='create training data')
    parser.add_argument('--test',  action='store_true' , help='create test data')
    parser.add_argument('--val',   action='store_true' , help='create validation data')


    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of options
    train = args.train
    test = args.test
    val = args.val

    image_size = (256, 256)

    # Number of images with random circles
    train_num = 1000
    test_num = 200
    Validation_num = 200

    # Create and save multiple images
    if train:
        for image_index in range(train_num):
            dst = create_random_fading_patterned_circle_image(image_size)
            cv2.imwrite(f"../datasets/white2pattern/trainB/0{image_index + 1}.png", dst)
            white = create_random_fading_white_circle_image(image_size)
            cv2.imwrite(f"../datasets/white2pattern/trainA/0{image_index + 1}.png", white)

    if test:
        for image_index in range(test_num):
            dst = create_random_fading_patterned_circle_image(image_size)
            cv2.imwrite(f"../datasets/white2pattern/testB/0{image_index + 1}.png", dst)
            white = create_random_fading_white_circle_image(image_size)
            cv2.imwrite(f"../datasets/white2pattern/testA/0{image_index + 1}.png", white)

    if val:
        for image_index in range(Validation_num):
            dst = create_random_fading_patterned_circle_image(image_size)
            cv2.imwrite(f"../datasets/white2pattern/valA/0{image_index + 1}.png", dst)
            white = create_random_fading_white_circle_image(image_size)
            cv2.imwrite(f"../datasets/white2pattern/valB/0{image_index + 1}.png", white)



if __name__ == "__main__":
    main()