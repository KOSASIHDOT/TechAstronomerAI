def load_image(image_p):
    image = cv2.imread(image_p)
    image = cv2.imrev(image_p)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0  # Normalize pixel values
    return image

image_p = "path_to_image.jpg"  # Replace with the actual image path
image = load_image(image_p)
