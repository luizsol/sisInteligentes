# PSI3472 Concepção e Implementação de Sistemas Eletrônicos Inteligentes
#### 1o exercício-programa
#### Aluno: Luiz Eduardo Sol (8586861)

### 1) Faça um programa "extrai" que lê o vídeo vid4.avi e extrai dez quadros distribuídos ao longo do vídeo: quad0.png, quad1.png, ..., quad9.png. Vamos usar os quadros extraídos para testar o resto do processamento.
>extrai vid4.avi quad => gera quad0.png ... quad9.png



### Códigos-fonte:

As classes e módulos desse EP estão estruturados da seguinte forma:
```shell
src
├── container.py
├── image_processor.py
└── video_processor.py
```

E estes são os conteúdos desses arquivos:
##### `container.py`
```python
# -*- coding: utf-8 -*-
"""A module to implement some basic image manipulations."""
__author__ = "Luiz Sol"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Luiz Sol"
__email__ = "luizedusol@gmail.com"

import copy

import cv2
import numpy as np


class Container(object):
    """A class to implement some basic image manipulations.

    Attributes:
        image (str): The image stored by the object.
    """

    def __init__(self, image=None):
        """Inits ImageProcessor with an optional image."""
        if image is not None:
            self.image = image
        else:
            self._image = None

    @property
    def image(self):
        """The image getter.

        The true image object is incapsulated on the _image attribute,
        therefore the user must use the image virtual attribute to interact
        with image stored in this class
        """
        return self._image

    @image.setter
    def image(self, value):
        """The image setter.

        Args:
            value: either a numpy.ndarray containing a image or the path to an
                image.

        Raises:
            TypeError: The value is neither an image nor the path to an image.
        """
        if isinstance(value, np.ndarray):
            self._image = value
        elif isinstance(value, str):
            self._image = cv2.imread(value, cv2.IMREAD_UNCHANGED)
        else:
            raise TypeError

    def has_image(self):
        """Determines whether an image is already stored."""
        if self.image is None:
            return False
        return True

    def lines(self):
        """Determines the number of lines of the image."""
        return self.image.shape[0]

    def cols(self):
        """Determines the number of columns of the image."""
        return self.image.shape[1]

    def colors(self):
        """Determines the number of colors of the image."""
        shape = self.image.shape
        if len(shape) == 2:
            return 1
        return shape[2]

    def copy(self):
        """Genetates a deep copy of this class instance."""
        return copy.deepcopy(self)

    def show(self, window_name='image'):
        """Displays the current image.

        Keyword Args:
            window_name (='image'): the name of the window in which the image
                will be displayed
        """
        cv2.imshow(window_name, self.image)

    def save(self, path='image_processor_output.bmp'):
        """Saves the current image on the file system.

        Keyword Args:
            path (='image_processor_output'): the path where the file should be
                saved
        """
        cv2.imwrite(path, self.image)

    def get_grayscale(self):
        """Gets the Grayscale version of the image."""
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def apply_grayscale(self):
        """Substitutes the current image for it's Grayscale version."""
        self.image = self.get_grayscale()

    def show_grayscale(self, window_name='grayscale'):
        """Shows the current image's Grayscale version.

        Keyword Args:
            window_name (='grayscale'): the name of the window in which the
                image will be displayed
        """
        cv2.imshow(window_name, self.get_grayscale())

    def get_grad(self, kernel_size=5, data_type=cv2.CV_32F):
        """Retrieves the image's complex matrix of the storaged image.

        Keyword Args:
            kernel_size (=5): The kernel size of the Scharr filter.
            data_type (=cv2.CV_32F): The data type of the image elements
        """
        grad_x = (1.0 / 4080) * cv2.Scharr(self.image, data_type, 1, 0,
                                           kernel_size)

        grad_y = (1.0 / 4080) * cv2.Scharr(self.image, data_type, 0, 1,
                                           kernel_size)

        return grad_x + (0 + 1j) * grad_y

    def get_grad_img(self, kernel_size=5, data_type=cv2.CV_32F):
        """Generates an image of this image's gradient.

        Keyword Args:
            kernel_size (=5): The kernel size of the Scharr filter.
            data_type (=cv2.CV_32F): The data type of the image elements
        """
        grad = self.get_grad(kernel_size=kernel_size, data_type=data_type)

        lines = self.lines()
        cols = self.cols()
        n_grad = np.zeros((lines, cols, 3), np.uint8)

        for line in range(0, lines):
                for col in range(0, cols):
                    g_sum = np.sum(grad[line, col])
                    g_abs = np.abs(g_sum) / 3.0
                    x = np.uint8(255 * g_abs * np.cos(g_sum))
                    y = np.uint8(255 * g_abs * np.sin(g_sum))
                    n_grad[line, col, 0] = x
                    n_grad[line, col, 2] = y

        return n_grad

    def show_grad(self, kernel_size=5, data_type=cv2.CV_32F,
                  window_name='grad'):
        """Displays the current image's gradient.

        Keyword Args:
            kernel_size (=5): The kernel size of the Scharr filter.
            data_type (=cv2.CV_32F): The data type of the image elements
            window_name (='grad'): the name of the window in which the image
                will be displayed
        """
        cv2.imshow(window_name, self.get_grad(kernel_size=kernel_size,
                   data_type=data_type))

        cv2.waitKey()

    def get_gaussian_blur(self, kernel_height=5, kernel_width=5,
                          data_type=cv2.CV_32F):
        """Retrieves the result of the gaussian blur of the storaged image.

        Keyword Args:
            kernel_height (=5): The kernel height of the Gaussian filter.
            kernel_width (=5): The kernel width of the Gaussian filter.
            data_type (=cv2.CV_32F): The data type of the image elements
        """
        return cv2.GaussianBlur(self.image, (kernel_height, kernel_width),
                                data_type)

    def apply_gaussian_blur(self, kernel_height=5, kernel_width=5,
                            data_type=cv2.CV_32F):
        """Applies the Gaussian Blur on the storaged image.

        Keyword Args:
            kernel_height (=5): The kernel height of the Gaussian filter.
            kernel_width (=5): The kernel width of the Gaussian filter.
            data_type (=cv2.CV_32F): The data type of the image elements
        """
        self.image = self.get_gaussian_blur(kernel_height=kernel_height,
                                            kernel_width=kernel_width,
                                            data_type=data_type)

    def get_canny(self, threshold_1=100, threshold_2=200):
        """Retrieves the result of the Canny filter of the storaged image.

        Keyword Args:
            threshold_1 (=100): The lower threshold of the filter.
            threshold_2 (=200): The higher threshold of the filter.
        """
        return cv2.Canny(self.image, threshold_1, threshold_2)

    def apply_canny(self, threshold_1=100, threshold_2=200):
        """Applies the the Canny filter to the storaged image.

        Keyword Args:
            threshold_1 (=100): The lower threshold of the filter.
            threshold_2 (=200): The higher threshold of the filter.
        """
        self.image = self.get_canny(threshold_1, threshold_2)

    def show_canny(self, threshold_1=100, threshold_2=200,
                   window_name='canny'):
        """Displays the current image's Canny.

        Keyword Args:
            threshold_1 (=100): The lower threshold of the filter.
            threshold_2 (=200): The higher threshold of the filter.
            window_name (='canny'): the name of the window in which the image
                will be displayed
        """
        cv2.imshow(window_name, self.get_canny(threshold_1, threshold_2))

    def __repr__(self):
        """The representation dunder of the class."""
        if self.has_image():
            return "Container(np.array)"
        else:
            return "Container(None)"

```

##### `image_processor.py`
```python
# -*- coding: utf-8 -*-
"""A module to implement some more complex image manipulations."""
__author__ = "Luiz Sol"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Luiz Sol"
__email__ = "luizedusol@gmail.com"


import cv2
import numpy as np


class ImageProcessor(object):
    """A class responsible for processing images."""

    def __init__(self):
        pass

    def get_circle_hough_transform_by_grad(self, container, radius,
                                           threshold=0):
        """Uses an image's gradient to determine the existence of a given
        circle in it by means of the hough transform.

        Args:
            container: a Container object containing the image to be analized.
            radius: the radius of the circle to be searched for

        Keyword Args:
            threshold (=0): The minimum absolute value of the gradient to be
                taken into consideration
        """
        channels = container.colors()

        if channels == 1:
            data_type = type(container.image[0][0])
        else:
            data_type = type(container.image[0][0][0])

        result = np.zeros(container.image.shape[:2], data_type)

        grad = container.get_grad()

        lines = container.lines()
        cols = container.cols()

        point_value = 1

        if channels == 1:
            grad.resize((lines, cols, 1), refcheck=False)

        for channel in range(0, channels):
            for line in range(0, lines):
                for col in range(0, cols):
                    value = grad[line][col][channel]
                    if abs(value) > threshold:
                        dx = int(np.cos(np.angle(value)) * radius)
                        dy = int(np.sin(np.angle(value)) * radius)
                        if ((line + dy) < lines and (line + dy) > 0) and \
                           ((col + dx) < cols and (col + dx) > 0):
                            result[line + dy, col + dx] += \
                                + data_type(point_value)

                        if ((line - dy) < lines and (line - dy) > 0) and \
                           ((col - dx) < cols and (col - dx) > 0):
                            result[line - dy, col - dx] += \
                                + data_type(point_value)

        return result

    def detect_circle(self, container, radius, threshold=44,
                      data_type=cv2.CV_32F):
        """Uses an image's resulting hough circle transform to determine the
        existence of a circle in it.

        Args:
            container: a Container object containing the image to be analized.
            radius: the radius of the circle to be searched for

        Keyword Args:
            threshold (=44): The minimum absolute value of a point to be
                be considered a circle center
            data_type (=cv2.CV_32F): The data type of the image elements
        """
        points = self.get_circle_hough_by_grad(container.image, radius,
                                               threshold=0)

        points = cv2.GaussianBlur(points, (5, 5), data_type)
        center = np.unravel_index(points.argmax(), points.shape)
        if points[center] > threshold:
            return center
        else:
            return None

    def detect_tyre(self, container, min_radius=50, max_radius=70,
                    show_text=True):
        """Uses an image's resulting hough circle transform to determine and
        highlight the existence of a tyre in it.

        Args:
            container: a Container object containing the image to be analized.

        Keyword Args:
            min_radius (=50): the minimum radius of the tyre to be searched for
            max_radius (=70): the maximu radius of the tyre to be searched for
            show_text (=True): whether information about the detected circle
                should be added to the image
        """
        for radius in range(min_radius, max_radius):
            center = self.detect_circle(container, radius)
            if center is not None:
                result = container.copy()
                cv2.circle(result.image, (center[1], center[0]), radius,
                           (0, 255, 255), 3)

                font = cv2.FONT_HERSHEY_SIMPLEX
                if show_text:
                    cv2.putText(result.image, 'Center: ' + str(center),
                                (3, 15), font, 0.5, (255, 128, 128), 1,
                                cv2.LINE_AA)

                    cv2.putText(result.image, 'Radius: ' + str(radius),
                                (3, 30), font, 0.5, (255, 128, 128), 1,
                                cv2.LINE_AA)

                return result

        return None
```

#### `video_processor.py`
```python
# -*- coding: utf-8 -*-
"""A module to implement some basic video manipulations."""
__author__ = "Luiz Sol"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Luiz Sol"
__email__ = "luizedusol@gmail.com"


import cv2


def get_frames(video_path, output_folder):
    """Retrieves and saves the frames from a video.

    Args:
        video: the path to the video to be analized.
        output_folder: the path to the folder into which the resulting frames
            will be stored.

    Source: https://stackoverflow.com/questions/33311153/python-extracting- \
        and-saving-video-frames
    """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        cv2.imwrite("%s/quad%d.png" % (output_folder, count), image)
        count += 1
```